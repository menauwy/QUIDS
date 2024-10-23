import json
import os
import sys
sys.path.append('..')
sys.path.append('.')
from tqdm import tqdm
import random
import logging
import pickle
from transformers import T5Tokenizer
import torch
# import argparse
from sentence_transformers import SentenceTransformer



def rank_sentences_by_similarity(target_embedding, sentence_embeddings):
    # Calculate cosine similarities using torch
    similarities = torch.nn.functional.cosine_similarity(target_embedding, sentence_embeddings)
    ranked_indices = similarities.argsort(descending=True)
    return ranked_indices, similarities[ranked_indices]

def get_hard_negatives(target, pos_docs, neg_docs, device):
    """
    get hard negatives based on similarity with pos_docs
    1.Encode all sentences (positive, negative, and query intent) using RoBERTa with mean pooling or sentence encoder.
    # training: query + intent
    # test: query
    2.Rank the positive sentences based on their cosine similarity to the query intent embedding.
    3.Concatenate the ranked positive sentences into one new sentence and get its embedding.
    4.Rank the hard negative samples based on their cosine similarity to the new positive sentence embedding.
    """
    model = SentenceTransformer("msmarco-bert-base-dot-v5").to(device)

    target_embedding = model.encode(target, convert_to_tensor=True, device=device)
    pos_embeddings = model.encode(pos_docs, convert_to_tensor=True, device=device)

    # Rank positive sentences based on their similarity to the query /intent
    ranked_indices, similarities = rank_sentences_by_similarity(target_embedding, pos_embeddings)

    # Get ranked positive sentences
    ranked_pos_docs = [pos_docs[i] for i in ranked_indices]

    # Concatenate ranked positive sentences into one new sentence
    new_pos_docs = " ".join(ranked_pos_docs)

    # Encode the new positive sentence
    # new_pos_embedding = encode_sentences_with_mean_pooling([new_pos_sentence], tokenizer, model)
    new_pos_embedding = model.encode([new_pos_docs], convert_to_tensor=True, device=device)

    # be aware of the length of neg_docs
    if len(neg_docs) <= 1:
        return ranked_pos_docs, neg_docs
    else: 
        neg_embeddings = model.encode(neg_docs, convert_to_tensor=True, device=device)
        # Rank negative sentences based on their similarity to the new positive sentence
        ranked_indices, similarities = rank_sentences_by_similarity(new_pos_embedding, neg_embeddings)
        ranked_neg_docs = [neg_docs[i] for i in ranked_indices]
        return ranked_pos_docs, ranked_neg_docs

def find_dissimilar_documents(target_embedding, candidates, model, device, top_k=5):
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True, device=device)
    similarities = torch.nn.functional.cosine_similarity(target_embedding, candidate_embeddings)
    # find the top_k most dissimilar documents: acsending order
    ranked_indices = similarities.argsort(descending=False)
    return [candidates[i] for i in ranked_indices[:top_k] if similarities[i].item() < 0.1]  # Adjust threshold as needed

def augment_easy_negatives(orig_hard_neg, orig_record, source_data, device, max_neg_ratio = 1, train = False):
    '''
    Argument the hard negatives to max_neg_ratio by adding similar sentences from the whole training dataset
    orig_hard_neg: the original ranked hard negatives from get_hard_negatives
    orig_record: record contains the query, pos, neg, intent, that need to be argumented
    source_data: source dataset to argument hard negatives
    max_neg_ratio: the max number of hard negatives
    '''
    if len(orig_hard_neg) >= max_neg_ratio:
        return orig_hard_neg
    
    model = SentenceTransformer("msmarco-bert-base-dot-v5").to(device)

    sample_id = orig_record['sample_id']
    if train:
        target_doc = orig_record['query'] + ' ' + orig_record['intent']
    else: # for test data
        target_doc = orig_record['query']
    target_embedding = model.encode(target_doc, convert_to_tensor=True, device=device)

    augmented_easy_neg = orig_hard_neg

    # Randomly sample from the source data and find similar documents
    while len(augmented_easy_neg) < max_neg_ratio:
        sample = random.choice(source_data)
        # skip the same sample
        if sample['sample_id'] == sample_id:
            continue
        candidate_docs = sample['pos'] + sample['neg']
        
        similar_docs = find_dissimilar_documents(target_embedding, candidate_docs, model, device)
        for doc in similar_docs:
            if doc not in augmented_easy_neg and doc not in orig_record['pos']:
                augmented_easy_neg.append(doc)
            if len(augmented_easy_neg) >= max_neg_ratio:
                break
    
    return list(augmented_easy_neg)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    option = 'augment' # 'empty' or 'augment'
    test_origianl = '/home/ywang1/data1/dataset/q2id/Q2ID_data/test_data_original.jsonl'
    output_file = f'/home/ywang1/workspace/IntentRewriter/src/degree_irrelevant/low_degree_{option}.jsonl'
    
    
    with open(test_origianl, 'r') as f:
        test_data = [json.loads(line) for line in f]
    

    with open(output_file, 'w') as f:
        for i, data in enumerate(test_data):
            record = {}
            if len(data['pos']) == 0:
                continue
            print('i: ', i)
            record['sample_id'] = data['sample_id']
            record['query'] = data['query']
            record['pos'] = list(set(data['pos']))
            record['intent'] = data['intent']
            record['neg'] = list(set(data['neg']))
            # get new list of docs
            new_pos_docs, new_neg_docs = get_hard_negatives(record["query"], record["pos"], record["neg"], device)
            record['pos'] = new_pos_docs

            # get the last ranked neg doc
            if len(new_neg_docs) >= 1:
                record['neg'] = [new_neg_docs[-1]] # there are 30 samples with empty neg docs
            else:
                # for empty neg docs
                # option 1: set to empty string
                if option == 'empty':
                    record['neg'] = ['']
                # option 2: augment hard negatives and select the last one: easy negative
                elif option == 'augment':
                    new_neg_docs = augment_easy_negatives(new_neg_docs, 
                                                                    record,
                                                                    output_data, 
                                                                    device=device, 
                                                                    max_neg_ratio = max_neg_ratio, 
                                                                    train = False)
                    record['neg'] = [new_neg_docs[-1]]

            f.write(json.dumps(record) + '\n')