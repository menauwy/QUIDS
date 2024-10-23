"""
QRECC dataset has 3 passage collection sources.
* Seems that they use zenodo version 2021-10-01: scai-qrecc21-test-turns.json
1. First need to generate a unified passage collection file pid2rawpid.pkl. 
2. Second need to generate a qrel file for the test set. but why just test set?
3. Third need to generate train and test files. Check the 'cur_utt_text' part --> CONQRR, so and ctx_query_utt. Also negative sampling for the training process.
* They use previous turns positive docs as the negatives of the current turn --> doubt that.
* They don't use bm25 hard negatives for training.
The question to be sovled is how to use negative sampling for training!!!

Q2ID:
1. randomly divide them into 5000/100/258 for train/val/test sets following the paper of Q2ID.
2. get unique pos.
3. randomly select 3 unique negative docs for each sample.
# TODO: more ways for negative sampling.
# TODO: put samples without negative docs into test dataset
"""
#from IPython import embed
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
import argparse
from sentence_transformers import SentenceTransformer
#from utils import pstore, pload
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))

def gen_q2id_train_valid_test_files(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               max_random_neg_raito = 5):
    '''
    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data_original.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data_original.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data_original.jsonl"
    train_dataset里的neg 取5个，valid 和 test的neg不变
    '''

    random.seed(42)
    data = []
    with open(inputfile, "r") as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            l['sample_id'] = i
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    random.shuffle(data)

    # split the lines into train/valid/test
    train_data = data[:5000]
    valid_data = data[5000:5100]
    test_data = data[5100:]


    # remove lines without positive docs & random select negative docs
    outputfiles = {train_outputfile: train_data, 
                   valid_outputfile: valid_data, 
                   test_outputfile: test_data}
    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                # if len(line["pos"]) == 0:
                #     continue
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = line["neg"]

                # Various negatives
                if output_data == train_data:
                    # 1. random negatives: less qual than 5
                    random_neg_docs = set()
                    potential_neg_docs = set()
                    for neg_sample in line["neg"]:
                        if neg_sample not in line["pos"]:
                            potential_neg_docs.add(neg_sample)
                    potential_neg_docs = set(potential_neg_docs)

                    if len(potential_neg_docs) < max_random_neg_raito:
                        random_neg_docs = potential_neg_docs
                    else:
                        random_neg_docs = random.sample(potential_neg_docs, max_random_neg_raito)
                    record["neg"] = list(random_neg_docs)
                    # 2. TODO: in batch negative sampling
                    # implemented in the training process?

                    # 3. TODO: CoNT: generation as negative samples
                    # implemented in the trainin process?
                f.write(json.dumps(record))
                f.write('\n')
    logger.info("QReCC train test file preprocessing (first stage) ok!")            

def gen_q2id_train_valid_test_files_v2(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               max_random_neg_raito = 3):
    '''
    remove samples without pos docs
    random pick <= 3 negatives, each is an individual sample
    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data.jsonl"
    '''

    random.seed(42)
    data = []
    with open(inputfile, "r") as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            l['sample_id'] = i
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    random.shuffle(data)

    # split the lines into train/valid/test
    train_data = data[:5000]
    valid_data = data[5000:5100]
    test_data = data[5100:]

    # remove lines without positive docs & random select negative docs
    outputfiles = {train_outputfile: train_data, 
                   valid_outputfile: valid_data, 
                   test_outputfile: test_data}
    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                if len(line["pos"]) == 0:
                    continue
                
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = line["neg"]

                # Various negatives
                if output_data == train_data:
                    # 1. random negatives: less qual than 3
                    random_neg_docs = set()
                    potential_neg_docs = set()
                    for neg_sample in line["neg"]:
                        if neg_sample not in line["pos"]:
                            potential_neg_docs.add(neg_sample)
                    potential_neg_docs = set(potential_neg_docs)

                    if len(potential_neg_docs) <= max_random_neg_raito:
                        random_neg_docs = potential_neg_docs
                    else:
                        random_neg_docs = random.sample(potential_neg_docs, max_random_neg_raito)
                    record["neg"] = list(random_neg_docs)

                # store each neg doc as an individual sample
                for i, neg in enumerate(record["neg"]):
                    new_record = {}
                    new_record["sample_id"] = record["sample_id"] + (i+1) * 10000 # >5388
                    new_record["query"] = record["query"]
                    new_record["pos"] = record["pos"]
                    new_record["intent"] = record["intent"]
                    new_record["neg"] = [neg]
                
                    f.write(json.dumps(new_record))
                    f.write('\n')

                if len(record["neg"]) == 0:
                    f.write(json.dumps(record))
                    f.write('\n')
    logger.info("QReCC train test file preprocessing (first stage) ok!")

def gen_q2id_train_valid_test_files_v3(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               max_random_neg_raito = 3):
    '''
    remove samples without pos docs
    random pick <= 3 negatives, each is an individual sample
    don't seperate neg docs in valid and test!

    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data_v3.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data_v3.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data_v3.jsonl"
    '''

    random.seed(42)
    data = []
    with open(inputfile, "r") as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            l['sample_id'] = i
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    random.shuffle(data)

    # split the lines into train/valid/test
    train_data = data[:5000]
    valid_data = data[5000:5100]
    test_data = data[5100:]

    # remove lines without positive docs & random select negative docs
    outputfiles = {train_outputfile: train_data, 
                   valid_outputfile: valid_data, 
                   test_outputfile: test_data}

    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                if len(line["pos"]) == 0:
                    continue
                
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = line["neg"]

                
                if output_data == train_data:
                    # 1. get random negatives: less qual than 3
                    random_neg_docs = set()
                    potential_neg_docs = set()
                    for neg_sample in line["neg"]:
                        if neg_sample not in line["pos"]:
                            potential_neg_docs.add(neg_sample)
                    potential_neg_docs = set(potential_neg_docs)

                    if len(potential_neg_docs) <= max_random_neg_raito:
                        random_neg_docs = potential_neg_docs
                    else:
                        random_neg_docs = random.sample(potential_neg_docs, max_random_neg_raito)
                    record["neg"] = list(random_neg_docs)

                # 2. store each neg doc as an individual sample for train dataset only!!!
                    for i, neg in enumerate(record["neg"]):
                        new_record = {}
                        new_record["sample_id"] = record["sample_id"] + (i+1) * 10000 # >5388
                        new_record["query"] = record["query"]
                        new_record["pos"] = record["pos"]
                        new_record["intent"] = record["intent"]
                        new_record["neg"] = [neg]

                        f.write(json.dumps(new_record))
                        f.write('\n')

                    if len(record["neg"]) == 0:
                        f.write(json.dumps(record))
                        f.write('\n')

                # TODO: for valid and test data: randomly select one negtive doc in th record, since the neg will not be used. 
                # to align with the data structure format. Need to be modified in the future.        
                
                if output_data == valid_data or output_data == test_data:                
                    neg_doc = random.sample(line["neg"], 1) if len(line["neg"]) > 0 else []
                    new_record = {}
                    new_record["sample_id"] = record["sample_id"]
                    new_record["query"] = record["query"]
                    new_record["pos"] = record["pos"]
                    new_record["intent"] = record["intent"]
                    new_record["neg"] = neg_doc
                    f.write(json.dumps(new_record))
                    f.write('\n')

    logger.info("QReCC train test file preprocessing (first stage) ok!")

def gen_q2id_train_valid_test_files_v4(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               max_random_neg_raito = 1):
    '''
    The same code as v3: but set max_random_neg_raito = 1
    remove samples without pos docs
    random pick <= 3 negatives, each is an individual sample
    !!!! New: compared to v3!!!
    !!! save one neg doc in train dataset, valid and test dataset are the same as v3

    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data_v4.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data_v4.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data_v4.jsonl"
    '''

    random.seed(42)
    data = []
    with open(inputfile, "r") as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            l['sample_id'] = i
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    random.shuffle(data)

    # split the lines into train/valid/test
    train_data = data[:5000]
    valid_data = data[5000:5100]
    test_data = data[5100:]

    # remove lines without positive docs & random select negative docs
    outputfiles = {train_outputfile: train_data, 
                   valid_outputfile: valid_data, 
                   test_outputfile: test_data}

    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                if len(line["pos"]) == 0:
                    continue
                
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = line["neg"]

                
                if output_data == train_data:
                    # 1. get random negatives: less qual than 3
                    random_neg_docs = set()
                    potential_neg_docs = set()
                    for neg_sample in line["neg"]:
                        if neg_sample not in line["pos"]:
                            potential_neg_docs.add(neg_sample)
                    potential_neg_docs = set(potential_neg_docs)

                    if len(potential_neg_docs) <= max_random_neg_raito:
                        random_neg_docs = potential_neg_docs
                    else:
                        random_neg_docs = random.sample(potential_neg_docs, max_random_neg_raito)
                    record["neg"] = list(random_neg_docs)

                # 2. store each neg doc as an individual sample for train dataset only!!!
                    for i, neg in enumerate(record["neg"]):
                        new_record = {}
                        new_record["sample_id"] = record["sample_id"] + (i+1) * 10000 # >5388
                        new_record["query"] = record["query"]
                        new_record["pos"] = record["pos"]
                        new_record["intent"] = record["intent"]
                        new_record["neg"] = [neg]

                        f.write(json.dumps(new_record))
                        f.write('\n')

                    if len(record["neg"]) == 0:
                        f.write(json.dumps(record))
                        f.write('\n')

                # TODO: for valid and test data: randomly select one negtive doc in th record, since the neg will not be used. 
                # to align with the data structure format. Need to be modified in the future.        
                
                if output_data == valid_data or output_data == test_data:                
                    neg_doc = random.sample(line["neg"], 1) if len(line["neg"]) > 0 else []
                    new_record = {}
                    new_record["sample_id"] = record["sample_id"]
                    new_record["query"] = record["query"]
                    new_record["pos"] = record["pos"]
                    new_record["intent"] = record["intent"]
                    new_record["neg"] = neg_doc
                    f.write(json.dumps(new_record))
                    f.write('\n')

    logger.info("QReCC train test file preprocessing (first stage) ok!")

def gen_q2id_train_valid_test_files_v5(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               max_random_neg_raito = 3):
    '''
    remove samples without pos docs
    random pick <= 3 negatives, each is an individual sample
    !!!! New: compared to v3!!!
    !!! Train data: take first 50 tokens of each rel doc, valid and test dataset are the same as v3

    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data.jsonl"
    '''

    random.seed(30)
    data = []
    with open(inputfile, "r") as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            l['sample_id'] = i
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    random.shuffle(data)

    # split the lines into train/valid/test
    train_data = data[:5000]
    valid_data = data[5000:5100]
    test_data = data[5100:]

    # remove lines without positive docs & random select negative docs
    outputfiles = {train_outputfile: train_data, 
                   valid_outputfile: valid_data, 
                   test_outputfile: test_data}

    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                if len(line["pos"]) == 0:
                    continue
                
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = line["neg"]

                if output_data == train_data:
                    # 0. Take first 50 tokens of each rel doc
                    tokenizer = T5Tokenizer.from_pretrained("t5-base")
                    record["pos"] = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc)[:50]) for doc in record["pos"]]
                    
                    # 1. get random negatives: less qual than 3
                    random_neg_docs = set()
                    potential_neg_docs = set()
                    for neg_sample in line["neg"]:
                        if neg_sample not in line["pos"]:
                            potential_neg_docs.add(neg_sample)
                    potential_neg_docs = set(potential_neg_docs)

                    if len(potential_neg_docs) <= max_random_neg_raito:
                        random_neg_docs = potential_neg_docs
                    else:
                        random_neg_docs = random.sample(potential_neg_docs, max_random_neg_raito)
                    record["neg"] = list(random_neg_docs)

                    # 2. store each neg doc as an individual sample for train dataset only!!!
                    for i, neg in enumerate(record["neg"]):
                        new_record = {}
                        new_record["sample_id"] = record["sample_id"] + (i+1) * 10000 # >5388
                        new_record["query"] = record["query"]
                        new_record["pos"] = record["pos"]
                        new_record["intent"] = record["intent"]
                        new_record["neg"] = [neg]

                        f.write(json.dumps(new_record))
                        f.write('\n')

                    if len(record["neg"]) == 0:
                        f.write(json.dumps(record))
                        f.write('\n')

                # TODO: for valid and test data: randomly select one negtive doc in th record, since the neg will not be used. 
                # to align with the data structure format. Need to be modified in the future.        
                
                if output_data == valid_data or output_data == test_data:                
                    neg_doc = random.sample(line["neg"], 1) if len(line["neg"]) > 0 else []
                    new_record = {}
                    new_record["sample_id"] = record["sample_id"]
                    new_record["query"] = record["query"]
                    new_record["pos"] = record["pos"]
                    new_record["intent"] = record["intent"]
                    new_record["neg"] = neg_doc
                    f.write(json.dumps(new_record))
                    f.write('\n')

    logger.info("QReCC train test file preprocessing (first stage) ok!")

# def modify_pos_docs(conv_sample, pos_docs_text):
#     '''
#     Modify the pos doc content based on the current conversational sample 
#     to avoid simply string-match, enhance model generalization ability
#     '''
#     return pos_docs_text

# def modify_neg_docs(conv_sample, neg_docs_text):
#     '''
#     Modify the neg doc content based on the current conversational sample 
#     to avoid simply string-match, enhance model generalization ability
#     '''
#     return neg_docs_text

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

def find_similar_documents(target_embedding, candidates, model, device, top_k=5):
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True, device=device)
    similarities = torch.nn.functional.cosine_similarity(target_embedding, candidate_embeddings)
    ranked_indices = similarities.argsort(descending=True)
    return [candidates[i] for i in ranked_indices[:top_k] if similarities[i].item() > 0.5]  # Adjust threshold as needed

def find_similar_documents_with_scores(target_embedding, candidates, model, device, top_k=5):
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True, device=device)
    similarities = torch.nn.functional.cosine_similarity(target_embedding, candidate_embeddings)
    sorted_similarities, ranked_indices = similarities.sort(descending=True)
    # 0.5 for dataversion before 11 (including 11)
    # 0.6 for dataset_ranked and afterwards
    # get the top_k sorted similarities that are greater than 0.4
    candidate_num = min(top_k, len(candidates))
    sim_scores = [sorted_similarities[i].item() for i in range(candidate_num) if sorted_similarities[i].item() > 0.8]
    sim_docs = [candidates[ranked_indices[i]] for i in range(len(sim_scores))]
    
    return sim_scores, sim_docs

def augment_hard_negatives(orig_hard_neg, orig_record, source_data, device, max_neg_ratio = 3, train = True):
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
    else:
        target_doc = orig_record['query']
    target_embedding = model.encode(target_doc, convert_to_tensor=True, device=device)
    # rename
    augmented_hard_neg = orig_hard_neg

    # Randomly sample from the source data and find similar documents
    while len(augmented_hard_neg) < max_neg_ratio:
        sample = random.choice(source_data)
        # skip the same sample
        if sample['sample_id'] == sample_id:
            continue
        candidate_docs = sample['pos'] + sample['neg']
        
        similar_docs = find_similar_documents(target_embedding, candidate_docs, model, device)
        for doc in similar_docs:
            if doc not in augmented_hard_neg and doc not in orig_record['pos']:
                augmented_hard_neg.append(doc)
            if len(augmented_hard_neg) >= max_neg_ratio:
                break
    
    return list(augmented_hard_neg)

def augment_hard_negatives_rank(orig_hard_neg, orig_record, source_data, device, max_neg_ratio = 3, train = True):
    '''
    Argument the hard negatives to max_neg_ratio by adding similar sentences from the whole training dataset
    # orig_hard_neg: the original ranked hard negatives from get_hard_negatives
    orig_record: record contains the query, pos, neg, intent, that need to be argumented
    source_data: source dataset to argument hard negatives
    max_neg_ratio: the max number of hard negatives
    return scores and ranks of the original hard negatives
    '''
    model = SentenceTransformer("msmarco-bert-base-dot-v5").to(device)

    sample_id = orig_record['sample_id']
    if train:
        target_doc = orig_record['query'] + ' ' + orig_record['intent']
    else:
        target_doc = orig_record['query']
    print('target doc:', target_doc)
    target_embedding = model.encode(target_doc, convert_to_tensor=True, device=device)

    candidate_hard_neg = []
    candidate_sim_scores = []

    # augment neg docs to max_neg_ratio
    # if True:
    if len(orig_hard_neg) < max_neg_ratio:
        # Randomly sample from the source data and find candidate neg docs base on similarity
        # while len(candidate_hard_neg) < max_neg_ratio:
        while len(candidate_hard_neg) +  len(orig_hard_neg) < max_neg_ratio:
            sample = random.choice(source_data)
            # skip the same sample
            if sample['sample_id'] == sample_id:
                continue
            candidate_docs = sample['pos'] + sample['neg']

            # ranked 
            similar_scores, similar_docs = find_similar_documents_with_scores(target_embedding, candidate_docs, model, device)
            for score, doc in zip(similar_scores, similar_docs):
                if doc not in candidate_hard_neg and doc not in orig_record['pos'] and doc not in orig_record['neg']:
                    candidate_hard_neg.append(doc)
                    candidate_sim_scores.append(score)
                    print('length of candidate_hard_neg: ', len(candidate_hard_neg))
                    print('candidate doc:', doc)
                if len(candidate_hard_neg) >= max_neg_ratio:
                    break

    if len(orig_hard_neg) == 0:
        return (candidate_hard_neg, candidate_sim_scores, [], [])            
    # rank the original neg docs based on similarity with the target doc
    orig_hard_neg_embeddings = model.encode(orig_hard_neg, convert_to_tensor=True, device=device)
    orig_similarities = torch.nn.functional.cosine_similarity(target_embedding, orig_hard_neg_embeddings)
    sorted_similarities, ranked_indices = orig_similarities.sort(descending=True)
    orig_neg_sim_scores = [sorted_similarities[i].item() for i in range(len(orig_hard_neg))]
    orig_hard_neg = [orig_hard_neg[i] for i in ranked_indices]

    # Merge candidate and original neg docs by ranking sim scores
    merged_sim_scores = candidate_sim_scores + orig_neg_sim_scores
    merged_neg_docs = candidate_hard_neg + orig_hard_neg

    # Pair each doc with its score and sort the pairs by score in descending order
    merged_pairs = sorted(zip(merged_neg_docs, merged_sim_scores), key=lambda pair: pair[1], reverse=True)
    merged_pairs = merged_pairs[:max_neg_ratio]

    # Initialize lists to hold the selected docs and scores, and the ranks and scores of the original neg docs
    selected_neg_docs, selected_sim_scores = zip(*merged_pairs)
    orig_neg_ranks = []
    orig_neg_ranked_scores = []

    # Iterate over the sorted pairs
    for i, (doc, score) in enumerate(merged_pairs):
        # If the doc is from the original neg docs and its rank hasn't been recorded yet, record its rank and score
        if doc in orig_hard_neg and doc not in orig_neg_ranks:
            orig_neg_ranks.append(i)
            orig_neg_ranked_scores.append(score)
    
    # return tuple of selected neg docs and their ranks and scores and the ranks and scores of the original neg docs
    return (selected_neg_docs, selected_sim_scores, orig_neg_ranks, orig_neg_ranked_scores)

def gen_q2id_train_valid_test_files_hard_negatives(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               version,
                               max_neg_ratio = 3,
                               argument = False):
    '''
    # Top k ranked negative doc as the hard negative sample
    # each hard negative as a single sample
    * Training: similarity with the query intent
    * Testing: similarity with the query


    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data_v6.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data_v6.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data_v6.jsonl"
    '''

    random.seed(42)
    data = []
    with open(inputfile, "r") as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            l['sample_id'] = i
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    random.shuffle(data)

    # split the lines into train/valid/test
    train_data = data[:5000]
    valid_data = data[5000:5100]
    test_data = data[5100:]
    # move all data to cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # remove lines without positive docs & random select negative docs
    outputfiles = {train_outputfile: train_data, 
                   valid_outputfile: valid_data, 
                   test_outputfile: test_data}

    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                if len(line["pos"]) == 0:
                    continue
                
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = list(set(line["neg"]))
           
                if output_data == train_data:
                    # 1. get new ranked pos docs and hard negatives
                    ##### similarity with intent
                    target = record['query'] + ' ' + record["intent"]
                    new_pos_docs, new_neg_docs = get_hard_negatives(target, record["pos"], record["neg"], device)
                    record["pos"] = new_pos_docs

                    if len(new_neg_docs) > max_neg_ratio:
                        record["neg"] = new_neg_docs[:max_neg_ratio]
                    else:
                        if argument:
                            new_neg_docs = augment_hard_negatives(new_neg_docs, 
                                                                   record,
                                                                   output_data, 
                                                                   device=device, 
                                                                   max_neg_ratio = max_neg_ratio, 
                                                                   train = True)
                        record["neg"] = new_neg_docs

                    # store each neg doc as an individual sample for train dataset only!!!
                    if version == 'v6' or version == 'v8' or version == 'v9' or version == 'v10' or version == 'v11':
                        for i, neg in enumerate(record["neg"]):
                            new_record = {}
                            new_record["sample_id"] = record["sample_id"] + (i+1) * 10000 # >5388
                            new_record["query"] = record["query"]
                            new_record["pos"] = new_pos_docs
                            new_record["intent"] = record["intent"]
                            new_record["neg"] = [neg]

                            f.write(json.dumps(new_record))
                            f.write('\n')

                        if len(new_neg_docs) == 0:
                            new_record = {}
                            new_record["sample_id"] = record["sample_id"]
                            new_record["query"] = record["query"]
                            new_record["pos"] = new_pos_docs
                            new_record["intent"] = record["intent"]
                            new_record["neg"] = []

                            f.write(json.dumps(new_record))
                            f.write('\n')
                    # or concatenate negatives
                    elif version == 'v7':
                        new_record = {}
                        new_record["sample_id"] = record["sample_id"]
                        new_record["query"] = record["query"]
                        new_record["pos"] = new_pos_docs
                        new_record["intent"] = record["intent"]
                        new_record["neg"] = ['.'.join(record["neg"][:max_neg_ratio])]
                        f.write(json.dumps(new_record))
                        f.write('\n')

                # TODO: for valid and test data: randomly select one negtive doc in th record, since the neg will not be used. 
                # to align with the data structure format. Need to be modified in the future.        
                
                if output_data == valid_data or output_data == test_data:    
                    new_pos_docs, new_neg_docs = get_hard_negatives(record["query"], record["pos"], record["neg"], device)
                    record['pos'] = new_pos_docs

                    if len(new_neg_docs) > max_neg_ratio:
                        record["neg"] = new_neg_docs[:max_neg_ratio]
                    else:
                        if argument:
                            new_neg_docs = augment_hard_negatives(new_neg_docs, 
                                                                    record,
                                                                    output_data, 
                                                                    device=device, 
                                                                    max_neg_ratio = max_neg_ratio, 
                                                                    train = False)
                        record["neg"] = new_neg_docs
                    
                    if version == 'v6' or version == 'v8' or version == 'v9' or version == 'v10' or version == 'v11':
                    # store each neg doc in 'v62', 'v82', 'v92'
                    # get output file name, and add 2 before .jsonl to get a new outputfile name
                        outputfile_2 = outputfile.split(".json")[0] + '2.jsonl'
                        with open(outputfile_2, "a") as f2:
                            for i, neg in enumerate(record["neg"]):
                                new_record = {}
                                new_record["sample_id"] = record["sample_id"] + (i+1) * 10000 # >5388
                                new_record["query"] = record["query"]
                                new_record["pos"] = new_pos_docs
                                new_record["intent"] = record["intent"]
                                new_record["neg"] = [neg]
                                f2.write(json.dumps(new_record))
                                f2.write('\n')

                            if len(record["neg"]) == 0:
                                new_record = {}
                                new_record["sample_id"] = record["sample_id"]
                                new_record["query"] = record["query"]
                                new_record["pos"] = new_pos_docs
                                new_record["intent"] = record["intent"]
                                new_record["neg"] = []
                                f2.write(json.dumps(new_record))
                                f2.write('\n')

                    if version == 'v6' or version == 'v8' or version == 'v9' or version == 'v10' or version == 'v11':    
                    # only store the top neg doc in: 'v6', 'v8', 'v9'
                        if len(new_neg_docs) >= 1:
                            new_neg_doc = [new_neg_docs[0]]
                        elif len(new_neg_docs) == 0:
                            new_neg_doc = ['']   
                    # concatenate negatives
                    elif version == 'v7':
                        new_neg_doc = ['.'.join(new_neg_docs[:max_neg_ratio])]
                    
                    new_record = {}
                    new_record["sample_id"] = record["sample_id"]
                    new_record["query"] = record["query"]
                    new_record["pos"] = new_pos_docs
                    new_record["intent"] = record["intent"]
                    new_record["neg"] = new_neg_doc

                    f.write(json.dumps(new_record))
                    f.write('\n')

    logger.info("Q2ID train test file preprocessing (first stage) ok!")

def gen_q2id_train_valid_test_files_hard_negatives_ranked(input_file_path,
                               output_file_path,
                               max_neg_ratio = 16,
                               argument = True,
                               mode = 'train'):
    '''
    # Top k ranked docs as the hard negative sample: can replace the GT negatives
    # each hard negative as a single sample
    * Training: similarity with the query intent
    * Testing: similarity with the query


    - inputfile = "data1/dataset/q2id/Q2ID_data/all_Q2ID_data.jsonl"
    - train_outputfile = "data1/dataset/q2id/train_data_ranked.jsonl"
    - valid_outputfile = "data1/dataset/q2id/valid_data_ranked.jsonl"
    - test_outputfile = "data1/dataset/q2id/test_data_ranked.jsonl"
    '''
    # random.seed(42)
    # data = []
    # with open(input_file_path, "r") as f:
    #     for i, line in enumerate(f):
    #         l = json.loads(line)
    #         l['sample_id'] = i
    #         data.append(l)
    #     print("The number of samples in the original file is {}".format(len(data)))
    # random.shuffle(data)

    # split the lines into train/valid/test
    # train_data = data[:2]
    # valid_data = data[5000:5005]
    # test_data = data[5100:5105]
    data = []
    with open(input_file_path, 'r')as f:
        for i, line in enumerate(f):
            l = json.loads(line)
            data.append(l)
        print("The number of samples in the original file is {}".format(len(data)))
    
    # data = data[:50]

    # move all data to cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # remove lines without positive docs & random select negative docs
    outputfiles = {output_file_path: data}

    for outputfile in outputfiles:
        output_data = outputfiles[outputfile]
        with open(outputfile, "w") as f:
            for line in tqdm(output_data):
                record = {}
                # Format: query, pos:[], neg:[], intent
                # remove lines without positive docs
                if len(line["pos"]) == 0:
                    continue
                
                record["sample_id"] = line["sample_id"]
                record["query"] = line["query"]
                record["pos"] = list(set(line["pos"]))
                record["intent"] = line["intent"]
                record["neg"] = list(set(line["neg"]))
           
                if mode == 'train':
                    # 1. get new ranked pos docs and hard negatives
                    ##### similarity with intent
                    target = record['query'] + ' ' + record["intent"]
                    new_pos_docs, new_neg_docs = get_hard_negatives(target, record["pos"], record["neg"], device)
                    record["pos"] = new_pos_docs

                elif mode == 'valid' or mode == 'test':
                    # 1. get new ranked pos docs and hard negatives
                    ##### similarity with intent
                    target = record['query']
                    new_pos_docs, new_neg_docs = get_hard_negatives(target, record["pos"], record["neg"], device)
                    record["pos"] = new_pos_docs

                train_flag = mode == 'train'
                if argument:
                    selected_neg_docs, selected_sim_scores, orig_neg_ranks, orig_neg_ranked_scores = augment_hard_negatives_rank(new_neg_docs, 
                                                           record,
                                                           output_data, 
                                                           device=device, 
                                                           max_neg_ratio = max_neg_ratio, 
                                                           train = train_flag)
                else:
                    raise ValueError("The argument should be True for this version!")

                    
                # store all infomation -> generate sub dataset for each neg doc in next step
                new_record = {}
                new_record["sample_id"] = record["sample_id"] # + (i+1) * 10000 # >5388
                new_record["query"] = record["query"]
                new_record["pos"] = record["pos"]
                new_record["intent"] = record["intent"]
                new_record["selected_neg"] = selected_neg_docs
                new_record["selected_sim_scores"] = selected_sim_scores
                new_record["orig_neg_ranks"] = orig_neg_ranks
                new_record["orig_neg_ranked_scores"] = orig_neg_ranked_scores
                f.write(json.dumps(new_record))
                f.write('\n')

    logger.info("Q2ID train test file preprocessing (first stage) ok!")


def statis_info(inputfile, statis_oracle):
    '''
    Get some useful statistic information of the QReCC dataset. Just print.
    '''
    def statis_this_conv(conv_id, pre_gold_doc_ids, pre_cur_utts, pre_oracle_utts, statis_oracle):
        d_gold = {}
        num_turns = len(pre_gold_doc_ids)
        num_valid_turns = 0
        num_has_same_gold_doc = 0
        num_same_gold_doc_turn_in_ctx = 0
        for _, doc_id in enumerate(pre_gold_doc_ids):
            if doc_id == -1:
                continue
            num_valid_turns += 1
            if doc_id in d_gold:
                num_has_same_gold_doc += 1
                num_same_gold_doc_turn_in_ctx += d_gold[doc_id]
                d_gold[doc_id] += 1
            else:
                d_gold[doc_id] = 1
        

        # statis of ideal terms
        num_utt_length = 0
        num_oracle_utt_length = 0
        num_utt_diff_length = 0
        num_utt_diff_percent_in_oracle = 0
        if statis_oracle:
            for i in range(len(pre_cur_utts)):
                cur_utt = pre_cur_utts[i].split(" ")
                num_utt_length += len(cur_utt)
                cur_oracle_utt = pre_oracle_utts[i].split(" ")
                num_oracle_utt_length += len(cur_oracle_utt)
                num_utt_diff_length += len(set(cur_oracle_utt) - set(cur_utt))
                num_utt_diff_percent_in_oracle += (len(set(cur_oracle_utt) - set(cur_utt))) / len(set(cur_oracle_utt))
      
        return num_turns, num_valid_turns, num_has_same_gold_doc, num_same_gold_doc_turn_in_ctx, \
                num_utt_length, num_oracle_utt_length, num_utt_diff_length, num_utt_diff_percent_in_oracle
    

    with open(inputfile, "r") as f:
        data = f.readlines()
    total_turns = 0
    total_valid_turns = 0   # the number of turns that have gold doc
    total_convs = 0
    total_has_same_gold_doc = 0
    total_num_same_gold_doc_turn_in_ctx = 0
    total_utt_length = 0
    total_oracle_utt_length = 0
    total_utt_diff_length = 0
    total_utt_diff_percent_in_oracle = 0

    pre_conv_id = -1 
    pre_gold_doc_ids = []
    pre_cur_utts = []
    pre_oracle_utts = []
    for line in tqdm(data):
        line = json.loads(line)
        conv_id = line["sample_id"].split("_")[1]
        if conv_id != pre_conv_id:
            assert len(pre_gold_doc_ids) == len(pre_cur_utts)
            num_turns, num_valid_turns, num_has_same_gold_doc, num_same_gold_doc_turn_in_ctx, \
                num_utt_length, num_oracle_utt_length, \
                num_utt_diff_length, num_utt_diff_percent_in_oracle = statis_this_conv(pre_conv_id, 
                                                                                       pre_gold_doc_ids,
                                                                                       pre_cur_utts, 
                                                                                       pre_oracle_utts,
                                                                                       statis_oracle)
            total_turns += num_turns
            total_valid_turns += num_valid_turns
            total_convs += 1
            total_has_same_gold_doc += num_has_same_gold_doc
            total_num_same_gold_doc_turn_in_ctx += num_same_gold_doc_turn_in_ctx
            total_utt_length += num_utt_length
            total_oracle_utt_length += num_oracle_utt_length
            total_utt_diff_length += num_utt_diff_length
            total_utt_diff_percent_in_oracle += num_utt_diff_percent_in_oracle

            pre_cur_utts, pre_gold_doc_ids, pre_oracle_utts = [], [], []
        
        if len(line["pos_docs_pids"]) == 0:
            pre_gold_doc_ids.append(-1)
        else:
            pre_gold_doc_ids.append(line["pos_docs_pids"][0])
        pre_cur_utts.append(line["cur_utt_text"])
        pre_oracle_utts.append(line["oracle_utt_text"])
        pre_conv_id = conv_id
    
    # last conversation
    num_turns, num_valid_turns, num_has_same_gold_doc, num_same_gold_doc_turn_in_ctx, \
                num_utt_length, num_oracle_utt_length, \
                num_utt_diff_length, num_utt_diff_percent_in_oracle = statis_this_conv(pre_conv_id,
                                                                                       pre_gold_doc_ids,
                                                                                       pre_cur_utts, 
                                                                                       pre_oracle_utts,
                                                                                       statis_oracle)
    total_turns += num_turns
    total_valid_turns += num_valid_turns
    total_convs += 1
    total_has_same_gold_doc += num_has_same_gold_doc
    total_num_same_gold_doc_turn_in_ctx += num_same_gold_doc_turn_in_ctx
    total_utt_length += num_utt_length
    total_oracle_utt_length += num_oracle_utt_length
    total_utt_diff_length += num_utt_diff_length
    total_utt_diff_percent_in_oracle += num_utt_diff_percent_in_oracle
    
    logger.info("The statis info of \"file {}\" is")
    logger.info("total turns: {}".format(total_turns))
    logger.info("total valid turns: {}".format(total_valid_turns))
    logger.info("total conversations: {}".format(total_convs))
    logger.info("percent of valid turns have same gold doc: {}".format(total_has_same_gold_doc / total_valid_turns))
    logger.info("average num of the same gold doc turn in ctx per valid turn: {}".format(total_num_same_gold_doc_turn_in_ctx / total_valid_turns))
    logger.info("average utt length: {}".format(total_utt_length / total_turns))
    logger.info("average oracle utt length: {}".format(total_oracle_utt_length / total_turns))
    logger.info("average utt diff length: {}".format(total_utt_diff_length / (total_turns - total_convs)))
    logger.info("average utt diff percent in oracle: {}".format(total_utt_diff_percent_in_oracle / (total_turns - total_convs)))
    
    return 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
  
def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--data_version", type=str, default='v6', help="data version")
    parser.add_argument("--input_file", type=str, default='/home/ywang1/data1/dataset/q2id/Q2ID_data/ranked/split', help="input file")
    parser.add_argument("--max_neg_ratio", type=int, default=3, help="max number of hard negatives")
    parser.add_argument("--argument", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # original local data dir

    # data_version = args.data_version
    input_file = args.input_file
    max_neg_ratio = args.max_neg_ratio
    argument = args.argument

    dataset_dir = "/home/ywang1/data1/dataset/q2id/Q2ID_data/ranked"
    # dataset_dir = '/home/ywang1/data1/dataset/q2id/Q2ID_data/ranked'
    input_file_path = dataset_dir + "/split/" + input_file

    output_file_path = dataset_dir + "/ranked_" + input_file
    mode = input_file.split('_')[0]
    print('mode is:', mode)
    # train_outputfile = dataset_dir + "/train_data_{}.jsonl".format(data_version)
    # valid_outputfile = dataset_dir + "/valid_data_{}.jsonl".format(data_version)
    # test_outputfile = dataset_dir + "/test_data_{}.jsonl".format(data_version)
    # gen_q2id_train_valid_test_files(inputfile, train_outputfile, valid_outputfile, test_outputfile, max_random_neg_raito = 5)
    # gen_q2id_train_valid_test_files_v2(inputfile, train_outputfile, valid_outputfile, test_outputfile, max_random_neg_raito = 3)
    # gen_q2id_train_valid_test_files_v3(inputfile, train_outputfile, valid_outputfile, test_outputfile, max_random_neg_raito = 3)
    # gen_q2id_train_valid_test_files_v4(inputfile, train_outputfile, valid_outputfile, test_outputfile, max_random_neg_raito = 1)
    # gen_q2id_train_valid_test_files_v5(inputfile, train_outputfile, valid_outputfile, test_outputfile, max_random_neg_raito = 3)
    gen_q2id_train_valid_test_files_hard_negatives(inputfile, train_outputfile, valid_outputfile, test_outputfile, version= data_version, max_neg_ratio = max_neg_ratio, argument=argument)
    # gen_q2id_train_valid_test_files_hard_negatives_ranked(input_file_path, output_file_path,  max_neg_ratio = max_neg_ratio, argument=argument, mode=mode)

