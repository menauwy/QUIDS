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


def gen_q2id_train_valid_test_files_hard_negatives(inputfile,
                               train_outputfile, 
                               valid_outputfile,
                               test_outputfile, 
                               version,
                               max_neg_ratio = 3,
                               argument = False):

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
  
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default='/home/XXX/data1/dataset/q2id/Q2ID_data/ranked/split', help="input file")
    parser.add_argument("--max_neg_ratio", type=int, default=3, help="max number of hard negatives")
    parser.add_argument("--argument", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # original local data dir

    input_file = args.input_file
    max_neg_ratio = args.max_neg_ratio
    argument = args.argument

    dataset_dir = "/home/XXX/data1/dataset/q2id/Q2ID_data"
    input_file_path = dataset_dir + "/split/" + input_file
    output_file_path = dataset_dir + "/" + input_file
    mode = input_file.split('_')[0]
    print('mode is:', mode)

    data_version = 'v10'
    train_outputfile = dataset_dir + "/train_data_{}.jsonl".format(data_version)
    valid_outputfile = dataset_dir + "/valid_data_{}.jsonl".format(data_version)
    test_outputfile = dataset_dir + "/test_data_{}.jsonl".format(data_version)
    gen_q2id_train_valid_test_files_hard_negatives(inputfile, train_outputfile, valid_outputfile, test_outputfile, version= data_version, max_neg_ratio = max_neg_ratio, argument=argument)

