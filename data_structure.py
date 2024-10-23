from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import argparse
import torch
#import toml
from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import numpy as np
# from src.models.models import load_model

class BartIntentIRDataset_q2id(Dataset):
    """
    Prepare BART inputs: <s>query</s>docs</s>
    """
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat_pos = []
            flat_concat_neg = []
            pos_doc_length = []
            neg_doc_length = []
            sample_id = record['sample_id']
            query = record['query']
            pos_docs = record['pos'] # list of strings
            neg_doc = record['neg'] # list of list of strings: [[strings]]
            intent = record['intent']

            # prefix for bart input and encode query
            query = f"<s>{query}</s>"
            query_encoding = tokenizer.encode(query, 
                                              add_special_tokens = False,
                                              max_length = args.max_query_length,
                                              truncation = True)
            flat_concat_pos.extend(query_encoding)
            flat_concat_neg.extend(query_encoding)
            pos_doc_length.append(len(query_encoding))
            neg_doc_length.append(len(query_encoding))

            #### process pos docs: add prefix + concatenate all pos and query
            if args.pos_num == "one" and len(pos_docs) > 0:
                pos_doc_encoding = tokenizer.encode(pos_docs[0], 
                                                    add_special_tokens=True, 
                                                    max_length=args.max_concat_length,
                                                    truncation=True)
                if len(flat_concat_pos) + len(pos_doc_encoding) > args.max_concat_length:
                    if args.max_concat_length != len(flat_concat_pos):
                        pos_doc_length.append(args.max_concat_length - len(flat_concat_pos)) # contains EOS
                        flat_concat_pos += pos_doc_encoding[:args.max_concat_length - len(flat_concat_pos) - 1] + [pos_doc_encoding[-1]] # must ended with [SEP] ([EOS])
                    break
                else:
                    flat_concat_pos.extend(pos_doc_encoding)
                    pos_doc_length.append(len(pos_doc_encoding))
            # opt 2: concatenate all pos docs, limit on max_doc_length
            if args.pos_num == "multiple":        
                for idx in range(len(pos_docs)):
                    pos_doc_encoding = tokenizer.encode(f"{pos_docs[idx]}</s>", 
                                                        add_special_tokens=False, 
                                                        max_length=args.max_doc_length,
                                                        truncation=True)
                    if len(flat_concat_pos) + len(pos_doc_encoding) > args.max_concat_length:
                        # don't add 0 value
                        if args.max_concat_length != len(flat_concat_pos):
                            pos_doc_length.append(args.max_concat_length - len(flat_concat_pos)) # contains EOS
                            flat_concat_pos += pos_doc_encoding[:args.max_concat_length - len(flat_concat_pos) - 1] + [pos_doc_encoding[-1]] # must ended with [SEP] ([EOS])
                        break
                    else:
                        flat_concat_pos.extend(pos_doc_encoding)
                        pos_doc_length.append(len(pos_doc_encoding))
            flat_concat_pos, flat_concat_pos_mask = padding_seq_to_same_length(flat_concat_pos, max_pad_length = args.max_concat_length)

            ##### process neg doc: add prefix + concat with query  
            # each neg doc will be stored as a seperate example, length: max_concat_length
            if len(neg_doc) == 1: # can be empty: [""]
                ##################### difference with T5: add </s>, so the length of neg_doc is at least 1
                neg_doc_encoding = tokenizer.encode(f"{neg_doc[0]}</s>", 
                                                     add_special_tokens=False, 
                                                     max_length=args.max_concat_length)
                if len(flat_concat_neg) + len(neg_doc_encoding) > args.max_concat_length:
                    if args.max_concat_length != len(flat_concat_neg):
                        neg_doc_length.append(args.max_concat_length - len(flat_concat_neg)) # contains EOS
                        flat_concat_neg += neg_doc_encoding[:args.max_concat_length - len(flat_concat_neg) - 1] + [neg_doc_encoding[-1]] # must ended with [SEP] ([EOS])
                else:
                    flat_concat_neg.extend(neg_doc_encoding)
                    neg_doc_length.append(len(neg_doc_encoding))
            elif len(neg_doc) == 0:
                #TODO: rethink the flat_concat here
                flat_concat_neg = []
            else:
                raise ValueError("Alert: Multiple negative documents detected!")
            
            flat_concat_neg, flat_concat_neg_mask = padding_seq_to_same_length(flat_concat_neg, max_pad_length = args.max_concat_length)

            ##### for training label encoding: see https://huggingface.co/docs/transformers/main/en/model_doc/t5#training
            if args.collate_fn_type == "flat_concat_for_train":
                target_seq = intent
                #千万注意，bart的tokenizer默认加上bos和eos
                target_encoding = tokenizer(target_seq, 
                                            # add_special_tokens=False,
                                            padding="max_length", 
                                            max_length=256, # args.max_new_tokens
                                            truncation=True)
                
                # change pad token to -100 for not being attended. Same for Bart and T5 models
                labels = torch.tensor(target_encoding.input_ids)
                # print('original labels:', labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()

                # labels = labels.tolist()
                decoder_attention_mask = target_encoding['attention_mask']
                self.examples.append([sample_id,
                                      flat_concat_pos,
                                      flat_concat_pos_mask,
                                      flat_concat_neg, # can be empty list []
                                      flat_concat_neg_mask,
                                      labels,
                                      decoder_attention_mask,
                                      pos_doc_length,
                                      neg_doc_length])
                i += 1
            ###### for reference
            elif args.collate_fn_type == "flat_concat_for_test":
                labels = []
                decoder_attention_mask = []
                self.examples.append([sample_id,
                                      flat_concat_pos, # query encoding
                                      flat_concat_pos_mask,
                                      flat_concat_neg,
                                      flat_concat_neg_mask,
                                      labels,
                                      decoder_attention_mask,
                                      pos_doc_length,
                                      neg_doc_length])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_doc": [],
                             "bt_neg_doc_mask": [],
                             "bt_labels": [],
                             "bt_decoder_attention_mask": [],
                             "bt_pos_doc_length": [],
                             "bt_neg_doc_length": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_pos_docs"].append(example[1])
                collated_dict["bt_pos_docs_mask"].append(example[2])
                collated_dict["bt_neg_doc"].append(example[3])
                collated_dict["bt_neg_doc_mask"].append(example[4])
                collated_dict["bt_labels"].append(example[5])
                collated_dict["bt_decoder_attention_mask"].append(example[6])
                collated_dict["bt_pos_doc_length"].append(example[7])
                collated_dict["bt_neg_doc_length"].append(example[8])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_pos_doc_length", "bt_neg_doc_length"])
            if args.collate_fn_type == "flat_concat_for_test":
                # may need to add pos & neg docs to test file
                not_need_to_tensor_keys.add("bt_labels") 
                not_need_to_tensor_keys.add("bt_decoder_attention_mask")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5IntentIRDataset_q2id(Dataset):
    """
    Add prefix for T5 input: 'question': xxx, 'context': xxx  <-- SQuAD prefix in T5 paper
    Determine input_ids and attention_mask and label encoding by max_length, padding.
    IRdataset: incorparate positive and negative docs for training
    1. positive documents: one / multiple; 
    2. neg: each is a new one. len(neg_docs) <= 1
    3. different prefixes

    For training: flat_concat_pos and flat_concat_neg as dual inputs.
    For reference: flat_concat_pos (query only) as inputs.
    """
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat_pos = []
            flat_concat_neg = []
            pos_doc_length = []
            neg_doc_length = []
            sample_id = record['sample_id']
            query = record['query']
            pos_docs = record['pos'] # list of strings
            neg_doc = record['neg'] # list of list of strings: [[strings]]
            intent = record['intent']

            # prefix for T5 input and encode query
            if args.use_prefix:
                if args.prefix_type == 'question':
                    query = "question: " + query
                if args.prefix_type == 'intent':
                    query = "query: " + query
                if args.prefix_type == 'prompt':
                    query = "Determine the intention behind the query based on the context provided in the document. query:" + query
            query_encoding = tokenizer.encode(query, 
                                              add_special_tokens = True, # default to true, add </s>
                                              max_length = args.max_query_length)
            flat_concat_pos.extend(query_encoding)
            flat_concat_neg.extend(query_encoding)
            pos_doc_length.append(len(query_encoding))
            neg_doc_length.append(len(query_encoding))

            #### process pos docs: add prefix + concatenate all pos and query
            for j in range(len(pos_docs)):
                if args.use_prefix:
                    if args.prefix_type == 'question':
                        pos_docs[j] = "context: " + pos_docs[j]
                    if args.prefix_type == 'intent':
                        pos_docs[j] = "relevant document: " + pos_docs[j]
                    if args.prefix_type == 'prompt':
                        pos_docs[j] = "relevant document: " + pos_docs[j]
            # concatenate pos docs
            # opt 1: use one postive doc, limit on max_doc_length
            if args.pos_num == "one" and len(pos_docs) > 0:
                pos_doc_encoding = tokenizer.encode(pos_docs[0], 
                                                    add_special_tokens=True, 
                                                    max_length=args.max_concat_length,
                                                    truncation=True)
                if len(flat_concat_pos) + len(pos_doc_encoding) > args.max_concat_length:
                    if args.max_concat_length != len(flat_concat_pos):
                        pos_doc_length.append(args.max_concat_length - len(flat_concat_pos)) # contains EOS
                        flat_concat_pos += pos_doc_encoding[:args.max_concat_length - len(flat_concat_pos) - 1] + [pos_doc_encoding[-1]] # must ended with [SEP] ([EOS])
                    break
                else:
                    flat_concat_pos.extend(pos_doc_encoding)
                    pos_doc_length.append(len(pos_doc_encoding))
            # opt 2: concatenate all pos docs, limit on max_doc_length
            if args.pos_num == "multiple":        
                for idx in range(len(pos_docs)):
                    pos_doc_encoding = tokenizer.encode(pos_docs[idx], 
                                                        add_special_tokens=True, 
                                                        max_length=args.max_doc_length,
                                                        truncation=True)
                    if len(flat_concat_pos) + len(pos_doc_encoding) > args.max_concat_length:
                        # don't add 0 value
                        if args.max_concat_length != len(flat_concat_pos):
                            pos_doc_length.append(args.max_concat_length - len(flat_concat_pos)) # contains EOS
                            flat_concat_pos += pos_doc_encoding[:args.max_concat_length - len(flat_concat_pos) - 1] + [pos_doc_encoding[-1]] # must ended with [SEP] ([EOS])
                        break
                    else:
                        flat_concat_pos.extend(pos_doc_encoding)
                        pos_doc_length.append(len(pos_doc_encoding))
            flat_concat_pos, flat_concat_pos_mask = padding_seq_to_same_length(flat_concat_pos, max_pad_length = args.max_concat_length)

            ##### process neg doc: add prefix + concat with query  
            if len(neg_doc) == 1:
                if args.use_prefix:
                    if args.prefix_type == 'question':
                        neg_doc[0] = "irrelevant context: " + neg_doc[0]
                    if args.prefix_type == 'intent':
                        neg_doc[0] = "irrelevant document: " + neg_doc[0]
                    if args.prefix_type == 'prompt':
                        neg_doc[0] = "irrelevant document: " + neg_doc[0]
            
            # each neg doc will be stored as a seperate example, length: max_concat_length
            if len(neg_doc) == 1:
                neg_doc_encoding = tokenizer.encode(neg_doc[0], 
                                                     add_special_tokens=True, 
                                                     max_length=args.max_concat_length)
                if len(flat_concat_neg) + len(neg_doc_encoding) > args.max_concat_length:
                    if args.max_concat_length != len(flat_concat_neg):
                        neg_doc_length.append(args.max_concat_length - len(flat_concat_neg)) # contains EOS
                        flat_concat_neg += neg_doc_encoding[:args.max_concat_length - len(flat_concat_neg) - 1] + [neg_doc_encoding[-1]] # must ended with [SEP] ([EOS])
                else:
                    flat_concat_neg.extend(neg_doc_encoding)
                    neg_doc_length.append(len(neg_doc_encoding))
            elif len(neg_doc) == 0:
                #TODO: rethink the flat_concat here
                flat_concat_neg = []
            else:
                raise ValueError("Alert: Multiple negative documents detected!")
            
            flat_concat_neg, flat_concat_neg_mask = padding_seq_to_same_length(flat_concat_neg, max_pad_length = args.max_concat_length)

            ##### for training label encoding: see https://huggingface.co/docs/transformers/main/en/model_doc/t5#training
            if args.collate_fn_type == "flat_concat_for_train":
                target_seq = intent
                target_encoding = tokenizer(target_seq, 
                                            padding="max_length", 
                                            max_length=256, # args.max_new_tokens
                                            truncation=True)
                # change pad token to -100 for not being attended
                labels = torch.tensor(target_encoding.input_ids)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                self.examples.append([sample_id,
                                      flat_concat_pos,
                                      flat_concat_pos_mask,
                                      flat_concat_neg, # can be empty list []
                                      flat_concat_neg_mask,
                                      labels,
                                      pos_doc_length,
                                      neg_doc_length])
                i += 1
            ###### for reference
            elif args.collate_fn_type == "flat_concat_for_test":
                labels = []
                self.examples.append([sample_id,
                                      flat_concat_pos, # query encoding
                                      flat_concat_pos_mask,
                                      flat_concat_neg,
                                      flat_concat_neg_mask,
                                      labels,
                                      pos_doc_length,
                                      neg_doc_length])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_doc": [],
                             "bt_neg_doc_mask": [],
                             "bt_labels": [],
                             "bt_pos_doc_length": [],
                             "bt_neg_doc_length": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_pos_docs"].append(example[1])
                collated_dict["bt_pos_docs_mask"].append(example[2])
                collated_dict["bt_neg_doc"].append(example[3])
                collated_dict["bt_neg_doc_mask"].append(example[4])
                collated_dict["bt_labels"].append(example[5])
                collated_dict["bt_pos_doc_length"].append(example[6])
                collated_dict["bt_neg_doc_length"].append(example[7])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_pos_doc_length", "bt_neg_doc_length"])
            if args.collate_fn_type == "flat_concat_for_test":
                # why is that? may need to add pos & neg docs to test file
                not_need_to_tensor_keys.add("bt_labels") 

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    # if padding_length <= 0, then truncate the input_ids
    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    # else pad the input_ids
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask
