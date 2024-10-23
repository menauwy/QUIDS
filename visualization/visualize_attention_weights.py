import logging
import re
import sys
import glob
import shutil
sys.path.append('..')
sys.path.append('.')
import os
from os import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.environ.get('CONDA_DEFAULT_ENV'))
print(os.environ.get('PYTHONPATH'))
print(sys.executable)
# import time
# import copy
# import pickle
# import random
import numpy as np
import csv
import argparse
import json
from tqdm import tqdm, trange
import statistics

import torch
# from torch import nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
# from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.bart.modeling_bart import BartConfig
from transformers import BartForConditionalGeneration, BartTokenizer
# from torchmetrics.text.rouge import ROUGEScore
# import evaluate
# from rouge import Rouge
# import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.models.model_builder import MyBartModel
from utils import set_seed
from data_structure import BartIntentIRDataset_q2id

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_attention_weights(args):
    # 1. setup model
    config = BartConfig().from_pretrained('facebook/bart-large')
    intent_model = MyBartModel('facebook/bart-large', args, config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

    checkpoint = torch.load(args.model_checkpoint_path+ "/checkpoint.tar")
    intent_model.load_state_dict(checkpoint['model_state_dict'])

    # 2. setup dataset
    test_dataset = BartIntentIRDataset_q2id(args, tokenizer, args.test_file_path)
    test_dataloader = DataLoader(test_dataset, 
                                shuffle=False,
                                batch_size=args.per_gpu_train_batch_size, 
                                collate_fn=test_dataset.get_collate_fn(args))
    # 3. inference
    with open('attention_weights.json', "w") as f:
        with torch.no_grad():
            intent_model.eval()
            for batch in tqdm(test_dataloader, desc="Step"):
                bt_sample_ids = batch['bt_sample_ids']
                print('bt_sample_ids: ', bt_sample_ids)
                bt_pos_docs = batch['bt_pos_docs'].to(args.device)
                bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
                bt_neg_doc = batch['bt_neg_doc'].to(args.device)
                print('bt_neg_doc: ', bt_neg_doc[:, :80])
                bt_neg_doc_mask = batch['bt_neg_doc_mask'].to(args.device)
                
                output_seqs = intent_model.generate(generation_config = intent_model.generation_config,
                                                    input_ids_pos = bt_pos_docs,
                                                    attention_mask_pos = bt_pos_docs_mask,
                                                    input_ids_neg = bt_neg_doc,
                                                    attention_mask_neg = bt_neg_doc_mask,
                                                    use_cache = True,
                                                    max_new_tokens=args.max_new_tokens,
                                                    min_new_tokens=args.min_new_tokens,
                                                    num_beams=args.num_beams,
                                                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                    do_sample=False,
                                                    early_stopping=args.early_stopping,
                                                    output_attentions=True, #explicitly
                                                    return_dict_in_generate=True, #explicitly
                                                    output_hidden_states=True, #explicitly
                                                    output_scores=True,
                                                    )
                # 4. get attention weights
                
                print('outputs attentions shape: ', len(output_seqs['cross_attentions'])) # 18 tokens
                print('outputs attentions shape: ', len(output_seqs['cross_attentions'][0])) # 12 layers
                print('outputs attentions shape: ', len(output_seqs['cross_attentions'][0][0])) # 2 cross attentions
                print('outputs pos attentions shape: ', output_seqs['cross_attentions'][0][0][0].shape) # torch.Size([8, 16, 1, 1024])
                print('outputs neg attentions shape: ', output_seqs['cross_attentions'][0][0][1].shape) # torch.Size([8, 16, 1, 1024])
                print('output sequences shape:', output_seqs['sequences'].shape) # torch.Size([2, 18])
                print('output sequences: ', output_seqs['sequences'])
                sequences = tokenizer.batch_decode(output_seqs['sequences'], skip_special_tokens=True)
                # output seq line by line
                print('sequences: \n')
                for seq in sequences:
                    print(seq)
                    print('\n')
                
                # get cross attention from beam 
                beam_indices = output_seqs['beam_indices'] # (batch_size * num_return_seq, seq_length)
                # for each sample in batch
                
                for i in range(args.per_gpu_train_batch_size):
                    _output_seq_id = output_seqs['sequences'][i] # (seq_length,)
                    print('_output_seq_id:', _output_seq_id)
                    _output_seq = tokenizer.batch_decode(_output_seq_id, skip_special_tokens=True)
                    print('_output_seq:', _output_seq)
                    _beam_indices = beam_indices[i] # (seq_length,)
                    print('_beam_indices:', _beam_indices)
                    ######## customized word ########
                    # if 'currently' in _output_seq:
                    #     _token_index = _output_seq.index('currently')
                    # elif ' currently' in _output_seq:
                    #     _token_index = _output_seq.index(' currently')
                    # elif 'currently ' in _output_seq:
                    #     _token_index = _output_seq.index('currently ')
                    # else:
                    #     print('token not found in the output sequence')
                    #     continue
                    if 'effects' in _output_seq:
                        _token_index = _output_seq.index('effects')
                        print(' Token "effects"found in the output sequence')
                    elif 'effects ' in _output_seq:
                        _token_index = _output_seq.index('effects ')
                        print(' Token "effects"found in the output sequence')
                    elif 'effect' in _output_seq:
                        _token_index = _output_seq.index('effect')
                        print(' Token "effects"found in the output sequence')
                    elif ' effects' in _output_seq:
                        _token_index = _output_seq.index(' effects')
                        print(' Token "effects"found in the output sequence')
                    elif 'impacted' in _output_seq:
                        _token_index = _output_seq.index('impacted')
                        print(' Token "impacted"found in the output sequence')
                    elif 'impacted ' in _output_seq:
                        _token_index = _output_seq.index('impacted ')
                        print(' Token "impacted"found in the output sequence')
                    elif ' impacted' in _output_seq:
                        _token_index = _output_seq.index(' impacted')
                        print(' Token "impacted"found in the output sequence')
                    elif 'impact' in _output_seq:
                        _token_index = _output_seq.index('impact')
                        print(' Token "impacted"found in the output sequence')
                    else:
                        print('Token impact not found in the output sequence')
                        continue

                    # _token_index = torch.where(_output_seq_id == token_id)[0].tolist()[0] # first 0: fixed, from a tupple; second 0: the first appearance of the token
                    print('_token_index: ', _token_index)
                    _token_beam_index = _beam_indices[_token_index]
                    print('_token_beam_index: ', _token_beam_index)
                    _cross_attention_pos = output_seqs['cross_attentions'][_token_index][-1][0][_token_beam_index, :, 0, :] # shape (num_heads, sequence_length)
                    _cross_attention_neg = output_seqs['cross_attentions'][_token_index][-1][1][_token_beam_index, :, 0, :] # shape
                    print('_cross_attention_pos: ', _cross_attention_pos.shape) # torch.Size([16, 1024])
                    print('_cross_attention_neg: ', _cross_attention_neg.shape) #  torch.Size([16, 1024])
                    print('range of pos_cross_attention: ', _cross_attention_pos.max(), _cross_attention_pos.min()) # 
                    print('range of neg_cross_attention: ', _cross_attention_neg.max(), _cross_attention_neg.min()) # 
                    # 5. store attention weights
                    # store sample_id, _cross_attention_pos and _cross_attention_neg in a json 
                    json.dump({
                        'sample_id': bt_sample_ids[i],
                        'pos': _cross_attention_pos.tolist(),
                        'neg': _cross_attention_neg.tolist()
                    }, f)
                    f.write('\n')
                    # # 5. visualize attention weights
                    print('input_ids_pos: ', bt_pos_docs.shape) # torch.Size([2, 1024])
                    print('\n')
                    # visualize_attention_weights(bt_pos_docs, _cross_attention_pos, tokenizer, 'pos')
                    # visualize_attention_weights(bt_neg_doc, _cross_attention_neg, tokenizer, 'neg')
                    # """

                ####### visualize pos/neg and combined hidden states #######
                # hidden states主要用于loss计算，并不直观，生成的图像没有太大意义。
                """decoder_hidden_states (tuple(tuple(torch.FloatTensor)), optional, returned when output_hidden_states=True is passed 
                or when config.output_hidden_states=True) — Tuple (one element for each generated token) of 
                tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size, generated_length, hidden_size).
                """
                # last_hidden_states_pos = output_seqs['last_hidden_state_archive'][0]
                # last_hidden_states_neg = output_seqs['last_hidden_state_archive'][1]
                # print('last_hidden_states_pos: ', last_hidden_states_pos.shape)
                # print('last_hidden_states_neg: ', last_hidden_states_neg.shape)
                
    

def get_args():
    parser = argparse.ArgumentParser()
    dataset_dir = '/home/ywang1/data1/dataset/'
    model_dir = '/home/ywang1/data1/model/'

    parser.add_argument("--load_from_checkpoint", type=str2bool, nargs='?', const=True, default=False) #if specified in command line, then true. Defalut false
    parser.add_argument("--checkpoint_dir", type=str, default="")

    parser.add_argument("--PTM", type=str, default='t5')
    parser.add_argument("--intent_model", type=str, default="t5-base") #t5-base, t5-large
    parser.add_argument("--loss_type", type=str, default="warmup", help="loss3, full")

    parser.add_argument("--train_dataset", type=str, default="q2id") # Q2ID for intent descripter
    parser.add_argument("--train_file_path", type=str, default=dataset_dir + "q2id/Q2ID_data/train_data.jsonl")
    parser.add_argument("--dev_file_path", type=str, default=dataset_dir + "q2id/Q2ID_data/valid_data.jsonl")
    parser.add_argument("--log_dir_path", type=str, default=model_dir+ "intentdescriptor/train_q2id_pos_one_question")
    parser.add_argument('--model_output_path', type=str, default=model_dir+"intentdescriptor/train_q2id_pos_one_question")
    # for test:
    parser.add_argument("--model_checkpoint_path", type=str, default="")
    parser.add_argument("--test_file_path", type=str, default="")
    parser.add_argument('--output_file_path', type=str, default="")
    parser.add_argument('--rouge_file_path', type=str, default="")

    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train") # data_structure; flat_concat_for_test
    # parser.add_argument("--decode_type", type=str, default="intent") # intent for intent descripter
    parser.add_argument("--use_prefix", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--prefix_type", type=str, default="question") #2/3 types: question, (summarization,) intent
    parser.add_argument("--pos_num", type=str, default="multiple") #one, multiple

    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=8)
    parser.add_argument("--use_data_percent", type=float, default=1)

    parser.add_argument("--num_train_epochs", type=int, default=15, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--min_new_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--disable_tqdm", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--print_steps", type=float, default=0.5)
    # parser.add_argument("--dev_frequency", type=int, default=1) # aware!
    parser.add_argument("--evaluate_every_steps", type=int, nargs='?', default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-5) # TODO
    parser.add_argument("--weight_decay", type=float, default=0.0) # TODO
    # new for beam search
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--early_stopping", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ignore_index", type=int, default=-100) #-100 for t5 and bart
    parser.add_argument("--data_version", type=str, default="v10data")

    parser.add_argument("--mode", type=str, default="train") # train, test
    parser.add_argument("--validate", type=str2bool, nargs='?', const=True, default=False) # if validate during training
    parser.add_argument("--model_variant", type=str, default="") # used to name file: useful for test script
    parser.add_argument("--encoder_loss", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--decoder_loss", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--twolinearlayers", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--patience", type=int, default=3)
    # parser.add_argument("--second_phase", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    args.device = device

    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    get_attention_weights(args)
