import logging
import re
import sys
import glob
import shutil
sys.path.append('..')
sys.path.append('.')
import os
from os import path
print(os.environ.get('CONDA_DEFAULT_ENV'))
print(os.environ.get('PYTHONPATH'))
print(sys.executable)
import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import json
from tqdm import tqdm, trange
import statistics

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.bart.modeling_bart import BartConfig
from models.model_builder import MyT5Model, MyBartModel
from torchmetrics.text.rouge import ROUGEScore
import evaluate
from rouge import Rouge
import datasets
from utils import check_dir_exist_or_build, set_seed, get_optimizer
from data_structure import T5IntentIRDataset_q2id, BartIntentIRDataset_q2id

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def save_checkpoint(model, checkpoint_dir, epoch, step, loss, loss_list, best_dev_score, epochs_no_improve, optimizer, scheduler):
    """
    To continue training a checkpoint, save optimizer, epoch, step and loss
    """
    check_dir_exist_or_build([checkpoint_dir])
    checkpoint_file = checkpoint_dir + "/checkpoint.tar"

    model_to_save = model.module if hasattr(model, 'module') else model
    # save optimizer and scheduler states, epoch and step.
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'step': step,
        'loss': loss,
        'loss_list': loss_list,
        'best_dev_score': best_dev_score,
        'epochs_no_improve': epochs_no_improve, 
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_file)

def load_checkpoint(args, model, optimizer, scheduler):
    """
    To continue training a checkpoint, load optimizer, epoch, step and loss
    """
    checkpoint_dir = args.checkpoint_dir
    checkpoint_file = checkpoint_dir + "/checkpoint.tar"

    # load optimizer and scheduler states, epoch and step.
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if 'best_dev_score' in checkpoint:
        best_dev_score = checkpoint['best_dev_score']
        epochs_no_improve = checkpoint['epochs_no_improve']
        loss_list = checkpoint['loss_list']
    else:
        best_dev_score = 0
        epochs_no_improve = 0
        loss_list = []
    step = checkpoint['step']
    
    return model, epoch, step, loss, loss_list, best_dev_score, epochs_no_improve, optimizer, scheduler

def get_average_rouge(predictions_dict, references_dict):
    predictions = []
    references = []
    for id in predictions_dict:
        if predictions_dict[id].strip().split():  # Check if prediction is not empty
            predictions.append(predictions_dict[id])
            references.append(references_dict[id])
    # rouge score
    rouge3 = datasets.load_metric("rouge", trust_remote_code=True, use_stemmer=True)
    rouge3_scores = rouge3.compute(predictions=predictions, references=references)
    # get average
    rouge_items = [rouge3_scores['rouge1'].mid.recall, rouge3_scores['rouge2'].mid.recall, rouge3_scores['rougeL'].mid.recall]
    avg_rouge = sum(rouge_items) / len(rouge_items)
    return avg_rouge, rouge_items

def train(args, logger):
    ################ load model ################
    # load our model
    if args.intent_model == "my-t5-base" or args.intent_model == "my-t5-large":
        model_type = "-".join(args.intent_model.split("-")[-2:])
        model_type = "google-t5/" + model_type
        tokenizer = T5Tokenizer.from_pretrained(model_type)
        config = T5Config.from_pretrained(model_type)
        intent_model = MyT5Model(model_type, args, config).to(args.device)
    elif args.intent_model in ["my-bart-large", "my-bart-base"]:
        model_type = "-".join(args.intent_model.split("-")[-2:])
        model_type = "facebook/" + model_type
        config = BartConfig.from_pretrained(model_type)
        intent_model = MyBartModel(model_type, args, config).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        # bart model has bos_token_id
    else:
        raise ValueError("Intent generator not supported.")
    
    # gpu setting
    if args.n_gpu > 1:
        intent_model = torch.nn.DataParallel(intent_model, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    ################ data loader ################
    if args.train_dataset == "q2id":
        if args.PTM == 'bart':
            train_dataset = BartIntentIRDataset_q2id(args, tokenizer, args.train_file_path)
            dev_dataset = BartIntentIRDataset_q2id(args, tokenizer, args.dev_file_path)
        elif args.PTM == 't5':
            train_dataset = T5IntentIRDataset_q2id(args, tokenizer, args.train_file_path)
            dev_dataset = T5IntentIRDataset_q2id(args, tokenizer, args.dev_file_path)
        
        train_loader = DataLoader(train_dataset, 
                            batch_size = args.batch_size, 
                            shuffle=True, 
                            collate_fn=train_dataset.get_collate_fn(args))
        dev_loader = DataLoader(dev_dataset, 
                            batch_size = args.batch_size, 
                            shuffle=True, 
                            collate_fn=dev_dataset.get_collate_fn(args))
    
    logger.info("train dataset samples num = {}".format(len(train_dataset)))
    logger.info("dev dataset samples num = {}".format(len(dev_dataset)))
    
    num_steps_per_epoch = (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    total_training_steps = args.num_train_epochs * num_steps_per_epoch
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, intent_model, weight_decay=args.weight_decay) #args.learning_rate
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    load_epoch = 0
    best_dev_score = float('-inf') #save
    epochs_no_improve = 0 #save
    # check if load from checkpoint: 
    if args.load_from_checkpoint:
        logger.info("Loading last epoch training checkpoint...")
        logger.info("Load from checkpoint path: {}".format(args.model_output_path))
        intent_model, load_epoch, last_step, latest_loss, loss_list, best_dev_score, epochs_no_improve, optimizer, scheduler = load_checkpoint(args, intent_model, optimizer, scheduler)
        logger.info("Load checkpoint, epoch = {}, last_local_step = {}, training_loss = {}, best_dev_score = {}, epochs_no_improve = {}".format(load_epoch, last_step, latest_loss, best_dev_score, epochs_no_improve))

    # begin to train
    logger.info("############## Start training ##############")
    logger.info("Training epochs = {}".format(args.num_train_epochs))
    logger.info("Trained epochs in previous stage = {}".format(load_epoch))
    logger.info("Total training steps = {}".format(total_training_steps))
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    if not args.load_from_checkpoint: # train from scratch
        # iterable object with tqdm progress bar
        epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
        best_train_loss = float('inf')
        loss_list = []
    else:
        # iterable object start from checkpoint epoch
        epoch_iterator = trange(load_epoch+1, args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
        checkpoint_pattern = os.path.join(args.model_output_path, 'checkpoint_besttrain_*')
        checkpoint_subfolders = glob.glob(checkpoint_pattern)
        filename = os.path.basename(checkpoint_subfolders[0])
        match = re.search('loss[^_]*_(-?\d+\.\d+)', filename)
        if match:
            best_train_loss = float(match.group(1))
            print("Best train loss so far: ", best_train_loss)
        else:
            raise ValueError("No best train loss found in checkpoint path.")

    ############# training begin #############
    early_stop = False
    for epoch in epoch_iterator:
        for step, batch in enumerate(tqdm(train_loader,  desc="Step", disable=args.disable_tqdm)):
            intent_model.train()
            intent_model.zero_grad()

            bt_sample_ids = batch['bt_sample_ids']# .to(args.device) # B * len
            bt_pos_docs = batch['bt_pos_docs'].to(args.device) # B * len one pos
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_doc = batch['bt_neg_doc'].to(args.device) # B * len batch size negs
            bt_neg_doc_mask = batch['bt_neg_doc_mask'].to(args.device)            
            bt_neg_sim_score = None # for v-like dataset
            if 'bt_labels' in batch:
                bt_labels = batch['bt_labels'].to(args.device)
            if 'bt_decoder_input_ids' in batch:# for my bart
                bt_decoder_input_ids = batch['bt_decoder_input_ids'].to(args.device)
            if 'bt_decoder_attention_mask' in batch: #for bart models
                bt_decoder_attention_mask = batch['bt_decoder_attention_mask'].to(args.device)
                
            bt_pos_doc_length = batch['bt_pos_doc_length']
            bt_neg_doc_length = batch['bt_neg_doc_length']

            # for our model:
            if args.intent_model in ["my-bart-large", "my-bart-base"]:
                output = intent_model(input_ids_pos=bt_pos_docs, 
                                  attention_mask_pos=bt_pos_docs_mask,
                                  input_ids_neg=bt_neg_doc,
                                  attention_mask_neg=bt_neg_doc_mask,
                                  neg_sim_score=bt_neg_sim_score,
                                  labels=bt_labels,
                                  decoder_attention_mask = bt_decoder_attention_mask,
                                  pos_doc_length=bt_pos_doc_length,
                                  neg_doc_length=bt_neg_doc_length,
                                  current_epoch=epoch,
                                  use_cache=False) ######## only true in autoregressive decoding! Not in training!
                loss = output.loss[0]
            elif args.intent_model == 'my-t5-large':
                # https://huggingface.co/docs/transformers/main/en/model_doc/t5
                output = intent_model(input_ids_pos=bt_pos_docs, 
                                  attention_mask_pos=bt_pos_docs_mask,
                                  input_ids_neg=bt_neg_doc,
                                  attention_mask_neg=bt_neg_doc_mask,
                                  neg_sim_score=bt_neg_sim_score,
                                  labels=bt_labels,
                                  pos_doc_length=bt_pos_doc_length,
                                  neg_doc_length=bt_neg_doc_length,
                                  current_epoch=epoch,
                                  use_cache=False) ######## only true in autoregressive decoding! Not in training!
                # loss is a tuple containing loss1, encoder_loss, decoder_loss
                loss = output.loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(intent_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # save best training checkpoint
            if loss < best_train_loss:
                checkpoint_prefix = os.path.join(args.model_output_path, 'checkpoint_besttrain')
                checkpoint_dir = checkpoint_prefix + '_{}_{:.6f}_epoch_{}_step_{}'.format(args.loss_type,loss, epoch, step)
                # save_checkpoint(intent_model, checkpoint_dir, epoch, step, loss, loss_list, best_dev_score, epochs_no_improve, optimizer, scheduler)
                logger.info("Epoch = {}, local Step = {}, total loss = {}".format(
                                epoch, # original to be epoch + 1
                                step,
                                output.loss))
                # if epoch <= 3:
                #     # remove all subfolders except the latest best train checkpoint
                #     for d in glob.glob(checkpoint_prefix + '*'):
                #         if d != checkpoint_dir:
                #             # os.rmdir(d)
                #             shutil.rmtree(d)
                best_train_loss = loss
            ################# evaluate performance on validation set periodically #################
            if args.validate: 
            ############################ evaluate on rouge score ############################
                if step % args.evaluate_every_steps == 0 and step != 0: # once per epoch
                    intent_model.eval()
                    with torch.no_grad():
                        predictions_dict = {}
                        references_dict = {}
                        for dev_step, dev_batch in enumerate(tqdm(dev_loader,  desc="Step", disable=args.disable_tqdm)):
                            dev_bt_pos_docs = dev_batch['bt_pos_docs'].to(args.device) # B * len one pos
                            dev_bt_pos_docs_mask = dev_batch['bt_pos_docs_mask'].to(args.device)
                            dev_bt_neg_doc = dev_batch['bt_neg_doc'].to(args.device) # B * len batch size negs
                            dev_bt_neg_doc_mask = dev_batch['bt_neg_doc_mask'].to(args.device)
                            
                            if args.PTM == 't5':
                                model_type = "-".join(args.intent_model.split("-")[-2:])
                                model_type = "google-t5/" + model_type
                                generation_config = GenerationConfig.from_pretrained(model_type)
                                generation_config.bos_token_id = intent_model.config.pad_token_id
                                intent_model.generation_config = generation_config
                            output_seqs = intent_model.generate(
                                                                generation_config = intent_model.generation_config,
                                                                input_ids_pos = dev_bt_pos_docs,
                                                                attention_mask_pos = dev_bt_pos_docs_mask,
                                                                input_ids_neg = dev_bt_neg_doc,
                                                                attention_mask_neg = dev_bt_neg_doc_mask,
                                                                use_cache = True,
                                                                max_new_tokens=args.max_new_tokens,
                                                                min_new_tokens=args.min_new_tokens,
                                                                do_sample=False,
                                                                num_beams=args.num_beams,
                                                                no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                early_stopping=args.early_stopping,
                                                                )

                            outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
                            # intent generation
                            for i, sample_id in enumerate(dev_batch['bt_sample_ids']):
                                predictions_dict[sample_id] = outputs[i]
                        
                        # load references
                        with open(args.dev_file_path, 'r') as f:
                            for line in f:
                                record = json.loads(line)
                                sample_id = record['sample_id']
                                reference = record['intent']
                                references_dict[sample_id] = reference
                        
                        # get average rouge scores
                        avg_rouge, rouge_items = get_average_rouge(predictions_dict, references_dict)
                        if avg_rouge > best_dev_score:
                            best_dev_score = avg_rouge
                            epochs_no_improve = 0
                            logger.info("Saving best checkpoint on dev rouge: Train_Epoch = {}, Local Step = {}, train loss = {}, avg_rouge = {}, rouge_scores = {}".format(
                                            epoch,
                                            step,
                                            output.loss,
                                            avg_rouge,
                                            rouge_items))
                            checkpoint_prefix = os.path.join(args.model_output_path, 'checkpoint_bestrouge')
                            checkpoint_dir = checkpoint_prefix + '_rouge_{:.6f}_epoch_{}_step_{}'.format(avg_rouge, epoch, step)
                            save_checkpoint(intent_model, checkpoint_dir, epoch, step, loss, loss_list, best_dev_score, epochs_no_improve, optimizer, scheduler)
                        else:
                            epochs_no_improve += 1
                            logger.info("No improvement on dev rouge: Train_Epoch = {}, Local Step = {}, train loss = {}, avg_rouge = {}".format(
                                            epoch,
                                            step,
                                            output.loss,
                                            # loss.item(),
                                            avg_rouge))
                            if epochs_no_improve >= args.patience:
                                early_stop = True
                                logger.info("########### Early stopping at epoch = {} ###############".format(epoch))
                                # break
            # save last step training checkpoint to resume training later on
            if step == num_steps_per_epoch - 1:
                checkpoint_prefix = os.path.join(args.model_output_path, 'checkpoint_lasttrain')
                checkpoint_dir = checkpoint_prefix + '_loss_{:.6f}_epoch_{}_step_{}'.format(loss, epoch, step)
                save_checkpoint(intent_model, checkpoint_dir, epoch, step, loss, loss_list, best_dev_score, epochs_no_improve, optimizer, scheduler)
                logger.info("Epoch = {}, local Step = {}, total loss = {}".format(
                                epoch, # original to be epoch + 1
                                step,
                                output.loss))

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, local Step = {}, total loss = {}".format(
                                epoch, # original to be epoch + 1
                                step,
                                output.loss))
            # store loss for plot
            if step % 2500 == 0:
                loss_list.append(loss.item())
            global_step += 1 # avoid saving the model of the first step.
                        
            if early_stop:
                break   
        if early_stop:
            break 
    logger.info("Training finish!")          

def inference(args):
    # load pretrained model
    if args.intent_model == "my-t5-base" or args.intent_model == "my-t5-large":
        model_type = "-".join(args.intent_model.split("-")[-2:])
        model_type = "google-t5/" + model_type
        tokenizer = T5Tokenizer.from_pretrained(model_type)
        config = T5Config.from_pretrained(model_type)
        intent_model = MyT5Model(model_type, args, config).to(args.device)

        # set BOS token id in generation config to avoid error
        generation_config = GenerationConfig.from_pretrained(model_type)
        generation_config.bos_token_id = intent_model.config.pad_token_id
        intent_model.generation_config = generation_config
    elif args.intent_model in ["my-bart-large", "my-bart-base"]:
        model_type = "-".join(args.intent_model.split("-")[-2:])
        model_type = "facebook/" + model_type
        config = BartConfig.from_pretrained(model_type)
        intent_model = MyBartModel(model_type, args, config).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        # bart model has bos_token_id

    # load checkpoint model
    checkpoint = torch.load(args.model_checkpoint_path+ "/checkpoint.tar")
    # print('checkpoint:', checkpoint)
    intent_model.load_state_dict(checkpoint['model_state_dict'])

    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.train_dataset == 'q2id':
        if args.PTM == 'bart':
            test_dataset = BartIntentIRDataset_q2id(args, tokenizer, args.test_file_path)
        elif args.PTM == 't5':
            test_dataset = T5IntentIRDataset_q2id(args, tokenizer, args.test_file_path)
            
    test_dataloader = DataLoader(test_dataset, 
                                # sampler=SequentialSampler(test_dataset),
                                # shuffle=False,
                                shuffle=True,
                                batch_size=args.batch_size, 
                                collate_fn=test_dataset.get_collate_fn(args))
    
    # begin to inference
    with open(args.output_file_path, "w") as f:
        with torch.no_grad():
            intent_model.eval()
            for batch in tqdm(test_dataloader, desc="Step"):
                bt_pos_docs = batch['bt_pos_docs'].to(args.device)
                bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
                bt_neg_doc = batch['bt_neg_doc'].to(args.device)
                bt_neg_doc_mask = batch['bt_neg_doc_mask'].to(args.device)
                if args.n_gpu > 1:
                    pass #TODO
                else:
                    # our model, input_ids is None. pos and neg docs inputs are model_kwargs.
                    if args.PTM == 't5':
                                    model_type = "-".join(args.intent_model.split("-")[-2:])
                                    model_type = "google-t5/" + model_type
                                    generation_config = GenerationConfig.from_pretrained(model_type)
                                    generation_config.bos_token_id = intent_model.config.pad_token_id
                                    intent_model.generation_config = generation_config

                    output_seqs = intent_model.generate(
                                                        generation_config = intent_model.generation_config,
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
                                                        )
                    
                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)

                for i in range(len(outputs)):
                    record = {}
                    if isinstance(batch["bt_sample_ids"][i], torch.Tensor):
                        record["sample_id"] = batch["bt_sample_ids"][i].item()
                    else:
                        record["sample_id"] = batch["bt_sample_ids"][i]
                    if args.train_dataset == "q2id":
                        record['intent'] = outputs[i]
                    
                    f.write(json.dumps(record) + '\n') 

    logger.info("Inference finsh!")

def evaluate_intent(args):
    """
    1. check if intent generation file exists, otherwise do inference
    2. load test data and intent generation file, get ground truth and inference texts
    3. get rouge score
    """
    #check if intent generation file exists, otherwise do inference
    if not os.path.exists(args.output_file_path):
        inference(args) # inference

    #check if final result file exists, otherwise do inference
    output_dir = os.path.dirname(args.output_file_path)
    if args.rouge_file_path != '':
        output_file = args.rouge_file_path
    else:
        if args.early_stopping:
            output_file = os.path.join(output_dir, 'rouge.json')
        else:
            output_file = os.path.join(output_dir, f'rouge_{args.min_new_tokens}-{args.max_new_tokens}_beam{args.num_beams}_{args.no_repeat_ngram_size}_notearly.json')
    
    if os.path.exists(output_file):
        return

    #load test data and intent generation file
    prediction_dict = {}
    with open(args.output_file_path, "r") as f:
        for line in f:
            sample_id = json.loads(line)["sample_id"]
            prediction = json.loads(line)["intent"]
            prediction_dict[sample_id] = prediction
    
    reference_dict = {}
    with open(args.test_file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            sample_id = record["sample_id"]
            reference = record["intent"]
            reference_dict[sample_id] = reference

    predictions = []
    references = []
    for id in prediction_dict:
        if prediction_dict[id].strip().split():  # Check if prediction is not empty
            predictions.append(prediction_dict[id])
            references.append(reference_dict[id])
    print('length:', len(predictions), len(references))

    # 1.1 get rouge score, need to remove empty predictions first, or generate at least one token
    rouge = Rouge()
    rouge_scores = None
    if predictions and references:
        rouge_scores = rouge.get_scores(predictions, references, avg=True)
    elif not predictions:
        print("Predictions are empty.")
    elif not references:
        print("References are empty.")

    # 1.2 rouge from evaluate:
    rouge2 = ROUGEScore(use_stemmer=True)
    results = rouge2(preds=predictions, target=references)
    rouge2_scores = {}
    rouge2_scores['rouge1_fmeasure'] = results['rouge1_fmeasure'] * 100
    rouge2_scores['rouge2_fmeasure'] = results['rouge2_fmeasure'] * 100
    rouge2_scores['rougeL_fmeasure'] = results['rougeL_fmeasure'] * 100
    rouge2_scores['rouge1_recall'] = results['rouge1_recall'] * 100
    rouge2_scores['rouge2_recall'] = results['rouge2_recall'] * 100
    rouge2_scores['rougeL_recall'] = results['rougeL_recall'] * 100
    rouge2_scores = {k: round(v.item(),4) for k, v in rouge2_scores.items()}
    print("Done with 2 rouge scores!")

    # 1.3 rouge from datasets:
    rouge3 = datasets.load_metric("rouge", trust_remote_code=True, use_stemmer=True)
    results = rouge3.compute(predictions=predictions, references=references)
    rouge3_scores = {}
    rouge3_scores['rouge1_fmeasure'] = results['rouge1'].mid.fmeasure * 100
    rouge3_scores['rouge2_fmeasure'] = results['rouge2'].mid.fmeasure * 100
    rouge3_scores['rougeL_fmeasure'] = results['rougeL'].mid.fmeasure * 100
    rouge3_scores['rouge1_recall'] = results['rouge1'].mid.recall * 100
    rouge3_scores['rouge2_recall'] = results['rouge2'].mid.recall * 100
    rouge3_scores['rougeL_recall'] = results['rougeL'].mid.recall * 100
    rouge3_scores = {k: round(v,4) for k, v in rouge3_scores.items()}
    print("Done with 3 rouge scores!")

    # 2. bleu score:
    bleu = evaluate.load("bleu")
    bleu_scores = bleu.compute(predictions=predictions, references=references)
    print("Done with bleu scores!")
    
    # 3. meteor score:
    meteor = evaluate.load('meteor')
    meteor_scores = meteor.compute(predictions=predictions, references=references)
    print("Done with meteor scores!")
    
    # 4. bert_score:
    bertscore = evaluate.load("bertscore")
    model_type='microsoft/deberta-xlarge-mnli'
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang='en', model_type = model_type)
    bert_scores['precision'] = statistics.mean(bert_scores['precision'])
    bert_scores['recall'] = statistics.mean(bert_scores['recall'])
    bert_scores['f1'] = statistics.mean(bert_scores['f1'])
    print("Done with bert_scores!")

    # score dict
    all_scores = {
    "rouge": rouge_scores,
    "rouge2":rouge2_scores,
    "rouge3":rouge3_scores,
    "bleu": bleu_scores,
    "meteor": meteor_scores,
    "bert": bert_scores
    }
    
    with open(output_file, "w") as f:
        json.dump(all_scores, f, indent=4)

def get_args():
    parser = argparse.ArgumentParser()
    dataset_dir = '/home/XXX/data1/dataset/'
    model_dir = '/home/XXX/data1/model/'

    parser.add_argument("--load_from_checkpoint", type=str2bool, nargs='?', const=True, default=False) #if specified in command line, then true. Defalut false
    parser.add_argument("--checkpoint_dir", type=str, default="")

    parser.add_argument("--PTM", type=str, default='t5')
    parser.add_argument("--intent_model", type=str, default="t5-base") #t5-base, t5-large
    parser.add_argument("--loss_type", type=str, default="loss3nce")

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
    parser.add_argument("--use_prefix", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--prefix_type", type=str, default="question")
    parser.add_argument("--pos_num", type=str, default="multiple")

    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=2)
    parser.add_argument("--use_data_percent", type=float, default=1)

    parser.add_argument("--num_train_epochs", type=int, default=10, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=1024, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=1024, help="Max doc length")
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=1024, help="Max concatenation length of the session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--disable_tqdm", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("--print_steps", type=float, default=0.5)
    parser.add_argument("--evaluate_every_steps", type=int, nargs='?', default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
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

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    args.device = device

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    logging.basicConfig(filename=args.log_dir_path,level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) #create a logger with the name of the current module
    if args.mode == 'train':
        train(args, logger)
    elif args.mode == 'test':
        evaluate_intent(args)