import transformers
import torch
import json
import os
import glob
import re
import argparse
import statistics
import evaluate
from rouge import Rouge
from torchmetrics.text.rouge import ROUGEScore
import datasets


def get_pipeline(model_id, model_type):
    if model_type in ["llama3_8b","llama3.1_8b"]:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
    )
    elif model_type == 'llama3_8b_quantized':
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
    return pipeline
    
def get_user_input(query = None, 
                    relevant_docs = None, 
                    irrelevant_docs = None):
    user_input = {"role": "user", 
                     "content": f"Query: {query} \n Relevant Documents: {relevant_docs} \n Irrelevant Documents: {irrelevant_docs} \n Intent:"}    
    return user_input

class Llama3:
    def __init__(self, model_path, model_type):
        self.model_id = model_path
        self.model_type = model_type
        self.pipeline = get_pipeline(self.model_id, self.model_type)
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
  
    def get_response(
          self, 
          query, 
          message_history=[], 
          max_tokens=256, 
          temperature=0.6, 
          top_p=0.9
      ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        # user_prompt = self.pipeline.tokenizer.apply_chat_template(
        #     user_prompt, tokenize=False, add_generation_prompt=True
        # )
#         user_prompt = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
        outputs = self.pipeline(
            user_prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(user_prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]
    
    def evaluate(self, system_instructions="", user_input="", output_length=128):
        # evaluate an example on one metric
        conversation = [{"role": "system", "content": system_instructions}]
        
        response, conversation = self.get_response(user_input, conversation, max_tokens=output_length)
        return response, conversation

def inference(model_path, model_type, test_data_path, output_path, output_length, zero_shot):
    # read data
    reference_dict = {} # id2intent
    test_data = []
    with open(test_data_path, "r") as f:
        for line in f:
            record = json.loads(line)
            sample_id = record["sample_id"]
            reference = record["intent"]
            reference_dict[sample_id] = reference
            test_data.append(record)

    # read instruction
    if not zero_shot:   
        instruction_path = '/home/XXX/workspace/IntentRewriter/src/evaluation/few_shot_prompt.jsonl'
    else:
        instruction_path = '/home/XXX/workspace/IntentRewriter/src/evaluation/zero_shot_prompt.jsonl'
    
    # initialize llama3
    llama3 = Llama3(model_path, model_type)

    # predictions = []
    with open(instruction_path, 'r')as f1:
        for line in f1:
            instruction = json.loads(line) # dict
        print("Fewshot instruction:", instruction['content'])
        with open(output_path, 'w')as f2:
            for record in test_data:
                sample_id = record["sample_id"]
                query = record["query"]
                relevant_docs = ' '.join(record['pos'])
                if len(record["neg"]) == 0:
                    irrelevant_docs = ""
                else:
                    irrelevant_docs = record["neg"][0]
                GT_intent = record['intent']
                # run 5 times
                for _ in range(1):
                    user_input = get_user_input(query, 
                                                    relevant_docs, 
                                                    irrelevant_docs)
                    response, conversation = llama3.evaluate(instruction['content'], user_input['content'], output_length)
                    # save score and reason
                    new_record = {"sample_id": sample_id, "intent": response}
                    # predictions.append(new_record)
                    json.dump(new_record, f2)
                    f2.write('\n')

def evaluate_intent(prediction_path, test_data_path, output_file):
    """
    1. check if intent generation file exists, otherwise do inference
    2. load test data and intent generation file, get ground truth and inference texts
    3. get rouge score
    """
    #load test data and intent generation file
    prediction_dict = {}
    with open(prediction_path, "r") as f:
        for line in f:
            sample_id = json.loads(line)["sample_id"]
            prediction = json.loads(line)["intent"][0]["content"]
            prediction_dict[sample_id] = prediction
    
    reference_dict = {}
    with open(test_data_path, "r") as f:
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
    
    # 4. bert_score:
    # Currently, the best model is microsoft/deberta-xlarge-mnli, please consider using 
    # it instead of the default roberta-large in order to have the best correlation with human evaluation.
    bertscore = evaluate.load("bertscore")
    model_type='microsoft/deberta-xlarge-mnli'
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang='en', model_type = model_type)
    bert_scores['precision'] = statistics.mean(bert_scores['precision'])
    bert_scores['recall'] = statistics.mean(bert_scores['recall'])
    bert_scores['f1'] = statistics.mean(bert_scores['f1'])
    print("Done with bert_scores!")

    # score dict
    all_scores = {
    "rouge3":rouge3_scores,
    "bleu": bleu_scores,
    "bert": bert_scores
    }
    
    with open(output_file, "w") as f:
        json.dump(all_scores, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='llama3_8b', help="model type")
    parser.add_argument("--data_type", type=str, default='v3', help="data type")
    parser.add_argument("--output_length", type=int, default=256, help="output length")
    parser.add_argument("--run", type=int, default=1, help="run number")
    parser.add_argument("--zero_shot", action="store_true", help="whether to run zero shot")

    args = parser.parse_args()
    model_type = args.model_type # 'llama3_8b' # llama3_8b_quantized
    output_length = args.output_length
    zero_shot = args.zero_shot
    if model_type == 'llama3_8b':
        model_id = '/home/XXX/data1/model/meta-llama/Meta-Llama-3-8B-Instruct'
    elif model_type == 'llama3.1_8b':
        model_id = '/home/XXX/data1/model/meta-llama/Meta-Llama-3.1-8B-Instruct'
    print('model_type:', model_type)
    print('model_id:', model_id)

    test_data_path = f'/home/XXX/data1/dataset/q2id/Q2ID_data/test_data_{args.data_type}.jsonl'   
    if not args.zero_shot:
        output_dir =f'/home/XXX/workspace/IntentRewriter/src/evaluation/output/{model_type}'
    else:
        output_dir = f'/home/XXX/workspace/IntentRewriter/src/evaluation/output/{model_type}_zero_shot'

    if not zero_shot:
        prediction_path = os.path.join(output_dir, f'predictions_{args.run}.jsonl')
    else:
        prediction_path = os.path.join(output_dir, f'clean_prediction_{args.run}.jsonl')
    evaluation_path = os.path.join(output_dir, f'rouge_{args.run}.json')
    
    # inference then evaluate
    if not os.path.exists(prediction_path):
        inference(model_id, model_type, test_data_path, prediction_path, output_length, zero_shot)
    
    if not os.path.exists(evaluation_path):
        evaluate_intent(prediction_path, test_data_path, evaluation_path)

    
    