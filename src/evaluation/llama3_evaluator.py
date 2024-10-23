import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import torch.nn.functional as F
import json
import os
import glob
import re
import time
import argparse
import openai
from openai import OpenAI

def get_pipeline(model_id, model_type):
    if model_type in ["llama3_8b", "llama3.1_8b"]:
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
    
def get_example_input(model_type,
                    metric = None,
                    query = None, 
                    relevant_docs = None, 
                    irrelevant_docs = None, 
                    GT_intent = None, 
                    generated_intent = None):
    if model_type in ['llama3_8b', 'llama3.1_8b']:
        example_input = {"role": "user", 
                     "content": f"Query: {query} \n Relevant Documents: {relevant_docs} \n Irrelevant Documents: {irrelevant_docs} \n Generated Intent: {generated_intent} \n Ground Truth Intent: {GT_intent} \n \n Evaluation Form: \n - {metric}: \n - Reason:"}    
    elif model_type in ['gpt-4o', 'gpt-4']:
        example_input = {"role": "user", 
                     "content": f"Query: {query} \n Relevant Documents: {relevant_docs} \n Irrelevant Documents: {irrelevant_docs} \n Generated Intent: {generated_intent} \n Ground Truth Intent: {GT_intent} \n \n Evaluation Form (scores ONLY): \n - {metric}:"}
    return example_input

class Llama3:
    def __init__(self, model_path, model_type):
        self.model_id = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        # self.pipeline = get_pipeline(self.model_id, model_type)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    def get_score_index(self, response_ids):
        # score tokens
        score_tokens = [str(i) for i in range(1, 6)]
        score_token_ids = self.tokenizer.convert_tokens_to_ids(score_tokens)
        score_token_ids = torch.tensor(score_token_ids).to(self.model.device) #tensor([16, 17, 18, 19, 20])

        # locate score index
        for i, token_id in enumerate(response_ids[:30]): # only check the first 15 tokens
            if token_id in score_token_ids:
                return i
        return None
    
    def get_score_prob(self, score_prob):
        # score tokens
        score_tokens = [str(i) for i in range(1, 6)]
        score_token_ids = self.tokenizer.convert_tokens_to_ids(score_tokens)

        # get the probability of the score token
        score_prob = score_prob[score_token_ids] # probability of the score tokens [1, 5]

        return score_prob
  
    def get_response(
          self, 
          query, 
          message_history=[], 
          max_tokens=4096, 
          temperature=0.6, 
          top_p=0.9
      ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        input_ids = self.tokenizer.apply_chat_template(
            user_prompt,
            add_generation_prompt=True, # add the generation prompt like <|im_start|>assistant to guide LLM to continue the conversation
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # response = outputs[0]["generated_text"][input_ids.shape[-1]:]
        # print('outputs sequence size:', outputs.sequences.size()) # [1, 4248]
        # print('input_ids size:', input_ids.size()) # [1, 4089]
        response_ids = outputs.sequences[:, input_ids.shape[-1]:][0] # [1, 159]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        # locate score index
        score_index = self.get_score_index(response_ids)
        print('score_index:', score_index)
        if not score_index:
            print("Score not found in the response.")
            score_prob = torch.tensor([1., 0., 0., 0., 0.]).to(self.model.device)
            normalized_score = torch.sum(torch.tensor([1, 2, 3, 4, 5]).to(score_prob.device) * score_prob)
        else:
            # add softmax to get the probability of the score tokens
            score_prob = F.softmax(outputs.scores[score_index].squeeze(), dim=0) #torch.Size([128256])
            score_prob = self.get_score_prob(score_prob) # probs of the 5 score tokens
            normalized_score = torch.sum(torch.tensor([1, 2, 3, 4, 5]).to(score_prob.device) * score_prob)
            print('score_prob:', score_prob)
            print('normalized_score:', normalized_score)

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response, score_prob, normalized_score, user_prompt + [{"role": "assistant", "content": response}]
    
    def evaluate(self, system_instructions="", user_input=""):
        # evaluate an example on one metric
        conversation = [{"role": "system", "content": system_instructions}]
        
        response, score_prob, normalized_score, conversation = self.get_response(user_input, conversation)
        return response, score_prob, normalized_score, conversation


class GPT4o:
    def __init__(self, model_path, model_type):
        self.model_type = model_type

    def get_response(
          self, 
          query, 
          message_history=[], 
          max_tokens=8, 
          temperature=2, 
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None,
          n=20
      ):

        user_prompt = message_history + [{"role": "user", "content": query}]
        # user_prompt = [{"role": "user", "content": 'give me an integer: [score only]'}]
        client = OpenAI()
        while True:
            try:
                _response = client.chat.completions.create(
                    model=self.model_type,
                    messages=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    # logprobs=40,
                    n=n
                )
                time.sleep(0.5)
                # change bracket notation to dot donation: 'ChatCompletion' object is not subscriptable
                # all_responses = [_response['choices'][i]['message']['content'] for i in
                #                  range(len(_response['choices']))]
                all_responses = [choice.message.content for choice in
                                 _response.choices]
                return all_responses
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    print('ignored')
                    return None

    def evaluate(self, system_instructions="", user_input=""):
        # evaluate an example on one metric
        conversation = [{"role": "system", "content": system_instructions}]
        
        response = self.get_response(user_input, conversation)
        return response


        
def evaluate(model_path, model_type, prediction_type, test_data_path, prediction_path, output_path, instruction, run):
    # read data
    reference_dict = {}
    test_data = []
    test_sampeld_ids = []
    with open(test_data_path, "r") as f:
        for line in f:
            record = json.loads(line)
            sample_id = record["sample_id"]
            reference = record["intent"]
            reference_dict[sample_id] = reference
            test_data.append(record)
            test_sampeld_ids.append(sample_id)
    
    # load all predictions
    prediction_dict = {}
    with open(prediction_path, "r") as f:
        for line in f:
            sample_id = json.loads(line)["sample_id"]
            prediction = json.loads(line)["intent"]
            prediction_dict[sample_id] = prediction

    print('number of predictions:', len(prediction_dict))
    print('number of test data:', len(test_data))

    # read metrics instructions
    # get all insturction jsonl file paths from the folder
    instruction_folder = '/home/XXX/workspace/IntentRewriter/src/evaluation'
    instruction_paths = glob.glob(instruction_folder + f'/*{instruction}_instruction.jsonl')

    if model_type in ['llama3_8b', 'llama3.1_8b']:
        # initialize llama3
        model = Llama3(model_path, model_type)
    elif model_type in ['gpt-4o', 'gpt-4']:
        model = GPT4o(model_path, model_type)

    # for each metric, evaluate all samples from the dataset
    # system instruction remain the same for each metrics
    for p in instruction_paths:
        data = []
        with open(p, 'r')as f1:
            for line in f1:
                data.append(json.loads(line)) # dict
            instruction = data[0]
            print("Evaluation on:", instruction['metric'])

            output_file = os.path.join(output_path, f'normalized_{model_type}_{prediction_type}_{instruction['metric']}.jsonl')
            with open(output_file, 'w')as f2:
                ignore = 0
                for record in test_data:
                    sample_id = record["sample_id"]
                    print("sample id:", sample_id)
                    query = record["query"]
                    relevant_docs = ' '.join(record['pos'])
                    irrelevant_docs = record["neg"][0]
                    GT_intent = record['intent']
                    generated_intent = prediction_dict[sample_id]

                    # run 5 times
                    for _ in range(1):
                        user_input = get_example_input(model_type,
                                                        instruction['metric'],
                                                        query, 
                                                        relevant_docs, 
                                                        irrelevant_docs, 
                                                        GT_intent, 
                                                        generated_intent)
                        if model_type in ['llama3_8b', 'llama3.1_8b']:
                            response, score_prob, normalized_score, conversation = model.evaluate(instruction['content'], user_input['content'])   
                            new_record = {sample_id: (normalized_score.tolist(), score_prob.tolist(), response)}
                            json.dump(new_record, f2)
                            f2.write('\n')
                        elif model_type == 'gpt-4o' or model_type == 'gpt-4':
                            response = model.evaluate(instruction['content'], user_input['content'])
                            if not response:
                                ignore += 1
                            new_record = {sample_id: response}
                            json.dump(new_record, f2)
                            f2.write('\n')
                    # break
                print('ignored total for gpt-4o', ignore)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True, help="openai key")
    parser.add_argument("--model_type", type=str, default='llama3.1_8b', help="model type")
    parser.add_argument("--prediction_type", type=str, default='var3loss1')
    parser.add_argument("--instruction", type=str, default='inclusion', help="instruction type")
    parser.add_argument("--run", type=int, default=1, help="run number")

    args = parser.parse_args()
    # openai.api_key = args.key
    model_type = args.model_type # 'llama3_8b' # llama3_8b_quantized
    prediction_type = args.prediction_type
    instruction= args.instruction

    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    if model_type == 'llama3_8b':
        model_id = '/home/XXX/data1/model/meta-llama/Meta-Llama-3-8B-Instruct'
    elif model_type == 'llama3.1_8b':
        model_id = '/home/XXX/data1/model/meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif model_type == 'gpt-4o' or model_type == 'gpt-4':
        model_id = ''

    print('model_type:', model_type)
    print('model_id:', model_id)

    if prediction_type == 'mybartlarge':
        prediction_path = '/home/XXX/data1/model/intentdescriptor/bart/v10data_train_var6loss3nce_my-bart-large_batch_2_lr_0.00001_twolinearlayers_new/checkpoint_lasttrain_loss_0.337129_epoch_8_step_7499/predictions_50.json'
        test_data_path = '/home/XXX/data1/dataset/q2id/Q2ID_data/test_data_v10_human_50.jsonl'
    elif prediction_type == 'sota': # wrong vanilla t5 # depracated
        # processed sota prediction ( see test_data.ipynb)
        prediction_path = '/home/XXX/data1/output-socratic/q2id_8_512_strided_socratic_30M_gpu_batch2_1e5_4/selected_checkpoint/predictions_50.jsonl'
        test_data_path = '/home/XXX/data1/dataset/q2id/Q2ID_data/test_data_v10_human_50.jsonl'
    output_path =f'/home/XXX/workspace/IntentRewriter/src/evaluation/output/{instruction}'
    
    evaluate(model_id, model_type, prediction_type, test_data_path, prediction_path, output_path, instruction, args.run)