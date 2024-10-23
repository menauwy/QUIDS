import openai
import json
import argparse
import tqdm
import time

def evaluate(prediction_type, test_data_path, prediction_path, output_path, instruction):
    # load data
    summeval = json.load(open(args.summeval_fp))
    prompt = open(args.prompt_fp).read()

    ct, ignore = 0, 0
    new_json = []
    for instance in tqdm.tqdm(summeval):
        source = instance['source']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt
        while True:
            try:
                _response = openai.ChatCompletion.create(
                    model=args.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=2,
                    max_tokens=5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    # logprobs=40,
                    n=20
                )
                time.sleep(0.5)

                all_responses = [_response['choices'][i]['message']['content'] for i in
                                 range(len(_response['choices']))]
                instance['all_responses'] = all_responses
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)

                    break

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--instruction', type=str, default='fluency')
    # argparser.add_argument('--summeval_fp', type=str, default='data\summeval.json')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-4o')
    argparser.add_argument('--prediction_type', type=str, default='mybartlarge')
    args = argparser.parse_args()
    openai.api_key = args.key

    prediction_type = args.prediction_type
    instruction = args.instruction

    # load data
    if prediction_type == 'mybartlarge':
        prediction_path = '/home/XXX/data1/model/intentdescriptor/bart/v10data_train_var6loss3nce_my-bart-large_batch_2_lr_0.00001_twolinearlayers_new/checkpoint_lasttrain_loss_0.337129_epoch_8_step_7499/predictions_50.json'
        test_data_path = '/home/XXX/data1/dataset/q2id/Q2ID_data/test_data_v10_human_50.jsonl'
    elif prediction_type == 'sota': # wrong vanilla t5 # depracated
        prediction_path = '/home/XXX/data1/output-socratic/q2id_8_512_strided_socratic_30M_gpu_batch2_1e5_4/selected_checkpoint/predictions_50.jsonl'
        test_data_path = '/home/XXX/data1/dataset/q2id/Q2ID_data/test_data_v10_human_50.jsonl'
    output_path =f'/home/XXX/workspace/IntentRewriter/src/evaluation/output/{instruction}'
    
    evaluate(prediction_type, test_data_path, prediction_path, output_path, instruction)

    