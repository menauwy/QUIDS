from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
import json
import re
import argparse



def print_correlations(result):
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    table.add_row(
        [round(result['pearson'], 4), round(result['spearman'], 4), round(result['kendalltau'], 4)])
    print(table)

def extract_fluency_scores(strings, instruction):
    scores = []
    for string in strings:
        # Normalize the string by making it lowercase
        normalized_string = string.lower()
        
        # Regular expression to match "fluency score: " followed by a number 1-
        pattern = r'{} score:\s*([1-5])'.format(instruction)
        match = re.search(pattern, normalized_string)
        
        # If a match is found, extract the score
        if match:
            scores.append(int(match.group(1)))  # Convert the score to an integer
        else:
            # Try to capture standalone numbers between 1 and 5 if "fluency score" is not present
            standalone_match = re.match(r'^\s*([1-5])\s*$', string)
            if standalone_match:
                scores.append(int(standalone_match.group(1)))
    
    return scores

def post_process(instruction, input_path, output_path):
    new_data = []
    # read the input jsonl file
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            key = list(data.keys())[0]
            strings = list(data.values())[0]
            scores = extract_fluency_scores(strings, instruction)
            if len(scores) == 0:
                print(f'No fluency scores found for sample {key}')
                print(key)
                mean = 0
            else:
                mean = sum(scores) / len(scores)
            new_dict = {'sample_id': key, 'mean': mean, 'scores': scores}
            new_data.append(new_dict)

    # write the output jsonl file
    with open(output_path, 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', type=str, default='llama3.1')
    parser.add_argument('--model', type=str, default='sota') # sota
    parser.add_argument('--instruction', type=str, default='exclusion') # fluency, alignment, inclusion, exclusion
    args = parser.parse_args()

    model = args.model
    instruction = args.instruction
    eval_model = args.eval_model

    if eval_model == 'gpt-4o':
        input_path = f'/home/ywang1/workspace/IntentRewriter/src/evaluation/output/{instruction}/normalized_gpt-4o_{model}_{instruction} score.jsonl'
        ouput_path = f'/home/ywang1/workspace/IntentRewriter/src/evaluation/output/{instruction}/processed_gpt-4o_{model}_{instruction} score.jsonl'
        if instruction == 'alignment':
            input_path = f'/home/ywang1/workspace/IntentRewriter/src/evaluation/output/{instruction}/normalized_gpt-4o_{model}_factual {instruction} score.jsonl'
            ouput_path = f'/home/ywang1/workspace/IntentRewriter/src/evaluation/output/{instruction}/processed_gpt-4o_{model}_factual {instruction} score.jsonl'

        # step 1: post process the gpt output 
        post_process(instruction, input_path, ouput_path)

    elif eval_model == 'llama3.1':
        output_path = f'/home/ywang1/workspace/IntentRewriter/src/evaluation/output/{instruction}/normalized_llama3.1_8b_{model}_{instruction} score.jsonl'
        if instruction == 'alignment':
            output_path = f'/home/ywang1/workspace/IntentRewriter/src/evaluation/output/{instruction}/normalized_llama3.1_8b_{model}_factual {instruction} score.jsonl'

    

    # step 2: get gpt scores
    llm_score = []
    with open(output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if eval_model == 'gpt-4o':
                llm_score.append(data['mean'])
            elif eval_model == 'llama3.1':
                llm_score.append(list(data.values())[0][0])
    
    print('len(pred_scores): {}'.format(len(llm_score)))
    print(llm_score)

    # step 3: get human scores
    fluency_mybartlarge = [4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 5.0, 5.0, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 4.666666666666667, 3.6666666666666665, 4.333333333333333, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 4.333333333333333, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 4.666666666666667, 5.0, 5.0, 5.0, 5.0, 4.333333333333333, 5.0, 4.666666666666667, 5.0, 3.6666666666666665, 5.0, 5.0, 5.0, 4.666666666666667, 4.0, 5.0, 5.0, 4.0, 4.666666666666667, 5.0, 4.333333333333333, 5.0]
    alignment_mybartlarge = [4.333333333333333, 3.6666666666666665, 4.0, 4.666666666666667, 4.666666666666667, 5.0, 4.0, 3.6666666666666665, 2.3333333333333335, 4.0, 3.3333333333333335, 3.6666666666666665, 2.3333333333333335, 1.6666666666666667, 3.0, 3.3333333333333335, 3.6666666666666665, 5.0, 2.3333333333333335, 2.3333333333333335, 4.666666666666667, 4.333333333333333, 1.6666666666666667, 2.3333333333333335, 3.3333333333333335, 3.6666666666666665, 3.0, 3.6666666666666665, 4.333333333333333, 3.6666666666666665, 5.0, 4.666666666666667, 2.0, 4.333333333333333, 5.0, 2.3333333333333335, 2.6666666666666665, 4.333333333333333, 3.6666666666666665, 4.666666666666667, 3.6666666666666665, 3.0, 3.3333333333333335, 3.6666666666666665, 3.6666666666666665, 3.0, 4.0, 3.3333333333333335, 4.666666666666667]
    inclusion_mybartlarge = [3.6666666666666665, 3.6666666666666665, 4.0, 4.0, 4.333333333333333, 4.333333333333333, 3.6666666666666665, 3.3333333333333335, 4.0, 4.333333333333333, 4.0, 4.0, 3.6666666666666665, 3.3333333333333335, 4.0, 4.0, 4.333333333333333, 4.666666666666667, 3.3333333333333335, 3.6666666666666665, 4.666666666666667, 4.666666666666667, 3.0, 2.3333333333333335, 4.333333333333333, 4.0, 3.6666666666666665, 4.0, 4.0, 3.3333333333333335, 5.0, 4.333333333333333, 3.3333333333333335, 4.0, 5.0, 3.6666666666666665, 4.0, 4.333333333333333, 3.6666666666666665, 5.0, 4.0, 4.0, 3.6666666666666665, 4.333333333333333, 4.333333333333333, 2.3333333333333335, 4.333333333333333, 4.333333333333333, 4.333333333333333]
    exclusion_mybartlarge = [4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 4.333333333333333, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.333333333333333, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 4.333333333333333, 5.0, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 4.333333333333333, 5.0, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667]

    fluency_sota = [5.0, 5.0, 4.666666666666667, 5.0, 4.666666666666667, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.666666666666667, 4.0, 4.333333333333333, 4.666666666666667, 5.0, 4.666666666666667, 5.0, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.0, 4.333333333333333, 5.0, 5.0, 4.333333333333333, 4.666666666666667, 4.0, 4.666666666666667, 4.333333333333333, 5.0, 4.333333333333333, 4.666666666666667, 5.0, 5.0, 5.0, 4.0, 4.666666666666667, 5.0, 4.333333333333333, 4.0, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667]
    alignment_sota = [4.0, 3.0, 4.333333333333333, 3.3333333333333335, 3.3333333333333335, 5.0, 4.0, 3.3333333333333335, 4.0, 4.666666666666667, 4.333333333333333, 3.3333333333333335, 2.3333333333333335, 3.0, 4.333333333333333, 3.3333333333333335, 3.3333333333333335, 4.0, 2.6666666666666665, 4.333333333333333, 4.333333333333333, 4.333333333333333, 3.3333333333333335, 3.0, 4.0, 4.333333333333333, 4.333333333333333, 1.6666666666666667, 4.333333333333333, 3.3333333333333335, 5.0, 3.6666666666666665, 3.0, 3.6666666666666665, 5.0, 3.0, 2.0, 4.333333333333333, 3.3333333333333335, 5.0, 3.6666666666666665, 3.0, 3.0, 4.0, 4.333333333333333, 2.0, 4.333333333333333, 3.3333333333333335, 4.666666666666667]
    inclusion_sota = [3.3333333333333335, 2.6666666666666665, 4.0, 4.333333333333333, 4.333333333333333, 4.333333333333333, 3.6666666666666665, 4.0, 4.0, 4.666666666666667, 4.0, 4.0, 4.0, 3.3333333333333335, 4.0, 3.6666666666666665, 4.0, 4.666666666666667, 3.6666666666666665, 4.333333333333333, 4.666666666666667, 4.666666666666667, 3.6666666666666665, 4.333333333333333, 4.333333333333333, 4.333333333333333, 4.0, 2.3333333333333335, 4.0, 3.3333333333333335, 5.0, 4.333333333333333, 3.3333333333333335, 3.3333333333333335, 5.0, 3.3333333333333335, 1.6666666666666667, 4.0, 4.0, 4.333333333333333, 4.333333333333333, 3.6666666666666665, 2.6666666666666665, 4.0, 4.333333333333333, 1.6666666666666667, 4.333333333333333, 3.6666666666666665, 4.333333333333333]
    exclusion_sota = [4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 3.3333333333333335, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.333333333333333, 5.0, 4.666666666666667, 4.0, 5.0, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 5.0, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 5.0, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667, 4.666666666666667, 5.0, 4.666666666666667, 4.666666666666667]

    human_scores = {'fluency_mybartlarge': fluency_mybartlarge, 
                    'alignment_mybartlarge': alignment_mybartlarge, 
                    'inclusion_mybartlarge': inclusion_mybartlarge, 
                    'exclusion_mybartlarge': exclusion_mybartlarge,
                    'fluency_sota': fluency_sota,
                    'alignment_sota': alignment_sota,
                    'inclusion_sota': inclusion_sota,
                    'exclusion_sota': exclusion_sota}
    human_score = human_scores[f'{instruction}_{model}']
    print('len(human_scores): {}'.format(len(human_score)))
    print(human_score)
    
    # step 4: calculate correlation
    result = {}
    result['pearson'] = pearsonr(llm_score, human_score)[0]
    result['spearman'] = spearmanr(llm_score, human_score)[0]
    result['kendalltau'] = kendalltau(llm_score, human_score)[0]
    print_correlations(result)