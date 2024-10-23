import evaluate
import json
import statistics
# from rouge import Rouge
import datasets
import json
from transformers import BartForConditionalGeneration, BartTokenizer



def report_rouge(output_file, test_file_path, prediction_file_path):
    ###### Read test data and create reference dict
    reference_dict = {}
    test_ids = []
    with open(test_file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            sample_id = record["sample_id"]
            test_ids.append(sample_id)
            reference_dict[sample_id] = record["intent"]
    
    # Read prediction file and create prediction dict based on test ids
    prediction_dict = {}
    with open(prediction_file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["sample_id"] in test_ids:
                sample_id = record["sample_id"]
                prediction_dict[sample_id] = record["intent"]

    ###### Calculate Rouge scores
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

    # # 2. bleu score:
    # bleu = evaluate.load("bleu")
    # bleu_scores = bleu.compute(predictions=predictions, references=references)
    # print("Done with bleu scores!")
    
    # # 3. meteor score:
    # meteor = evaluate.load('meteor')
    # meteor_scores = meteor.compute(predictions=predictions, references=references)
    # print("Done with meteor scores!")
    
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
    "bert": bert_scores
    }
    
    with open(output_file, "w") as f:
        json.dump(all_scores, f, indent=4)

if __name__ == "__main__":
    # split the input length data
    test_path = '/home/ywang1/data1/dataset/q2id/Q2ID_data/test_data_v10.jsonl'

    # concatenate all pos docs, and calculate the token length
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    with open(test_path, 'r') as f:
        data = [json.loads(line) for line in f]

    short_id = []
    mid_id = []
    long_id = []

    for record in data:
        pos = record['pos']
        pos_docs = ' '.join(pos)

        # calculate the token length
        token_length = len(tokenizer.encode(pos_docs))

        if token_length < 512:
            short_id.append(record['sample_id'])
        elif token_length < 1024:
            mid_id.append(record['sample_id'])
        else:
            long_id.append(record['sample_id'])

    print(f'short_id: {len(short_id)}, mid_id: {len(mid_id)}, long_id: {len(long_id)}')


    # get rouge scores
    length = ['short', 'mid', 'long']
    len_2_id = {'short': short_id, 'mid': mid_id, 'long': long_id}
    prediction_file_path = '/home/ywang1/workspace/IntentRewriter/src/input_length/predictions.json'

    for l in length:
        test_file_path = f'/home/ywang1/workspace/IntentRewriter/src/input_length/{l}_data.jsonl'
        output_file = f'/home/ywang1/workspace/IntentRewriter/src/input_length/rouge_{l}.json'
        report_rouge(output_file, test_file_path, prediction_file_path)
