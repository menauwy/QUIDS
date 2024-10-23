# QUIDS: Query Intent Generation via Dual Space Modeling

This repository contains the code for the paper "QUIDS: Query Intent Generation via Dual Space Modeling".

## Table of Contents

- [Preprocessing](#preprocessing)
- [Training and Inference](#training-and-inference)
- [LLM-based Evaluation](#llm-based-evaluation)

## Preprocessing

To preprocess the data and generate an intent-driven hard negative augmented dataset:

1. Update the file paths in the `preprocess_q2id.py` main block to match your data directory.
2. Run the following command:

```bash
python preprocess_q2id.py \
	--input_file='valid_data.jsonl' \
	--max_neg_ratio=3 \
	--argument='True'
```

## Training and Inference
Install an enviroment using the requirements.txt file under the main directory.
### Training

To train the model:

1. Navigate to the `src` directory.
2. Run the following script:

```bash
./train_general_v10data_v5_mybartlarge.sh
```

### Inference

To perform inference:

1. Navigate to the `src` directory.
2. Run the following script:

```bash
./test_general_v10data_v5_mybartlarge.sh
```

## LLM-based Evaluation
For evaluation, we use a separate environment to run LLaMa3.1 and GPT-4. This environment requires both a Hugging Face API key (for LLaMa3.1) and an OpenAI API key (for GPT-4).

To set up the evaluation environment:
1. Navigate to the `src/evaluation` directory.
2. Install the required dependencies using the `requirements.txt` file in this directory:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have valid API keys for both Hugging Face and OpenAI before proceeding with the evaluation.

### GPT-4 Evaluation

To evaluate using GPT-4:

1. Update the following paths in the `gpt_evaluator.py` script:
- `prediction_path`
- `test_data_path`
- `output_path`
2. Replace `'OPENAI_API_KEY'` in the `run_gpt.sh` script with your OpenAI API key.
3. Run the evaluation:

```bash
./run_gpt.sh
```

### Llama-3 Evaluation

To evaluate using Llama-3:

1. Update the following variables in the `llama3_evaluator.py` script:
- `prediction_path`
- `test_data_path`
- `output_path`
- `model_id`
2. Run the evaluation, updating the `instruction` to load your desired metric:

```bash
./run_llama.sh
```

