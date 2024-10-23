source ~/.bashrc
source /home/XXX/miniconda3/etc/profile.d/conda.sh
conda activate llama
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH

TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# # run script!
export OPENAI_API_KEY="xxxxxxxx"
instruction='alignment'
model_type='gpt-4o' #llama3_8b
prediction_type='sota' #mybartlarge / sota

python llama3_evaluator.py \
  --key=${KEY} \
  --model_type=${model_type} \
  --prediction_type=${prediction_type} \
  --instruction=${instruction}

echo "## Training Done!"