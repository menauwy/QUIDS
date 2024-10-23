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
model_type='llama3.1_8b' #llama3_8b_quantized # llama3.1_8b # llama3_8b
data_type='v3' #'v3'
ouput_length=256

NUM_RUNS=5
START=1
for RUN in $(seq $START $NUM_RUNS)
do
  python llama3_few_shot.py \
  --model_type=${model_type} \
  --data_type=${data_type} \
  --output_length=${ouput_length} \
  --run=${RUN} \
  --zero_shot
done
echo "## Testing Done!"