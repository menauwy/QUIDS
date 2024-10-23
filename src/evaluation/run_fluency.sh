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
model_type='llama3.1_8b' #llama3_8b
prediction_type='sota' #mybartlarge / sota
instruction='fluency' #inclusion / exclusion / alignment  / fluency
NUM_RUNS=1
START=1
for RUN in $(seq $START $NUM_RUNS)
do
  python llama3_evaluator.py \
  --model_type=${model_type} \
  --prediction_type=${prediction_type} \
  --instruction=${instruction} \
  --run=${RUN} 
done
      
echo "## Training Done!"