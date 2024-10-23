#!/bin/bash
#SBATCH --job-name=run-rouge
#SBATCH --mail-type="ALL"
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1

source ~/.bashrc
# conda init bash doesn't work
source /home/ywang1/miniconda3/etc/profile.d/conda.sh
conda activate intent
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH

TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

python rouge.py
