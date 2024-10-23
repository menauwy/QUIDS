#!/bin/bash
#SBATCH --job-name=preprocess_data_ranked
#SBATCH --mail-type="ALL"
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=10G

source /home/ywang1/miniconda3/etc/profile.d/conda.sh
conda activate embed

python generate_degree_data.py

