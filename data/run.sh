#!/bin/bash
#SBATCH --job-name=preprocess_data_ranked
#SBATCH --mail-type="ALL"
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=10G

source /home/ywang1/miniconda3/etc/profile.d/conda.sh
conda activate embed

# v6 3 False
# v7 3 False
# v8 10 False
# v9 10 True
# v10 3 True
# v11 20 True
# ranked 16 True
# need to add version in preprocess_q2id.py
# data_version='ranked'
max_neg_ratio=32
argument='True'

python preprocess_q2id.py \
        --input_file='valid_data.jsonl' \
        --max_neg_ratio=$max_neg_ratio \
        --argument=$argument

# dir="/home/ywang1/data1/dataset/q2id/Q2ID_data/ranked/split"

# # Iterate over all files in the directory
# for file in "$dir"/*; do
#     # Extract the base name of the file to use as the data version
#     input_file=$(basename "$file")

#     # Set the max_neg_ratio and argument variables
#     max_neg_ratio=16
#     argument='True'

#     # Submit a job to SLURM for each file
#     sbatch --wrap="python preprocess_q2id.py --input_file=$input_file --max_neg_ratio=$max_neg_ratio --argument=$argument"
# done

