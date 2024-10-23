#!/bin/bash

# Directory containing files
INPUT_DIR="/home/ywang1/data1/dataset/q2id/Q2ID_data/ranked/split"

# Path to the Python script
PYTHON_SCRIPT="preprocess_q2id.py"

# Arguments for the Python script
MAX_NEG_RATIO=32  # Set your max_neg_ratio value
ARGUMENT="True"  # Set your argument value

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Directory $INPUT_DIR does not exist."
  exit 1
fi

# Loop over each file in the directory
for FILE in "$INPUT_DIR"/*; do
  # Skip if not a file
  if [ ! -f "$FILE" ]; then
    continue
  fi
  
  # Submit a Slurm job for each file
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=preprocess_data_ranked
#SBATCH --mail-type="ALL"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=10G

source /home/ywang1/miniconda3/etc/profile.d/conda.sh
conda activate embed

# Run the Python script with the specific arguments
python "$PYTHON_SCRIPT" --input_file="$FILE" --max_neg_ratio=$MAX_NEG_RATIO --argument=$ARGUMENT
EOF

done
