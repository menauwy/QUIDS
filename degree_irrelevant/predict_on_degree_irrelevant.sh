####### RUN the same file in src/ #######
#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --mail-type="ALL"
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=10G

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

# # run script!
echo "## Running training script on q2id dataset with concatenation of pos docs!"
dataset_dir='/home/ywang1/data1/dataset/'
model_dir='/home/ywang1/data1/model/'
PTM='bart'
intent_model='my-bart-large' # my-t5-base / 'google-t5/t5-base'
load_from_checkpoint='False'

train_dataset='q2id'
data_version='v10data' # used in data structure: v9data, v11data, ranked8, ranked16, ranked32, ranked for vanilla T5
loss_type='loss3nce' # for decoder loss
model_variant='var6' #vanilla, var3, var6, var7, var8, var9, var10, var11
train_file_path="${dataset_dir}${train_dataset}/Q2ID_data/train_data_v10.jsonl"
dev_file_path="${dataset_dir}${train_dataset}/Q2ID_data/valid_data_v10.jsonl"

# doc_type='both'
# prefix_type='question'
# pos_num='multiple'
batch_size=2
epoch=10 # if epoch is changed, remember to change the corresponding model_load_path
min_new_tokens=0
max_new_tokens=256
max_query_length=1024
max_doc_length=1024
max_concat_length=1024
# if alpha, beta, gamma is -1, then the model will use the default values tensor([1.0, 1.0, 1.0])
# otherwies, the model will use the customized values
alpha=-1
beta=-1
gamma=-1
learning_rate=0.00001
weigth_decay=0.0
num_beams=4
no_repeat_ngram_size=3
early_stopping='True'
twolinearlayers='True'

validate='True'
patience=100 # 7 * 5epochs
encoder_loss='True'
decoder_loss='True'

########### check for test ###########
degree='low_empty'
test_file_path="/home/ywang1/workspace/IntentRewriter/src/degree_irrelevant/low_empty/low_degree_empty.jsonl"

mode='test'
collate_fn_type="flat_concat_for_test"
# model_checkpoint_dir="${model_dir}intentdescriptor/${PTM}/${data_version}_train_${model_variant}${loss_type}_${intent_model}_batch_${batch_size}_lr_${learning_rate}_twolinearlayers_2"
# echo $model_checkpoint_dir
checkpoint_path="/home/ywang1/data1/model/intentdescriptor/bart/v10data_train_var6loss3nce_my-bart-large_batch_2_lr_0.00001_twolinearlayers_new/checkpoint_lasttrain_loss_0.337129_epoch_8_step_7499"
# for checkpoint_path in "${checkpoint_paths[@]}";

python /home/ywang1/workspace/IntentRewriter/src/train.py \
      --load_from_checkpoint=$load_from_checkpoint \
      --loss_type=$loss_type \
      --PTM=$PTM \
      --intent_model=$intent_model \
      --train_file_path=$train_file_path \
      --dev_file_path=$dev_file_path \
      --test_file_path=$test_file_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \
      --model_checkpoint_path=$checkpoint_path \
      --output_file_path="/home/ywang1/workspace/IntentRewriter/src/degree_irrelevant/${degree}/predictions.json" \
      --collate_fn_type=$collate_fn_type \
      --per_gpu_train_batch_size=$batch_size \
      --num_train_epochs=$epoch \
      --data_version=$data_version \
      --max_new_tokens=$max_new_tokens \
      --min_new_tokens=$min_new_tokens \
      --max_query_length=$max_query_length \
      --max_doc_length=$max_doc_length \
      --max_concat_length=$max_concat_length \
      --alpha=$alpha \
      --beta=$beta \
      --gamma=$gamma \
      --learning_rate=$learning_rate \
      --weight_decay=$weigth_decay \
      --num_beams=$num_beams \
      --no_repeat_ngram_size=$no_repeat_ngram_size \
      --early_stopping=$early_stopping \
      --twolinearlayers=$twolinearlayers \
      --mode=$mode \
      --validate=$validate \
      --model_variant=$model_variant \
      --encoder_loss=$encoder_loss \
      --decoder_loss=$decoder_loss \
      --patience=$patience
