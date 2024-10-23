source ~/.bashrc
source /home/XXX/miniconda3/etc/profile.d/conda.sh
conda activate intent
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH

TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

echo "## Running training script on q2id dataset with concatenation of pos docs!"
dataset_dir='/home/XXX/data1/dataset/'
model_dir='/home/XXX/data1/model/'
PTM='bart'
intent_model='my-bart-large' # my-t5-large
load_from_checkpoint='False'

train_dataset='q2id'
data_version='v10data'
loss_type='loss3nce' # for decoder loss
model_variant='var6'
train_file_path="${dataset_dir}${train_dataset}/Q2ID_data/train_data_v10.jsonl"
dev_file_path="${dataset_dir}${train_dataset}/Q2ID_data/valid_data_v10.jsonl"

batch_size=2
epoch=10
min_new_tokens=0
max_new_tokens=256
max_query_length=1024
max_doc_length=1024
max_concat_length=1024
# if alpha, beta, gamma is -1, then the model will use the default values tensor([1.0, 1.0, 1.0])
# otherwies, the model will use the customized values
alpha=0.2
beta=0.2
gamma=0.6
learning_rate=0.00001
weigth_decay=0.000001
num_beams=4
no_repeat_ngram_size=3
twolinearlayers='True'

mode='train'
validate='True'
patience=100 #set to 100 or more to deactivate early stopping
encoder_loss='True'
decoder_loss='True'

model_output_path="${model_dir}intentdescriptor/${PTM}/${data_version}_train_${model_variant}${loss_type}_${intent_model}_batch_${batch_size}_lr_${learning_rate}_twolinearlayers"
log_dir_path="${model_output_path}"
python train.py \
      --load_from_checkpoint=$load_from_checkpoint \
      --loss_type=$loss_type \
      --PTM=$PTM \
      --intent_model=$intent_model \
      --train_file_path=$train_file_path \
      --dev_file_path=$dev_file_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \
      --collate_fn_type="flat_concat_for_train" \
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
      --twolinearlayers=$twolinearlayers \
      --mode=$mode \
      --validate=$validate \
      --model_variant=$model_variant \
      --encoder_loss=$encoder_loss \
      --decoder_loss=$decoder_loss \
      --patience=$patience
done   
echo "## Training Done!"