task=yahoo
gpu=0 # the id of the GPU
model_type=roberta-base


method=train
max_seq_len=128
lr=1e-5
batch_size=16
train_label=128 # number of labels 16/32/64/128


eval_batch_size=256
dev_labels=100
steps=100
epochs=15


model_type=${model_type}
output_dir=${task}/model
mkdir -p ${output_dir}
echo ${method}
for train_seed in  121 122 123 124 125 126 127 128 129 130
do
  train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 LMBFF.py --do_train --do_eval --task=${task} \
    --train_file=train_${train_label}.json --dev_file=valid.json --test_file=test.json \
    --unlabel_file=unlabeled.json \
    --data_dir=${task} --train_seed=${train_seed} \
    --cache_dir=${task}/pt_cache \
    --output_dir=${output_dir} --lr=${lr} \
    --logging_steps=20 --dev_labels=${dev_labels} \
    --num_train_epochs=${epochs} \
    --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
    --max_seq_len=${max_seq_len} \
    --max_steps=${steps} --model_type=${model_type} \
    --sample_labels=${train_label}"
  echo $train_cmd
  eval $train_cmd
done