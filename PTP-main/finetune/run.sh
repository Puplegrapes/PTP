task=trec # dataset

# 'agnews'num_label = 4
# 'yahoo'num_label = 10
# 'YelpReviewPolarity','imdb','youtube', 'amazon-polarity', 'SST-2', 'elec' num_label = 2
# 'yelp-full','amazon-full', 'SST-5', 'pubmed' num_label = 5
# 'trec'num_label = 6
# 'mnli'num_label = 3
# 'dbpedia'num_label = 14
#  'tacred'num_label = 42
gpu=0
n_gpu=1



model_type=roberta-base
method=train
eval_batch_size=256
dev_labels=100
steps=100

#train_seed=139 # random seed

train_label=32 # number of labels 32/64/128
lr=2e-5
batch_size=4
max_seq_len=64
epochs=15

##############################################################################
#''' Evalutation on OOD datasets, for IMDB dataset only '''
#contra_datasets='sst2val.json,sst2test.json,IMDB-contrast.json,IMDB-counter.json'
#extra_cmd="--do_extra_eval --extra_dataset=${contra_datasets}"
##############################################################################

model_type=${model_type}
output_dir=${task}/model
mkdir -p ${output_dir}
echo ${method}
mkdir -p ${task}/cache
for train_seed in  121 122 123 124 125 126 127 128 129 130
do
  train_cmd="CUDA_VISIBLE_DEVICES=${gpu} python3 main.py --do_train --do_eval --task=${task} \
    --train_file=train_${train_label}.json --dev_file=valid.json --test_file=test.json \
    --unlabel_file=unlabeled.json \
    --data_dir="${task}" --train_seed=${train_seed} \
    --cache_dir="${task}/cache" \
    --output_dir=${output_dir} --dev_labels=${dev_labels} \
    --gpu=${gpu} --n_gpu=${n_gpu} --num_train_epochs=${epochs} \
    --learning_rate=${lr} --weight_decay=1e-8 \
    --method=${method} --batch_size=${batch_size} --eval_batch_size=${eval_batch_size} \
    --max_seq_len=${max_seq_len} --auto_load=1 \
    --max_steps=${steps} --model_type=${model_type} \
    --sample_labels=${train_label} ${extra_cmd}"
  echo $train_cmd
  eval $train_cmd
done