# Fine-tuning Experiments

## Requirements
```
python 3.8
transformers==4.2.0
pytorch==1.6.0
tqdm
scikit-learn
faiss-cpu==1.6.4
```

## Dataset Preprocessing
Suppose we have the selected id from the training data from `ptp_sample.py`. An example of the file name should be `train_idx_roberta-base_ptp_sample32.json`. 

Then, we need to sample the unlabeled data into the *training* set and _validation_ set. 
The number in the above file is the *index* of the selected data, which will be used as the training set. 
The validation set is randomly selected 100 samples from the unlabeled data.

The corresponding train/dev dataset are `train_[budget].json` and `valid.json`.

## Training Commands
Run the following commands `run.sh` for fine-tuning the PLM with the selected data.



## Hyperparameters
Note: the three numbers in each grid indicate the parameter for 16/32/64/128 labels.

|            | IMDB                | Yelp-full | AG News             | Yahoo!   | TREC     |
|------------|---------------------|-----------|---------------------|----------|----------|
| BSZ        | 2/4/8/16            | 2/4/8/16  | 2/4/8/16            | 2/4/8/16 | 2/4/8/16 |
| LR         | 2e-5/2e-5/1e-5/1e-5 | 2e-5      | 3e-5/3e-5/3e-5/1e-5 | 5e-5     | 2e-5     |
| EPOCH      | 15                  | 15        | 15                  | 15       | 15       |
| Max Tokens | 512                 | 512       | 128                 | 256      | 64       |
