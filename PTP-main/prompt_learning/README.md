# Prompt-based Learning Experiments
## Requirements
```
python 3.8
transformers==4.2.0
pytorch==1.6.0
tqdm
scikit-learn
faiss-cpu==1.6.4
openprompt
```

## Dataset Preprocessing
The input for prompt-based learning is the same as the vanilla fine-tuning. 
Suppose we have the selected id from the training data from `patron_sample.py`. An example of the file name should be `train_idx_roberta-base_sample32.json`. 

Then, we need to sample the unlabeled data into the *training* set and _validation_ set. 
The number in the above file is the *index* of the selected data, which will be used as the training set. 
The validation set is randomly selected from the unlabeled data.

The corresponding train/dev dataset are `train_[budget].json` and `valid.json`.


## Training Commands
Run the following commands `run.sh` for fine-tuning the PLM with the selected data.

_Note_: the relevant configurations for prompts (e.g. verbalizers, templates) are in `utils.py`. 

## Hyperparameters
Note: the three numbers in each grid indicate the parameter for 16/32/64/128 labels.

|            | IMDB                | Yelp-full | AG News             | Yahoo!  | TREC    |
|------------|---------------------|-----------|---------------------|---------|---------|
| BSZ        | 2/4/8/16            | 2/4/8/16  | 2/4/8/16            | 2/4/8/8 | 1/2/2/4 |
| LR         | 2e-5/2e-5/1e-5/1e-5 | 1e-5      | 2e-5/2e-5/1e-5/1e-5 | 1e-5    | 1e-5    |
| EPOCH      | 15                  | 15        | 15                  | 15      | 15      |
| Max Tokens | 512                 | 512       | 128                 | 256     | 64      |
