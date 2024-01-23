import re
import torch
import numpy as np
from collections import Counter
import time
from datetime import datetime 

from scipy import stats
import pandas as pd
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import faiss
from tqdm import tqdm, trange 
from sklearn.metrics import pairwise_distances
import copy 
import pickle
import json
import argparse
import math 
import os
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use("agg")


def select_instances_kmeans_random(embeddings,n_sample):
    unlabeled_feat = embeddings
    d = unlabeled_feat.shape[-1]
    n_data = unlabeled_feat.shape[0]

    kmeans = faiss.Kmeans(d, n_sample, niter=100, verbose=True, nredo=5)
    kmeans.train(unlabeled_feat)
    D, I = kmeans.index.search(unlabeled_feat, 1)
    cluster_id = I.flatten()

    sample_idx = []
    for i in range(n_sample):
        idxs_i = np.arange(n_data)[cluster_id == i]
        feat_i = unlabeled_feat[cluster_id == i]
        print(len(feat_i))
        # random.seed(42)
        # random.seed(43)
        random.seed(44)
        # random.seed(45)
        random_idx = random.sample(range(len(feat_i)), 1)
        index = idxs_i[random_idx]
        sample_idx.append(int(index))
    print(sample_idx)
    return sample_idx

def select_instances_PTP(embeddings,n_sample):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f'yahoo/embedding_simcse_unlabeled.pkl', 'rb') as f:
        train_emb = pickle.load(f)
    n_embedding = train_emb.shape[0]
    feat_i_tensor = torch.tensor(train_emb, dtype=torch.float32, device=device)
    allscores = []
    for i in trange(n_embedding):
        cur_emb = feat_i_tensor[i].view(1, -1)
        cur_scores = torch.sum(torch.abs(torch.nn.functional.cosine_similarity(feat_i_tensor, cur_emb, dim=1)))
        allscores.append(cur_scores.item())

    max_indices = np.argsort(allscores)[-n_sample:]
    print(max_indices)
    scores = [allscores[i] for i in max_indices]
    print(scores)
    print(max_indices)
    return max_indices
''' loading embedding and predictions '''

''' loading training data '''
def load_id(method = 'badge', dataset = 'agnews', nlabel = 16, model = 'roberta-base'):
    path = f'{dataset}/'
    train_name = path + f'train_idx_{model}_{method}_{nlabel}.json'
    with open(train_name, 'r') as f:
        train_idx = json.load(f)
    train_idx = np.array(train_idx, dtype = int)
    return train_idx


''' loading training arguments '''
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='agnews',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--n_sample",
        default=32,
        type=int,
        help="The number of acquired data size",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    Suppose all the data is in the folder X, where X = {AGNews, IMDB, TREC, Yahoo, Yelp-full}
    '''
    start = time.perf_counter()

    args = get_arguments()
    n_sample = args.n_sample

    print(f"Using, Prop: {prop}")
    path = f'{dataset}/'
    with open(path + f'embedding_{embedding_model}_roberta.pkl', 'rb') as f:
        train_emb = pickle.load(f)
    # sample_idxs = select_instances_kmeans_random(train_emb, n_sample)
    sample_idxs = select_instances_PTP(train_emb, n_sample)
    print(sample_idxs)
    with open(f"{args.dataset}/train_idx_roberta-base_ptp_{n_sample}.json", 'w') as f:
        for sample_idx in enumerate(sample_idxs):
            json.dump(sample_idx, f)

    dur = time.perf_counter() - start
    print("dur:")
    print(dur)


            
