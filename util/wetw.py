import sys 
import numpy as np 
import time 
import pandas as pd
import torch
import itertools
import random
import glob
import re 
import os 
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw
from itertools import combinations
from fastdtw import fastdtw
from tslearn.metrics import dtw, soft_dtw, cdist_dtw
from dtaidistance import dtw
from concurrent.futures import ThreadPoolExecutor
import copy
import time
import polars as pl 
import gc
from tqdm import tqdm 
import numpy as np
from joblib import Parallel, delayed
from tslearn.metrics import soft_dtw
from dtaidistance.dtw import distance_matrix_fast
import csv
project_rootfolder = '/home/cpanourg/projects/1-dtwrl/'
module_path = project_rootfolder

if module_path not in sys.path:
    sys.path.append(module_path)

from util.messi import run_messi
from projects.wesee_wetw.model.wetw_model import WeightBilinear, WeightMLP, WeightTransformer


def get_divisors(N):
        divisors = []
        for i in range(1, int(N**0.5) + 1):
            if N % i == 0:
                divisors.append(i)
                if i != N // i:  # Avoid adding the square root twice for perfect squares
                    divisors.append(N // i)
        return sorted(divisors)


def weighted_pairwise_distance(E_i, E_j, w, p=2, dim=0):

    return torch.sum(w * torch.abs(E_i - E_j) ** p, dim=dim).pow(1.0 / p)

def load_warper(warp_model_state_dict_fp, dim):
    nheads = get_divisors(dim)[1]

    warp_model = WeightTransformer(dim, n_heads=nheads, 
                                    temperature=0.5, dropout=0.1).cuda()


    with open(warp_model_state_dict_fp, 'rb') as fin:
        warp_model.load_state_dict(torch.load(fin))
    return warp_model


def process_on_gpu(start_idx, end_idx, warp_model, device, 
                   indices, eval_batch_size, data, dists):


    for i in tqdm(range(start_idx, end_idx, eval_batch_size), desc='distance batches calculated'):
        # Select a batch of query embeddings
        batch_indices = indices[i:i + eval_batch_size]

        left_vectors = data[batch_indices[:, 0]]
        right_vectors = data[batch_indices[:, 1]]

        # Move tensors to the current GPU
        left_vectors = left_vectors.to(device)
        right_vectors = right_vectors.to(device)
        warp_model.change_device(device)

        warp_model.eval()
        with torch.no_grad():
            # Use the model on the current GPU
            w = warp_model(left_vectors, right_vectors)

            # Compute the weighted pairwise distance for the batch and reshape
            dists_batch = (
                weighted_pairwise_distance(left_vectors, right_vectors, w, dim=1)
                .cpu()
                .detach()
                .numpy()
                .reshape(len(batch_indices))
            )

        # Store the computed distances in the appropriate slice of the distance matrix
        dists[i:i + eval_batch_size] = dists_batch

        # Clean up to free memory
        del left_vectors, right_vectors, dists_batch, w
        torch.cuda.empty_cache()

def wetw_set(X, warp_model, indices, eval_batch_size=2, num_parallel_jobs=2):
    dim_emb = X.shape[0]

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    # Initialize an empty array to store distances
    dists = np.empty(indices.shape[0], dtype=np.float32)

    # Calculate the number of queries per job
    divd_queries = indices.shape[0] // num_parallel_jobs
    # Initialize model instances for each GPU
    models = [copy.deepcopy(warp_model).to(devices[i % len(devices)]) for i in range(num_parallel_jobs)]
    data_emb = torch.from_numpy(X)
    
    # Define parallel execution
    with ThreadPoolExecutor(max_workers=num_parallel_jobs) as executor:
        futures = []
        for job_id in range(num_parallel_jobs):
            start_idx = job_id * divd_queries
            end_idx = (job_id + 1) * divd_queries
            device = devices[job_id % len(devices)]  # Alternate between cuda:0 and cuda:1
            futures.append(executor.submit(process_on_gpu, start_idx, end_idx, models[job_id], device,
                                           indices, eval_batch_size, data_emb, dists))

        # Wait for all jobs to finish processing
        for future in futures:
            future.result()

    # Clear resources
    del models
    torch.cuda.empty_cache()
    gc.collect()

    return dists


def wetw_pair(x, y, warp_model, device='cuda'):

    x = torch.from_numpy(x).to(device=device)
    y = torch.from_numpy(y).to(device=device)

    warp_model.eval()
    with torch.no_grad():

        start = time.time()
        warp_weights = warp_model(x, y)


        wetw_dist = weighted_pairwise_distance(x, y, w=warp_weights, dim=1).detach().cpu().numpy()
        end = time.time()

    return 
