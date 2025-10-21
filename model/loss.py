# coding = utf-8

import logging
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from util.helper_functions import weighted_pairwise_distance


class PairwiseDistDiff(torch.nn.Module):
    def __init__(self, cfg, dataset_size, default_batch_size, device='cuda', logger=None):
        super(PairwiseDistDiff, self).__init__()

        self.cfg = cfg
        if logger is None:
            logging.basicConfig(filename=cfg.logging.log_filepath,
                                filemode='a+',
                                format='%(asctime)s:%(msecs)d %(levelname).3s [%(filename).4s:%(lineno)d] %(message)s',
                                level=logging.DEBUG,
                                datefmt='%y%m%d:%I%M%S')

            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger
        
        self.__device = device

        self.__dataset_size = dataset_size
        self.__targets_size = int(
            self.__dataset_size * (self.__dataset_size - 1) / 2)


        self.__l2 = torch.nn.PairwiseDistance(p=2).to(device)
        self.__l1 = torch.nn.PairwiseDistance(p=1).to(device)
        self.train_all_pairs = cfg.training.train_all_pairs

        self.__dist_targets_np = None

        self.__pair_indices_array = [
            None for _ in range(default_batch_size + 1)]
        self.__num_samples_array = np.zeros(default_batch_size + 1, dtype=int)
        
        for batch_size in range(2, default_batch_size + 1):
            pair_indices = []

            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    pair_indices.append([i, j])

            self.__pair_indices_array[batch_size] = np.asarray(pair_indices)

            self.__num_samples_array[batch_size] = batch_size * \
                int(np.ceil(np.log2(batch_size)))

            if self.__num_samples_array[batch_size] > self.__pair_indices_array[batch_size].shape[0]:
                self.__num_samples_array[batch_size] = self.__pair_indices_array[batch_size].shape[0]

        self.__epsilon = 1e-3

        self.__num_example2check = 5
        
            

    def update_targets(self, dist_ndarray):
       
        
        if self.cfg.training.target_dist_function == 'dtw':
            self.__dist_targets_np = np.sqrt(dist_ndarray) 
        else:
            self.__dist_targets_np = dist_ndarray 

        if int(self.__dataset_size * (self.__dataset_size - 1) / 2) != self.__dist_targets_np.shape[0] and not self.augment_data:
            self.logger.error('dataset_size={:d} - {:d}, dist_targets_np.shape[0]={:d}'.format(
                self.__dataset_size, int(self.__dataset_size * (self.__dataset_size - 1) / 2), self.__dist_targets_np.shape[0]))
            raise ValueError('mismatched targets size')
    
        return self.__dist_targets_np
           
    def forward(self, batch, indices, wetw_model):
        
        batch = torch.squeeze(batch)
        batch_size = batch.shape[0]

        # sampling pairs to calculate loss, instead of taking their (n 2) which would be intractable
        sample_pair_idx = np.random.choice(self.__pair_indices_array[batch_size].shape[0],
                                           size=self.__num_samples_array[batch_size],
                                           replace=False)
        

        if self.train_all_pairs: 
            sample_idx_pairs = self.__pair_indices_array[batch_size]
        else:
            sample_idx_pairs = self.__pair_indices_array[batch_size][sample_pair_idx]

        left_sample_idx = sample_idx_pairs[:, 0]
        left_sample_global_idx = indices[left_sample_idx]
        right_sample_idx = sample_idx_pairs[:, 1]
        right_sample_global_idx = indices[right_sample_idx]

        global_row_idx = torch.minimum(
            left_sample_global_idx, right_sample_global_idx).to(torch.int64)
        global_column_idx = torch.maximum(
            left_sample_global_idx, right_sample_global_idx).to(torch.int64)

        targets_cutoff_rows = self.__dataset_size - global_row_idx

        targets_cutoff_sizes = torch.bitwise_right_shift(
            targets_cutoff_rows * (targets_cutoff_rows - 1), 1)
        targets_row_offsets = global_column_idx - global_row_idx - 1

        target_dist_idx = targets_row_offsets + \
            self.__targets_size - targets_cutoff_sizes
        
        if (torch.any(left_sample_global_idx >= self.__dataset_size) or torch.any(left_sample_global_idx < 0)) and not self.augment_data:
            raise ValueError("left_sample_global_idx contains invalid values")

        if (torch.any(right_sample_global_idx >= self.__dataset_size) or torch.any(right_sample_global_idx < 0)) and not self.augment_data:
            raise ValueError("right_sample_global_idx contains invalid values")


        try:
            batch_dist_targets = torch.Tensor(
                self.__dist_targets_np[target_dist_idx]).to(self.__device)
        except IndexError as ie:
            self.logger.error('IndexError: {:s}'.format(str(ie)))
            self.logger.error('left_sample_global_idx: {:s}'.format(
                np.array2string(left_sample_global_idx, separator=', ', suppress_small=True)))
            self.logger.error('right_sample_global_idx: {:s}'.format(
                np.array2string(right_sample_global_idx, separator=', ', suppress_small=True)))
            self.logger.error('target_dist_idx: {:s}'.format(
                np.array2string(target_dist_idx, separator=', ', suppress_small=True)))
            raise IndexError(str(ie))

        bleft, bright = batch[left_sample_idx],  batch[right_sample_idx]

        warp_weights = wetw_model(bleft, bright) 

        batch_dist_embedded = weighted_pairwise_distance(bleft, bright, w=warp_weights)
        
        dist_diff = self.__l1(batch_dist_embedded, batch_dist_targets) / sample_pair_idx.shape[0]

        loss = dist_diff

        return loss
