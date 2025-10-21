# coding = utf-8

import os
import struct
from os.path import isfile
from ctypes import CDLL, c_char_p, c_long
from _ctypes import dlclose

import torch
import numpy as np
from torch.utils.data import Dataset




class DatasetwithIdx(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, indices):
        return self.data[indices], indices


class TSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, indices):
        return self.data[indices]

class FileContainer(object):
    def __init__(self, filename, binary=True):
        self.filename = filename
        self.binary = binary
        if self.binary:
            self.f = open(filename, "wb")
        else:
            self.f = open(filename, "w")

    def write(self, ts):
        if self.binary:
            s = struct.pack('f' * len(ts), *ts)
            self.f.write(s)
        else:
            self.f.write(" ".join(map(str, ts)) + "\n")

    def close(self):
        self.f.close()


def embedData(model, data_filepath, embedding_filepath, data_size, 
              batch_size = 2000, original_dim = 256, embedded_dim = 16, device = 'cuda'):
 
    if data_size < batch_size:
        num_segments = 1
        batch_size = data_size
    else:
        num_segments = int(np.ceil(data_size / batch_size))        

    nan_replacement_original = np.array([0.] * original_dim).reshape([1, original_dim])
    nan_replacement_embedding = [0.] * embedded_dim

    writer = FileContainer(embedding_filepath)
    
    try:
        with torch.no_grad():
            total_nans = 0
                
            for segment in range(num_segments):
                if num_segments > 1 and segment == num_segments - 1 and data_size % batch_size != 0:
                    last_batch_size = data_size % batch_size
                    
                    batch = np.fromfile(data_filepath, dtype=np.float32, count=original_dim * last_batch_size, offset=4 * original_dim * batch_size * segment)
                                                                
                else:
                    batch = np.fromfile(data_filepath, dtype=np.float32, count=original_dim * batch_size, offset=4 * original_dim * batch_size * segment)              
                    
                batch = batch.reshape([-1, 1, original_dim])

                nan_indices = set()
                for i, sequence in zip(range(batch.shape[0]), batch):
                    if np.isnan(np.sum(sequence)):
                        nan_indices.add(i)
                        batch[i] = nan_replacement_original

                embedding = model.encode(torch.from_numpy(batch).to(device)).detach().cpu().numpy()

                for i in nan_indices:
                    embedding[i] = nan_replacement_embedding

                writer.write(embedding.flatten())
                        
                total_nans += len(nan_indices)

            print('nans = {:d}'.format(total_nans))
    finally:
        writer.close()
