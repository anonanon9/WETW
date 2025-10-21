import copy
import logging  
from timeit import default_timer as timer
from ctypes import CDLL, c_char_p, c_long
from _ctypes import dlclose
from omegaconf import DictConfig
import numpy as np
from torch.utils.data import DataLoader
from lib.mpmath import mp
import numpy as np
from multiprocessing import Pool
from dtaidistance import dtw
import ctypes
import time 


class Sampler: 
    def __init__(self, cfg: DictConfig, size_db, train_size, valid_size, logger=None):
        
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

    
            self.size_db = size_db
            self.dataset_filepath = cfg.dataset.processed_train_fp
            self.db_sorted_indices_filepath = self.cfg.paths.db_sorted_indices_path

            self.dim_series = cfg.dataset.dim_series

            self.size_train = train_size
            self.size_valid = valid_size
            self.size_extra = 17

            self.batch_loading_size = cfg.training.batch_loading_size

            self.target_dist_function = cfg.training.target_dist_function

            self.warping_window = int(self.cfg.training.warping_window * self.dim_series)

    def sample(self, all_data=False) -> np.ndarray:
        
        # seasam sampling 
        if not all_data:
            self.logger.info('initialize sorted indices')

            c_functions = CDLL(self.cfg.seasam_libpath)
            
            return_code = c_functions.sample_coconut(c_char_p(self.dataset_filepath.encode('ASCII')), 
                                                    c_long(self.size_db),
                                                    c_char_p(self.db_sorted_indices_filepath.encode('ASCII')), 
                                                    c_char_p(b'skip'), 
                                                    self.size_train,
                                                    c_char_p(b'skip'),  
                                                    self.size_valid, 
                                                    self.dim_series, 
                                                    self.cfg.training.seasam_cardinality,
                                                    self.cfg.training.dim_seasam) 

            dlclose(c_functions._handle)
            
            assert return_code == 0
            
            self.logger.info('SEAsam Done.')
            print("SEAsam Done.")


            # seasam indices 
            indices = np.fromfile(self.db_sorted_indices_filepath, dtype=int, count=self.size_train + self.size_valid + self.size_extra)
            indices = np.sort(indices)
                    
            samples = []
            samples_idx = []
            num_loaded = 0
            num2load = self.size_train + self.size_valid
            
            for batch_loading_offset in range(0, self.size_db, self.batch_loading_size):
                        series_batch = np.fromfile(self.dataset_filepath,
                                                dtype=np.float32,
                                                count=self.dim_series * self.batch_loading_size,
                                                offset=4 * self.dim_series * batch_loading_offset)  # 4 because each float is 4 bytes and we are skipping the previous batch 
                        series_batch = series_batch.reshape([-1, self.dim_series])

                        batch_end = batch_loading_offset + self.batch_loading_size
                        if batch_end > self.size_db:
                            batch_end = self.size_db
                        
                        # as long as the number to load has not been reached (train_size + query_size) and the current index is not bigger than the batch
                        # add the new sample 
                        while num_loaded < num2load and indices[num_loaded] < batch_end:
                            new_sample = series_batch[indices[num_loaded] -
                                                    batch_loading_offset]

                            if not self.__is_nan(new_sample):
                                samples.append(copy.deepcopy(new_sample))
                                samples_idx.append(indices[num_loaded])

                                num_loaded += 1
                            else:
                                self.logger.info('nan found at {:d}: {:s}'.format(
                                    indices[num_loaded], str(new_sample)))
                                num_loaded += 1
                                

            rand_perm = np.random.permutation(len(samples))

            samples = np.asarray(samples)[rand_perm]
            samples_idx = np.asarray(samples_idx)[rand_perm]

            samples.tofile(self.cfg.path.seasam_sample_filepath)
            samples_idx.tofile(self.cfg.path.seasam_sample_idx_filepath)

            train_samples = samples[: self.size_train]
            valid_samples = samples[self.size_train:]
            
            assert train_samples.shape[0] == self.size_train
            assert valid_samples.shape[0] == self.size_valid

        else:
            # load all data 
            samples = np.fromfile(self.dataset_filepath, dtype=np.float32, count=self.size_db * self.dim_series).reshape([-1, self.dim_series])

            rand_perm = np.random.permutation(len(samples))

            samples = np.asarray(samples)[rand_perm]

            train_samples = samples[: self.size_train]
            valid_samples = samples[self.size_train:]

            assert train_samples.shape[0] == self.size_train
            assert valid_samples.shape[0] == self.size_valid

        dists = []
        for sample_data, sample_type, sample_size in zip(
            [train_samples, valid_samples],
            ['-train-', '-valid-'],
            [self.size_train, self.size_valid]):
            
            if self.target_dist_function == 'dtw':

                start_time = time.time()
                w = self.warping_window # getting the warping window of the sample 
              
                d = dtw.distance_matrix_fast(   sample_data.astype(np.double),
                                                window=w,
                                                compact=True,
                                                parallel=True) 
                d = np.array(d) ** 2 
                d = d.astype(np.float32)
                end_time  = time.time()

                self.logger.info('distances calculated')
                self.logger.info(f'Elaped time: { (end_time - start_time) / 60} min')
                print('distances calculated')
                print("Elapsed time:", end_time - start_time)


                dists.append(d)

                assert len(dists[-1]) == int(sample_data.shape[0]
                                            * (sample_data.shape[0] - 1) / 2)
                

                dist_filepath = f"{self.cfg.paths.train_val_dist_prefix}_{sample_type[1:-1]}.bin"
                d.tofile(dist_filepath)

                print(f'min distance = {d.min()}')
                print(f'max distance = {d.max()}')
                  


            elif self.target_dist_function == 'l2' or self.target_dist_function == 'euclidean':
                local_dists = []

                for i in range(sample_data.shape[0]):
                    for j in range(i + 1, sample_data.shape[0]):
                        local_dists.append(np.linalg.norm(sample_data[i] - sample_data[j]))

                local_dists = np.asarray(local_dists, dtype=np.float32)
                dists.append(local_dists)
                assert len(dists[-1]) == int(sample_data.shape[0]
                                            * (sample_data.shape[0] - 1) / 2)

                dist_filepath = f"{self.cfg.paths.train_val_dist_prefix}_{sample_type[1:-1]}.bin"
                local_dists.tofile(dist_filepath)

        
        return train_samples, dists[0], valid_samples, dists[1]


    def __is_nan(self, series: np.ndarray):
        return np.isnan(np.sum(series))
    
