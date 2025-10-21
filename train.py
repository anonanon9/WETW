# coding = utf-8

import os
import logging
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
from dtaidistance import dtw
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
import wandb 
from omegaconf import OmegaConf
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.sampleSEAsamEMB import Sampler
from util.data import DatasetwithIdx, embedData
from model.builder import ModelBuilder
from model.loss import PairwiseDistDiff
from model.wetw_model import PairConvModel



class Train:
    def __init__(self, cfg: DictConfig):

        self.wandb = wandb
        self.cfg = cfg
        self.epoch = 0
        self.has_setup = False

    def __setup(self) -> None:
        if not self.has_setup:
            self.has_setup = True

            logging.basicConfig(filename=self.cfg.logging.log_filepath,
                                filemode='a+',
                                format='%(asctime)s:%(msecs)d %(levelname).3s [%(filename).4s:%(lineno)d] %(message)s',
                                level=logging.DEBUG,
                                datefmt='%y%m%d:%I%M%S')

            self.logger = logging.getLogger(self.__class__.__name__)

            self.device = self.cfg.system.device

            torch.manual_seed(self.cfg.system.torch_manual_seed)
            if self.device == 'cuda':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.manual_seed_all(
                        self.cfg.system.torch_cuda_manual_seed)
                else:
                    raise ValueError('cuda is not available')
            self.max_epoch = self.cfg.training.num_epoch

            self.dim_series = self.cfg.dataset.dim_series
            self.dim_embedding = self.cfg.encoder_model.dim_emb
            self.db = np.fromfile(self.cfg.dataset.processed_train_fp, dtype=np.float32).reshape(-1, self.dim_series)
            self.test_db = np.fromfile(self.cfg.dataset.processed_test_fp, dtype=np.float32).reshape(-1, self.dim_series)
            self.size_db = self.db.shape[0]
            self.size_test_db = self.test_db.shape[0]

            self.valid_size = int(self.cfg.training.valid_split * self.size_db)
            self.train_size = int(self.size_db - self.valid_size)
            self.batch_size = self.cfg.training.size_batch

            self.train_data_np = None
            self.train_dist_np = None
            self.train_dataloader = None
            self.valid_data_np = None
            self.valid_dist_np = None
            self.valid_dataloader = None


            self.sampler = Sampler(self.cfg, 
                                   self.size_db, 
                                   self.train_size,
                                   self.valid_size) # if scheduler is used then resampling happens independently of this variable


            self.encoder_model = None
            self.wetw_model = None

            self.lossF_distances_train = PairwiseDistDiff(  self.cfg,
                                                            dataset_size=self.train_size,
                                                            default_batch_size=self.batch_size,
                                                            device=self.device)

            self.lossF_distances_valid = PairwiseDistDiff(  self.cfg,
                                                            dataset_size=self.valid_size,
                                                            default_batch_size=self.batch_size,
                                                            device=self.device)
            
                
            # encoder Model ======================================================
            self.encoder_model = ModelBuilder(self.cfg)
                        
            if self.cfg.encoder_model.optimizer == 'sgd':
                encoder_init_lr = self.cfg.encoder_model.lr_init
                encoder_momentum = self.cfg.encoder_model.momentum
                encoder_init_wd = self.cfg.encoder_model.weight_decay

                self.encoder_optimizer = torch.optim.SGD(self.encoder_model.parameters(), 
                                                         lr=encoder_init_lr, 
                                                         momentum=encoder_momentum, 
                                                         weight_decay=encoder_init_wd)
            
            self.encoder_scheduler = ReduceLROnPlateau( optimizer=self.encoder_optimizer, 
                                                        mode=self.cfg.encoder_model.mode, 
                                                        factor=self.cfg.encoder_model.lr_reduce_factor,
                                                        patience=self.cfg.encoder_model.sched_pat, 
                                                        threshold_mode=self.cfg.encoder_model.threshold_mode, 
                                                        threshold=self.cfg.encoder_model.threshold,  
                                                        min_lr=self.cfg.encoder_model.min_lr * self.cfg.encoder_model.lr_reduce_factor,
                                                        cooldown=self.cfg.encoder_model.sched_cooldown) 

            # ===================================================================

            # wetw model =========================================================
            self.wetw_model = PairConvModel(embed_dim=self.cfg.weight_model.hidden_dim, 
                                            num_tokens=self.cfg.weight_model.num_tokens, 
                                            token_dim=self.cfg.weight_model.token_dim, 
                                            num_layers=self.cfg.weight_model.weight_model_layers, 
                                            conv_channels=self.cfg.weight_model.weight_model_channels,
                                            kernel_size=self.cfg.weight_model.kernel_size, 
                                            output_dim=self.dim_embedding, 
                                            use_positional_encoding=self.cfg.weight_model.use_pos_enc).to(self.device)
            
            if self.cfg.weight_model.optimizer == 'adam':
                self.wetw_optimizer = optim.Adam(self.wetw_model.parameters(), lr=self.cfg.weight_model.lr_init)
   
                    
            self.wetw_scheduler = ReduceLROnPlateau(optimizer=self.wetw_optimizer, 
                                                    mode=self.cfg.weight_model.mode, 
                                                    factor=self.cfg.weight_model.lr_reduce_factor,
                                                    patience=self.cfg.weight_model.lr_scheduler_patience, 
                                                    threshold_mode=self.cfg.weight_model.threshold_mode, 
                                                    threshold=self.cfg.weight_model.threshold,  
                                                    min_lr=self.cfg.weight_model.min_lr * self.cfg.weight_model.lr_reduce_factor,
                                                    cooldown=self.cfg.weight_model.sched_cooldown)

            # ===================================================================

            print(self.encoder_model)
            print(self.wetw_model)
  
            # -------------------------------------------------------
            # Count parameters 
            # -------------------------------------------------------

            encoder_params = sum(p.numel() for p in self.encoder_model.parameters() if p.requires_grad)
            wetw_params    = sum(p.numel() for p in self.wetw_model.parameters() if p.requires_grad)

            parameters_num = encoder_params + wetw_params

            self.logger.info(f"Encoder parameters: {encoder_params:,}")
            self.logger.info(f"WETW parameters: {wetw_params:,}")
            self.logger.info(f"Total trainable parameters: {parameters_num:,}")

            print("--------------------------------------------------")
            print(f"Encoder parameters: {encoder_params:,}")
            print(f"WETW parameters: {wetw_params:,}")
            print(f"Total trainable parameters: {parameters_num:,}")
            print("--------------------------------------------------")

    def train(self, extra_epoch: int = None) -> None:
        self.__setup()

        local_max_epoch = self.max_epoch

        self.best_valid_loss = float('inf')
        
        while self.epoch < local_max_epoch:
            
            if self.epoch == 0:
                self.__sample()
                        
            self.__adjust_wd(self.cfg.encoder_model.wd_max, self.cfg.encoder_model.wd_min)

            epoch_train_loss = self.__train_epoch()
            epoch_valid_loss = self.__validate_epoch()

            if self.epoch >= self.cfg.training.start_sched_after_epoch:

                self.encoder_scheduler.step(epoch_valid_loss)
                self.wetw_scheduler.step(epoch_valid_loss)

            self.epoch += 1


            if epoch_valid_loss < self.best_valid_loss:
                self.best_valid_loss = epoch_valid_loss
                encoder_best_sd = self.encoder_model.state_dict()
                wetw_best_sd = self.wetw_model.state_dict()

                torch.save(encoder_best_sd, self.cfg.paths.best_encoder_sd_fp)
                torch.save(wetw_best_sd, self.cfg.paths.best_wetw_sd_fp)
                OmegaConf.save(self.cfg, self.cfg.paths.conf_fp)

                
            for param_group in self.encoder_optimizer.param_groups:
                lr_current_enc = param_group['lr'] 
            for param_group in self.wetw_optimizer.param_groups:
                lr_current_wetw = param_group['lr']
            # if early stopping and no improvement or max epoch then save the model and end training
            if (self.cfg.training.early_stopping and \
                lr_current_enc < self.cfg.encoder_model.min_lr and \
                lr_current_wetw < self.cfg.weight_model.min_lr) or \
                (self.epoch >= local_max_epoch):


                self.logger.info(f'Stopping after a total of {self.epoch + 1} epochs.')
                print(f'Stopping after a total of {self.epoch + 1} epochs.')

                # loading best models so far 
                self.encoder_model.load_state_dict(encoder_best_sd)
                self.wetw_model.load_state_dict(wetw_best_sd)

                embedData(self.encoder_model,
                        self.cfg.dataset.processed_train_fp,
                        self.cfg.paths.train_embedding_filepath,
                        self.size_db,
                        batch_size=self.cfg.training.embed_batch,
                        original_dim=self.dim_series,
                        embedded_dim=self.dim_embedding,
                        device=self.device )

                embedData(self.encoder_model,
                        self.cfg.dataset.processed_test_fp,
                        self.cfg.paths.test_embedding_filepath,
                        self.size_test_db,
                        batch_size=self.cfg.training.embed_batch,
                        original_dim=self.dim_series,
                        embedded_dim=self.dim_embedding,
                        device=self.device)
                                    
                
                break                
    

    def __train_epoch(self) -> None:
        start = timer()
      
        self.encoder_model.train()
        self.wetw_model.train()

        losses = []
        for batch_idx, (batch_data, batch_indices) in enumerate(self.train_dataloader):

            self.encoder_optimizer.zero_grad()
            self.wetw_optimizer.zero_grad()

            batch_embedded = self.encoder_model.encode(batch_data)

            loss = self.lossF_distances_train(batch_embedded, batch_indices, wetw_model=self.wetw_model)


            loss.backward()

            self.encoder_optimizer.step()
            self.wetw_optimizer.step()

            
            losses.append(loss.detach().item())

            self.logger.info('t{:d}={:.3f}s: l/dd={:.3f}'.format(self.epoch, timer() - start, np.mean(losses)))

        return np.mean(losses)

    def __validate_epoch(self) -> None:
        start = timer()

        self.encoder_model.eval()
        self.wetw_model.eval()

        losses = []

        with torch.no_grad():
            for batch_idx, (batch_data, batch_indices) in enumerate(self.valid_dataloader):

                batch_embedded = self.encoder_model.encode(batch_data)
                   
                loss = self.lossF_distances_valid(batch_embedded, batch_indices, wetw_model=self.wetw_model)

                losses.append(loss.detach().item())

 
            # getting the learning rate for the logger    
            for param_group in self.encoder_optimizer.param_groups:
                lr = param_group['lr']
                break

            self.logger.info('v{:d}={:.3f}s: l/dd={:.3f}, lr={}'.format(
                self.epoch, timer() - start, np.mean(losses), lr ))
      
        return np.mean(losses)

    def __sample(self) -> None:
        start = timer()
        
        if self.train_size >= 100_000:
                self.train_data_np, self.train_dist_np, self.valid_data_np, self.valid_dist_np = self.sampler.sample()
        else:
            self.train_data_np, self.train_dist_np, self.valid_data_np, self.valid_dist_np = self.sampler.sample(all_data=True)


        train_data_tensor = torch.from_numpy(
            self.train_data_np.reshape([-1, 1, self.dim_series])).to(self.device)
        
        self.train_dataloader = DataLoader(DatasetwithIdx(
            train_data_tensor), batch_size=self.batch_size, shuffle=True)
        
        self.lossF_distances_train.update_targets(self.train_dist_np)
        
        valid_data_tensor = torch.from_numpy(
            self.valid_data_np.reshape([-1, 1, self.dim_series])).to(self.device)
        
        self.valid_dataloader = DataLoader(DatasetwithIdx(
            valid_data_tensor), batch_size=self.batch_size, shuffle=True)
        
        self.lossF_distances_valid.update_targets(self.valid_dist_np)


    def __adjust_wd(self, wd_max, wd_min):
        
        for param_group in self.encoder_optimizer.param_groups:
            current_lr = param_group['lr']

            if self.cfg.encoder_model.weight_decay_mode == 'linear':
                progress = current_lr / self.cfg.encoder_model.lr_init
                new_wd = wd_min + progress * (wd_max - wd_min) 
                param_group['weight_decay'] = new_wd

