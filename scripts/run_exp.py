import os
import logging
import datetime
import numpy as np
import time 
import torch
import hydra
from omegaconf import DictConfig
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.helper_functions import max_layers_before_collapse
from util.train import Train

@hydra.main(config_path="/home/cpanourg/projects/wesee_wetw/config/", config_name="config")
def main(cfg: DictConfig):

    torch.manual_seed(1229)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1998)
        cfg.system.device = 'cuda'
    else:
        cfg.system.device = 'cpu'
        
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    cfg.exps_dir = f"{cfg.exps_dir}/{cfg.dataset.dataset_name}-{datetime_string}"
    os.makedirs(cfg.exps_dir, exist_ok=True)
    print(f"Created directory: {cfg.exps_dir}")



    cfg.encoder_model.strided_conv_num = max_layers_before_collapse(cfg.dataset.dim_series)
        
            
    train_manager = Train(cfg)

        
    start_time = time.time() 

    train_manager.train()

    end_time = time.time() 

    logger = logging.getLogger()

    print('-------------------------')
    print(f'Training time = {end_time - start_time}')
    print('-------------------------')

    logger.info('-------------------------')
    logger.info(f'Training time = {end_time - start_time}')
    logger.info('-------------------------')


if __name__ == "__main__":
    main()