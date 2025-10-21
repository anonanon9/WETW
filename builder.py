# coding = utf-8

from torch import nn, Tensor
from omegaconf import DictConfig
from model.ResidualAE import ResidualEncoder

class ModelBuilder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ModelBuilder, self).__init__()
        self.cfg = cfg
        device = cfg.system.device
        
        self.encoder = ResidualEncoder(cfg)

        self.encoder.to(device)
        
        

    def encode(self, input: Tensor) -> Tensor:

        out = self.encoder(input)
        
        return out
                    
    def forward(self, input: Tensor) -> Tensor:
        embedding = self.encode(input)
        
        return embedding

