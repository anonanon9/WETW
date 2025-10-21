# coding = utf-8

from torch import nn, Tensor
import torch.nn.functional as F
import torch 
from model.commons import Squeeze, Reshape
import numpy as np 
from omegaconf import DictConfig
from util.helper_functions import getActivation, getWeightNorm, getDilation, getLayerNorm

class _OriginalResBlock(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, out_channels, dilation):
        super(_OriginalResBlock, self).__init__()
        self.cfg = cfg
        dim_series = cfg.dataset.dim_series
        kernel_size = cfg.encoder_model.conv_kernel_size
        padding = int(kernel_size / 2) * dilation
        activation_name = cfg.encoder_model.activation_conv
        bias = cfg.encoder_model.layernorm_type == 'none'

        self.__residual_link = nn.Sequential(getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg),
                                             getLayerNorm(dim_series, cfg), 
                                             getActivation(activation_name, cfg),

                                             getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg),
                                             getLayerNorm(dim_series, cfg))
        
        if in_channels != out_channels:
            self.__identity_link = getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias), cfg)
        else:
            self.__identity_link = nn.Identity()

        self.__after_addition = getActivation(activation_name, cfg)
        
        
    def forward(self, input: Tensor) -> Tensor:
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)



class _PreActivatedResBlock(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, out_channels, dilation, first = False, last = False):
        super(_PreActivatedResBlock, self).__init__()

        self.cfg = cfg
        self.dim_series = cfg.dataset.dim_series
        self.last = last
        kernel_size = cfg.encoder_model.conv_kernel_size
        padding = int(kernel_size / 2) * dilation
        activation_name = cfg.encoder_model.activation_conv

        bias = cfg.encoder_model.layernorm_type == 'none' or not cfg.encoder_model.layernorm_elementwise_affine
                
        if first:
           
            self.__first_block = getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg)
            in_channels = out_channels
        else:
            self.__first_block = nn.Identity()

        self.__residual_link = nn.Sequential(getLayerNorm(self.dim_series, cfg), 
                                             getActivation(activation_name, cfg),
                                             getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg),
                                            
                                             getLayerNorm(self.dim_series, cfg),
                                             getActivation(activation_name, cfg),
                                             getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg))


        if in_channels != out_channels:
            self.__identity_link = getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias), cfg)
            
        else:
            self.__identity_link = nn.Identity()

        if last:
            self.__after_addition = nn.Sequential(getLayerNorm(self.dim_series, cfg), getActivation(activation_name, cfg))
        else:
            self.__after_addition = nn.Identity()


    def forward(self, input: Tensor) -> Tensor:
            
        input = self.__first_block(input)
        
        residual = input
        identity = input

        residual = self.__residual_link(residual)
                    
        identity = self.__identity_link(identity)
                
        out = self.__after_addition(identity + residual)
        
        return out


class _ResNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(_ResNet, self).__init__()

        self.cfg = cfg
        num_resblock = cfg.encoder_model.num_en_resblock

        if cfg.encoder_model.dilation_type == 'exponential':
            assert num_resblock > 1 and 2 ** (num_resblock + 1) <= cfg.dataset.dim_series + 1

        inner_channels = cfg.encoder_model.num_en_channels
        out_channels = cfg.encoder_model.dim_en_latent
      
        
        if cfg.encoder_model.resblock_pre_activation:
            layers = [_PreActivatedResBlock(cfg, 1, inner_channels, getDilation(1, cfg), first=True)]         
            layers += [_PreActivatedResBlock(cfg, inner_channels, inner_channels, getDilation(depth, cfg)) for depth in range(2, num_resblock)]
            layers += [_PreActivatedResBlock(cfg, inner_channels, out_channels, getDilation(num_resblock, cfg), last=True)]
       
        else:
            layers = [_OriginalResBlock(cfg, 1, inner_channels, getDilation(1, cfg))]
            layers += [_OriginalResBlock(cfg, inner_channels, inner_channels, getDilation(depth, cfg)) for depth in range(2, num_resblock)]
            layers += [_OriginalResBlock(cfg, inner_channels, out_channels, getDilation(num_resblock, cfg))]

        self.__model = nn.Sequential(*layers)

        
    def forward(self, input: Tensor) -> Tensor:
        
        input = self.__model(input)

        return input



class ResidualEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ResidualEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg.system.device
        dim_latent = cfg.encoder_model.dim_en_latent
        dim_embedding = cfg.encoder_model.dim_emb
        self.__resnet = _ResNet(cfg)
        
        self.identity = None
        self.downsample_layer = nn.Conv1d(in_channels=dim_latent, out_channels=dim_latent, kernel_size=3, stride=2, padding=0).to(cfg.system.device)
            
        self.__model = nn.Sequential(nn.AdaptiveMaxPool1d(1),
                                    Squeeze(),

                                    nn.Linear(dim_latent, dim_latent),
                                    getActivation(cfg.encoder_model.activation_linear, cfg),

                                    nn.Linear(dim_latent, dim_embedding, bias=False),
                                    nn.LayerNorm(dim_embedding, elementwise_affine=False) if cfg.encoder_model.encoder_normalize_embedding else nn.Identity())

        self.__model.to(self.device)


    def forward(self, input: Tensor) -> Tensor: 
    
        input = self.__resnet(input)
    
        for layer in self.__model:
            if self.cfg.encoder_model.residual_strided_convs:
                if type(layer) == torch.nn.modules.conv.Conv1d:
                    if self.identity == None:
                        self.identity = input 
                    else:
                        input = input + self.downsample_layer(self.identity)
                        self.identity = input 

                    input = layer(input)
                else:
                    input = layer(input)

            else:
                input = layer(input)
                    
        if self.cfg.encoder_model.residual_strided_convs:
            self.identity = None

        return input


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:

        return input

