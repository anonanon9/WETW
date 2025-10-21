import numpy as np 
import torch 
from model.activation import LeCunTanh
from omegaconf import DictConfig
from typing import Union, List
from model.normalization import AdaNorm

def pad_with_zeros(arr, target_length):
    if arr.shape[1] < target_length:
        padding = target_length - arr.shape[1]
        arr_padded = np.pad(arr, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    else:
        arr_padded = arr[:, :target_length]  # Truncate if longer
    return arr_padded


def weighted_pairwise_distance(E_i, E_j, w, p=2):
    """
    Efficiently compute the weighted pairwise distance between two embeddings.

    Parameters:
    - E_i: First embedding (Tensor of shape [batch_size, embedding_dim])
    - E_j: Second embedding (Tensor of shape [batch_size, embedding_dim])
    - w: Weight vector (Tensor of shape [batch_size, embedding_dim])
    - p: Power parameter for the distance (p=2 for Euclidean distance)

    Returns:
    - Weighted distance (Tensor of shape [batch_size])
    """
    # Compute weighted p-norm distance in a single operation
    # E_i = F.normalize(E_i, p=2, dim=1)
    # E_j = F.normalize(E_j, p=2, dim=1)
    epsilon = 1e-6
    w = torch.clamp(w, min=1e-6)

    distance = torch.sum(w * torch.abs(E_i - E_j) ** p, dim=1).add(epsilon).pow(1.0 / p)

    return distance



def getActivation(name: str, cfg: DictConfig) -> torch.nn.Module:
    if name == 'tanh':
        return torch.nn.Tanh()
    elif name == 'lecuntanh':
        return LeCunTanh()
    elif name == 'relu':
        return torch.nn.ReLU()
    elif name == 'leakyrelu':
        relu_slope = cfg.encoder_model.relu_slope
        return torch.nn.LeakyReLU(relu_slope)
    elif name == 'prelu':
        prelu_params = cfg.encoder_model.prelu_params
        return torch.nn.PReLU(prelu_params)
    return torch.nn.Identity()

def getWeightNorm(model: torch.nn.Module, cfg: DictConfig) -> torch.nn.Module:
    weightnorm_type = cfg.encoder_model.weightnorm_type

    if weightnorm_type == 'weightnorm':
        return torch.nn.utils.weight_norm(model, dim=cfg.encoder_model.weightnorm_dim)

    return model

def getLayerNorm(shape: Union[int, List[int], torch.Size], cfg: DictConfig) -> torch.nn.Module:
    
    layernorm_type = cfg.encoder_model.layernorm_type
    if layernorm_type == 'layernorm':
        return torch.nn.LayerNorm(shape, elementwise_affine=cfg.encoder_model.layernorm_elementwise_affine)
    
    elif layernorm_type == 'adanorm':
        return AdaNorm(shape, cfg.encoder_model.adanorm_k, cfg.encoder_model.adanorm_scale, cfg.encoder_model.eps, cfg.encoder_model.layernorm_elementwise_affine)

    return torch.nn.Identity()


def getDilation(depth: int, cfg: DictConfig) -> int:
    dilation_type = cfg.encoder_model.dilation_type

    if dilation_type == 'exponential':
        return int(2 ** (depth - 1))
    
    elif dilation_type == 'linear':
        return cfg.encoder_model.dilation_base + cfg.encoder_model.dilation_slope * (depth - 1)

    return cfg.encoder_model.dilation_constant


def max_layers_before_collapse(initial_dim, kernel_size=3, stride=2, padding=0):
    """
    Compute how many Strided conv layers can be applied 
    before spatial dimension collapses to <= 1.
    """
    dim = initial_dim
    num_layers = 0

    while True:
        next_dim = (dim - kernel_size + 2 * padding) // stride + 1
        if next_dim <= 1:
            break
        dim = next_dim
        num_layers += 1

    return num_layers


def get_divisors(N):
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:  # Avoid adding the square root twice for perfect squares
                divisors.append(N // i)
    return sorted(divisors)
