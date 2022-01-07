import copy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

def repeat(layer: nn.Module, times: int) -> nn.ModuleList:
    """Repeats the layer module given as input N times, specified by the second parameter.

    Args:
        layer (nn.Module): layer to be repeated
        times (int): how many repetitions

    Returns:
        nn.ModuleList: Torch list of modules
    """
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(times)])


def mask_future_positions(size: int) -> Tensor:
    att_matrix = np.ones((1, size, size))
    mask = np.tril(att_matrix, k=1).astype(np.uint8)
    return torch.from_numpy(mask)


def attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    length = torch.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) / length
