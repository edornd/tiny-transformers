import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as func
from typing import Callable


class EncoderDecoder(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 source_emb: nn.Module,
                 target_emb: nn.Module,
                 generator: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_emb = source_emb
        self.target_emb = target_emb
        self.generator = generator

    def encode(self, source: Tensor, source_mask: Tensor) -> Tensor:
        return self.encoder(self.source_emb(source), source_mask)

    def decode(self, memory: Tensor, source_mask: Tensor, target: Tensor, target_mask: Tensor) -> Tensor:
        return self.decoder(self.target_emb(target), memory, source_mask, target_mask)

    def forward(self, source: Tensor, target: Tensor, source_mask: Tensor, target_mask: Tensor):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)


class Generator(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return func.log_softmax(self.linear(x), dim=-1)


class ResidualConnection(nn.Module):
    """Implements a residual connection, similar to ResNets.
    The flow can be seen as:
    x ------------------------ + ---> output
    \---> f(x) ---> dropout --/

    This block corresponds to Multi-Head Attention and Feed Forward blocks in fig.1 (left).
    """

    def __init__(self, function: Callable, size: int, dropout: float = 0.5):
        super().__init__()
        self.sublayer = function
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.sublayer(self.norm(x)))
