from typing import Callable
from torch import Tensor
import torch.nn as nn

from tinytrf import ResidualConnection
from tinytrf.utils import repeat


class EncoderLayer(nn.Module):
    """Layer base, containing multi-head attention, feed forward and residual connections.
    Left component of fig.1 in the paper.
    """

    def __init__(self, size: int, self_attention: nn.Module, feed_forward: nn.Module, dropout: float = 0.5):
        super().__init__()
        self.size = size
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer_attn = ResidualConnection(self.self_attention, size=size, dropout=dropout)
        self.sublayer_ffwd = ResidualConnection(self.feed_forward, size=size, dropout=dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sublayer_attn(x)
        return self.sublayer_ffwd(x)


class Encoder(nn.Module):
    """Stack of N repeated layers, encoding input into semantically rich vectors.
    """
    def __init__(self, layer: EncoderLayer, times: int = 6):
        super().__init__()
        self.layers = repeat(layer, times=times)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
