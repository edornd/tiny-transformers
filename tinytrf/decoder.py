from torch import Tensor
import torch.nn as nn

from tinytrf import ResidualConnection
from tinytrf.utils import repeat


class DecoderLayer(nn.Module):
    """Basic block for a decoder: consists of the *masked* multi-head attention, multi-head attention
    with input from encoder and final feed forward layer, again with three residual connections.
    Corresponds to the right part of fig.1 in the paper.
    """

    def __init__(self, size: int, self_attention, source_attention, feed_forward, dropout: float = 0.5):
        super().__init__()
        self.size = size
        self.self_attention = self_attention
        self.source_attention = source_attention
        self.feed_forward = feed_forward
        self.sublayer1 = ResidualConnection(self.self_attention, size, dropou=dropout)
        self.sublayer2 = ResidualConnection(self.source_attention, size, dropout=dropout)
        self.sublayer3 = ResidualConnection(self.feed_forward, size, dropout=dropout)

    def forward(self, x: Tensor, memory: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
        x = self.sublayer1(x, target_mask)
        x = self.sublayer2(x, memory, source_mask)
        return self.sublayer3(x)

class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, times: int = 6):
        super().__init__()
        self.layers = repeat(layer=layer, times=times)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: Tensor, memory: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)
