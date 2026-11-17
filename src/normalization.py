import torch
from torch import nn


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mean = x.mean((1, 2), keepdim=True)
        variance = (x - mean).square().mean((1, 2), keepdim=True)
        return ((x - mean) * torch.rsqrt(variance + 1e-8) * self.weight + self.bias).to(dtype)
