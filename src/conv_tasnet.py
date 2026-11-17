import torch
from torch import nn
from torch.nn import functional as F

from src.normalization import GlobalLayerNorm


class TemporalBlock(nn.Module):
    def __init__(self, bottleneck, hidden, skip, kernel, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(bottleneck, hidden, 1), nn.PReLU(), GlobalLayerNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation,
                      padding=(kernel - 1) * dilation // 2, groups=hidden),
            nn.PReLU(), GlobalLayerNorm(hidden),
        )
        self.residual = nn.Conv1d(hidden, bottleneck, 1)
        self.skip = nn.Conv1d(hidden, skip, 1)

    def forward(self, x):
        features = self.layers(x)
        return x + self.residual(features), self.skip(features)


class ConvTasNet(nn.Module):
    def __init__(self, sources, encoder_channels, kernel_size, stride, bottleneck_channels,
                 hidden_channels, skip_channels, temporal_kernel, blocks, repeats):
        super().__init__()
        if min(sources, encoder_channels, kernel_size, stride, bottleneck_channels,
               hidden_channels, skip_channels, temporal_kernel, blocks, repeats) <= 0:
            raise ValueError("Model dimensions must be positive")
        if temporal_kernel % 2 != 1 or stride > kernel_size:
            raise ValueError("Use an odd temporal kernel and stride <= encoder kernel")
        self.sources, self.kernel, self.stride = sources, kernel_size, stride
        self.encoder = nn.Conv1d(1, encoder_channels, kernel_size, stride, bias=False)
        self.bottleneck = nn.Sequential(
            GlobalLayerNorm(encoder_channels), nn.Conv1d(encoder_channels, bottleneck_channels, 1)
        )
        self.blocks = nn.ModuleList([
            TemporalBlock(bottleneck_channels, hidden_channels, skip_channels, temporal_kernel,
                          2 ** block)
            for _ in range(repeats) for block in range(blocks)
        ])
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(skip_channels, sources * encoder_channels, 1),
                                  nn.ReLU())
        self.decoder = nn.ConvTranspose1d(encoder_channels, 1, kernel_size, stride, bias=False)

    def forward(self, mixture):
        """[B, T] waveform -> [B, sources, T], with exact length preservation."""
        if mixture.ndim != 2 or mixture.shape[-1] == 0:
            raise ValueError("Expected a nonempty [batch, time] waveform")
        length = mixture.shape[-1]
        # Context padding covers waveform boundaries even for sub-kernel inputs.
        extra = (-(length + self.kernel)) % self.stride
        encoded = F.relu(self.encoder(F.pad(mixture[:, None], (self.kernel, self.kernel + extra))))
        x = self.bottleneck(encoded)
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        batch, channels, frames = encoded.shape
        masks = self.mask(skip_sum).reshape(batch, self.sources, channels, frames)
        decoded = self.decoder((encoded[:, None] * masks).reshape(-1, channels, frames))
        return decoded.reshape(batch, self.sources, -1)[..., self.kernel:self.kernel + length]
