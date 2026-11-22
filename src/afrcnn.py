"""A-FRCNN (Method B, concatenation fusion), adapted from Kai Li's MIT reference.

https://github.com/JusperLee/AFRCNN-For-Speech-Separation
Copyright (c) 2021 Kai Li. See THIRD_PARTY_NOTICES.md for the license.
"""

from math import lcm

import torch
from torch import nn
from torch.nn import functional as F

from src.normalization import GlobalLayerNorm


def conv_norm(inputs, outputs, kernel, stride=1, groups=1, activation=False):
    layers = [nn.Conv1d(inputs, outputs, kernel, stride=stride,
                        padding=(kernel - 1) // 2, groups=groups), GlobalLayerNorm(outputs)]
    if activation:
        layers.append(nn.PReLU())
    return nn.Sequential(*layers)


class AsynchronousBlock(nn.Module):
    def __init__(self, bottleneck, hidden, stages):
        super().__init__()
        self.project = conv_norm(bottleneck, hidden, 1, activation=True)
        self.downsample = nn.ModuleList([
            conv_norm(hidden, hidden, 5, stride=1 if index == 0 else 2, groups=hidden)
            for index in range(stages)
        ])
        self.lateral_downsample = nn.ModuleList([
            conv_norm(hidden, hidden, 5, stride=2, groups=hidden)
            for _ in range(stages - 1)
        ])
        self.fuse = nn.ModuleList([
            conv_norm(hidden * (2 if index in (0, stages - 1) else 3), hidden, 1,
                      activation=True)
            for index in range(stages)
        ])
        self.aggregate = conv_norm(hidden * stages, hidden, 1, activation=True)
        self.residual = nn.Conv1d(hidden, bottleneck, 1)

    def modulate(self, features, coarser, index):
        return features

    def forward(self, x):
        levels = []
        current = self.project(x)
        for layer in self.downsample:
            current = layer(current)
            levels.append(current)
        fused = []
        # Fuse the original adjacent levels simultaneously, then gather at full resolution.
        for index, level in enumerate(levels):
            neighbors = []
            if index:
                neighbors.append(self.lateral_downsample[index - 1](levels[index - 1]))
            neighbors.append(level)
            if index + 1 < len(levels):
                neighbors.append(F.interpolate(levels[index + 1], size=level.shape[-1],
                                               mode="nearest"))
            features = self.fuse[index](torch.cat(neighbors, dim=1))
            if index + 1 < len(levels):
                features = self.modulate(features, levels[index + 1], index)
            fused.append(F.interpolate(features, size=levels[0].shape[-1], mode="nearest")
                         if index else features)
        return x + self.residual(self.aggregate(torch.cat(fused, dim=1)))


class AFRCNN(nn.Module):
    block_type = AsynchronousBlock

    def __init__(self, sources, encoder_channels, kernel_size, stride, bottleneck_channels,
                 hidden_channels, stages, iterations):
        super().__init__()
        if min(sources, encoder_channels, kernel_size, stride, bottleneck_channels,
               hidden_channels, iterations) < 1 or stages < 2:
            raise ValueError("A-FRCNN requires positive dimensions and at least two stages")
        if kernel_size % 2 != 1 or not 1 <= stride <= kernel_size // 2:
            raise ValueError("A-FRCNN requires an odd encoder kernel and stride <= kernel_size // 2")
        self.sources, self.iterations = sources, iterations
        self.padding_multiple = lcm(stride, 2 ** stages)
        self.encoder = nn.Conv1d(1, encoder_channels, kernel_size, stride=stride,
                                 padding=kernel_size // 2, bias=False)
        self.input_norm = GlobalLayerNorm(encoder_channels)
        self.bottleneck = nn.Conv1d(encoder_channels, bottleneck_channels, 1)
        self.block = self.block_type(bottleneck_channels, hidden_channels, stages)
        self.refine = nn.Sequential(
            nn.Conv1d(bottleneck_channels, bottleneck_channels, 1, groups=bottleneck_channels),
            nn.PReLU(),
        )
        self.mask = nn.Sequential(nn.PReLU(),
                                  nn.Conv1d(bottleneck_channels, sources * encoder_channels, 1),
                                  nn.ReLU())
        # The reference uses a joint decoder across the masked source channels.
        self.decoder = nn.ConvTranspose1d(sources * encoder_channels, sources, kernel_size,
                                          stride=stride, padding=kernel_size // 2,
                                          output_padding=stride - 1, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, mixture):
        """[B, T] -> [B, sources, T], preserving arbitrary nonempty input lengths."""
        if mixture.ndim != 2 or mixture.shape[-1] == 0:
            raise ValueError("Expected a nonempty [batch, time] waveform")
        length = mixture.shape[-1]
        encoded = self.encoder(F.pad(mixture[:, None], (0, -length % self.padding_multiple)))
        initial = self.bottleneck(self.input_norm(encoded))
        features = initial
        for index in range(self.iterations):
            if index:
                features = self.refine(initial + features)
            features = self.block(features)
        batch, channels, frames = encoded.shape
        masks = self.mask(features).reshape(batch, self.sources, channels, frames)
        estimates = self.decoder((masks * encoded[:, None]).reshape(batch, -1, frames))
        return estimates[..., :length]
