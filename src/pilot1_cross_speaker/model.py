"""Shared convolutional exchange between the two early-split speaker streams."""

import torch
from torch import nn

from src.earlier_speaker_split import EarlySplitAFRCNN


class CrossSpeakerAFRCNN(EarlySplitAFRCNN):
    def __init__(self, sources, bottleneck_channels, exchange_channels, exchange_iterations,
                 **options):
        super().__init__(sources=sources, bottleneck_channels=bottleneck_channels, **options)
        if sources != 2:
            raise ValueError("Cross-speaker refinement requires exactly two sources")
        if not exchange_iterations or any(
            type(index) is not int or not 1 <= index <= self.iterations - self.split_iteration
            for index in exchange_iterations
        ):
            raise ValueError("Exchange iterations must index the speaker refinement loop")
        self.exchange_iterations = frozenset(exchange_iterations)
        self.exchange = nn.Sequential(
            nn.Conv1d(2 * bottleneck_channels, exchange_channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(exchange_channels, exchange_channels, 5, padding=2,
                      groups=exchange_channels, bias=False),
            nn.SiLU(),
            nn.Conv1d(exchange_channels, bottleneck_channels, 1, bias=False),
        )
        nn.init.zeros_(self.exchange[-1].weight)

    def refine_speakers(self, features, iteration):
        if iteration not in self.exchange_iterations:
            return features
        channels, frames = features.shape[1:]
        speakers = features.reshape(-1, 2, channels, frames)
        pairs = torch.cat((speakers, speakers.flip(1)), dim=2)
        residual = self.exchange(pairs.reshape(-1, 2 * channels, frames))
        return features + residual
