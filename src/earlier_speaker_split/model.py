"""Early speaker splitting with shared A-FRCNN refinement and reconstruction.

Reuses the MIT-licensed A-FRCNN adaptation in src/afrcnn.py.
Copyright (c) 2021 Kai Li. See THIRD_PARTY_NOTICES.md.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.afrcnn import AFRCNN
from src.normalization import GlobalLayerNorm


class EarlySplitAFRCNN(AFRCNN):
    def __init__(self, sources, encoder_channels, kernel_size, stride, bottleneck_channels,
                 hidden_channels, stages, iterations, split_iteration):
        if not isinstance(split_iteration, int) or not 1 <= split_iteration < iterations:
            raise ValueError("split_iteration must be an integer between 1 and iterations - 1")
        super().__init__(sources, encoder_channels, kernel_size, stride, bottleneck_channels,
                         hidden_channels, stages, iterations)
        self.split_iteration = split_iteration
        self.speaker_split = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 2 * sources * bottleneck_channels, 1),
            nn.GLU(dim=1),
        )
        self.speaker_norm = GlobalLayerNorm(bottleneck_channels)
        self.mask = nn.Sequential(nn.PReLU(),
                                  nn.Conv1d(bottleneck_channels, encoder_channels, 1),
                                  nn.ReLU())
        # Each source uses the same decoder, without mixing the source channels.
        self.decoder = nn.ConvTranspose1d(encoder_channels, 1, kernel_size, stride=stride,
                                          padding=kernel_size // 2,
                                          output_padding=stride - 1, bias=False)
        nn.init.xavier_uniform_(self.decoder.weight)

    def refine_speakers(self, features, iteration):
        return features

    def _decode_sources(self, features, encoded, length):
        batch, channels, frames = encoded.shape
        masks = self.mask(features).reshape(batch, self.sources, channels, frames)
        masked = (masks * encoded[:, None]).reshape(batch * self.sources, channels, frames)
        estimates = self.decoder(masked).reshape(batch, self.sources, -1)
        return estimates[..., :length]

    def forward(self, mixture, *, tap_iteration=None):
        """Separate waveforms, optionally returning a second estimate at a speaker step."""
        if mixture.ndim != 2 or mixture.shape[-1] == 0:
            raise ValueError("Expected a nonempty [batch, time] waveform")
        length = mixture.shape[-1]
        encoded = self.encoder(F.pad(mixture[:, None], (0, -length % self.padding_multiple)))
        initial = self.bottleneck(self.input_norm(encoded))
        features = initial
        for index in range(self.split_iteration):
            if index:
                features = self.refine(initial + features)
            features = self.block(features)

        batch, channels, frames = features.shape
        # Folding sources into the batch keeps normalization and refinement independent.
        initial = self.speaker_norm(
            self.speaker_split(features).reshape(batch * self.sources, channels, frames)
        )
        features = initial
        tapped = None
        for index in range(self.iterations - self.split_iteration):
            if index:
                features = self.refine(initial + features)
            # Recompute speaker-block activations during backward to limit GPU memory.
            features = (checkpoint(self.block, features, use_reentrant=False)
                        if self.training and torch.is_grad_enabled() else self.block(features))
            features = self.refine_speakers(features, index + 1)
            if index + 1 == tap_iteration:
                tapped = features

        estimates = self._decode_sources(features, encoded, length)
        if tap_iteration is None:
            return estimates
        if tapped is None:
            raise ValueError("tap_iteration must index a speaker refinement step")
        return estimates, self._decode_sources(tapped, encoded, length)
