import torch
from torch import nn
from torch.nn import functional as F

from src.afrcnn import AsynchronousBlock
from src.pilot1_cross_speaker.model import CrossSpeakerAFRCNN


class GatedAsynchronousBlock(AsynchronousBlock):
    def __init__(self, bottleneck, hidden, stages):
        super().__init__(bottleneck, hidden, stages)
        # The control's remaining layers must see the same CPU initialization RNG.
        with torch.random.fork_rng(devices=[]):
            self.gates = nn.ModuleList([
                nn.Conv1d(hidden, hidden, 1, groups=hidden) for _ in range(stages - 1)
            ])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def modulate(self, features, coarser, index):
        gains = 2 * self.gates[index](coarser).sigmoid()
        return features * F.interpolate(gains, size=features.shape[-1], mode="nearest")


class CoarseToFineAFRCNN(CrossSpeakerAFRCNN):
    block_type = GatedAsynchronousBlock
