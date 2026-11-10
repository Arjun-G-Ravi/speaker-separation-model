import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.audio import read_audio
from src.settings import project_path


class SeparationDataset(Dataset):
    def __init__(self, root, manifest, sample_rate, segment_seconds=None, random_crop=False):
        self.root = project_path(root)
        self.records = json.loads((self.root / manifest).read_text())["examples"]
        if not self.records:
            raise ValueError(f"Empty dataset: {manifest}")
        self.rate = sample_rate
        self.segment = round(segment_seconds * sample_rate) if segment_seconds else None
        self.random_crop = random_crop

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        signals = []
        for path in [record["mixture"], *record["sources"]]:
            rate, audio = read_audio(self.root / path)
            if rate != self.rate or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid waveform: {path}")
            signals.append(torch.from_numpy(audio.copy()))
        if len(signals) != 4 or len({len(signal) for signal in signals}) != 1:
            raise ValueError("Expected aligned mixture and three sources")
        length = len(signals[0])
        if self.segment and length > self.segment:
            start = (random.randint(0, length - self.segment) if self.random_crop
                     else (length - self.segment) // 2)
            signals = [signal[start:start + self.segment] for signal in signals]
            length = self.segment
        target = torch.stack(signals[1:])
        if (target.var(-1, unbiased=False) <= 1e-10).any():
            raise ValueError(f"Silent source in example {record['id']}")
        return signals[0], target, length, record["id"]


def collate_audio(examples):
    lengths = torch.tensor([example[2] for example in examples])
    width = int(lengths.max())
    mixture = torch.zeros(len(examples), width)
    sources = torch.zeros(len(examples), 3, width)
    for index, (x, y, length, _) in enumerate(examples):
        mixture[index, :length] = x
        sources[index, :, :length] = y
    return mixture, sources, lengths, [example[3] for example in examples]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(config, split, epoch=0, crop_seconds=None):
    data, training = config["data"], config["training"]
    is_train = split == "train"
    dataset = SeparationDataset(
        data["root"], data[f"{split}_manifest"], data["sample_rate"],
        data["segment_seconds"] if is_train else crop_seconds, is_train,
    )
    generator = torch.Generator().manual_seed(training["seed"] + epoch)
    return DataLoader(dataset, batch_size=training["batch_size"] if is_train else 1,
                      shuffle=is_train, num_workers=training["workers"],
                      collate_fn=collate_audio, worker_init_fn=seed_worker, generator=generator,
                      pin_memory=select_pin_memory(training["device"]))


def select_pin_memory(device):
    return device != "cpu" and torch.cuda.is_available()
