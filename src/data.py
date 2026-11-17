import json
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.audio import read_audio, resample
from src.settings import project_path, resolve_training_config


class SeparationDataset(Dataset):
    def __init__(self, root, manifest, sample_rate, segment_seconds=None, random_crop=False,
                 fixed_crops=False, dynamic_mixing=None, seed=0, sources=3,
                 speed_perturbation=None):
        self.root = project_path(root)
        manifest_path = self.root / manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing prepared manifest: {manifest_path}. "
                                    "Prepare the selected LibriMix dataset before training.")
        self.records = json.loads(manifest_path.read_text())["examples"]
        if not self.records:
            raise ValueError(f"Empty dataset: {manifest}")
        self.rate = sample_rate
        self.sources = sources
        if any(len(record["sources"]) != sources for record in self.records):
            raise ValueError(f"Expected {sources} sources per example in {manifest_path}")
        self.segment = round(segment_seconds * sample_rate) if segment_seconds else None
        self.random_crop = random_crop
        self.seed = seed
        self.mixing = dynamic_mixing if dynamic_mixing and dynamic_mixing["enabled"] else None
        self.speeds = None
        if speed_perturbation and speed_perturbation["enabled"]:
            if not self.mixing:
                raise ValueError("Source speed perturbation requires dynamic mixing")
            self.speeds = speed_perturbation["speeds"]
            if not self.speeds or any(type(speed) is not int or speed <= 0
                                      for speed in self.speeds):
                raise ValueError("speed_perturbation.speeds must be positive integer percentages")
        if fixed_crops or self.mixing:
            if not self.segment or self.segment <= 0:
                raise ValueError("Fixed crops and dynamic mixing require a positive segment_seconds")
            if any(record["length"] < self.segment for record in self.records):
                raise ValueError("Training sources must be at least segment_seconds long; use a shorter crop")
        if self.mixing:
            low, high = self.mixing["gain_db"]
            if not (np.isfinite([low, high, self.mixing["source_rms"],
                                 self.mixing["peak_limit"]]).all()
                    and low <= high and self.mixing["source_rms"] > 0
                    and 0 < self.mixing["peak_limit"] <= 1
                    and isinstance(self.mixing["crop_attempts"], int)
                    and self.mixing["crop_attempts"] >= 1):
                raise ValueError("Invalid dynamic_mixing levels or crop_attempts")
            self.speakers = {}
            for record in self.records:
                utterances = record["id"].split("_")
                if len(utterances) != self.sources:
                    raise ValueError(f"Dynamic mixing requires {self.sources} LibriMix source IDs per record")
                for utterance, path in zip(utterances, record["sources"]):
                    speaker = utterance.split("-")[0]
                    self.speakers.setdefault(speaker, []).append((path, record["length"], utterance))
            if len(self.speakers) < self.sources:
                raise ValueError(f"Dynamic mixing requires at least {self.sources} training speakers")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Per-example RNG also makes workers=0 reproducible across epoch recovery.
        rng = random.Random(self.seed + index)
        if self.mixing:
            return self.mix_sources(rng)
        record = self.records[index]
        length = record["length"]
        start = 0
        if self.segment and length > self.segment:
            start = (rng.randint(0, length - self.segment) if self.random_crop
                     else (length - self.segment) // 2)
            length = self.segment
        signals = []
        for path in [record["mixture"], *record["sources"]]:
            rate, audio = read_audio(self.root / path, start=start,
                                     frames=length if self.segment else None)
            if rate != self.rate or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid waveform: {path}")
            signals.append(torch.from_numpy(audio))
        if any(len(signal) != length for signal in signals):
            raise ValueError("Expected aligned mixture and sources")
        target = torch.stack(signals[1:])
        if (target.var(-1, unbiased=False) <= 1e-10).any():
            raise ValueError(f"Silent source in example {record['id']}")
        return signals[0], target, length, record["id"]

    def mix_sources(self, rng):
        targets, identifiers = [], []
        for speaker in rng.sample(list(self.speakers), self.sources):
            speed = rng.choice(self.speeds) if self.speeds else 100
            for _ in range(self.mixing["crop_attempts"]):
                path, length, utterance = rng.choice(self.speakers[speaker])
                frames = min(length, math.ceil(self.segment * speed / 100))
                start = rng.randint(0, length - frames)
                rate, audio = read_audio(self.root / path, start=start, frames=frames)
                if (rate != self.rate or audio.ndim != 1 or len(audio) != frames
                        or not np.isfinite(audio).all()):
                    raise ValueError(f"Invalid training source: {path}")
                audio = resample(audio, speed, 100)[:self.segment]
                # Short utterances may end before a speed-up supplies a complete crop.
                if len(audio) < self.segment:
                    audio = np.pad(audio, (0, self.segment - len(audio)))
                if np.var(audio) > 1e-10:
                    break
            else:
                raise ValueError(f"Could not sample an active crop for training speaker {speaker}")
            rms = float(np.sqrt(np.mean(audio ** 2)))
            gain = self.mixing["source_rms"] * 10 ** (rng.uniform(*self.mixing["gain_db"]) / 20)
            targets.append(torch.from_numpy(audio * (gain / rms)))
            identifiers.append(f"{utterance}@{start}" + (f":speed={speed}" if self.speeds else ""))
        target = torch.stack(targets)
        mixture = target.sum(0)
        peak = max(float(mixture.abs().max()), float(target.abs().max()))
        gain = min(1.0, self.mixing["peak_limit"] / peak)
        target = target * gain
        return target.sum(0), target, self.segment, "_".join(identifiers)


def collate_audio(examples):
    lengths = torch.tensor([example[2] for example in examples])
    width = int(lengths.max())
    mixture = torch.zeros(len(examples), width)
    sources = torch.zeros(len(examples), examples[0][1].shape[0], width)
    for index, (x, y, length, _) in enumerate(examples):
        mixture[index, :length] = x
        sources[index, :, :length] = y
    return mixture, sources, lengths, [example[3] for example in examples]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(config, split, epoch=0, crop_seconds=None):
    config = resolve_training_config(config)
    data, training = config["data"], config["training"]
    is_train = split == "train"
    dataset = SeparationDataset(
        data["root"], data[f"{split}_manifest"], data["sample_rate"],
        data["segment_seconds"] if is_train else crop_seconds, is_train,
        fixed_crops=is_train and data.get("fixed_crops", False),
        dynamic_mixing=data.get("dynamic_mixing") if is_train else None,
        speed_perturbation=data.get("speed_perturbation") if is_train else None,
        seed=training["seed"] + epoch * 1_000_003,
        sources=config["model"]["sources"],
    )
    generator = torch.Generator().manual_seed(training["seed"] + epoch)
    return DataLoader(dataset, batch_size=training["batch_size"] if is_train else 1,
                      shuffle=is_train, num_workers=training["workers"],
                      collate_fn=collate_audio, worker_init_fn=seed_worker, generator=generator,
                      pin_memory=select_pin_memory(training["device"]))


def select_pin_memory(device):
    return device != "cpu" and torch.cuda.is_available()
