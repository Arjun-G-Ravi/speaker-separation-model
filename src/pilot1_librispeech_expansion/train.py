"""Run one matched Pilot 1 fine-tuning arm."""

import hashlib
import json
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.audio import read_audio
from src.data import collate_audio, make_loader
from src.model import build_model
from src.settings import digest, load_config, project_path, seed_all, select_device
from src.train import run_training, train_epoch

CONFIG = "pilot1_librispeech_expansion.yaml"


def load_source_pool(config, manifest):
    manifest = project_path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"Run pilot source preparation first: {manifest}")
    content = manifest.read_bytes()
    pool = json.loads(content)
    data = config["data"]
    min_samples = round(data["segment_seconds"] * data["sample_rate"])
    if pool["sample_rate"] != data["sample_rate"] or pool["min_samples"] > min_samples:
        raise ValueError("Source pool configuration differs from training")
    speakers = {}
    seen = set()
    for row in pool["sources"]:
        utterance = row["id"]
        path = manifest.parent / row["path"]
        speaker = row["speaker"]
        if (utterance in seen or speaker != utterance.split("-")[0]
                or row["samples"] < min_samples or not path.is_file()):
            raise ValueError(f"Invalid prepared source: {row}")
        seen.add(utterance)
        speakers.setdefault(speaker, []).append((path, row["samples"], utterance))
    if len(speakers) < config["model"]["sources"]:
        raise ValueError("Source pool contains too few eligible speakers")
    for split in ("validation", "test"):
        held_out = project_path(data["root"]) / data[f"{split}_manifest"]
        records = json.loads(held_out.read_text())["examples"]
        identities = {utterance.split("-")[0] for row in records
                      for utterance in row["id"].split("_")}
        if identities.intersection(speakers):
            raise ValueError(f"Training source pool overlaps {split} speakers")
    return speakers, hashlib.sha256(content).hexdigest()


def pilot_loader(config, split, epoch=0, *, speakers):
    loader = make_loader(config, split, epoch)
    if split == "train":
        loader.dataset.speakers = speakers
    return loader


@torch.inference_mode()
def fit_decoder_gain(model, config, examples, device):
    data = config["data"]
    root = project_path(data["root"])
    rows = json.loads((root / data["train_manifest"]).read_text())["examples"]
    if not 1 <= examples <= len(rows):
        raise ValueError("calibration_examples must fit the training manifest")
    samples = round(data["segment_seconds"] * data["sample_rate"])
    gains = []
    model.eval()
    for index in np.linspace(0, len(rows) - 1, examples, dtype=int):
        row = rows[index]
        rate, signal = read_audio(root / row["mixture"], start=(row["length"] - samples) // 2,
                                  frames=samples)
        if rate != data["sample_rate"] or signal.shape != (samples,):
            raise ValueError(f"Invalid calibration waveform: {row['mixture']}")
        mixture = torch.from_numpy(signal).to(device)
        summed = model(mixture[None])[0].sum(dim=0).double()
        gains.append((mixture.double().dot(summed) / summed.square().sum()).item())
    if not np.isfinite(gains).all() or not (gain := float(np.median(gains))):
        raise ValueError("Failed to fit a finite, nonzero decoder gain")
    return gain


@torch.no_grad()
def initialize_weights(model, weights, gain):
    model.load_state_dict(weights)
    if gain != 1:
        model.decoder.weight.mul_(gain)


def preflight(model, loader, config, device):
    options = config["training"]
    batch_size = options["batch_size"]
    batch = Subset(loader.dataset, range(batch_size))
    batches = DataLoader(batch, batch_size=batch_size, collate_fn=collate_audio)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"],
                                 weight_decay=options["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and options["amp"])
    train_epoch(model, batches, optimizer, scaler, device, options)
    print(f"Preflight forward and backward passed at batch {batch_size} on {device}", flush=True)


def run_pilot(settings):
    arm = settings["arm"]
    if arm not in ("A", "B", "C"):
        raise ValueError("Pilot arm must be A, B, or C")
    device = select_device(settings["training"]["device"])
    baseline = project_path(settings["baseline"])
    parent = torch.load(baseline, map_location="cpu", weights_only=True)
    config = deepcopy(parent["config"])
    if (config["model"]["name"] != "early_split_afrcnn" or config["model"]["sources"] != 2
            or not config["data"]["dynamic_mixing"]["enabled"]):
        raise ValueError("Pilot requires a two-speaker early-split dynamic-mixing checkpoint")
    for split in ("train", "validation", "test"):
        data = config["data"]
        manifest = project_path(data["root"]) / data[f"{split}_manifest"]
        if digest(manifest) != parent["manifest_hashes"][split]:
            raise ValueError(f"{split} manifest differs from the baseline checkpoint")
    config["experiment"] = f"pilot1_librispeech_expansion_{arm}"
    config["training"].update(settings["training"])
    config["checkpoint"] = {"resume": settings["resume"]}
    pool_path = settings["source_pool"]["manifest"] if arm == "C" else None
    loader_factory, pool_hash = make_loader, {}
    if pool_path:
        speakers, pool_hash["source_pool"] = load_source_pool(config, pool_path)
        loader_factory = partial(pilot_loader, speakers=speakers)

    provenance = {"arm": arm, "baseline": str(baseline), "baseline_sha256": digest(baseline),
                  "source_pool": pool_path}
    config["pilot1"] = provenance
    if settings["resume"]:
        previous = torch.load(project_path(settings["resume"]), map_location="cpu",
                              weights_only=True)
        gain = previous["config"].get("pilot1", {}).get("decoder_gain")
        provenance["decoder_gain"] = gain
        if gain is None or previous["config"]["pilot1"] != provenance:
            raise ValueError("Recovery checkpoint belongs to a different pilot arm or baseline")
        del previous
    else:
        seed_all(config["training"]["seed"])
        probe = build_model(config["model"]).to(device)
        initialize_weights(probe, parent["model"], 1)
        gain = (fit_decoder_gain(probe, config, settings["calibration_examples"], device)
                if arm != "A" else 1.0)
        provenance["decoder_gain"] = gain
        initialize_weights(probe, parent["model"], gain)
        print(f"Pilot {arm}: decoder gain {gain:.6g}", flush=True)
        preflight(probe, loader_factory(config, "train"), config, device)
        del probe

    return run_training(config, loader_factory=loader_factory, extra_manifest_hashes=pool_hash,
                        initialize_model=partial(initialize_weights, weights=parent["model"],
                                                 gain=gain))


if __name__ == "__main__":
    print(f"Selected pilot checkpoint: {run_pilot(load_config(CONFIG))}", flush=True)
