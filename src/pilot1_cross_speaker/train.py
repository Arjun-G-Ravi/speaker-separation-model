"""Fine-tune cross-speaker refinement from the calibrated Pilot B starting point."""

import math
import time
from copy import deepcopy
from functools import partial

import torch

from src.data import make_loader
from src.loss import pit_objective
from src.model import build_model
from src.pilot1_librispeech_expansion.train import initialize_weights
from src.settings import digest, load_config, project_path, seed_all, select_device
from src.train import run_training

CONFIG = "pilot1_cross_speaker.yaml"
EXPERIMENT = "pilot1_cross_speaker"


def matched_control_config(settings, experiment):
    """Inherit the archived cross-speaker recipe and record its identity as control."""
    path = project_path(settings["control"])
    control = torch.load(path, map_location="cpu", weights_only=True)
    config = control["config"]
    pilot = config.get("pilot1", {})
    if pilot.get("experiment") != EXPERIMENT or config["model"]["name"] != EXPERIMENT:
        raise ValueError("Control must be a cross-speaker pilot checkpoint")
    gain = pilot["decoder_gain"]
    if not math.isfinite(gain) or gain == 0:
        raise ValueError("Control must record a finite, nonzero decoder gain")
    data = config["data"]
    for split in ("train", "validation", "test"):
        manifest = project_path(data["root"]) / data[f"{split}_manifest"]
        if digest(manifest) != control["manifest_hashes"][split]:
            raise ValueError(f"{split} manifest differs from the cross-speaker control")
    config["experiment"] = experiment
    config["checkpoint"] = {"resume": settings["resume"]}
    config["pilot1"] = {**pilot, "experiment": experiment, "control": str(path),
                        "control_sha256": digest(path)}
    return config


def load_experiment(settings):
    baseline, control = (project_path(settings[key]) for key in ("baseline", "control"))
    parent = torch.load(baseline, map_location="cpu", weights_only=True)
    reference = torch.load(control, map_location="cpu", weights_only=True)
    baseline_hash = digest(baseline)
    reference_config = reference["config"]
    pilot = reference_config.get("pilot1", {})
    if (pilot.get("arm") != "B" or pilot.get("baseline_sha256") != baseline_hash
            or pilot.get("source_pool") is not None
            or reference_config["model"] != parent["config"]["model"]
            or reference["manifest_hashes"] != parent["manifest_hashes"]):
        raise ValueError("Control must be Pilot B trained from the configured baseline")
    gain = pilot.get("decoder_gain")
    if gain is None or not math.isfinite(gain) or gain == 0:
        raise ValueError("Pilot B checkpoint must contain a finite, nonzero decoder gain")

    config = deepcopy(reference_config)
    data = config["data"]
    for split in ("train", "validation", "test"):
        manifest = project_path(data["root"]) / data[f"{split}_manifest"]
        if digest(manifest) != reference["manifest_hashes"][split]:
            raise ValueError(f"{split} manifest differs from Pilot B")

    config["experiment"] = EXPERIMENT
    config["model"].update(settings["model"], name=EXPERIMENT)
    config["training"].update(settings["training"])
    config["checkpoint"] = {"resume": settings["resume"]}
    config["pilot1"] = {"experiment": EXPERIMENT, "baseline": str(baseline),
                        "baseline_sha256": baseline_hash, "control": str(control),
                        "control_sha256": digest(control), "decoder_gain": gain}
    return config, reference_config["model"], parent["model"]


@torch.no_grad()
def initialize_pilot(model, weights, gain):
    missing, unexpected = model.load_state_dict(weights, strict=False)
    expected = {f"exchange.{key}" for key in model.exchange.state_dict()}
    if set(missing) != expected or unexpected:
        raise ValueError(f"Incompatible baseline weights. Missing {missing}, "
                         f"unexpected {unexpected}")
    model.decoder.weight.mul_(gain)


def preflight(config, reference_model, weights, device):
    training = config["training"]
    seed_all(training["seed"])
    mixture, sources, lengths, _ = next(iter(make_loader(config, "train")))
    mixture, sources, lengths = mixture.to(device), sources.to(device), lengths.to(device)
    gain = config["pilot1"]["decoder_gain"]
    reference = build_model(reference_model).to(device).eval()
    initialize_weights(reference, weights, gain)
    with torch.inference_mode():
        expected = reference(mixture[:1])
    del reference

    model = build_model(config["model"]).to(device).eval()
    initialize_pilot(model, weights, gain)
    with torch.inference_mode():
        torch.testing.assert_close(model(mixture[:1]), expected, rtol=1e-5, atol=1e-6)

    model.train()
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=cuda and training["amp"]):
        estimate = model(mixture)
    loss = pit_objective(estimate, sources, lengths, training["scale_loss_weight"])["loss"]
    loss.backward()
    grad = model.exchange[-1].weight.grad
    if (not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all()
            or not torch.count_nonzero(grad)):
        raise FloatingPointError("Preflight requires finite, nonzero exchange projection gradients")
    elapsed = time.perf_counter() - started
    peak = (f", {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB peak allocated"
            if cuda else "")
    print(f"Preflight passed on {device} with {len(mixture)} examples. "
          f"Forward/backward took {elapsed:.2f} s{peak}.", flush=True)


def run_pilot(settings):
    device = select_device(settings["training"]["device"])
    config, reference_model, weights = load_experiment(settings)
    if settings["resume"]:
        previous = torch.load(project_path(settings["resume"]), map_location="cpu",
                              weights_only=True)
        if previous["config"].get("pilot1") != config["pilot1"]:
            raise ValueError("Recovery checkpoint belongs to a different experiment or control")
        del previous
    else:
        preflight(config, reference_model, weights, device)
    initializer = partial(initialize_pilot, weights=weights, gain=config["pilot1"]["decoder_gain"])
    return run_training(config, initialize_model=initializer)


if __name__ == "__main__":
    print(f"Selected pilot checkpoint: {run_pilot(load_config(CONFIG))}", flush=True)
