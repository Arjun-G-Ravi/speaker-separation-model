"""Train identity-initialized coarse-to-fine gates against the archived control."""

from functools import partial

import torch

from src.data import make_loader
from src.model import build_model
from src.pilot1_cross_speaker.train import EXPERIMENT as CONTROL
from src.pilot1_cross_speaker.train import initialize_pilot, matched_control_config
from src.settings import digest, load_config, project_path, seed_all, select_device
from src.train import run_training, separation_step

EXPERIMENT = "pilot1_coarse_to_fine"
CONFIG = f"{EXPERIMENT}.yaml"


def starting_model(config):
    pilot = config["pilot1"]
    baseline = project_path(pilot["baseline"])
    if digest(baseline) != pilot["baseline_sha256"]:
        raise ValueError("Baseline differs from the control's starting checkpoint")
    weights = torch.load(baseline, map_location="cpu", weights_only=True)["model"]
    seed_all(config["training"]["seed"])
    model = build_model({**config["model"], "name": CONTROL})
    initialize_pilot(model, weights, pilot["decoder_gain"])
    return model


def initialize_gated(model, weights):
    missing, unexpected = model.load_state_dict(weights, strict=False)
    gates = {f"block.gates.{key}" for key in model.block.gates.state_dict()}
    if set(missing) != gates or unexpected:
        raise ValueError("Only the coarse-to-fine gates may be absent from the control weights")


def preflight(config, reference, device):
    options = config["training"]
    mixture, sources, lengths, _ = next(iter(make_loader(config, "train")))
    mixture, sources, lengths = mixture.to(device), sources.to(device), lengths.to(device)
    reference.to(device).eval()
    with torch.inference_mode():
        expected = reference(mixture[:1])
    reference.cpu()
    model = build_model(config["model"]).to(device).eval()
    initialize_gated(model, reference.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(model(mixture[:1]), expected, rtol=1e-5, atol=1e-6)
    model.train()
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    _, metrics = separation_step(model, mixture, sources, lengths, options, cuda and options["amp"])
    loss = metrics["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite preflight loss")
    loss.backward()
    for index, gate in enumerate(model.block.gates):
        for parameter in gate.parameters():
            grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
                raise FloatingPointError(f"Invalid gradient in gate {index}")
    peak = f", peak {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB" if cuda else ""
    print(f"Gate preflight passed on {device} with batch size {len(mixture)}{peak}", flush=True)


def run_pilot(settings):
    config = matched_control_config(settings, EXPERIMENT)
    config["model"]["name"] = EXPERIMENT
    initializer = None
    if settings["resume"]:
        previous = torch.load(project_path(settings["resume"]), map_location="cpu", weights_only=True)
        if previous["config"].get("pilot1") != config["pilot1"]:
            raise ValueError("Recovery checkpoint belongs to a different pilot or control")
        del previous
    else:
        device = select_device(config["training"]["device"])
        reference = starting_model(config)
        preflight(config, reference, device)
        initializer = partial(initialize_gated, weights=reference.state_dict())
    return run_training(config, initialize_model=initializer)


if __name__ == "__main__":
    print(f"Selected pilot checkpoint: {run_pilot(load_config(CONFIG))}", flush=True)
