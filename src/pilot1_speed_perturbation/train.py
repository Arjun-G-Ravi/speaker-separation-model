"""Matched cross-speaker pilot with independent source speed perturbation."""

from functools import partial

import torch

from src.pilot1_cross_speaker.train import initialize_pilot, matched_control_config
from src.settings import digest, load_config, project_path
from src.train import run_training

EXPERIMENT = "pilot1_speed_perturbation"
CONFIG = f"{EXPERIMENT}.yaml"


def run_pilot(settings):
    config = matched_control_config(settings, EXPERIMENT)
    config["data"]["speed_perturbation"] = settings["speed_perturbation"]
    pilot = config["pilot1"]
    initializer = None
    if settings["resume"]:
        previous = torch.load(project_path(settings["resume"]), map_location="cpu", weights_only=True)
        if previous["config"].get("pilot1") != pilot:
            raise ValueError("Recovery checkpoint belongs to another experiment or control")
        del previous
    else:
        baseline = project_path(pilot["baseline"])
        if digest(baseline) != pilot["baseline_sha256"]:
            raise ValueError("Baseline differs from the cross-speaker starting checkpoint")
        weights = torch.load(baseline, map_location="cpu", weights_only=True)["model"]
        initializer = partial(initialize_pilot, weights=weights, gain=pilot["decoder_gain"])
    return run_training(config, initialize_model=initializer)


if __name__ == "__main__":
    print(f"Selected pilot checkpoint: {run_pilot(load_config(CONFIG))}", flush=True)
