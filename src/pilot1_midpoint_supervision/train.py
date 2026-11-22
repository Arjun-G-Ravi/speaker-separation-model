"""Train the cross-speaker separator with a supervised intermediate prediction."""

import math
from functools import partial

import torch

from src.loss import pairwise_si_sdr, pit_objective
from src.pilot1_cross_speaker.train import initialize_pilot, matched_control_config
from src.settings import digest, load_config, project_path
from src.train import run_training

EXPERIMENT = "pilot1_midpoint_supervision"
CONFIG = f"{EXPERIMENT}.yaml"


def midpoint_step(model, mixture, sources, lengths, options, amp, *, speaker_iteration,
                  loss_weight):
    with torch.autocast(device_type=mixture.device.type, dtype=torch.float16, enabled=amp):
        estimate, midpoint = model(mixture, tap_iteration=speaker_iteration)
    metrics = pit_objective(estimate, sources, lengths, options["scale_loss_weight"],
                            return_assignment=True)
    assignment = metrics.pop("assignment")
    # The final output decides which reference each speaker stream should predict.
    scores = pairwise_si_sdr(midpoint, sources, lengths)
    midpoint_si_sdr = scores.gather(2, assignment.unsqueeze(-1)).mean()
    metrics["midpoint_si_sdr"] = midpoint_si_sdr
    metrics["loss"] = metrics["loss"] - loss_weight * midpoint_si_sdr
    return estimate, metrics


def load_experiment(settings):
    config = matched_control_config(settings, EXPERIMENT)
    midpoint = settings["midpoint"]
    tap = midpoint["speaker_iteration"]
    speaker_steps = config["model"]["iterations"] - config["model"]["split_iteration"]
    if type(tap) is not int or not 1 <= tap < speaker_steps:
        raise ValueError("midpoint.speaker_iteration must precede the final speaker step")
    if not 0 < midpoint["loss_weight"] < math.inf:
        raise ValueError("midpoint.loss_weight must be finite and positive")

    config["pilot1"]["midpoint"] = midpoint
    return config


def run_pilot(settings):
    config = load_experiment(settings)
    pilot = config["pilot1"]
    initializer = None
    if settings["resume"]:
        previous = torch.load(project_path(settings["resume"]), map_location="cpu",
                              weights_only=True)
        if previous["config"].get("pilot1") != pilot:
            raise ValueError("Recovery checkpoint belongs to a different midpoint experiment")
    else:
        baseline = project_path(pilot["baseline"])
        if digest(baseline) != pilot["baseline_sha256"]:
            raise ValueError("Baseline differs from the cross-speaker pilot's starting checkpoint")
        weights = torch.load(baseline, map_location="cpu", weights_only=True)["model"]
        initializer = partial(initialize_pilot, weights=weights, gain=pilot["decoder_gain"])
    return run_training(config, initialize_model=initializer,
                        training_step=partial(midpoint_step, **pilot["midpoint"]))


if __name__ == "__main__":
    print(f"Selected pilot checkpoint: {run_pilot(load_config(CONFIG))}", flush=True)
