"""Continue a saved midpoint run with the same recipe and a longer epoch budget."""

from functools import partial

import torch

from src.pilot1_midpoint_supervision.train import EXPERIMENT, midpoint_step
from src.settings import load_config, project_path
from src.train import run_training


def continue_training(settings):
    path = project_path(settings["resume"])
    previous = torch.load(path, map_location="cpu", weights_only=True)
    config = previous["config"]
    if config.get("pilot1", {}).get("experiment") != EXPERIMENT:
        raise ValueError("Expected a midpoint-supervision training checkpoint")
    if type(settings["total_epochs"]) is not int or settings["total_epochs"] <= previous["next_epoch"]:
        raise ValueError("total_epochs must exceed the number of completed epochs")
    config["training"]["epochs"] = settings["total_epochs"]
    config["checkpoint"] = {"resume": str(path)}
    del previous
    step = partial(midpoint_step, **config["pilot1"]["midpoint"])
    return run_training(config, training_step=step)


if __name__ == "__main__":
    selected = continue_training(load_config("pilot1_midpoint_continuation.yaml"))
    print(f"Selected midpoint checkpoint: {selected}", flush=True)
