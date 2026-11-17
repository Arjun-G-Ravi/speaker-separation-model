import hashlib
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(name):
    with (ROOT / "config" / name).open() as handle:
        return yaml.safe_load(handle)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_training_config(config):
    """Resolve the selected dataset root and output count without modifying source YAML."""
    config = deepcopy(config)
    data = config["data"]
    if "dataset" in data:
        counts = {"libri2mix": 2, "libri3mix": 3}
        if data["dataset"] not in counts:
            raise ValueError("data.dataset must be libri2mix or libri3mix")
        if "roots" in data:
            data["root"] = data.pop("roots")[data["dataset"]]
        sources = counts[data["dataset"]]
        if config["model"].get("sources", sources) != sources:
            raise ValueError("model.sources disagrees with data.dataset; omit it for automatic selection")
        config["model"]["sources"] = sources
    return config


def select_device(value):
    if value not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available())
                        else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def create_run():
    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    artifact = ROOT / "out" / run_id
    log = ROOT / "logs" / run_id
    artifact.mkdir(parents=True, exist_ok=False)
    log.mkdir(parents=True, exist_ok=False)
    return run_id, artifact, log


def load_model(path, device):
    if not path:
        raise ValueError("Set checkpoint in the corresponding YAML configuration")
    from src.model import build_model

    state = torch.load(project_path(path), map_location="cpu", weights_only=True)
    model = build_model(state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def append_json(path, record):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
