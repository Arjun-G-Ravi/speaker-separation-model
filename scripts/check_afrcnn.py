"""Offline A-FRCNN checks for two/three sources and the shared training pipeline."""

import copy
import json
import tempfile
from pathlib import Path

import torch

from scripts.prepare_data import prepare
from scripts.smoke_check import make_fixture, training_data_checks
from src.audio import read_audio
from src.data import make_loader
from src.evaluate import evaluate
from src.inference import separate
from src.loss import pit_loss
from src.model import build_model
from src.settings import load_config, load_model, resolve_training_config, seed_all
from src.train import run_training


def main(model_key="afrcnn"):
    settings = load_config("smoke.yaml")
    torch.set_num_threads(2)
    seed_all(settings["seed"])
    source_config = load_config("train.yaml")
    assert "sources" not in source_config["model"] and "root" not in source_config["data"]
    for count in (2, 3):
        config = copy.deepcopy(source_config)
        dataset = f"libri{count}mix"
        config["data"]["dataset"] = dataset
        resolved = resolve_training_config(config)
        assert resolved["model"]["sources"] == count
        assert resolved["data"]["root"] == config["data"]["roots"][dataset]
        assert "sources" not in config["model"]
        model = build_model({**settings[model_key], "sources": count})
        for length in (1, 15, 21, 159, 160, 257, 24000, 80003):
            with torch.inference_mode():
                output = model(torch.randn(2, length))
            assert output.shape == (2, count, length) and torch.isfinite(output).all()
        parameter_count = sum(p.numel() for p in model.parameters())
        model.iterations = 1
        assert sum(p.numel() for p in model.parameters()) == parameter_count
        model.iterations = settings[model_key]["iterations"]
        target = torch.randn(2, count, 257)
        pit_loss(model(target.sum(1)), target, scale_weight=1).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

        with tempfile.TemporaryDirectory(prefix=f"afrcnn-{count}-") as folder:
            root = Path(folder)
            archives, metadata = make_fixture(root, settings, source_count=count)
            prep = {"root": str(root / "prepared"), "sources": count,
                    "seed": settings["seed"], "revision": "synthetic-fixture",
                    "sample_rate": settings["sample_rate"], "budget_bytes": 10000000,
                    "reserve_bytes": 1000, "splits": archives}
            prepare(prep, metadata)
            before = (root / "prepared" / "train.json").read_bytes()
            prepare(prep, metadata)
            assert (root / "prepared" / "train.json").read_bytes() == before
            manifest = json.loads(before)
            assert manifest["dataset"] == f"Libri{count}Mix"
            assert all(len(row["sources"]) == count for row in manifest["examples"])
            config["model"] = copy.deepcopy(settings[model_key])
            config["data"]["roots"][dataset] = prep["root"]
            config["data"]["segment_seconds"] = settings["seconds"]
            config["training"].update(epochs=2, workers=0, device="cpu", batch_size=2)
            config["logging"]["wandb_mode"] = "disabled"
            training_data_checks(resolve_training_config(config))
            checkpoint = run_training(config)
            restored, state = load_model(checkpoint, torch.device("cpu"))
            assert restored.sources == count
            assert state["config"]["data"]["dataset"] == dataset
            assert state["config"]["model"]["sources"] == count
            assert "roots" not in state["config"]["data"]
            batch = next(iter(make_loader(state["config"], "validation")))
            with torch.inference_mode():
                expected = restored(batch[0])
            duplicate, _ = load_model(checkpoint, torch.device("cpu"))
            with torch.inference_mode():
                torch.testing.assert_close(duplicate(batch[0]), expected, atol=0, rtol=0)
            evaluate(checkpoint, "cpu", crop_seconds=settings["seconds"])
            row = manifest["examples"][0]
            paths = separate(checkpoint, root / "prepared" / row["mixture"], root / "outputs", "cpu")
            assert len(paths) == count
            assert all(len(read_audio(path)[1]) == row["length"] for path in paths)
            # A different dataset/source count cannot resume the previous optimizer run.
            incompatible = copy.deepcopy(config)
            incompatible["model"]["sources"] = 5 - count
            try:
                resolve_training_config(incompatible)
            except ValueError:
                pass
            else:
                raise AssertionError("Dataset/model source-count mismatch was accepted")
        print(f"{model_key}: {count}-source shapes, gradients, preparation, mixing, "
              "training, checkpoint reload, evaluation, and inference passed", flush=True)


if __name__ == "__main__":
    main()
