import json

import torch

from src.data import make_loader
from src.loss import pit_si_sdr
from src.settings import load_config, load_model, project_path, select_device


@torch.inference_mode()
def evaluate(checkpoint, device_name, split="test", crop_seconds=10):
    if split not in ("test", "validation"):
        raise ValueError("Evaluation split must be test or validation")
    device = select_device(device_name)
    model, state = load_model(checkpoint, device)
    config = state["config"]
    config["training"]["device"] = device_name
    from src.settings import digest
    manifest = project_path(config["data"]["root"]) / config["data"][f"{split}_manifest"]
    if digest(manifest) != state["manifest_hashes"][split]:
        raise ValueError("Evaluation manifest differs from training dataset identity")
    log = project_path("logs") / state["run_id"]
    log.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for label, crop in (("full", None), ("crop", crop_seconds)):
        records = []
        for mixture, sources, lengths, names in make_loader(config, split, crop_seconds=crop):
            mixture, sources, lengths = mixture.to(device), sources.to(device), lengths.to(device)
            score, _ = pit_si_sdr(model(mixture), sources, lengths)
            baseline, _ = pit_si_sdr(mixture[:, None].expand_as(sources), sources, lengths)
            records.append({"id": names[0], "si_sdr": score.item(),
                            "mixture_si_sdr": baseline.item(),
                            "si_sdri": (score - baseline).item(), "samples": int(lengths[0])})
        name = f"{split}_{label}"
        (log / f"{name}_examples.json").write_text(json.dumps(records, indent=2, allow_nan=False))
        summaries[name] = {"examples": len(records),
                           "si_sdr": sum(r["si_sdr"] for r in records) / len(records),
                           "si_sdri": sum(r["si_sdri"] for r in records) / len(records),
                           "crop_seconds": crop}
    (log / f"{split}_metrics.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    settings = load_config("evaluate.yaml")
    evaluate(settings["checkpoint"], settings["device"], settings["split"], settings["crop_seconds"])
