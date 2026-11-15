import json
import math
import platform
import time
from importlib.metadata import version

import torch
import yaml

from src.data import make_loader
from src.loss import pit_loss, pit_si_sdr
from src.model import build_model
from src.settings import (append_json, atomic_save, create_run, digest, load_config,
                          project_path, restore_rng, rng_state, seed_all, select_device)


@torch.inference_mode()
def validate(model, loader, device, log_every_steps=None):
    model.eval()
    total, count = 0.0, 0
    for index, (mixture, sources, lengths, _) in enumerate(loader):
        mixture, sources, lengths = mixture.to(device), sources.to(device), lengths.to(device)
        scores, _ = pit_si_sdr(model(mixture), sources, lengths)
        total += scores.sum().item()
        count += len(scores)
        if log_every_steps and ((index + 1) % log_every_steps == 0 or index + 1 == len(loader)):
            print(f"Validation: {count}/{len(loader.dataset)} examples, "
                  f"SI-SDR {total / count:.3f} dB", flush=True)
    return total / count


def train_epoch(model, loader, optimizer, scaler, device, options, step=0, log_step=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    accumulation = options["accumulation_steps"]
    amp = options["amp"] and device.type == "cuda"
    for index, (mixture, sources, lengths, _) in enumerate(loader):
        mixture, sources, lengths = mixture.to(device), sources.to(device), lengths.to(device)
        group_start = index // accumulation * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        # Weight the final smaller batch by examples, not by microbatch count.
        group_examples = min(group_size * loader.batch_size,
                             len(loader.dataset) - group_start * loader.batch_size)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            estimate = model(mixture)
        loss = pit_loss(estimate, sources, lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss * len(mixture) / group_examples).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"])
            if not torch.isfinite(norm):
                if not amp:
                    raise FloatingPointError("Non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % options["log_every_steps"] == 0:
                print(f"Training: batch {index + 1}/{len(loader)}, step {step}, "
                      f"loss {loss.item():.3f}", flush=True)
                if log_step:
                    log_step({"step": step, "train/batch_loss": loss.item(),
                              "train/gradient_norm": float(norm) if torch.isfinite(norm) else None})
        total += loss.item() * len(mixture)
        count += len(mixture)
    return total / count, step


def run_training(config):
    options = config["training"]
    for key in ("batch_size", "accumulation_steps", "epochs", "log_every_steps",
                "early_stopping_patience"):
        if options[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if options["workers"] < 0 or options["learning_rate"] <= 0:
        raise ValueError("workers must be nonnegative and learning_rate positive")
    seed_all(options["seed"])
    device = select_device(options["device"])
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"],
                                 weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=options["scheduler_patience"],
        factor=options["scheduler_factor"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and options["amp"])
    hashes = {split: digest(project_path(config["data"]["root"]) /
                            config["data"][f"{split}_manifest"])
              for split in ("train", "validation", "test")}
    start, step, best, stale = 0, 0, -math.inf, 0
    previous = None
    if config["checkpoint"]["resume"]:
        previous = torch.load(project_path(config["checkpoint"]["resume"]),
                              map_location="cpu", weights_only=True)
        if previous["manifest_hashes"] != hashes:
            raise ValueError("Resume dataset manifests differ from the checkpoint")
        for section in ("model", "data", "training"):
            before, after = dict(previous["config"][section]), dict(config[section])
            if section == "training":
                for key in ("epochs", "device", "workers", "log_every_steps"):
                    before.pop(key)
                    after.pop(key)
            if before != after:
                raise ValueError(f"Resume changes incompatible {section} settings")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        if scaler.is_enabled() and previous["scaler"]:
            scaler.load_state_dict(previous["scaler"])
        start, step, best, stale = (previous[k] for k in ("next_epoch", "step", "best", "stale"))
        restore_rng(previous["rng"])
    run_id, artifact, log = create_run()
    print(f"Run {run_id}: device={device}, AMP={scaler.is_enabled()}", flush=True)
    print(f"Artifacts: {artifact}\nLogs: {log}", flush=True)
    (log / "train.yaml").write_text(yaml.safe_dump(config))
    metadata = {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": version("numpy"), "scipy": version("scipy"), "yaml": version("PyYAML"),
                "cuda_runtime": torch.version.cuda, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "amp": scaler.is_enabled(), "manifest_hashes": hashes,
                "parameters": sum(p.numel() for p in model.parameters())}
    (log / "environment.json").write_text(json.dumps(metadata, indent=2))
    if previous:
        # Carry the selected model forward even when resuming from a worse recovery epoch.
        selected = project_path(config["checkpoint"]["resume"]).parent / "best.pt"
        if not selected.exists():
            raise FileNotFoundError("Resume requires best.pt beside the recovery checkpoint")
        selected_state = torch.load(selected, map_location="cpu", weights_only=True)
        selected_state["run_id"] = run_id
        selected_state["config"] = config
        atomic_save(selected_state, artifact / "best.pt")
    wandb_run = None
    mode = config["logging"]["wandb_mode"]
    if mode not in ("disabled", "offline", "online"):
        raise ValueError("wandb_mode must be disabled, offline, or online")
    if mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install wandb in your environment, or set wandb_mode: disabled") from error
        wandb_run = wandb.init(project=config["logging"]["wandb_project"],
                               entity=config["logging"]["wandb_entity"], name=run_id,
                               config=config, mode=mode, dir=str(log))

    def record(metrics):
        append_json(log / "metrics.jsonl", metrics)
        if wandb_run:
            wandb_run.log(metrics)

    try:
        for epoch in range(start, options["epochs"]):
            began = time.monotonic()
            print(f"Epoch {epoch + 1}/{options['epochs']}", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            loss, step = train_epoch(model, make_loader(config, "train", epoch), optimizer,
                                     scaler, device, options, step, record)
            score = validate(model, make_loader(config, "validation"), device,
                             options["log_every_steps"])
            if not math.isfinite(score):
                raise FloatingPointError("Non-finite validation SI-SDR")
            scheduler.step(score)
            improved = score > best
            best, stale = (score, 0) if improved else (best, stale + 1)
            metrics = {"epoch": epoch + 1, "step": step, "train/loss": loss,
                       "validation/si_sdr": score, "validation/loss": -score,
                       "learning_rate": optimizer.param_groups[0]["lr"],
                       "elapsed_seconds": time.monotonic() - began,
                       "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)
                       if device.type == "cuda" else 0}
            record(metrics)
            print(json.dumps(metrics), flush=True)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "config": config, "next_epoch": epoch + 1, "step": step,
                     "best": best, "stale": stale, "rng": rng_state(),
                     "manifest_hashes": hashes, "run_id": run_id}
            if improved:
                atomic_save(state, artifact / "best.pt")
            atomic_save(state, artifact / "recovery.pt")
            if stale >= options["early_stopping_patience"]:
                break
        if not (artifact / "best.pt").exists():
            raise ValueError("No epochs ran and no selected checkpoint was available")
        (artifact / "recovery.pt").unlink(missing_ok=True)
        (log / "result.json").write_text(json.dumps({"best_validation_si_sdr": best,
                                                     "checkpoint": str(artifact / "best.pt")}))
    finally:
        if wandb_run:
            wandb_run.finish()
    return artifact / "best.pt"


if __name__ == "__main__":
    run_training(load_config("train.yaml"))
