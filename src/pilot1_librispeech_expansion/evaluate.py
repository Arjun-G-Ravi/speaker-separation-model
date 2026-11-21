"""Evaluate full recordings and select Pilot 1 candidates on validation only."""

import json
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch

from src.data import make_loader
from src.loss import pairwise_si_sdr
from src.settings import digest, load_config, load_model, project_path, select_device
from src.pilot1_librispeech_expansion.train import CONFIG


def mixture_project(estimate, mixture):
    sources = estimate.detach().cpu().numpy().astype(np.float64)
    mix = mixture.detach().cpu().numpy().astype(np.float64)
    gains, _, rank, _ = np.linalg.lstsq(sources.T, mix, rcond=1e-5)
    if rank < len(sources) or not np.isfinite(gains).all():
        return estimate
    fitted = sources * gains[:, None]
    corrected = fitted + (mix - fitted.sum(axis=0)) / len(sources)
    output = torch.from_numpy(corrected).to(estimate)
    return output if torch.isfinite(output).all() else estimate


def speaker_scores(estimate, sources, lengths):
    pairs = pairwise_si_sdr(estimate, sources, lengths)[0]
    direct, swapped = pairs.diag(), pairs.flip(1).diag()
    return (direct if direct.sum() >= swapped.sum() else swapped).tolist()


def summarize(rows, variant):
    scores = np.asarray([row[variant] for row in rows])
    worst = scores.min(axis=1)
    baseline = np.asarray([row["mixture_si_sdr"] for row in rows])
    return {"examples": len(rows), "si_sdr": float(scores.mean()),
            "si_sdri": float((scores.mean(axis=1) - baseline).mean()),
            "worst_speaker_p10": float(np.percentile(worst, 10)),
            "either_speaker_below_zero": float(np.mean(worst < 0))}


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, device_name, split, *, projection=True):
    if split not in ("validation", "test"):
        raise ValueError("Expected a held-out split")
    device = select_device(device_name)
    model, state = load_model(checkpoint, device)
    if model.sources != 2:
        raise ValueError("Pilot evaluation expects two sources")
    config = deepcopy(state["config"])
    config["training"]["device"] = device_name
    data = config["data"]
    manifest = project_path(data["root"]) / data[f"{split}_manifest"]
    if digest(manifest) != state["manifest_hashes"][split]:
        raise ValueError("Evaluation manifest differs from the checkpoint")
    rows = []
    for mixture, sources, lengths, names in make_loader(config, split):
        mixture, sources, lengths = mixture.to(device), sources.to(device), lengths.to(device)
        estimate = model(mixture)
        baseline = speaker_scores(mixture[:, None].expand_as(sources), sources, lengths)
        row = {"id": names[0], "mixture_si_sdr": sum(baseline) / 2,
               "raw": speaker_scores(estimate, sources, lengths)}
        if projection:
            corrected = mixture_project(estimate[0], mixture[0])
            row["projected"] = speaker_scores(corrected[None], sources, lengths)
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"{state['run_id']} {split}: {len(rows)} examples", flush=True)
    metrics = {"raw": summarize(rows, "raw")}
    if projection:
        metrics["projected"] = summarize(rows, "projected")
    return {"checkpoint": str(project_path(checkpoint)), "manifest_sha256": digest(manifest),
            "run_id": state["run_id"], "pilot": state["config"].get("pilot1"),
            "metrics": metrics, "examples": rows}


def compare(checkpoints, device, options):
    results = {}
    baseline_hash = digest(checkpoints["baseline"])
    for role in ("baseline", "A", "B", "C"):
        checkpoint = checkpoints[role]
        result = evaluate_checkpoint(checkpoint, device, "validation")
        if role != "baseline":
            pilot = result["pilot"] or {}
            if pilot.get("arm") != role or pilot.get("baseline_sha256") != baseline_hash:
                raise ValueError(f"Checkpoint does not belong to pilot arm {role}: {checkpoint}")
            if result["manifest_sha256"] != results["baseline"]["manifest_sha256"]:
                raise ValueError("Pilot and baseline use different validation recordings")
        results[role] = result
    minimum = options["min_si_sdr_gain_db"]
    tail_drop = options["max_worst_speaker_p10_drop_db"]
    baseline = results["baseline"]["metrics"]["raw"]
    eligible = []
    for role in ("A", "B", "C"):
        raw = results[role]["metrics"]["raw"]
        if (raw["si_sdr"] >= baseline["si_sdr"] + minimum
                and raw["worst_speaker_p10"] >= baseline["worst_speaker_p10"] - tail_drop):
            eligible.append(role)
    winner = max(eligible, key=lambda role: results[role]["metrics"]["raw"]["si_sdr"],
                 default=None)
    locked = results[winner or "baseline"]
    raw, projected = locked["metrics"]["raw"], locked["metrics"]["projected"]
    projection = (projected["si_sdr"] >= raw["si_sdr"] + options["min_projection_gain_db"]
                  and projected["worst_speaker_p10"] >= raw["worst_speaker_p10"] - tail_drop)
    selected_paths = {role: results[role]["checkpoint"] for role in ("baseline", winner) if role}
    return {"split": "validation", "winner": winner, "use_projection": projection,
            "checkpoint_sha256": {role: digest(path) for role, path in selected_paths.items()},
            "results": results}


def main():
    settings = load_config(CONFIG)
    options = settings["evaluation"]
    device = settings["training"]["device"]
    if options["selection_report"]:
        selection = project_path(options["selection_report"])
        report = json.loads(selection.read_text())
        if report.get("split") != "validation":
            raise ValueError("Choose a Pilot 1 validation selection report")
        output = selection.with_name("test.json")
        if output.exists():
            raise FileExistsError(f"Test evaluation already exists: {output}")
        checkpoints = {}
        for role, sha256 in report["checkpoint_sha256"].items():
            path = project_path(report["results"][role]["checkpoint"])
            if digest(path) != sha256:
                raise ValueError(f"Selected checkpoint changed: {path}")
            checkpoints[role] = path
        results = {role: evaluate_checkpoint(path, device, "test",
                                             projection=report["use_projection"])
                   for role, path in checkpoints.items()}
        if len({result["manifest_sha256"] for result in results.values()}) != 1:
            raise ValueError("Selected checkpoints use different test recordings")
        report = {"split": "test", "selection": str(selection), "winner": report["winner"],
                  "use_projection": report["use_projection"], "results": results}
    else:
        checkpoints = {"baseline": settings["baseline"], **options["checkpoints"]}
        if any(not checkpoints.get(role) for role in ("baseline", "A", "B", "C")):
            raise ValueError("Set checkpoint paths for pilot arms A, B and C")
        checkpoints = {role: project_path(path) for role, path in checkpoints.items()}
        report = compare(checkpoints, device, options)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output = project_path("logs") / f"pilot1_{stamp}" / "validation.json"
        output.parent.mkdir(parents=True, exist_ok=False)
    with output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    metrics = {role: result["metrics"] for role, result in report["results"].items()}
    print(json.dumps({"winner": report["winner"], "use_projection": report["use_projection"],
                      "report": str(output), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
