"""Compare the speaker-exchange pilot with Pilot B on full validation recordings."""

import json
from datetime import datetime
from statistics import mean, median

from src.pilot1_cross_speaker.train import CONFIG, EXPERIMENT
from src.pilot1_librispeech_expansion.evaluate import evaluate_checkpoint
from src.settings import digest, load_config, project_path


def training_runtime(run_id):
    path = project_path("logs") / run_id / "metrics.jsonl"
    if not path.is_file():
        return None
    with path.open() as file:
        epochs = [row for row in map(json.loads, file) if "elapsed_seconds" in row]
    if not epochs:
        return None
    step_seconds = [after["elapsed_seconds"] / (after["step"] - before["step"])
                    for before, after in zip(epochs, epochs[1:])]
    return {"peak_allocated_gib": max(row["gpu_peak_bytes"] for row in epochs) / 2**30,
            "median_epoch_seconds": median(row["elapsed_seconds"] for row in epochs),
            "median_seconds_per_step_including_validation": (
                median(step_seconds) if step_seconds else None),
            "recorded_epoch_hours": sum(row["elapsed_seconds"] for row in epochs) / 3600}


def compare(control, candidate, options, *, experiment=EXPERIMENT, control_label="B"):
    reference_rows, candidate_rows = control["examples"], candidate["examples"]
    if (control["manifest_sha256"] != candidate["manifest_sha256"]
            or [row["id"] for row in reference_rows] != [row["id"] for row in candidate_rows]):
        raise ValueError("Control and candidate must use the same validation recordings")
    gains = {variant: {
        metric: candidate["metrics"][variant][metric] - control["metrics"][variant][metric]
        for metric in ("si_sdri", "worst_speaker_p10")
    } for variant in ("raw", "projected")}
    before, after = control["metrics"]["raw"], candidate["metrics"]["raw"]
    required_si_sdri = before["si_sdri"] + options["min_si_sdr_gain_db"]
    required_tail = before["worst_speaker_p10"] - options["max_worst_speaker_p10_drop_db"]
    advance = after["si_sdri"] >= required_si_sdri and after["worst_speaker_p10"] >= required_tail
    paired = [{"id": before["id"],
               "raw_gain_db": mean(after["raw"]) - mean(before["raw"]),
               "projected_gain_db": mean(after["projected"]) - mean(before["projected"])}
              for before, after in zip(reference_rows, candidate_rows)]
    return {"experiment": experiment, "split": "validation", "advance": advance,
            "criteria": {key: options[key] for key in ("min_si_sdr_gain_db",
                                                        "max_worst_speaker_p10_drop_db")},
            "delta_db": gains, "paired": paired,
            "results": {control_label: control, "candidate": candidate}}


def run_comparison(config_name, experiment, control_label, *, summarize=None):
    settings = load_config(config_name)
    options = settings["evaluation"]
    if not options["checkpoint"]:
        raise ValueError(f"Set evaluation.checkpoint in config/{config_name} to the pilot's best.pt")
    device = options.get("device", settings.get("training", {}).get("device", "auto"))
    control_path = project_path(settings["control"])
    candidate = evaluate_checkpoint(options["checkpoint"], device, "validation")
    provenance = candidate["pilot"] or {}
    if (provenance.get("experiment") != experiment
            or provenance.get("control_sha256") != digest(control_path)):
        raise ValueError(f"Candidate must belong to {experiment} and its configured control")
    control = evaluate_checkpoint(control_path, device, "validation")
    report = compare(control, candidate, options, experiment=experiment, control_label=control_label)
    summary = summarize(report, options) if summarize else {}
    report.update(summary)
    runtime = {role: training_runtime(result["run_id"])
               for role, result in report["results"].items()}
    report["training_runtime"] = runtime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = project_path("logs") / f"{experiment}_{stamp}" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    with output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps({"advance": report["advance"], "delta_db": report["delta_db"],
                      "training_runtime": runtime, **summary, "report": str(output)}, indent=2), flush=True)
    return output


def main():
    run_comparison(CONFIG, EXPERIMENT, "B")


if __name__ == "__main__":
    main()
