"""Compare the continued midpoint model with its archived validation scores."""

import json
from datetime import datetime

import numpy as np

from src.pilot1_librispeech_expansion.evaluate import evaluate_checkpoint
from src.settings import load_config, project_path


def compare(reference, candidate):
    before, after = reference["examples"], candidate["examples"]
    if (reference["manifest_sha256"] != candidate["manifest_sha256"]
            or [row["id"] for row in before] != [row["id"] for row in after]):
        raise ValueError("Expected the same ordered validation recordings")
    rng = np.random.default_rng(42)
    results = {}
    for variant in ("raw", "projected"):
        gains = np.array([np.mean(b[variant]) - np.mean(a[variant])
                          for a, b in zip(before, after)])
        means = rng.choice(gains, (10000, len(gains))).mean(axis=1)
        results[variant] = {
            "reference": reference["metrics"][variant],
            "candidate": candidate["metrics"][variant],
            "mean_gain_db": float(gains.mean()),
            "wins": int((gains > 0).sum()),
            "bootstrap_95pct_db": np.quantile(means, [0.025, 0.975]).tolist(),
        }
    return results


def main():
    settings = load_config("pilot1_midpoint_continuation.yaml")
    options = settings["evaluation"]
    archived = json.loads(project_path(options["reference_report"]).read_text())
    if archived["split"] != "validation":
        raise ValueError("Expected the original midpoint validation report")
    reference = archived["results"]["candidate"]
    candidate = evaluate_checkpoint(options["checkpoint"], options["device"], "validation")
    results = compare(reference, candidate)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = project_path("logs") / f"pilot1_midpoint_continuation_{stamp}" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    with output.open("x") as file:
        json.dump({"split": "validation", "summary": results,
                   "reference": reference, "candidate": candidate}, file, indent=2, allow_nan=False)
    print(json.dumps({"summary": results, "report": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
