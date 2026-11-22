"""Evaluate the gate pilot and decide whether it merits a second seed."""

import json

import numpy as np

from src.pilot1_coarse_to_fine.train import CONFIG, EXPERIMENT
from src.pilot1_cross_speaker.evaluate import run_comparison
from src.pilot1_midpoint_supervision.train import EXPERIMENT as MIDPOINT
from src.settings import project_path


def summarize(report, options):
    archived = json.loads(project_path(options["midpoint_report"]).read_text())
    if archived["experiment"] != MIDPOINT or archived["split"] != "validation":
        raise ValueError("Expected the midpoint supervision validation report")
    midpoint = archived["results"]["candidate"]
    candidate = report["results"]["candidate"]
    if (midpoint["manifest_sha256"] != candidate["manifest_sha256"]
            or midpoint["pilot"]["control_sha256"] != candidate["pilot"]["control_sha256"]
            or [row["id"] for row in midpoint["examples"]] != [row["id"] for row in report["paired"]]):
        raise ValueError("Midpoint comparison requires the same control and validation recordings")
    projected = candidate["metrics"]["projected"]["si_sdri"]
    previous_best = midpoint["metrics"]["projected"]["si_sdri"]
    paired = {}
    for variant in ("raw", "projected"):
        gains = np.asarray([row[f"{variant}_gain_db"] for row in report["paired"]])
        rng = np.random.default_rng(42)
        means = rng.choice(gains, (10000, len(gains))).mean(axis=1)
        paired[variant] = {"wins": int((gains > 0).sum()), "mixes": len(gains),
                           "mean_gain_db": float(gains.mean()),
                           "bootstrap_95pct_db": np.quantile(means, [0.025, 0.975]).tolist()}
    return {"paired_vs_cross_speaker": paired, "projected_si_sdri_db": projected,
            "projected_vs_midpoint_db": projected - previous_best,
            "confirm_with_second_seed": report["advance"] and projected > previous_best}


if __name__ == "__main__":
    run_comparison(CONFIG, EXPERIMENT, "cross_speaker", summarize=summarize)
