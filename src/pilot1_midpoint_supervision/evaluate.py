"""Compare midpoint supervision with the cross-speaker control on validation."""

from src.pilot1_cross_speaker.evaluate import run_comparison
from src.pilot1_midpoint_supervision.train import CONFIG, EXPERIMENT


if __name__ == "__main__":
    run_comparison(CONFIG, EXPERIMENT, "cross_speaker")
