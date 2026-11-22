"""Compare source speed perturbation with the archived cross-speaker control."""

from src.pilot1_coarse_to_fine.evaluate import summarize
from src.pilot1_cross_speaker.evaluate import run_comparison
from src.pilot1_speed_perturbation.train import CONFIG, EXPERIMENT


if __name__ == "__main__":
    run_comparison(CONFIG, EXPERIMENT, "cross_speaker", summarize=summarize)
