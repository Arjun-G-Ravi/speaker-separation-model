import numpy as np
import torch
from scipy.io import wavfile

from src.audio import read_audio, resample
from src.settings import load_config, load_model, project_path, select_device


@torch.inference_mode()
def separate_waveform(model, audio, source_rate, target_rate, device):
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = resample(audio, source_rate, target_rate)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Input must contain finite nonempty audio")
    estimate = model(torch.from_numpy(audio.copy())[None].to(device))[0].float().cpu().numpy()
    if not np.isfinite(estimate).all():
        raise FloatingPointError("Model produced nonfinite audio")
    return audio, estimate


@torch.inference_mode()
def separate(checkpoint, input_path, output, device_name="auto"):
    if not input_path:
        raise ValueError("Set input in config/inference.yaml")
    device = select_device(device_name)
    model, state = load_model(checkpoint, device)
    rate, audio = read_audio(project_path(input_path))
    target_rate = state["config"]["data"]["sample_rate"]
    _, estimate = separate_waveform(model, audio, rate, target_rate, device)
    destination = project_path(output)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / f"speaker_{index + 1}.wav" for index in range(len(estimate))]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output tracks already exist; choose a new output directory")
    for path, track in zip(paths, estimate):
        wavfile.write(path, target_rate, track.astype(np.float32))
    print(f"Saved {len(paths)} tracks at {target_rate} Hz to {destination}")
    return paths


if __name__ == "__main__":
    settings = load_config("inference.yaml")
    separate(settings["checkpoint"], settings["input"], settings["output"], settings["device"])
