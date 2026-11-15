"""Offline regression checks for playback overflow from scale-invariant training."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import wavfile

from src.inference import playback_audio, separate
from src.loss import pit_si_sdr


def main():
    time = np.arange(8000, dtype=np.float32) / 8000
    target = np.stack([0.1 * np.sin(2 * np.pi * frequency * time)
                       for frequency in (173, 337, 691)])
    raw = target * 5000 + np.random.default_rng(42).normal(0, 5, target.shape).astype(np.float32)
    before = raw.copy()
    safe, gain = playback_audio(raw)
    assert np.array_equal(raw, before)
    assert np.max(np.abs(safe)) <= 0.950001
    np.testing.assert_allclose(safe, raw * gain)
    a = pit_si_sdr(torch.from_numpy(raw)[None], torch.from_numpy(target)[None])[0]
    b = pit_si_sdr(torch.from_numpy(safe)[None], torch.from_numpy(target)[None])[0]
    torch.testing.assert_close(a, b, atol=1e-3, rtol=0)
    for signal in (target, np.zeros_like(target)):
        unchanged, unity = playback_audio(signal)
        assert unity == 1 and np.array_equal(signal, unchanged)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        state = {"config": {"data": {"sample_rate": 8000}}}
        with patch("src.inference.load_model", return_value=(None, state)), \
             patch("src.inference.read_audio", return_value=(8000, target.sum(0))), \
             patch("src.inference.separate_waveform", return_value=(target.sum(0), raw)):
            paths = separate("fixture.pt", "fixture.wav", root, "cpu")
        for index, path in enumerate(paths):
            rate, signal = wavfile.read(path)
            assert rate == 8000 and np.isfinite(signal).all()
            np.testing.assert_array_equal(signal, safe[index])
    print("Audio export checks passed: shared gain, no overflow, unchanged metrics, silence, WAVs")


if __name__ == "__main__":
    main()
