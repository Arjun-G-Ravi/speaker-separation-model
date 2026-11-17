"""Offline correctness checks and a tiny synthetic optimization experiment."""

import copy
import csv
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from scripts.prepare_data import StorageBudget, prepare
from src.audio import read_audio
from src.data import collate_audio, make_loader
from src.evaluate import evaluate
from src.inference import separate
from src.loss import pairwise_si_sdr, pit_loss, pit_objective
from src.model import build_model
from src.settings import ROOT, load_config, resolve_training_config, seed_all
from src.train import run_training, train_epoch


def make_fixture(root, settings, source_count=3):
    archives, metadata = {}, {}
    rate = settings["sample_rate"] * 2
    fields = ["mixture_ID", *[field for i in range(1, source_count + 1)
                              for field in (f"source_{i}_path", f"source_{i}_gain")]]
    for split_index, split in enumerate(("train", "validation", "test")):
        archive = root / f"{split}.tar.gz"
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        with tarfile.open(archive, "w:gz") as tar:
            for example in range(settings["examples"]):
                row, ids = {}, []
                for index in range(1, source_count + 1):
                    identifier = f"{split_index * 100 + example * 3 + index}-1-0000"
                    ids.append(identifier)
                    path = root / f"{identifier}.wav"
                    time = np.arange(round(rate * (settings["seconds"] + index * 0.02))) / rate
                    frequency = 140 + index * 173 + example * 19
                    audio = 0.12 * np.sin(2 * np.pi * frequency * time)
                    audio *= 0.7 + 0.3 * np.sin(2 * np.pi * (index + 1) * time)
                    wavfile.write(path, rate, np.round(audio * 32767).astype(np.int16))
                    flac = path.with_suffix(".flac")
                    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), str(flac)], check=True)
                    relative = f"{split}/{identifier}.flac"
                    tar.add(flac, arcname=f"LibriSpeech/{relative}")
                    row[f"source_{index}_path"] = relative
                    row[f"source_{index}_gain"] = 0.5 + index * 0.1
                row["mixture_ID"] = "_".join(ids)
                writer.writerow(row)
        archives[split] = {"source": split, "count": settings["examples"], "archive": str(archive)}
        metadata[split] = stream.getvalue().encode()
    return archives, metadata


def loss_checks(model_config):
    target = torch.randn(2, 3, 251)
    estimate = target + 0.2 * torch.randn_like(target)
    expected = pit_loss(estimate, target)
    for order in permutations(range(3)):
        assert torch.allclose(pit_loss(estimate[:, order], target), expected, atol=1e-5)
    assert torch.allclose(pit_loss(estimate * 2, target), expected, atol=1e-4)
    padded_target = torch.cat((target, torch.randn(2, 3, 83)), -1)
    padded_estimate = torch.cat((estimate, torch.randn(2, 3, 83)), -1)
    assert torch.allclose(pit_loss(padded_estimate, padded_target, torch.tensor([251, 251])),
                          expected, atol=1e-5)
    assert pit_loss(target[:, [2, 0, 1]], target) < -60
    assert pit_loss(torch.zeros_like(target), target) > 70
    combined = pit_loss(estimate, target, scale_weight=1.0)
    for order in permutations(range(3)):
        assert torch.allclose(pit_loss(estimate[:, order], target, scale_weight=1.0),
                              combined, atol=1e-5)
    assert torch.allclose(pit_loss(padded_estimate, padded_target, torch.tensor([251, 251]),
                                  scale_weight=1.0), combined, atol=1e-5)
    assert pit_loss(estimate * 100, target, scale_weight=1.0) > combined + 20
    assert pit_loss(-estimate, target, scale_weight=1.0) > combined
    # Check joint assignment against independently computed waveform errors.
    scores = pairwise_si_sdr(estimate, target)
    candidates = []
    for order in permutations(range(3)):
        reference = target[:, order]
        error = (estimate - reference).square().sum(-1) / reference.square().sum(-1)
        candidates.append(-scores[:, range(3), order].mean(-1)
                          + (10 * torch.log10(1 + error)).mean(-1))
    manual_joint = torch.stack(candidates, -1).min(-1).values.mean()
    torch.testing.assert_close(combined, manual_joint)
    gain = torch.tensor(1000.0, requires_grad=True)
    pit_loss(estimate * gain, target, scale_weight=1.0).backward()
    assert torch.isfinite(gain.grad) and gain.grad > 0
    gain = torch.tensor(0.1, requires_grad=True)
    pit_objective(target * gain, target)["scale_loss"].backward()
    assert torch.isfinite(gain.grad) and gain.grad < 0
    try:
        pit_loss(target, torch.zeros_like(target))
    except ValueError:
        pass
    else:
        raise AssertionError("Silent references were accepted")
    # Verify pairwise SI-SDR against explicit projection/residual computation.
    ref = target[0, 0] - target[0, 0].mean()
    est = estimate[0, 1] - estimate[0, 1].mean()
    projection = (est @ ref) * ref / ref.square().sum()
    manual = 10 * torch.log10(projection.square().sum() / (est - projection).square().sum())
    assert torch.allclose(pairwise_si_sdr(estimate, target)[0, 1, 0], manual, atol=1e-4)
    model = build_model(model_config)
    for length in (1, 15, 16, 257, 80000, 80003):
        with torch.inference_mode():
            output = model(torch.randn(1, length))
        assert output.shape == (1, 3, length) and torch.isfinite(output).all()
    loss = pit_loss(model(torch.randn(2, 251)), target)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    full = build_model(resolve_training_config(load_config("train.yaml"))["model"])
    full(torch.randn(1, 257)).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in full.parameters())


def accumulation_check(model_config):
    from torch.utils.data import DataLoader

    targets = torch.randn(5, 3, 251)
    mixtures = targets.sum(1)
    examples = [(mixtures[i], targets[i], 251, str(i)) for i in range(5)]
    loader = DataLoader(examples, batch_size=2, collate_fn=collate_audio)
    accumulated = build_model(model_config)
    reference = copy.deepcopy(accumulated)
    options = load_config("train.yaml")["training"]
    options.update(amp=False, accumulation_steps=2)
    optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    _, steps, _ = train_epoch(accumulated, loader, optimizer,
                           torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), options)
    assert steps == 2
    for start, stop in ((0, 4), (4, 5)):
        reference_optimizer.zero_grad(set_to_none=True)
        pit_loss(reference(mixtures[start:stop]), targets[start:stop],
                 scale_weight=options["scale_loss_weight"]).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), options["gradient_clip"])
        reference_optimizer.step()
    for actual, expected in zip(accumulated.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected, atol=1e-6)


def training_data_checks(config):
    first = list(make_loader(config, "train", epoch=0))
    repeated = list(make_loader(config, "train", epoch=0))
    next_epoch = list(make_loader(config, "train", epoch=1))
    workers = copy.deepcopy(config)
    workers["training"]["workers"] = 2
    parallel = list(make_loader(workers, "train", epoch=0))
    segment = round(config["data"]["segment_seconds"] * config["data"]["sample_rate"])
    manifest = json.loads((Path(config["data"]["root"]) / "train.json").read_text())
    allowed = {source for row in manifest["examples"] for source in row["id"].split("_")}
    for batch, repeat, worker_batch in zip(first, repeated, parallel):
        mixture, sources, lengths, names = batch
        assert mixture.shape[-1] == segment and (lengths == segment).all()
        torch.testing.assert_close(mixture, sources.sum(1), atol=0, rtol=0)
        assert max(mixture.abs().max(), sources.abs().max()) <= config["data"]["dynamic_mixing"]["peak_limit"] + 1e-6
        assert (sources.var(-1) > 1e-10).all()
        for name in names:
            utterances = [part.split("@")[0] for part in name.split("_")]
            assert set(utterances) <= allowed
            assert len({utterance.split("-")[0] for utterance in utterances}) == config["model"]["sources"]
        for other in (repeat, worker_batch):
            assert names == other[3]
            torch.testing.assert_close(sources, other[1], atol=0, rtol=0)
    assert first[0][3] != next_epoch[0][3]
    for split in ("validation", "test"):
        loader = make_loader(config, split)
        assert loader.dataset.mixing is None and loader.dataset.segment is None
    fixed = copy.deepcopy(config)
    fixed["data"]["dynamic_mixing"]["enabled"] = False
    fixed["data"]["speed_perturbation"] = {"enabled": False}
    assert all((batch[2] == segment).all() for batch in make_loader(fixed, "train"))
    loud = copy.deepcopy(config)
    loud["data"]["dynamic_mixing"]["source_rms"] = 5
    mixture, sources, _, _ = next(iter(make_loader(loud, "train")))
    assert max(mixture.abs().max(), sources.abs().max()) <= loud["data"]["dynamic_mixing"]["peak_limit"] + 1e-6
    fixed["data"]["segment_seconds"] = 100
    try:
        make_loader(fixed, "train")
    except ValueError:
        pass
    else:
        raise AssertionError("Fixed training crops allowed short inputs")
    dataset = make_loader(config, "train").dataset
    with patch("src.data.read_audio", return_value=(dataset.rate, np.zeros(segment, np.float32))):
        try:
            dataset[0]
        except ValueError as error:
            assert "active crop" in str(error)
        else:
            raise AssertionError("Dynamic mixing accepted silent crops")
    speed_perturbation_checks(config)


def speed_perturbation_checks(config):
    augmented = copy.deepcopy(config)
    augmented["data"]["speed_perturbation"] = {"enabled": True, "speeds": [95, 100, 105]}
    first = list(make_loader(augmented, "train", epoch=2))
    augmented["training"]["workers"] = 2
    parallel = list(make_loader(augmented, "train", epoch=2))
    for batch, other in zip(first, parallel):
        assert batch[3] == other[3]
        torch.testing.assert_close(batch[1], other[1], atol=0, rtol=0)
        torch.testing.assert_close(batch[0], batch[1].sum(1), atol=0, rtol=0)
        assert torch.isfinite(batch[1]).all()
        assert batch[1].abs().max() <= config["data"]["dynamic_mixing"]["peak_limit"] + 1e-6
    assert first[0][3] != next(iter(make_loader(augmented, "train", epoch=3)))[3]
    for split in ("validation", "test"):
        assert make_loader(augmented, split).dataset.speeds is None
        original = next(iter(make_loader(config, split)))
        actual = next(iter(make_loader(augmented, split)))
        torch.testing.assert_close(original[0], actual[0], atol=0, rtol=0)
        torch.testing.assert_close(original[1], actual[1], atol=0, rtol=0)

    dataset = make_loader(augmented, "train").dataset
    segment = dataset.segment
    frequency = 1000

    def tone(path, *, start, frames):
        audio = np.sin(2 * np.pi * frequency * np.arange(frames) / dataset.rate)
        return dataset.rate, audio.astype(np.float32)

    for speed in (95, 100, 105):
        dataset.speeds = [speed]
        with patch("src.data.read_audio", side_effect=tone):
            mixture, sources, length, _ = dataset[0]
        assert sources.shape == (dataset.sources, segment) and length == segment
        torch.testing.assert_close(mixture, sources.sum(0), atol=0, rtol=0)
        spectrum = np.abs(np.fft.rfft(sources[0].numpy()))
        observed = np.argmax(spectrum) * dataset.rate / segment
        assert abs(observed - frequency * speed / 100) <= dataset.rate / segment
    # An utterance exactly one crop long still works at the fastest speed.
    dataset.speeds = [105]
    dataset.speakers = {speaker: [(path, segment, utterance) for path, _, utterance in tracks]
                        for speaker, tracks in dataset.speakers.items()}
    with patch("src.data.read_audio", side_effect=tone):
        _, sources, _, _ = dataset[0]
    assert sources.shape[-1] == segment and (sources[:, -1] == 0).all()
    for speeds in ([], [0], [-95], [95.5], [True]):
        augmented["data"]["speed_perturbation"]["speeds"] = speeds
        try:
            make_loader(augmented, "train")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid speeds accepted: {speeds}")


def main():
    settings = load_config("smoke.yaml")
    torch.set_num_threads(2)
    seed_all(settings["seed"])
    loss_checks(settings["model"])
    accumulation_check(settings["model"])
    raw = yaml.safe_load((ROOT / "config" / "train.yaml").read_text())
    for name, weight, enabled in (("baseline_200", 1.0, False),
                                   ("si_sdr_only", 0.0, False),
                                   ("speed_perturbation", 0.0, True)):
        selected = copy.deepcopy(raw)
        selected["experiment"] = name
        with patch("src.settings.yaml.safe_load", return_value=selected):
            resolved = load_config("train.yaml")
        assert "experiments" not in resolved
        assert resolved["training"]["scale_loss_weight"] == weight
        assert resolved["data"]["speed_perturbation"]["enabled"] == enabled
        assert resolved["training"]["epochs"] == 200
    invalid = copy.deepcopy(raw)
    invalid["experiment"] = "unknown"
    with patch("src.settings.yaml.safe_load", return_value=invalid):
        try:
            load_config("train.yaml")
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown experiment was accepted")
    with tempfile.TemporaryDirectory(prefix="speech-smoke-") as folder:
        root = Path(folder)
        archives, metadata = make_fixture(root, settings)
        config = {"root": str(root / "prepared"), "seed": settings["seed"],
                  "revision": "synthetic-fixture", "sample_rate": settings["sample_rate"],
                  "budget_bytes": 10000000, "reserve_bytes": 1000, "splits": archives}
        report = prepare(config, metadata)
        assert report["peak_bytes"] < config["budget_bytes"]
        assert report["final_bytes"] == sum(p.stat().st_size for p in
                                             (root / "prepared").rglob("*") if p.is_file())
        assert report["peak_bytes"] >= report["final_bytes"]
        first = (root / "prepared" / "train.json").read_bytes()
        prepare(config, metadata)
        assert (root / "prepared" / "train.json").read_bytes() == first
        # Rebuild a partial example, preserving deterministic manifests.
        manifest = json.loads(first)
        source = root / "prepared" / manifest["examples"][0]["sources"][0]
        source.unlink()
        prepare(config, metadata)
        assert (root / "prepared" / "train.json").read_bytes() == first
        try:
            StorageBudget(root / "prepared", 100, 0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Storage ceiling was not enforced")
        current_size = sum(p.stat().st_size for p in (root / "prepared").rglob("*") if p.is_file())
        budget = StorageBudget(root / "prepared", current_size + 20, 0)
        try:
            budget.write_bytes(root / "prepared" / "too_big", b"x" * 21)
        except RuntimeError:
            assert not (root / "prepared" / "too_big").exists()
        else:
            raise AssertionError("An over-budget write was allowed")
        row = manifest["examples"][0]
        _, original = read_audio(root / (row["id"].split("_")[0] + ".flac"))
        from src.audio import resample
        _, prepared_source = read_audio(root / "prepared" / row["sources"][0])
        reference = resample(original * 0.6, settings["sample_rate"] * 2, settings["sample_rate"])
        assert np.max(np.abs(prepared_source - reference[:len(prepared_source)])) < 1 / 32768
        assert row["length"] == round((settings["seconds"] + 0.02) * settings["sample_rate"])

        sources = torch.stack([torch.from_numpy(read_audio(root / "prepared" / p)[1])
                               for p in row["sources"]])[None]
        mixture = sources.sum(1)
        model = build_model(settings["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        initial = pit_loss(model(mixture), sources).item()
        model.train()
        for _ in range(settings["overfit_steps"]):
            optimizer.zero_grad(set_to_none=True)
            loss = pit_loss(model(mixture), sources)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        final = pit_loss(model(mixture), sources).item()
        assert initial - final > settings["minimum_improvement_db"], (initial, final)
        print(f"Synthetic overfit: loss {initial:.3f} -> {final:.3f} dB", flush=True)

        training = load_config("train.yaml")
        training["model"] = settings["model"]
        training["data"].pop("dataset", None)
        training["data"].pop("roots", None)
        training["data"]["root"] = str(root / "prepared")
        training["data"]["segment_seconds"] = settings["seconds"]
        training["training"].update({"epochs": 2, "workers": 0, "device": "cpu",
                                     "batch_size": 2, "accumulation_steps": 2})
        training["logging"]["wandb_mode"] = "disabled"
        training_data_checks(training)
        training["data"]["speed_perturbation"] = {"enabled": True, "speeds": [95, 100, 105]}
        training["training"]["scale_loss_weight"] = 0.0
        worker_config = copy.deepcopy(training)
        worker_config["training"]["workers"] = 2
        assert sum(len(batch[0]) for batch in make_loader(worker_config, "train")) == settings["examples"]
        uninterrupted = run_training(copy.deepcopy(training))
        from src.train import atomic_save

        class SimulatedInterruption(Exception):
            pass

        recovery = None

        def interrupt_after_save(state, path):
            nonlocal recovery
            atomic_save(state, path)
            if path.name == "recovery.pt":
                recovery = path
                raise SimulatedInterruption()

        with patch("src.train.atomic_save", interrupt_after_save):
            try:
                run_training(copy.deepcopy(training))
            except SimulatedInterruption:
                pass
        assert recovery is not None and recovery.exists()
        training["checkpoint"]["resume"] = str(recovery)
        resumed = run_training(training)
        assert not (resumed.parent / "recovery.pt").exists()
        baseline = torch.load(uninterrupted, weights_only=True)
        restored = torch.load(resumed, weights_only=True)
        for name in baseline["model"]:
            assert torch.equal(baseline["model"][name], restored["model"][name]), name
        evaluate(resumed, "cpu", crop_seconds=settings["seconds"])
        paths = separate(resumed, root / "prepared" / row["mixture"], root / "separated", "cpu")
        assert len(paths) == 3
        for path in paths:
            rate, audio = read_audio(path)
            assert rate == settings["sample_rate"] and len(audio) == row["length"]
            assert np.isfinite(audio).all()
        shutil.copytree(root / "separated", resumed.parent / "synthetic_separated")
        shutil.copy2(root / "prepared" / row["mixture"], resumed.parent / "synthetic_input.wav")
        # Inference restores sample rate and architecture from the checkpoint.
        inference_input = root / "resample.wav"
        wavfile.write(inference_input, 16000, np.tile(original[:, None], (1, 2)))
        paths = separate(resumed, inference_input, root / "resampled_output", "cpu")
        assert len(read_audio(paths[0])[1]) == (len(original) + 1) // 2
        ten_seconds = np.resize(mixture[0].numpy(), 10 * settings["sample_rate"])
        ten_input = resumed.parent / "synthetic_10s_input.wav"
        wavfile.write(ten_input, settings["sample_rate"], ten_seconds)
        paths = separate(resumed, ten_input, resumed.parent / "synthetic_10s_separated", "cpu")
        assert all(len(read_audio(path)[1]) == 80000 for path in paths)

    cuda_result = "unavailable; CUDA/AMP checks skipped"
    if torch.cuda.is_available():
        cuda_model = build_model(resolve_training_config(load_config("train.yaml"))["model"]).cuda().train()
        target = torch.randn(1, cuda_model.sources, 80000, device="cuda")
        optimizer = torch.optim.Adam(cuda_model.parameters(), lr=settings["learning_rate"])
        scaler = torch.amp.GradScaler("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = cuda_model(target.sum(1))
        loss = pit_loss(output, target,
                        scale_weight=load_config("train.yaml")["training"]["scale_loss_weight"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cuda_model.parameters())
        scaler.step(optimizer)
        scaler.update()
        cuda_result = "10-second standard-model AMP forward/backward/optimizer step passed"
    summary = {"synthetic_initial_loss": initial, "synthetic_final_loss": final,
               "cpu_checks": "passed", "cuda_checks": cuda_result,
               "training_pipeline_checks": "fixed crops, train-only remixing, joint scale loss, worker reproducibility, recovery passed",
               "real_dataset_downloaded": False, "real_speech_metrics": None}
    destination = ROOT / "logs" / "smoke_result.json"
    destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
