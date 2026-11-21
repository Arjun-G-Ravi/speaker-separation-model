"""Prepare the pilot's training-only source pool from a local archive."""

import json
import re
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scipy.io import wavfile

from scripts.download_dataset import checksum
from scripts.prepare_data import StorageBudget
from src.audio import read_audio, resample, write_pcm16
from src.settings import load_config, project_path


def prepare_pool(archive, manifest, expected_md5, budget_bytes, reserve_bytes,
                 sample_rate, min_samples):
    import fcntl

    archive, manifest = project_path(archive), project_path(manifest)
    if not archive.is_file():
        raise FileNotFoundError(f"Cached training archive is missing: {archive}")
    if checksum(archive) != expected_md5:
        raise ValueError(f"Training archive checksum differs: {archive}")
    root = manifest.parent
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if (previous["archive_md5"], previous["sample_rate"], previous["min_samples"]) != (
                    expected_md5, sample_rate, min_samples):
                raise ValueError("Source pool configuration changed; choose a new directory")
            if all((root / row["path"]).is_file() for row in previous["sources"]):
                print(f"Reusing {len(previous['sources'])} prepared sources: {manifest}", flush=True)
                return manifest
        budget = StorageBudget(root, budget_bytes, reserve_bytes)
        rows = {}
        with TemporaryDirectory(dir=root) as temporary, tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or not re.fullmatch(r"\d+-\d+-\d+\.flac", name):
                    continue
                parts = Path(member.name).parts
                if parts[:2] != ("LibriSpeech", "train-clean-100"):
                    raise ValueError(f"Unexpected speech split in training archive: {member.name}")
                utterance = Path(name).stem
                if utterance in rows:
                    raise ValueError(f"Duplicate training utterance: {utterance}")
                speaker = utterance.split("-")[0]
                relative = Path("sources") / speaker / f"{utterance}.wav"
                destination = root / relative
                if destination.is_file():
                    rate, signal = wavfile.read(destination, mmap=True)
                    if rate != sample_rate or signal.ndim != 1 or len(signal) < min_samples:
                        raise ValueError(f"Invalid prepared source: {destination}")
                else:
                    staged_flac = Path(temporary) / name
                    budget.check(member.size)
                    with tar.extractfile(member) as source, staged_flac.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    rate, signal = read_audio(staged_flac)
                    staged_flac.unlink()
                    if signal.ndim != 1:
                        raise ValueError(f"Expected mono speech: {member.name}")
                    signal = resample(signal, rate, sample_rate)
                    if len(signal) < min_samples:
                        continue
                    budget.check(len(signal) * 2 + 44)
                    staged_wav = destination.with_suffix(".partial")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_pcm16(staged_wav, sample_rate, signal)
                    staged_wav.replace(destination)
                    budget.used += destination.stat().st_size
                rows[utterance] = {"id": utterance, "speaker": speaker,
                                   "path": relative.as_posix(), "samples": len(signal)}
                if len(rows) % 1000 == 0:
                    print(f"Prepared {len(rows)} training sources", flush=True)
        if not rows:
            raise ValueError("Training archive contained no eligible sources")
        record = {"archive_md5": expected_md5, "sample_rate": sample_rate,
                  "min_samples": min_samples, "sources": [rows[key] for key in sorted(rows)]}
        budget.write_bytes(manifest, json.dumps(record, indent=2).encode())
    print(f"Prepared {len(rows)} training sources: {manifest}", flush=True)
    return manifest


def main():
    settings = load_config("pilot1_librispeech_expansion.yaml")
    md5 = load_config("download_libri2mix.yaml")["splits"]["train"]["md5"]
    prepare_pool(**settings["source_pool"], expected_md5=md5, sample_rate=8000, min_samples=24000)


if __name__ == "__main__":
    main()
