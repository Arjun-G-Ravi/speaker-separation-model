"""Stream only clean source archives; never store an archive on disk."""

import csv
import hashlib
import io
import http.client
import json
import random
import re
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np

from src.audio import read_audio, resample, write_pcm16
from src.settings import digest, load_config, project_path


class StorageBudget:
    def __init__(self, root, limit, reserve):
        self.root, self.limit, self.reserve = Path(root), limit, reserve
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.peak = self.used
        self.check(0)

    def check(self, extra):
        if self.used + extra + self.reserve >= self.limit:
            raise RuntimeError(
                f"Preparation storage budget exceeded: {self.used} used + {extra} requested; "
                f"limit {self.limit}, reserve {self.reserve}. Existing data is preserved."
            )
        self.peak = max(self.peak, self.used + extra)

    def write_bytes(self, path, content):
        path = Path(path)
        self.check(len(content))
        old = path.stat().st_size if path.exists() else 0
        temporary = path.with_suffix(path.suffix + ".partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(content)
        temporary.replace(path)
        self.used += len(content) - old

    def remove(self, path):
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            self.used -= size


def output_paths(split, row):
    return [Path(split) / kind / (row["mixture_ID"] + ".wav")
            for kind in ("mix_clean", "s1", "s2", "s3")]


def validate_example(root, paths, rate):
    signals = []
    for path in paths:
        actual_rate, audio = read_audio(root / path)
        if actual_rate != rate or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid prepared audio: {path}")
        signals.append(audio)
    if len({len(x) for x in signals}) != 1:
        raise ValueError("Prepared signals have different lengths")
    if any(np.var(x) <= 1e-10 for x in signals[1:]):
        raise ValueError("Prepared source is silent")
    if np.max(np.abs(signals[0] - sum(signals[1:]))) > 4 / 32768:
        raise ValueError("Prepared mixture does not equal its sources within PCM16 tolerance")
    return len(signals[0])


def generate_example(root, split, row, cache, rate, budget):
    sources = []
    for index in range(1, 4):
        original_rate, audio = read_audio(cache / Path(row[f"source_{index}_path"]).name)
        if original_rate != 16000 or audio.ndim != 1:
            raise ValueError("Expected mono 16 kHz LibriSpeech sources")
        sources.append(resample(audio * float(row[f"source_{index}_gain"]), 16000, rate))
    length = min(map(len, sources))
    sources = [source[:length] for source in sources]
    mixture = sum(sources)
    if any(not np.isfinite(x).all() or np.max(np.abs(x)) > 1 for x in [mixture, *sources]):
        raise ValueError("Nonfinite or clipping audio; refusing to alter official source gains")
    paths = output_paths(split, row)
    budget.check(4 * (44 + length * 2))
    for path, audio in zip(paths, [mixture, *sources]):
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        old = full.stat().st_size if full.exists() else 0
        write_pcm16(full, rate, audio)
        budget.used += full.stat().st_size - old
    validate_example(root, paths, rate)


class ResumableHTTPStream:
    """Keep the gzip decoder alive while reconnecting at compressed byte offsets."""

    def __init__(self, url, timeout_seconds, retries, retry_delay_seconds):
        self.url, self.timeout = url, timeout_seconds
        self.retries, self.delay = retries, retry_delay_seconds
        self.offset, self.total, self.validator = 0, None, None
        self.response = None

    def close(self):
        if self.response is not None:
            self.response.close()
            self.response = None

    def connect(self):
        headers = {"Accept-Encoding": "identity"}
        if self.offset:
            headers["Range"] = f"bytes={self.offset}-"
            if self.validator:
                headers["If-Range"] = self.validator
        self.response = urllib.request.urlopen(
            urllib.request.Request(self.url, headers=headers), timeout=self.timeout
        )
        validator = self.response.headers.get("ETag") or self.response.headers.get("Last-Modified")
        if self.validator and validator != self.validator:
            self.close()
            raise ValueError("Source archive changed during download; refusing to combine versions")
        self.validator = validator
        if self.response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 self.response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != self.offset:
                self.close()
                raise ValueError("Server returned an invalid download resume range")
            self.total = int(match[3])
        elif self.response.status == 200:
            length = self.response.headers.get("Content-Length")
            self.total = int(length) if length is not None else None
            # Some mirrors ignore Range. Replay the prefix without storing it.
            remaining = self.offset
            if remaining:
                print(f"Server ignored Range; replaying {remaining / 1e6:.1f} MB prefix", flush=True)
            while remaining:
                chunk = self.response.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise EOFError("Archive ended while replaying the downloaded prefix")
                remaining -= len(chunk)
        else:
            self.close()
            raise ValueError("Unexpected archive HTTP response")

    def read(self, size):
        for attempt in range(self.retries + 1):
            try:
                if self.response is None:
                    self.connect()
                chunk = self.response.read(size)
                if not chunk and self.total is not None and self.offset < self.total:
                    raise EOFError("Source archive connection ended early")
                self.offset += len(chunk)
                return chunk
            except (OSError, EOFError, http.client.HTTPException) as error:
                self.close()
                if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429) and error.code < 500:
                    raise
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Download failed after {self.retries + 1} attempts at compressed byte "
                        f"{self.offset}. Completed data is preserved; rerun preparation to continue."
                    ) from error
                delay = min(self.delay * 2**attempt, 30)
                print(f"Download interrupted ({error}); reconnecting at {self.offset / 1e6:.1f} MB "
                      f"in {delay}s, retry {attempt + 1}/{self.retries}", flush=True)
                time.sleep(delay)


def stream_sources(archive, wanted, cache, budget, timeout_seconds=120, retries=6,
                   retry_delay_seconds=3):
    """Yield stored source names; tar members outside wanted never touch disk."""
    if timeout_seconds <= 0 or retries < 0 or retry_delay_seconds < 0:
        raise ValueError("Download timeout must be positive; retries and delay must be nonnegative")
    connection = (closing(ResumableHTTPStream(archive, timeout_seconds, retries, retry_delay_seconds))
                  if str(archive).startswith("http") else open(archive, "rb"))
    print(f"Reading source archive: {archive}", flush=True)
    with connection as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            full = cache / name
            if not full.exists():
                budget.check(member.size)
                partial = full.with_suffix(full.suffix + ".partial")
                with tar.extractfile(member) as source, partial.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                if partial.stat().st_size != member.size:
                    raise IOError(f"Truncated archive source: {name}")
                partial.replace(full)
                budget.used += member.size
            yield name


def prepare(config, metadata=None):
    if config["sample_rate"] != 8000:
        raise ValueError("This experiment requires sample_rate: 8000")
    root = project_path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(root, config["budget_bytes"], config["reserve_bytes"])
    # Partial files are owned by this preparer and never represent completed examples.
    for path in root.rglob("*.partial"):
        budget.remove(path)
    identities = {}
    for split, spec in config["splits"].items():
        if metadata is None:
            url = (f"https://raw.githubusercontent.com/JorisCos/LibriMix/{config['revision']}/"
                   f"metadata/Libri3Mix/libri3mix_{spec['source']}.csv")
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
        else:
            raw = metadata[split]
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        if not 0 < spec["count"] <= len(rows):
            raise ValueError(f"Invalid requested count for {split}")
        selected = random.Random(config["seed"]).sample(rows, spec["count"])
        if len({row["mixture_ID"] for row in selected}) != len(selected):
            raise ValueError("Duplicate mixture IDs in the selected metadata")
        for row in selected:
            if len({Path(row[f"source_{i}_path"]).stem.split("-")[0] for i in (1, 2, 3)}) != 3:
                raise ValueError("A mixture must contain three distinct speakers")
        identity = {"revision": config["revision"], "source": spec["source"],
                    "seed": config["seed"], "sample_rate": config["sample_rate"],
                    "metadata_sha256": hashlib.sha256(raw).hexdigest(), "rows": selected}
        selection_path = root / f"{split}_selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != identity:
            raise ValueError("Preparation configuration changed; use a new data root")
        budget.write_bytes(selection_path, json.dumps(identity, indent=2).encode())
        speaker_ids = {Path(row[f"source_{i}_path"]).stem.split("-")[0]
                       for row in selected for i in (1, 2, 3)}
        if any(speaker_ids & previous for previous in identities.values()):
            raise ValueError("Speaker leakage across dataset splits")
        identities[split] = speaker_ids
        cache = root / ".sources" / split
        cache.mkdir(parents=True, exist_ok=True)
        pending = {}
        for row in selected:
            paths = output_paths(split, row)
            complete = all((root / path).exists() for path in paths)
            if complete:
                try:
                    validate_example(root, paths, config["sample_rate"])
                except (ValueError, OSError):
                    complete = False
            if not complete:
                for path in paths:
                    budget.remove(root / path)
                pending[row["mixture_ID"]] = row
        references = Counter()
        dependencies = defaultdict(set)
        for key, row in pending.items():
            for i in (1, 2, 3):
                name = Path(row[f"source_{i}_path"]).name
                references[name] += 1
                dependencies[name].add(key)
        for path in cache.iterdir():
            if path.name not in references:
                budget.remove(path)

        def consume(name):
            for key in list(dependencies[name]):
                if key not in pending:
                    continue
                row = pending[key]
                names = [Path(row[f"source_{i}_path"]).name for i in (1, 2, 3)]
                if not all((cache / source).exists() for source in names):
                    continue
                generate_example(root, split, row, cache, config["sample_rate"], budget)
                del pending[key]
                for source in names:
                    references[source] -= 1
                    if references[source] == 0:
                        budget.remove(cache / source)
                if len(pending) % 100 == 0:
                    print(f"{split}: {len(pending)} remaining; {budget.used / 1e9:.3f} GB", flush=True)

        for path in list(cache.iterdir()):
            consume(path.name)
        if pending:
            wanted = {name for name, count in references.items() if count and not (cache / name).exists()}
            for name in stream_sources(spec["archive"], wanted, cache, budget,
                                       **config.get("network", {})):
                consume(name)
                if not pending:
                    break
        if pending:
            raise ValueError(f"Archive missing sources for {len(pending)} {split} examples")
        examples = []
        for row in selected:
            paths = output_paths(split, row)
            length = validate_example(root, paths, config["sample_rate"])
            examples.append({"id": row["mixture_ID"], "mixture": str(paths[0]),
                             "sources": [str(path) for path in paths[1:]], "length": length})
        manifest = {"dataset": "Libri3Mix", "variant": "8k/min/mix_clean",
                    "selection_sha256": digest(selection_path), "sample_rate": config["sample_rate"],
                    "examples": examples}
        budget.write_bytes(root / f"{split}.json", json.dumps(manifest, indent=2).encode())
    report = {"peak_bytes": budget.peak, "final_bytes": budget.used,
              "budget_bytes": budget.limit,
              "manifest_hashes": {split: digest(root / f"{split}.json") for split in identities}}
    report_path = root / "preparation.json"
    old_size = report_path.stat().st_size if report_path.exists() else 0
    # Include the report itself and its atomic replacement in storage accounting.
    while True:
        payload = json.dumps(report, indent=2).encode()
        final = budget.used - old_size + len(payload)
        peak = max(budget.peak, budget.used + len(payload))
        if report["final_bytes"] == final and report["peak_bytes"] == peak:
            break
        report.update(final_bytes=final, peak_bytes=peak)
    budget.write_bytes(report_path, payload)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    prepare(load_config("prepare.yaml"))
