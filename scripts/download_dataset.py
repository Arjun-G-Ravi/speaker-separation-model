"""Download resumable LibriSpeech archives, then prepare the 5,000-mixture dataset."""

import hashlib
import http.client
import logging
import re
import shutil
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing

from scripts.prepare_data import ResumableHTTPStream, StorageBudget, prepare
from src.settings import load_config, project_path

LOGGER = logging.getLogger(__name__)


def checksum(path):
    LOGGER.info("Verifying MD5: %s", path.name)
    result = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def probe(url, timeout):
    began = time.monotonic()
    request = urllib.request.Request(url, headers={
        "Range": "bytes=0-262143", "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)",
                             response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise ValueError("Mirror does not support byte-range downloads")
        content = response.read(262144)
        if len(content) != min(262144, int(match[2])):
            raise IOError("Truncated mirror probe")
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
    return len(content) / max(time.monotonic() - began, 1e-6), int(match[2]), validator


def select_mirror(filename, options):
    candidates = []
    with ThreadPoolExecutor(max_workers=len(options["mirrors"])) as pool:
        futures = {pool.submit(probe, f"{base.rstrip('/')}/{filename}",
                               options["timeout_seconds"]): base for base in options["mirrors"]}
        for future in futures:
            base = futures[future]
            try:
                speed, total, validator = future.result()
            except (OSError, ValueError, http.client.HTTPException) as error:
                LOGGER.warning("Mirror unavailable: %s (%s)", base, error)
                continue
            LOGGER.info("Mirror probe: %s, %.2f MB/s", base, speed / 1e6)
            candidates.append((speed, f"{base.rstrip('/')}/{filename}", total, validator))
    if not candidates:
        raise RuntimeError("No mirror supports downloads; check the network or download.mirrors")
    _, url, total, validator = max(candidates, key=lambda item: item[0])
    LOGGER.info("Selected mirror: %s (%.2f GB)", url, total / 1e9)
    return url, total, validator


def download_archive(destination, expected_md5, options, budget):
    if destination.exists():
        if checksum(destination) != expected_md5:
            raise ValueError(f"Checksum mismatch: {destination}; move this file aside and rerun")
        LOGGER.info("Reusing verified archive: %s", destination)
        return
    url, total, validator = select_mirror(destination.name, options)
    size = options["chunk_bytes"]
    chunks = destination.parent / f"{destination.name}.{expected_md5}.{size}.chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + size, total) - 1) for start in range(0, total, size)]
    completed = {start for start, end in ranges
                 if (chunks / f"{start}.chunk").exists()
                 and (chunks / f"{start}.chunk").stat().st_size == end - start + 1}
    reused = sum(end - start + 1 for start, end in ranges if start in completed)
    # Assembly temporarily coexists with the chunks, so account for both copies.
    budget.used = sum(p.stat().st_size for p in budget.root.rglob("*") if p.is_file())
    required = total - reused + total
    budget.check(required)
    if shutil.disk_usage(destination.parent).free < required + budget.reserve:
        raise RuntimeError(f"Need {required / 1e9:.2f} GB free for download and assembly")
    LOGGER.info("%s: resuming %.2f/%.2f GB; %d connections, %.0f MiB chunks",
                destination.name, reused / 1e9, total / 1e9, options["workers"], size / 2**20)
    transferred = 0
    lock = threading.Lock()
    stopped = threading.Event()

    def fetch(start, end):
        nonlocal transferred
        partial = chunks / f"{start}.partial"
        with closing(ResumableHTTPStream(url, options["timeout_seconds"], options["retries"],
                                         options["retry_delay_seconds"], stopped)) as stream:
            stream.offset, stream.end, stream.validator = start, end, validator
            with partial.open("wb") as target:
                while content := stream.read(min(1024 * 1024, end + 1 - stream.offset)):
                    target.write(content)
                    with lock:
                        transferred += len(content)
            if partial.stat().st_size != end - start + 1:
                raise IOError(f"Truncated range {start}-{end}")
        partial.replace(chunks / f"{start}.chunk")

    pending_ranges = iter((start, end) for start, end in ranges if start not in completed)
    began = last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=options["workers"]) as pool:
        pending = set()

        def schedule():
            item = next(pending_ranges, None)
            if item is not None:
                pending.add(pool.submit(fetch, *item))

        for _ in range(options["workers"]):
            schedule()
        try:
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    schedule()
                now = time.monotonic()
                if now - last_report >= options["progress_seconds"] or not pending:
                    with lock:
                        received = transferred
                    speed = received / max(now - began, 1e-6)
                    current = reused + received
                    eta = f"{(total - current) / speed / 60:.1f} min" if speed else "waiting"
                    LOGGER.info("%s: %5.1f%% | %.3f/%.3f GB | %.2f MB/s | ETA %s",
                                destination.name, 100 * current / total, current / 1e9,
                                total / 1e9, speed / 1e6, eta)
                    last_report = now
        except BaseException:
            stopped.set()
            for future in pending:
                future.cancel()
            LOGGER.warning("Download interrupted; completed chunks are saved. Rerun to resume.")
            raise
    LOGGER.info("Assembling %s from verified-length chunks", destination.name)
    assembled = destination.with_suffix(destination.suffix + ".assembling")
    with assembled.open("wb") as target:
        for start, _ in ranges:
            with (chunks / f"{start}.chunk").open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    if checksum(assembled) != expected_md5:
        raise ValueError(f"Archive checksum mismatch; move aside {chunks} and {assembled}, then rerun")
    assembled.replace(destination)
    for start, _ in ranges:
        (chunks / f"{start}.chunk").unlink()
    for partial in chunks.glob("*.partial"):
        partial.unlink()
    chunks.rmdir()
    LOGGER.info("Download verified: %s", destination)


def download_dataset(config):
    options = config["download"]
    if (not options["mirrors"] or options["workers"] < 1 or options["chunk_bytes"] < 1
            or options["timeout_seconds"] <= 0 or options["retries"] < 0
            or options["retry_delay_seconds"] < 0 or options["progress_seconds"] <= 0):
        raise ValueError("Invalid download settings; check config/download.yaml")
    root = project_path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    # Prevent two processes from modifying the same download/preparation files.
    import fcntl

    with (root / ".download.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another downloader is using {root}") from error
        archives = root / ".archives"
        archives.mkdir(exist_ok=True)
        budget = StorageBudget(root, config["budget_bytes"], config["reserve_bytes"])
        LOGGER.info("Dataset destination: %s", root)
        LOGGER.info("Mixtures: %s", ", ".join(f"{k}={v['count']}" for k, v in config["splits"].items()))
        prepared_config = {key: value for key, value in config.items() if key != "download"}
        prepared_config["splits"] = {}
        for split, spec in config["splits"].items():
            destination = archives / f"{spec['source']}.tar.gz"
            download_archive(destination, spec["md5"], options, budget)
            prepared_config["splits"][split] = {**spec, "archive": str(destination)}
        LOGGER.info("All archives verified. Preparing selected sources and mixtures.")
        report = prepare(prepared_config)
        LOGGER.info("Dataset ready: %s (%.2f GB including cached archives)",
                    root, report["final_bytes"] / 1e9)
        LOGGER.info("For training, set data.root: %s in config/train.yaml", config["root"])
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        download_dataset(load_config("download.yaml"))
    except KeyboardInterrupt:
        LOGGER.warning("Stopped. Run the same command to resume.")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("Dataset download/preparation failed. Existing files are preserved.")
        raise SystemExit(1)
