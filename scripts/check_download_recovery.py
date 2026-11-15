"""Offline regression checks for interrupted source archive transfers."""

import hashlib
import io
import random
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.download_dataset import download_archive
from scripts.prepare_data import StorageBudget, stream_sources


class Response(io.BytesIO):
    def __init__(self, content, status, headers, fail=False):
        super().__init__(content)
        self.status, self.headers, self.fail = status, headers, fail

    def read(self, size=-1):
        if self.fail and self.tell() >= 10240:
            raise TimeoutError("simulated stalled socket")
        return super().read(size)


def archive_fixture():
    files = {f"source_{i}.flac": random.Random(i).randbytes(50000) for i in range(3)}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"LibriSpeech/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue(), files


def check_recovery(range_supported):
    content, files = archive_fixture()
    offsets = []

    def urlopen(request, timeout):
        header = request.get_header("Range", "bytes=0-") if hasattr(request, "get_header") else "bytes=0-"
        offset = int(header.split("=")[1].split("-")[0])
        offsets.append(offset)
        actual = offset if range_supported else 0
        headers = {"Content-Length": str(len(content) - actual), "ETag": '"fixture"'}
        if actual:
            headers["Content-Range"] = f"bytes {actual}-{len(content) - 1}/{len(content)}"
        return Response(content[actual:], 206 if actual else 200, headers, fail=len(offsets) == 1)

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        budget = StorageBudget(root, 1000000, 0)
        with patch("scripts.prepare_data.urllib.request.urlopen", urlopen), \
             patch("time.sleep"):
            names = list(stream_sources("https://fixture/archive.tar.gz", set(files), root, budget))
        assert set(names) == set(files)
        assert len(offsets) == 2 and offsets[1] > 0, offsets
        assert not list(root.glob("*.partial"))
        for name, expected in files.items():
            assert (root / name).read_bytes() == expected
        assert budget.used == sum(map(len, files.values()))


def check_exhaustion():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        budget = StorageBudget(root, 1000000, 0)
        with patch("scripts.prepare_data.urllib.request.urlopen", side_effect=TimeoutError("stall")) as open_mock, \
             patch("time.sleep"):
            try:
                list(stream_sources("https://fixture/archive.tar.gz", set(), root, budget,
                                    timeout_seconds=1, retries=2, retry_delay_seconds=0))
            except RuntimeError as error:
                assert "rerun" in str(error).lower()
            else:
                raise AssertionError("Retry exhaustion did not report an error")
            assert open_mock.call_count == 3


def check_parallel(range_supported=True):
    content, files = archive_fixture()

    def urlopen(request, timeout):
        header = request.get_header("Range")
        start, end = 0, len(content) - 1
        status = 200
        headers = {"ETag": '"fixture"'}
        if range_supported:
            headers["Accept-Ranges"] = "bytes"
        if header and range_supported:
            left, right = header.split("=")[1].split("-")
            start = int(left)
            end = int(right) if right else len(content) - 1
            headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
            status = 206
        headers["Content-Length"] = str(end - start + 1)
        return Response(content[start:end + 1], status, headers)

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        budget = StorageBudget(root, 1000000, 0)
        with patch("scripts.prepare_data.urllib.request.urlopen", urlopen):
            names = list(stream_sources("https://fixture/archive.tar.gz", set(files), root, budget,
                                        download_workers=4, chunk_bytes=8192))
        assert set(names) == set(files)
        for name, expected in files.items():
            assert (root / name).read_bytes() == expected
        assert budget.used == sum(map(len, files.values()))


def check_disk_resume():
    content, _ = archive_fixture()
    expected = hashlib.md5(content).hexdigest()
    size = 8192
    calls = []

    def urlopen(request, timeout):
        left, right = request.get_header("Range").split("=")[1].split("-")
        start, end = int(left), int(right)
        calls.append(start)
        return Response(content[start:end + 1], 206, {
            "Content-Range": f"bytes {start}-{end}/{len(content)}", "ETag": '"fixture"',
        })

    options = {"workers": 4, "chunk_bytes": size, "timeout_seconds": 1,
               "retries": 0, "retry_delay_seconds": 0, "progress_seconds": 1}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        destination = root / "archive.tar.gz"
        chunks = root / f"archive.tar.gz.{expected}.{size}.chunks"
        chunks.mkdir()
        # Simulate an earlier process stopping after one complete chunk.
        (chunks / "0.chunk").write_bytes(content[:size])
        (chunks / f"{size}.partial").write_bytes(b"unfinished")
        budget = StorageBudget(root, 1000000, 0)
        with patch("scripts.download_dataset.select_mirror",
                   return_value=("https://fixture/archive.tar.gz", len(content), '"fixture"')), \
             patch("scripts.prepare_data.urllib.request.urlopen", urlopen):
            download_archive(destination, expected, options, budget)
            assert destination.read_bytes() == content
            assert 0 not in calls and size in calls
            assert not chunks.exists()
            calls.clear()
            download_archive(destination, expected, options, budget)
            assert not calls, "A verified archive should not be downloaded again"
            destination.write_bytes(b"corrupt")
            try:
                download_archive(destination, expected, options, budget)
            except ValueError as error:
                assert "Checksum mismatch" in str(error)
            else:
                raise AssertionError("Corrupt archive was accepted")


if __name__ == "__main__":
    check_recovery(True)
    check_recovery(False)
    check_exhaustion()
    check_parallel()
    check_parallel(False)
    check_disk_resume()
    print("Download checks passed (resume, replay, retries, ranges, fallback, disk resume, checksums)")
