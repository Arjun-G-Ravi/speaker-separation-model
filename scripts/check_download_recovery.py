"""Offline regression checks for interrupted source archive transfers."""

import io
import random
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    check_recovery(True)
    check_recovery(False)
    check_exhaustion()
    print("Download recovery checks passed (range resume, replay fallback, retry exhaustion)")
