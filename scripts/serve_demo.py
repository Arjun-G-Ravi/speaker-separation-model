"""Local dataset listening room and upload inference, without web dependencies."""

import json
import mimetypes
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from scipy.io import wavfile

from src.audio import read_audio
from src.inference import playback_audio, separate_waveform
from src.loss import pairwise_si_sdr, pit_si_sdr
from src.settings import ROOT, digest, load_config, load_model, project_path, select_device

STATIC = Path(__file__).with_name("demo")


class Demo:
    def __init__(self, config):
        self.config = config
        self.device = select_device(config["device"])
        if self.device.type == "cpu":
            torch.set_num_threads(config["cpu_threads"])
        self.model, self.state = load_model(config["checkpoint"], self.device)
        self.rate = self.state["config"]["data"]["sample_rate"]
        self.root = project_path(self.state["config"]["data"]["root"])
        self.examples = {}
        for split in ("train", "validation"):
            manifest = self.root / self.state["config"]["data"][f"{split}_manifest"]
            if digest(manifest) != self.state["manifest_hashes"][split]:
                raise ValueError(f"{split} manifest differs from the checkpoint's dataset")
            self.examples[split] = json.loads(manifest.read_text())["examples"]
        self.output = project_path(config["output"]) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.output.mkdir(parents=True, exist_ok=False)
        self.files, self.predictions = {}, {}
        self.lock = threading.Lock()

    def example(self, split, index):
        if split not in self.examples or not 0 <= index < len(self.examples[split]):
            raise ValueError("Unknown dataset example")
        return self.examples[split][index]

    def source_path(self, value):
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Dataset audio must be inside its data root")
        return path

    def track(self, path, label):
        rate, audio = read_audio(path)
        if audio.ndim == 2:
            audio = audio.mean(1)
        blocks = np.array_split(audio, min(400, len(audio)))
        peaks = [[float(block.min()), float(block.max())] for block in blocks]
        key = uuid.uuid4().hex
        self.files[key] = Path(path)
        return {"label": label, "url": f"/audio/{key}", "peaks": peaks,
                "duration": len(audio) / rate, "rate": rate}

    def info(self):
        metrics = ROOT / "logs" / self.state["run_id"] / "metrics.jsonl"
        history = []
        if metrics.exists():
            history = [row for line in metrics.read_text().splitlines()
                       if "epoch" in (row := json.loads(line))]
        return {"run": self.state["run_id"], "epoch": self.state["next_epoch"],
                "validation_si_sdr": self.state["best"], "device": str(self.device),
                "rate": self.rate, "history": history,
                "max_upload_mb": self.config["max_upload_mb"],
                "max_upload_seconds": self.config["max_upload_seconds"],
                "splits": {split: [{"id": row["id"], "index": index,
                                    "duration": row["length"] / self.rate}
                                   for index, row in enumerate(rows)]
                           for split, rows in self.examples.items()}}

    def detail(self, split, index):
        row = self.example(split, index)
        return {"id": row["id"], "index": index, "split": split,
                "mixture": self.track(self.source_path(row["mixture"]), "Mixture"),
                "references": [self.track(self.source_path(path), f"Speaker {i + 1} reference")
                               for i, path in enumerate(row["sources"])],
                "prediction": self.predictions.get((split, index))}

    def predict(self, path, references=None):
        rate, audio = read_audio(path)
        seconds = len(audio) / rate
        if references is None and seconds > self.config["max_upload_seconds"]:
            raise ValueError(f"Use audio up to {self.config['max_upload_seconds']} seconds long")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("The model is busy. Wait for the current separation to finish.")
        started = time.monotonic()
        try:
            mixture, estimate = separate_waveform(self.model, audio, rate, self.rate, self.device)
            metrics = None
            if references:
                target = torch.from_numpy(np.stack([read_audio(p)[1] for p in references]))[None]
                predicted = torch.from_numpy(estimate)[None]
                _, order = pit_si_sdr(predicted, target)
                estimate = estimate[torch.argsort(order[0]).numpy()]
                scores = pairwise_si_sdr(torch.from_numpy(estimate)[None], target)[0].diag()
                baseline = pairwise_si_sdr(torch.from_numpy(mixture)[None, None].expand_as(target),
                                           target)[0].diag()
                metrics = {"si_sdr": scores.mean().item(),
                           "si_sdri": (scores - baseline).mean().item(),
                           "per_speaker": scores.tolist()}
            destination = self.output / uuid.uuid4().hex
            destination.mkdir()
            input_path = destination / "mixture.wav"
            wavfile.write(input_path, self.rate, mixture)
            tracks = []
            estimate, playback_gain = playback_audio(estimate)
            for index, signal in enumerate(estimate):
                output = destination / f"speaker_{index + 1}.wav"
                wavfile.write(output, self.rate, signal.astype(np.float32))
                tracks.append(self.track(output, f"Speaker {index + 1} model output"))
            return {"tracks": tracks, "mixture": self.track(input_path, "Uploaded mixture"),
                    "seconds": time.monotonic() - started, "metrics": metrics,
                    "matched": bool(references), "playback_gain": playback_gain}
        finally:
            self.lock.release()

    def predict_example(self, split, index):
        row = self.example(split, index)
        if (split, index) not in self.predictions:
            self.predictions[split, index] = self.predict(
                self.source_path(row["mixture"]), [self.source_path(p) for p in row["sources"]]
            )
        return self.predictions[split, index]


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            content = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_file(self, path):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            header = self.headers.get("Range")
            if header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                start = int(match[1]) if match[1] else max(0, size - int(match[2]))
                end = min(int(match[2]), size - 1) if match[1] and match[2] else size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    content = handle.read(min(65536, remaining))
                    if not content:
                        break
                    self.wfile.write(content)
                    remaining -= len(content)

        def dispatch(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if self.command == "GET":
                if parsed.path in ("/", "/app.js", "/style.css"):
                    self.send_file(STATIC / ("index.html" if parsed.path == "/" else parsed.path[1:]))
                elif parsed.path == "/api/info":
                    self.send_json(demo.info())
                elif parsed.path == "/api/example":
                    self.send_json(demo.detail(query.get("split", ["train"])[0],
                                               int(query.get("index", ["0"])[0])))
                elif parsed.path.startswith("/audio/") and parsed.path[7:] in demo.files:
                    self.send_file(demo.files[parsed.path[7:]])
                else:
                    self.send_json({"error": "Not found"}, 404)
                return
            if parsed.path not in ("/api/separate", "/api/upload"):
                self.send_json({"error": "Not found"}, 404)
                return
            if self.headers.get("X-Requested-With") != "separation-viewer":
                self.send_json({"error": "Use the local viewer to submit audio"}, 403)
                return
            limit = demo.config["max_upload_mb"] * 1024**2 if parsed.path == "/api/upload" else 16384
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= limit:
                self.close_connection = True
                self.send_json({"error": f"Request must contain audio within the {demo.config['max_upload_mb']} MB limit"}, 413)
                return
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("Upload was interrupted; select your file and try again")
            if parsed.path == "/api/separate":
                body = json.loads(content)
                self.send_json(demo.predict_example(body["split"], int(body["index"])))
            else:
                name = query.get("name", ["audio"])[0]
                extension = Path(name).suffix.lower()
                if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
                    extension = ".audio"
                path = demo.output / (uuid.uuid4().hex + extension)
                try:
                    path.write_bytes(content)
                    self.send_json(demo.predict(path))
                finally:
                    path.unlink(missing_ok=True)

        def handle_request(self):
            try:
                self.dispatch()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BlockingIOError as error:
                self.send_json({"error": str(error)}, 409)
            except (ValueError, KeyError, IndexError) as error:
                self.send_json({"error": str(error)}, 400)
            except subprocess.CalledProcessError:
                self.send_json({"error": "Could not decode this file. Try a valid WAV, MP3, FLAC, OGG, or M4A file."}, 400)
            except torch.OutOfMemoryError:
                self.send_json({"error": "Not enough memory. Try a shorter audio clip."}, 422)
            except FileNotFoundError as error:
                self.send_json({"error": f"Required file or audio decoder not found: {error.filename}"}, 404)

        do_GET = handle_request
        do_POST = handle_request

    return Handler


def main():
    config = load_config("demo.yaml")
    if config["host"] not in ("localhost", "127.0.0.1"):
        raise ValueError("The demo binds to localhost only; use SSH forwarding for remote access")
    demo = Demo(config)
    with ThreadingHTTPServer((config["host"], config["port"]), make_handler(demo)) as server:
        print(f"Listening room: http://{config['host']}:{config['port']}\n"
              f"Checkpoint: {config['checkpoint']} | device: {demo.device}\n"
              f"Dataset: {len(demo.examples['train'])} training / "
              f"{len(demo.examples['validation'])} validation examples\n"
              "Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped listening room.", flush=True)


if __name__ == "__main__":
    main()
