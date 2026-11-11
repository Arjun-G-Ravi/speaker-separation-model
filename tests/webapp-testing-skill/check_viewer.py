import json
import subprocess
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright
from scipy.io import wavfile

from scripts.serve_demo import Demo, make_handler
from src.audio import read_audio
from src.settings import load_config


def main():
    config = load_config("demo.yaml")
    config["device"] = "cpu"
    demo = Demo(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(demo))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as p, tempfile.TemporaryDirectory() as folder:
            browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                        args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1080})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base)
            page.wait_for_load_state("networkidle")
            page.wait_for_function("document.querySelectorAll('#comparison audio').length === 3")
            assert page.locator("#sample-count").inner_text() == "500 examples"
            page.locator("#next").click()
            assert page.locator("#page").inner_text() == "2 / 25"
            page.locator("#split").select_option("validation")
            page.wait_for_function("document.querySelector('#sample-count').textContent === '50 examples'")
            page.wait_for_function("!document.querySelector('#run-example').disabled")
            page.locator("#run-example").click()
            page.wait_for_function("document.querySelectorAll('#comparison audio').length === 6", timeout=120000)
            assert "dB" in page.locator("#sample-metrics").inner_text()
            audio_url = page.locator("#comparison .model-track audio").first.get_attribute("src")
            request = urllib.request.Request(base + audio_url, headers={"Range": "bytes=0-43"})
            with urllib.request.urlopen(request) as response:
                assert response.status == 206 and len(response.read()) == 44
            page.locator("#comparison .model-track audio").first.evaluate("audio => audio.load()")
            page.wait_for_function("document.querySelector('#comparison .model-track audio').readyState >= 1")
            shots = Path(__file__).with_name("screenshots")
            shots.mkdir(exist_ok=True)
            page.screenshot(path=str(shots / f"{server.server_port}_dataset.png"), full_page=True)
            rate, signal = read_audio(demo.source_path(demo.examples["train"][0]["mixture"]))
            original = Path(folder) / "upload.wav"
            wavfile.write(original, rate, signal[:rate])
            encoded = original.with_suffix(".mp3")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), str(encoded)], check=True)
            page.locator("#upload-mode").click()
            page.locator("#audio-file").set_input_files(encoded)
            page.locator("#run-upload").click()
            page.wait_for_function("!document.querySelector('#upload-results').hidden", timeout=120000)
            assert page.locator("#upload-tracks audio").count() == 3
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(shots / f"{server.server_port}_upload_mobile.png"), full_page=True)
            assert not errors, errors
            browser.close()
        print(json.dumps({"dataset_browsing": "passed", "reference_comparison": "passed",
                          "model_inference": "passed", "audio_range_playback": "passed",
                          "mp3_upload": "passed", "mobile_layout": "passed"}))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
