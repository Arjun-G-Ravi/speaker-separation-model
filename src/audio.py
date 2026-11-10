"""WAV I/O and FLAC decoding through the system ffmpeg executable."""

import io
import struct
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_audio(path):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        rate, audio = wavfile.read(path)
        if audio.dtype.kind in "iu":
            if audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128
            else:
                audio = audio.astype(np.float32) / (np.iinfo(audio.dtype).max + 1)
        return int(rate), audio.astype(np.float32)
    if path.suffix.lower() != ".flac":
        raise ValueError("Input must be WAV or FLAC")
    # WAV output preserves sample rate/channels and avoids torchcodec dependencies.
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "wav", "-acodec", "pcm_f32le", "-"],
        check=True, capture_output=True,
    )
    # ffmpeg cannot seek back on a pipe and writes UINT_MAX chunk lengths.
    # Finalize those lengths before passing the complete buffer to scipy.
    buffer = bytearray(result.stdout)
    struct.pack_into("<I", buffer, 4, len(buffer) - 8)
    position = 12
    while position + 8 <= len(buffer):
        size = struct.unpack_from("<I", buffer, position + 4)[0]
        if buffer[position:position + 4] == b"data":
            struct.pack_into("<I", buffer, position + 4, len(buffer) - position - 8)
            break
        position += 8 + size + size % 2
    rate, audio = wavfile.read(io.BytesIO(buffer))
    return int(rate), audio.astype(np.float32)


def resample(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return audio
    from math import gcd

    factor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // factor, source_rate // factor, axis=0).astype(
        np.float32
    )


def write_pcm16(path, rate, audio):
    # Match libsndfile's PCM16 quantization, including its asymmetric full scale.
    pcm = np.floor(np.clip(audio, -1, 32767 / 32768) * 32768).astype(np.int16)
    wavfile.write(path, rate, pcm)
