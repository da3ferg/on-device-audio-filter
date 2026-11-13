"""Local Zipformer adapter (non-streaming GigaSpeech model; LibriSpeech-trained
models transcribe profanity as clean look-alikes). Only matching words and timestamps leave this module."""
import math
import re

from engine import Word

WORD = re.compile(rb"[a-zA-Z]+(?:'[a-zA-Z]+)*")


def vocabulary(path):
    entries = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.partition("#")[0].strip().lower().encode()
        if entry:
            if not WORD.fullmatch(entry):
                raise ValueError(f"{path}:{number} must contain one English word")
            entries.add(entry)
    if not entries:
        raise ValueError("Vocabulary is empty")
    return entries


def matched_words(tokens, entries):
    """Match reconstructed words, including words split across BPE tokens."""
    contents = bytearray()
    timing = []
    for token, start, end in tokens:
        contents.extend(token)
        timing.extend([(start, end)] * len(token))
    for match in WORD.finditer(contents):
        if match.group().lower() not in entries:
            continue
        bounds = timing[match.start():match.end()]
        if any(start < 0 or end < start for start, end in bounds):
            continue
        start = min(start for start, _ in bounds)
        end = max(end for _, end in bounds)
        if start < end:
            yield Word(match.group().lower(), start, end)


def decimation_kernel(rate):
    cutoff = 7200 / rate
    radius = 40
    weights = []
    for i in range(-radius, radius + 1):
        sinc = math.sin(2 * math.pi * cutoff * i) / (math.pi * i) if i else 2 * cutoff
        window = 0.54 + 0.46 * math.cos(math.pi * i / radius)
        weights.append(sinc * window)
    total = sum(weights)
    return [weight / total for weight in weights]


class Zipformer:
    def __init__(self, model, entries, rate, threads):
        paths = {"tokens": model / "tokens.txt"}
        for kind in ("encoder", "decoder", "joiner"):
            paths[kind] = model / f"{kind}-epoch-30-avg-1.int8.onnx"
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f"Local model file missing: {path}")
        try:
            import numpy
            import sherpa_onnx
        except ImportError as error:
            raise RuntimeError("Install requirements.txt to run local recognition") from error
        self.np = numpy
        self.entries = entries
        self.rate = rate
        self.stride = rate // 16000
        self.kernel = numpy.array(decimation_kernel(rate), dtype=numpy.float32)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            **{key: str(path) for key, path in paths.items()}, num_threads=threads,
            provider="cpu", decoding_method="greedy_search",
        )

    def mono16k(self, pcm):
        np = self.np
        mono = np.frombuffer(pcm, dtype="<i2").reshape(-1, 2).mean(axis=1, dtype=np.float32)
        if self.stride > 1:
            mono = np.convolve(mono, self.kernel, mode="same")[::self.stride]
        return mono

    def decode(self, pcm):
        frames = len(pcm) // 4
        if frames < self.rate // 10:
            return ()
        mono = self.mono16k(pcm)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, mono / 32768)
        self.recognizer.decode_stream(stream)
        result = stream.result
        starts = [round(timestamp * self.rate) for timestamp in result.timestamps]
        if len(starts) != len(result.tokens):
            raise RuntimeError("Recognition returned inconsistent token timestamps")

        tokens = []
        for index, (token, start) in enumerate(zip(result.tokens, starts)):
            # The model reports onsets. Bound the last-piece estimate when the next
            # token is absent, coincident, or separated by silence.
            end = min(start + self.rate * 180 // 1000, frames)
            if index + 1 < len(starts) and starts[index + 1] > start:
                end = min(end, starts[index + 1])
            tokens.append((token.replace("▁", " ").encode(), start, end))
        return tuple(matched_words(tokens, self.entries))


class Whisper(Zipformer):
    """Whisper large-v3-turbo through a local whisper.cpp server (loopback only).

    Hears sung and accented speech the Zipformer misses, but each decode takes
    2-4 s, so run it with --delay-ms 6000 or more. The engine's worker keeps only
    the newest pending window, so windows are skipped by design while one decodes.
    """

    def __init__(self, model, entries, rate, threads):
        import atexit
        import io
        import socket
        import subprocess
        import time
        import urllib.request
        import wave
        import numpy
        self.np, self.io, self.wave, self.request = numpy, io, wave, urllib.request
        self.entries = entries
        self.rate = rate
        self.stride = rate // 16000
        self.kernel = numpy.array(decimation_kernel(rate), dtype=numpy.float32)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/inference"
        self.server = subprocess.Popen(
            ["whisper-server", "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
             "-l", "en", "-ml", "1", "-sow", "-t", str(threads)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.server.terminate)
        for _ in range(300):  # model load takes a few seconds
            if self.server.poll() is not None:
                raise RuntimeError("whisper-server exited while loading the model")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("whisper-server did not start")
        self.decode(bytes(rate * 4))  # warm up so the first real window is not slow

    def decode(self, pcm):
        frames = len(pcm) // 4
        if frames < self.rate // 10:
            return ()
        audio = self.io.BytesIO()
        with self.wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(self.mono16k(pcm).astype("<i2").tobytes())
        mark = b"----filter-boundary"
        body = b"".join([
            b"--", mark, b'\r\nContent-Disposition: form-data; name="file"; filename="w.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n", audio.getvalue(),
            b"\r\n--", mark, b'\r\nContent-Disposition: form-data; name="response_format"\r\n\r\n'
            b"verbose_json\r\n--", mark, b"--\r\n"])
        request = self.request.Request(self.url, body, {
            "Content-Type": "multipart/form-data; boundary=" + mark.decode()})
        with self.request.urlopen(request, timeout=30) as response:
            import json
            segments = json.load(response)["segments"]
        tokens = [(segment["text"].encode("ascii", "ignore"),
                   round(segment["start"] * self.rate), min(round(segment["end"] * self.rate), frames))
                  for segment in segments]
        return tuple(matched_words(tokens, self.entries))
