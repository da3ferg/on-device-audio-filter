#!/usr/bin/env python3
"""Offline device-output audio filter."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import tempfile
from time import monotonic, sleep
import wave

from engine import BLOCK_MS, FRAME_BYTES, Engine, Worker
from recognition import Whisper, Zipformer, vocabulary

RATES = (16000, 32000, 48000)
PERIOD = BLOCK_MS / 1000


def report(status, engine=None, **extra):
    fields = {"status": status, **extra}
    if engine is not None:
        fields.update(asdict(engine.stats), captured_frames=engine.captured,
                      rendered_frames=engine.rendered, skipped_windows=engine.worker.dropped)
    print(json.dumps(fields), file=sys.stderr, flush=True)


def health(engine):
    return "degraded" if engine.worker.dropped or engine.stats.late_hits else "running"


def sleep_until(deadline):
    remaining = deadline - monotonic()
    while remaining > 0:
        sleep(remaining)
        remaining = deadline - monotonic()


def pace_wav(source, target, engine, clock=monotonic, wait=sleep_until):
    """Availability is block-clocked. Output audio keeps the input file's timeline."""
    remaining = source.getnframes()
    epoch = clock()
    tick = 0
    max_release_slip = 0
    first_release_ms = None
    while remaining or engine.buffer:
        tick += 1
        deadline = epoch + tick * PERIOD
        wait(deadline)
        if remaining:
            count = min(remaining, engine.block)
            pcm = source.readframes(count)
            if len(pcm) != count * FRAME_BYTES:
                raise ValueError("Input WAV ended before its declared frame count")
            engine.capture(pcm)
            remaining -= count
            if not remaining:
                engine.finish()
        if tick > engine.delay_blocks and engine.buffer:
            pcm = engine.render(deadline)
            target.writeframesraw(pcm)
            emitted = clock()
            max_release_slip = max(max_release_slip, emitted - deadline)
            if first_release_ms is None:
                first_release_ms = (emitted - epoch - PERIOD) * 1000
        else:
            engine.collect(deadline)
        if tick % 250 == 0:
            report(health(engine) if engine.rendered else "priming", engine)
    return {
        "first_release_delay_ms": rounded(first_release_ms),
        "max_release_slip_ms": rounded(max_release_slip * 1000),
        "paced_seconds": rounded(clock() - epoch),
    }


def rounded(value):
    return None if value is None else round(value, 3)


def open_model(args, rate):
    report("loading_model")
    # A single .bin file is a whisper.cpp model; a folder is a Zipformer.
    kind = Whisper if args.model.is_file() else Zipformer
    return kind(args.model, vocabulary(args.words), rate, args.threads)


def run_wav(args):
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Use different input and output WAV paths")
    temp_path = None
    try:
        with wave.open(str(args.input), "rb") as source:
            rate = source.getframerate()
            if (source.getnchannels(), source.getsampwidth()) != (2, 2) or rate not in RATES:
                raise ValueError("WAV input must be stereo signed 16-bit PCM at 16, 32 or 48 kHz")
            model = open_model(args, rate)
            worker = Worker(model.decode)
            engine = Engine(rate, args.delay_ms, worker, args.window_ms)
            try:
                report("priming", engine, rate=rate, delay_ms=args.delay_ms, mode="wav")
                with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".wav",
                                                 delete=False) as temp:
                    temp_path = Path(temp.name)
                    with wave.open(temp, "wb") as target:
                        target.setparams(source.getparams())
                        timing = pace_wav(source, target, engine)
                worker.close(drain=True)
                # Count results after the final deadline without modifying the file.
                engine.collect(monotonic())
                temp_path.replace(args.output)
            finally:
                worker.close()
        report("complete", engine, health=health(engine), **timing)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def device_selector(value):
    return int(value) if value.lstrip("-").isdecimal() else value


def audio_api():
    try:
        import sounddevice
    except ImportError as error:
        raise RuntimeError("Install requirements-live.txt to use live audio") from error
    return sounddevice


def run_live(args):
    sd = audio_api()
    rate = args.rate
    sd.check_input_settings(args.input_device, channels=2, dtype="int16", samplerate=rate)
    sd.check_output_settings(args.output_device, channels=2, dtype="int16", samplerate=rate)
    model = open_model(args, rate)
    worker = Worker(model.decode)
    engine = Engine(rate, args.delay_ms, worker, args.window_ms)
    adc_times = deque()
    failure = None
    min_io_delay_ms = None
    max_io_delay_ms = None

    def callback(indata, outdata, frames, times, flags):
        nonlocal failure, min_io_delay_ms, max_io_delay_ms
        deadline = monotonic()
        try:
            if flags or frames != engine.block:
                raise RuntimeError(f"Audio device discontinuity ({flags or frames})")
            engine.capture(bytes(indata))
            adc_times.append(times.inputBufferAdcTime)
            if len(engine.buffer) > engine.delay_blocks:
                outdata[:] = engine.render(deadline)
                delay_ms = (times.outputBufferDacTime - adc_times.popleft()) * 1000
                min_io_delay_ms = delay_ms if min_io_delay_ms is None else min(min_io_delay_ms, delay_ms)
                max_io_delay_ms = delay_ms if max_io_delay_ms is None else max(max_io_delay_ms, delay_ms)
            else:
                engine.collect(deadline)
                outdata[:] = engine.silence
        except Exception as error:
            failure = error
            outdata[:] = engine.silence
            raise sd.CallbackAbort from error

    try:
        with sd.RawStream(samplerate=rate, blocksize=engine.block, dtype="int16",
                          channels=2, device=(args.input_device, args.output_device),
                          latency="high", callback=callback) as stream:
            report("priming", engine, rate=rate, delay_ms=args.delay_ms, mode="live")
            next_report = monotonic() + 2
            try:
                while stream.active:
                    sleep(0.1)
                    if monotonic() >= next_report:
                        report(health(engine) if engine.rendered else "priming", engine,
                               min_adc_to_dac_ms=rounded(min_io_delay_ms),
                               max_adc_to_dac_ms=rounded(max_io_delay_ms))
                        next_report = monotonic() + 2
                raise RuntimeError(failure or "Audio stream stopped unexpectedly")
            except KeyboardInterrupt:
                stream.abort()
    finally:
        worker.close()
    report("stopped", engine, health=health(engine),
           discarded_frames=engine.captured - engine.rendered,
           min_adc_to_dac_ms=rounded(min_io_delay_ms),
           max_adc_to_dac_ms=rounded(max_io_delay_ms))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, required=True)
    common.add_argument("--words", type=Path, default=Path(__file__).with_name("words.txt"))
    common.add_argument("--delay-ms", type=int, default=1600)
    common.add_argument("--threads", type=int, default=2)
    # Slow recognisers (Whisper) need a window longer than one decode, or audio is skipped.
    common.add_argument("--window-ms", type=int, default=3200)
    modes = parser.add_subparsers(dest="mode", required=True)
    wav = modes.add_parser("wav", parents=[common])
    wav.add_argument("input", type=Path)
    wav.add_argument("output", type=Path)
    live = modes.add_parser("live", parents=[common])
    live.add_argument("--rate", type=int, choices=RATES, default=48000)
    live.add_argument("--input-device", type=device_selector)
    live.add_argument("--output-device", type=device_selector)
    modes.add_parser("devices")
    args = parser.parse_args()
    if args.mode != "devices":
        if not BLOCK_MS <= args.delay_ms <= 15000 or args.delay_ms % BLOCK_MS:
            parser.error("--delay-ms must be a multiple of 20 between 20 and 15000")
        if not 1040 <= args.window_ms <= 12000 or args.window_ms % BLOCK_MS:
            parser.error("--window-ms must be a multiple of 20 between 1040 and 12000")
        if not 1 <= args.threads <= 32:
            parser.error("--threads must be between 1 and 32")
    return args


def main():
    args = parse_args()
    if sys.byteorder != "little":
        report("failed", error="A little-endian host is required")
        return 1
    try:
        if args.mode == "devices":
            sd = audio_api()
            for number, device in enumerate(sd.query_devices()):
                print(f"{number}: {device['name']}  in={device['max_input_channels']} "
                      f"out={device['max_output_channels']}")
        elif args.mode == "wav":
            run_wav(args)
        else:
            run_live(args)
        return 0
    except KeyboardInterrupt:
        report("interrupted")
        return 130
    except Exception as error:
        report("failed", error=str(error))
        return 1


if __name__ == "__main__":
    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    raise SystemExit(main())
