# On-device audio filter

Mutes listed words in any audio routed through a device output. Recognition, buffering and muting all run locally; no audio leaves the machine.

It is built for speech (podcasts, shows, calls). Music is not supported: the recogniser misses most sung words.

## Layout

- `outputs/on-device-filter-go/` — the current version, in Go. Commands: `wav`, `live`, `devices`.
- `outputs/on-device-filter/` — the original Python prototype and its notes.
- `outputs/on-device-filter-rs/` — an unfinished Rust port (not compiled).

## Go version

Needs Go 1.27+, a C compiler, and the `sherpa-onnx-zipformer-gigaspeech-2023-12-12` model from the [sherpa-onnx ASR model releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models), extracted to a folder.

```sh
cd outputs/on-device-filter-go
go test ./...
go build -o audiofilter .
./audiofilter devices
./audiofilter wav --model MODEL_DIR input.wav output.wav
./audiofilter live --model MODEL_DIR --input-device "BlackHole 2ch" --output-device "MacBook Air Speakers"
```

WAV input must be stereo signed 16-bit PCM at 16, 32 or 48 kHz. Live mode needs a loopback input such as BlackHole 2ch as the system output, with the filter playing to the real speakers.

Options: `--delay-ms` (default 2000) is how far the output lags the input; `--hop-ms` (default 160) is the time between recognition checks; `--words FILE` replaces the built-in word list. Status is printed to stderr as JSON lines.

## Known limits

- Tested on macOS on Apple Silicon only.
- Only exact words on the list are muted; there is no contextual detection.
- Some words are recognised late. At a 1 second delay a few play in full; 2 seconds is the tested default.
- Test audio and models are not included in this repository.
