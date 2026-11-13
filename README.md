# On-device audio filter

Mutes listed words in any audio routed through a device output. Recognition, buffering and muting all run locally; no audio leaves the machine.

It works on any audio with words in it, music included, and most live testing so far has been on music. Accuracy on music is uneven: it muted listed words live in some songs, but on one 90-second dancehall track (Jamaican Patois and English) it caught none of the 9 words marked by ear. Plain speech is recognised more reliably.

How it works, part by part: [ARCHITECTURE.md](ARCHITECTURE.md).

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

## Word list

The words to mute live in one plain text file: [`outputs/on-device-filter-go/words.txt`](outputs/on-device-filter-go/words.txt).

- One English word per line. Case does not matter. Anything after `#` is a comment.
- Only exact words are muted, so list each form you want (`word`, `words`, `wording`).

Two ways to change it:

- Edit `words.txt` and rebuild (`go build -o audiofilter .`). The list is built into the program.
- Keep your own file anywhere and pass it at run time, no rebuild needed: `./audiofilter live --model MODEL_DIR --words my-words.txt`. This replaces the built-in list.

The Python prototype has its own copy at `outputs/on-device-filter/words.txt` and takes the same `--words` option.

## Known limits

- Tested on macOS on Apple Silicon only.
- Only exact words on the list are muted; there is no contextual detection.
- Sung, heavily accented or Patois words are often missed, because the recogniser was trained on spoken English.
- Some words are recognised late. At a 1 second delay a few play in full; 2 seconds is the tested default.
- Test audio and models are not included in this repository.
