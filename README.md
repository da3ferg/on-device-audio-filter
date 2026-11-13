# On-device audio filter

Mutes the words you list in whatever is playing on your Mac. Everything runs on the machine: no audio or text is sent anywhere.

It is written in Rust. Sound comes out about 2 seconds behind the original, because the filter needs that time to recognise a word before it reaches the speakers.

## What to expect

- It works on any audio with words in it, music included. Most live testing so far has been on music.
- Accuracy on music is uneven. It muted listed words live in several songs, but on one 90-second dancehall track (Jamaican Patois and English) it caught none of the 9 words marked by ear. Plain speech is recognised more reliably.
- Only the exact words in the list are muted.

## Set up (once)

You need a Mac with Apple Silicon, about 1 GB of free disk space, and an internet connection for the downloads below.

1. Install Rust 1.85 or newer from [rustup.rs](https://rustup.rs).

2. Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole). It is a virtual audio device that lets the filter sit between your apps and your speakers. Restart the Mac afterwards.

3. Get the code and build it. The first build downloads the speech recognition library and takes a few minutes.

   ```sh
   git clone https://github.com/da3ferg/on-device-audio-filter.git
   cd on-device-audio-filter
   cargo build --release
   ```

4. Download the speech model (310 MB) into the same folder and unpack it.

   ```sh
   curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
   tar xf sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2
   ```

## Filter live audio

1. Open System Settings → Sound → Output and choose BlackHole 2ch. Your Mac goes silent; that is expected.

2. Start the filter. It plays the filtered sound on your real speakers.

   ```sh
   ./target/release/audio-filter live \
     --model sherpa-onnx-zipformer-gigaspeech-2023-12-12 \
     --input-device "BlackHole 2ch" \
     --output-device "MacBook Air Speakers"
   ```

3. Play anything. You hear it about 2 seconds late, with listed words silenced.

4. To stop, press Ctrl-C, then set Sound → Output back to your speakers.

If your speakers or headphones have a different name, list the names with `./target/release/audio-filter devices` and use the one you want. Part of a name is enough.

## Filter a file

```sh
./target/release/audio-filter wav \
  --model sherpa-onnx-zipformer-gigaspeech-2023-12-12 \
  input.wav output.wav
```

The input must be a stereo, 16-bit WAV at 16, 32 or 48 kHz. The file is processed at playing speed, so a 3-minute file takes 3 minutes. The output has the same length as the input.

## Change the word list

The words to mute are in [`words.txt`](words.txt):

- One English word per line. Capitals do not matter. Anything after `#` is a comment.
- Only exact words are muted, so add each form you want: `word`, `words`, `wording`.

There are two ways to use a changed list:

- Edit `words.txt`, then run `cargo build --release` again. The list is built into the program.
- Or keep your own file anywhere and add `--words my-words.txt` to the command. No rebuild needed. Your file replaces the built-in list.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--model DIR` | required | Folder holding the speech model |
| `--words FILE` | built-in list | Your own word list |
| `--delay-ms N` | 2000 | How far the sound lags. Higher catches more words in time; lower lets more slip through |
| `--hop-ms N` | 160 | Time between recognition checks. Lower reacts sooner and uses more CPU |
| `--window-ms N` | 3200 | How much recent audio the recogniser sees per check |
| `--threads N` | 2 | CPU threads for recognition |
| `--input-device NAME` | system default | Live only: where audio comes from |
| `--output-device NAME` | system default | Live only: where filtered audio plays |
| `--rate N` | 48000 | Live only: sample rate (16000, 32000 or 48000) |

## Reading the status lines

While it runs, the filter prints one line of JSON every 2 seconds. The useful fields:

- `hits`: words muted so far.
- `late_hits`: words that were partly played before being recognised. `fully_late_hits` were played in full.
- `input_peak`: loudness of incoming audio. `0` means the filter is receiving silence; check that Sound → Output is BlackHole 2ch.
- `underrun_frames` and `drift_dropped_frames`: glitches in the output. Both should stay at or near 0.
- `max_callback_gap_ms`: longest wait between two output callbacks since the last line. It should sit near 11; a spike means the sound stalled for a moment.
- `status`: `running` is healthy. `degraded` means a word was late or a check was skipped.

## Troubleshooting

- No sound at all: the filter is not running, or the output device name is wrong. Run the `devices` command and check the name.
- Sound but nothing is ever muted: look at `input_peak`. If it is 0, audio is not reaching BlackHole.
- You hear everything twice: an app is playing straight to the speakers as well. Set that app's output to BlackHole 2ch.
- Words slip through: raise `--delay-ms` (try 3000), or add the missing form of the word to the list.

## Known limits

- Tested on macOS with Apple Silicon only.
- No contextual detection: a listed word is muted wherever it is recognised, and unlisted words never are.
- Sung, heavily accented or Patois words are often missed, because the recogniser was trained on spoken English.
- Video is not delayed, so picture and sound are out of sync by the delay.
- The input and output devices run on separate clocks. Over a long session the filter may drop 20 ms of audio now and then to stay in step.

## How it works

See [ARCHITECTURE.md](ARCHITECTURE.md) for a diagram and an explanation of each part.

## Tests

```sh
cargo test --release
```
