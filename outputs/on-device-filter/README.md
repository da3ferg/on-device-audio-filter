# On-device audio filter

A standalone prototype for filtering the stereo PCM sent to an audio output. It accepts any source that can deliver compatible PCM to the processor. The local English model timestamps vocabulary matches, and the engine mutes both stereo channels around each match. Audio, inference, and status stay on the host.

## Setup

Use Python 3.12 on a little-endian host. macOS on Apple Silicon is the verified setup. Dependency and model downloads are one-time setup steps.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Install this separately for physical or virtual audio devices.
.venv/bin/python -m pip install -r requirements-live.txt
mkdir -p models
```

## Run

From the folder containing this README, set `MODEL` to the extracted model folder. WAV input must be signed 16-bit stereo PCM at 16, 32, or 48 kHz.

```sh
.venv/bin/python audio_filter.py wav --model "$MODEL" \
  fixtures/synthetic-profanity.wav /tmp/filtered.wav
.venv/bin/python audio_filter.py devices
.venv/bin/python audio_filter.py live --model "$MODEL" --rate 48000 \
  --input-device INPUT_ID --output-device OUTPUT_ID
.venv/bin/python -m unittest discover -s tests -v
```

WAV processing reads 20 ms blocks at real-time pace, including a delay before writing the first processed block. The saved file preserves the original timeline and duration. Use `--words path/to/words.txt` for one case-insensitive English word per line with optional `#` comments. Use `--delay-ms` to set the playback buffer in 20 ms increments. Ctrl+C or SIGTERM stops the process. The CLI reports status and counts on stderr.

## Mac app audio

The current Mac has built-in stereo speakers and mono microphone inputs. A Parrot audio plug-in was present on disk, but Core Audio did not enumerate it as a usable input during this check. To test system audio through this build, install a stereo loopback device such as BlackHole 2ch separately. Set **System Settings → Sound → Output** to **BlackHole 2ch only**, then run live with BlackHole as the input and the physical speakers or headphones as the explicit output. For apps with their own output selector, route them to BlackHole there. A Multi-Output Device that includes the physical speakers would also play the original. Restore the previous system output after the test. If the clocks drift, configure a two-device Aggregate Device with drift correction in Audio MIDI Setup and verify the exposed input/output channel counts. [BlackHole system-audio instructions](https://github.com/ExistentialAudio/BlackHole/wiki/Getting-Started%3A-Sending-System-Audio-to-BlackHole/fda6278a3fb77a4f45aea47f178cc83aa011787f)

A packaged macOS version could use Apple's Core Audio process taps for stereo capture and optional suppression of direct playback. That integration needs a native app and the relevant audio-capture permission. [Apple Core Audio taps](https://developer.apple.com/documentation/CoreAudio/capturing-system-audio-with-core-audio-taps)

## LG C6 hookup

LG lists optical S/PDIF output and eARC on HDMI 2 for OLED65C6PUA. Start with **TV optical out → stereo PCM optical capture interface → processor → wired DAC/amp or powered speakers**. A stereo optical DAC into a USB line-input interface is another way to supply stereo capture. Select the wired optical output alone and set **Sound → Advanced Settings → Digital Sound Output → PCM**. In eARC experiments, use a receiver or extractor that explicitly accepts the TV's eARC return channel and exposes stereo PCM to the host. Confirm output with each intended TV app or HDMI source. [LG C6 connectivity](https://www.lg.com/us/tvs/lg-oled65c6pua-oled-4k-tv?tab=specs) · [LG PCM menu](https://www.lg.com/us/support/help-library/lg-tv-how-to-fix-echoing-or-outofsync-sound-when-using-tv-speakers-and-external-speakers-together--20155062023904)

Audio-only buffering pushes sound behind the TV picture by the filter delay plus the external audio path. An audio delay control cannot recover that time. A practical TV design needs substantially lower detection latency, control of video timing, or both. See [SPEC.md](SPEC.md), [EVALUATION.md](EVALUATION.md), and [HANDOFF.md](HANDOFF.md).
