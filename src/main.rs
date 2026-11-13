//! On-device audio filter: mutes listed words in stereo PCM, from a WAV file or live.
mod engine;
mod recognition;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use engine::{now, Engine, Recognition, Worker, BLOCK_MS, MIN_WINDOW_MS, WINDOW_MS};
use recognition::{vocabulary, Zipformer};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const RATES: [usize; 3] = [16000, 32000, 48000];
const PERIOD: f64 = BLOCK_MS as f64 / 1000.0;
const USAGE: &str = "usage: audio-filter {wav|live|devices} --model DIR [options] [input.wav output.wav]
  --words FILE          word list, one per line (default: built in)
  --delay-ms N          output delay (default 2000)
  --hop-ms N            time between recognition checks (default 160)
  --window-ms N         audio the recogniser sees per check (default 3200)
  --threads N           recognition threads (default 2)
  live only: --rate N, --input-device NAME|NUMBER, --output-device NAME|NUMBER";

#[derive(Debug)]
struct Options {
    model: PathBuf,
    words: Option<PathBuf>,
    delay_ms: usize,
    hop_ms: usize,
    window_ms: usize,
    threads: i32,
    rate: usize,
    input_device: Option<String>,
    output_device: Option<String>,
    files: Vec<PathBuf>,
}

fn parse(mode: &str, args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        model: PathBuf::new(),
        words: None,
        delay_ms: 2000,
        hop_ms: 160,
        window_ms: WINDOW_MS,
        threads: 2,
        rate: 48000,
        input_device: None,
        output_device: None,
        files: Vec::new(),
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let Some(name) = arg.strip_prefix("--") else {
            o.files.push(arg.into());
            continue;
        };
        let value = args.next().ok_or(format!("--{name} needs a value"))?;
        let number = || value.parse::<usize>().map_err(|_| format!("--{name} must be a number"));
        match (name, mode) {
            ("model", _) => o.model = value.into(),
            ("words", _) => o.words = Some(value.into()),
            ("delay-ms", _) => o.delay_ms = number()?,
            ("hop-ms", _) => o.hop_ms = number()?,
            ("window-ms", _) => o.window_ms = number()?,
            ("threads", _) => o.threads = number()? as i32,
            ("rate", "live") => o.rate = number()?,
            ("input-device", "live") => o.input_device = Some(value.clone()),
            ("output-device", "live") => o.output_device = Some(value.clone()),
            _ => return Err(format!("unknown option --{name}\n{USAGE}")),
        }
    }
    let blocks = |ms: usize, low: usize, high: usize| (low..=high).contains(&ms) && ms % BLOCK_MS == 0;
    if o.model.as_os_str().is_empty() {
        return Err("--model is required".into());
    }
    if !blocks(o.delay_ms, BLOCK_MS, 15000) {
        return Err("--delay-ms must be a multiple of 20 between 20 and 15000".into());
    }
    if !blocks(o.hop_ms, BLOCK_MS, 1000) {
        return Err("--hop-ms must be a multiple of 20 between 20 and 1000".into());
    }
    if !blocks(o.window_ms, MIN_WINDOW_MS, 12000) {
        return Err("--window-ms must be a multiple of 20 between 1040 and 12000".into());
    }
    if !(1..=32).contains(&o.threads) {
        return Err("--threads must be between 1 and 32".into());
    }
    if !RATES.contains(&o.rate) {
        return Err("--rate must be 16000, 32000 or 48000".into());
    }
    if o.files.len() != if mode == "wav" { 2 } else { 0 } {
        return Err(USAGE.into());
    }
    Ok(o)
}

fn quoted(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' | '\\' => out.extend(['\\', c]),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out + "\""
}

/// One JSON object per line on stderr. `extra` values are already JSON.
fn report<R: Recognition>(status: &str, engine: Option<&Engine<R>>, extra: &[(&str, String)]) {
    eprintln!("{}", status_line(status, engine, extra));
}

fn status_line<R: Recognition>(status: &str, engine: Option<&Engine<R>>, extra: &[(&str, String)]) -> String {
    let mut fields = vec![("status".to_string(), quoted(status))];
    if let Some(e) = engine {
        let s = &e.stats;
        for (key, value) in [
            ("windows", s.windows.to_string()),
            ("max_decode_ms", s.max_decode_ms.to_string()),
            ("hits", s.hits.to_string()),
            ("late_hits", s.late_hits.to_string()),
            ("fully_late_hits", s.fully_late_hits.to_string()),
            ("muted_frames", s.muted_frames.to_string()),
            ("max_late_core_ms", s.max_late_core_ms.to_string()),
            ("max_hit_age_ms", s.max_hit_age_ms.to_string()),
            ("captured_frames", e.captured.to_string()),
            ("rendered_frames", e.rendered.to_string()),
            ("skipped_windows", e.worker.dropped().to_string()),
        ] {
            fields.push((key.into(), value));
        }
    }
    fields.extend(extra.iter().map(|(key, value)| (key.to_string(), value.clone())));
    let body: Vec<String> = fields.iter().map(|(key, value)| format!("{}:{value}", quoted(key))).collect();
    format!("{{{}}}", body.join(","))
}

fn health<R: Recognition>(engine: &Engine<R>) -> &'static str {
    if engine.worker.dropped() > 0 || engine.stats.late_hits > 0 { "degraded" } else { "running" }
}

fn progress<R: Recognition>(engine: &Engine<R>) -> &'static str {
    if engine.rendered > 0 { health(engine) } else { "priming" }
}

fn sleep_until(deadline: f64) {
    let remaining = deadline - now();
    if remaining > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(remaining));
    }
}

fn rounded(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

struct Timing {
    first_release_ms: Option<f64>,
    max_release_slip_ms: f64,
    paced_seconds: f64,
}

/// Make audio available one block per tick. Output keeps the input's timeline.
fn pace_wav<R: Recognition>(
    frames: usize,
    read: &mut dyn FnMut(usize) -> Result<Vec<i16>, String>,
    write: &mut dyn FnMut(&[i16]) -> Result<(), String>,
    engine: &mut Engine<R>,
    clock: &dyn Fn() -> f64,
    wait: &mut dyn FnMut(f64),
) -> Result<Timing, String> {
    let mut remaining = frames;
    let epoch = clock();
    let mut timing = Timing { first_release_ms: None, max_release_slip_ms: 0.0, paced_seconds: 0.0 };
    let mut tick = 0usize;
    while remaining > 0 || !engine.buffer.is_empty() {
        tick += 1;
        let deadline = epoch + tick as f64 * PERIOD;
        wait(deadline);
        if remaining > 0 {
            let count = remaining.min(engine.block);
            engine.capture(&read(count)?)?;
            remaining -= count;
            if remaining == 0 {
                engine.finish();
            }
        }
        if tick > engine.delay_blocks && !engine.buffer.is_empty() {
            write(&engine.render(deadline)?)?;
            let emitted = clock();
            timing.max_release_slip_ms = timing.max_release_slip_ms.max((emitted - deadline) * 1000.0);
            timing.first_release_ms.get_or_insert(rounded((emitted - epoch - PERIOD) * 1000.0));
        } else {
            engine.collect(deadline)?;
        }
        if tick % 250 == 0 {
            report(progress(engine), Some(engine), &[]);
        }
    }
    timing.max_release_slip_ms = rounded(timing.max_release_slip_ms);
    timing.paced_seconds = rounded(clock() - epoch);
    Ok(timing)
}

fn open_model(opts: &Options, rate: usize) -> Result<Zipformer, String> {
    report::<Worker>("loading_model", None, &[]);
    let entries = match &opts.words {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            vocabulary(&path.to_string_lossy(), &text)?
        }
        None => vocabulary("words.txt", include_str!("../words.txt"))?,
    };
    Zipformer::new(&opts.model, entries, rate, opts.threads)
}

fn start_engine(opts: &Options, rate: usize) -> Result<Engine<Worker>, String> {
    let model = open_model(opts, rate)?;
    let worker = Worker::new(move |pcm| model.decode(pcm));
    Ok(Engine::new(rate, opts.delay_ms, worker).with_timing(opts.window_ms, opts.hop_ms))
}

fn run_wav(opts: &Options) -> Result<(), String> {
    let (input, output) = (&opts.files[0], &opts.files[1]);
    let absolute = |path: &Path| std::path::absolute(path).unwrap_or(path.to_path_buf());
    if absolute(input) == absolute(output) {
        return Err("Use different input and output WAV paths".into());
    }
    let bad = "WAV input must be stereo signed 16-bit PCM at 16, 32 or 48 kHz";
    let mut reader = hound::WavReader::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let spec = reader.spec();
    let rate = spec.sample_rate as usize;
    if (spec.channels, spec.bits_per_sample, spec.sample_format) != (2, 16, hound::SampleFormat::Int)
        || !RATES.contains(&rate)
    {
        return Err(bad.into());
    }
    let frames = reader.duration() as usize;
    let mut engine = start_engine(opts, rate)?;
    report(
        "priming",
        Some(&engine),
        &[("rate", rate.to_string()), ("delay_ms", opts.delay_ms.to_string()), ("mode", quoted("wav"))],
    );

    // Write beside the output and rename, so a failed run never leaves a partial file there.
    let temp = absolute(output).with_extension(format!("{}.partial", std::process::id()));
    let result = (|| -> Result<Timing, String> {
        let mut writer = hound::WavWriter::create(&temp, spec).map_err(|e| e.to_string())?;
        let mut samples = reader.samples::<i16>();
        let mut read = |count: usize| {
            let pcm: Vec<i16> = samples.by_ref().take(count * 2).filter_map(Result::ok).collect();
            if pcm.len() != count * 2 {
                return Err("Input WAV ended before its declared frame count".to_string());
            }
            Ok(pcm)
        };
        let mut write = |pcm: &[i16]| {
            pcm.iter().try_for_each(|&sample| writer.write_sample(sample)).map_err(|e: hound::Error| e.to_string())
        };
        let timing = pace_wav(frames, &mut read, &mut write, &mut engine, &now, &mut sleep_until)?;
        engine.worker.close(true);
        // Count results after the final deadline without modifying the file.
        engine.collect(now())?;
        writer.finalize().map_err(|e| e.to_string())?;
        std::fs::rename(&temp, output).map_err(|e| e.to_string())?;
        Ok(timing)
    })();
    let timing = result.inspect_err(|_| drop(std::fs::remove_file(&temp)))?;
    report(
        "complete",
        Some(&engine),
        &[
            ("health", quoted(health(&engine))),
            ("first_release_delay_ms", timing.first_release_ms.map_or("null".into(), |ms| ms.to_string())),
            ("max_release_slip_ms", timing.max_release_slip_ms.to_string()),
            ("paced_seconds", timing.paced_seconds.to_string()),
        ],
    );
    Ok(())
}

fn devices(input: bool) -> Result<Vec<cpal::Device>, String> {
    let host = cpal::default_host();
    let list = if input { host.input_devices() } else { host.output_devices() };
    Ok(list.map_err(|e| e.to_string())?.collect())
}

fn name(device: &cpal::Device) -> String {
    device.name().unwrap_or_else(|_| "unknown".into())
}

/// Resolve a number from `devices` or part of a name; none means the system default.
fn find_device(input: bool, selector: &Option<String>) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    let Some(selector) = selector else {
        let device = if input { host.default_input_device() } else { host.default_output_device() };
        return device.ok_or("No default audio device".into());
    };
    let mut list = devices(input)?;
    let wanted = selector.to_lowercase();
    let index = match selector.parse::<usize>() {
        Ok(number) if number < list.len() => Some(number),
        _ => list.iter().position(|device| name(device).to_lowercase().contains(&wanted)),
    };
    index.map(|i| list.swap_remove(i)).ok_or(format!("No audio device matches {selector:?}"))
}

fn run_devices() -> Result<(), String> {
    for (label, input) in [("input", true), ("output", false)] {
        for (number, device) in devices(input)?.iter().enumerate() {
            println!("{label} {number}: {}", name(device));
        }
    }
    Ok(())
}

/// Everything the two audio callbacks and the status loop share.
struct Live {
    engine: Engine<Worker>,
    pending_in: Vec<i16>,
    pending_out: VecDeque<i16>,
    failure: Option<String>,
    input_peak: i32,
    underrun_frames: u64, // counted in samples; halved when reported
    drift_dropped_frames: u64,
    last_play: Option<f64>,
    max_callback_gap_ms: f64, // longest wait between output callbacks; a spike means a glitch
}

impl Live {
    /// Re-cut device input into exact 20 ms blocks and queue the filtered result.
    fn capture(&mut self, input: &[f32]) {
        for &sample in input {
            let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            self.input_peak = self.input_peak.max((value as i32).abs());
            self.pending_in.push(value);
        }
        let block = self.engine.block * 2;
        while self.failure.is_none() && self.pending_in.len() >= block {
            let pcm: Vec<i16> = self.pending_in.drain(..block).collect();
            let deadline = now();
            let rendered = self.engine.capture(&pcm).and_then(|_| {
                if self.engine.buffer.len() > self.engine.delay_blocks {
                    self.engine.render(deadline)
                } else {
                    self.engine.collect(deadline).map(|_| self.engine.silence())
                }
            });
            match rendered {
                Ok(out) => self.pending_out.extend(out),
                Err(error) => self.failure = Some(error),
            }
            // ponytail: input and output devices run on separate clocks. If output is the
            // slower one this queue grows, so drop a block (20 ms click) once it holds
            // 200 ms. Use an Aggregate Device with drift correction to avoid it entirely.
            if self.pending_out.len() > block * 10 {
                self.pending_out.drain(..block);
                self.drift_dropped_frames += self.engine.block as u64;
            }
        }
    }

    fn play(&mut self, output: &mut [f32]) {
        let time = now();
        if let Some(last) = self.last_play.replace(time) {
            self.max_callback_gap_ms = self.max_callback_gap_ms.max((time - last) * 1000.0);
        }
        for sample in output.iter_mut() {
            *sample = match self.pending_out.pop_front() {
                Some(value) => value as f32 / 32768.0,
                None => {
                    self.underrun_frames += 1;
                    0.0
                }
            };
        }
    }
}

fn run_live(opts: &Options) -> Result<(), String> {
    let input = find_device(true, &opts.input_device)?;
    let output = find_device(false, &opts.output_device)?;
    let engine = start_engine(opts, opts.rate)?;
    // Two blocks of silence up front so the output callback never finds the queue
    // empty merely because the two devices' callbacks are out of step.
    let cushion = vec![0i16; engine.block * 4];
    let live = Arc::new(Mutex::new(Live {
        engine,
        pending_in: Vec::new(),
        pending_out: cushion.into(),
        failure: None,
        input_peak: 0,
        underrun_frames: 0,
        drift_dropped_frames: 0,
        last_play: None,
        max_callback_gap_ms: 0.0,
    }));
    let config = cpal::StreamConfig {
        channels: 2,
        sample_rate: cpal::SampleRate(opts.rate as u32),
        buffer_size: cpal::BufferSize::Default,
    };
    let fail = |live: &Arc<Mutex<Live>>| {
        let live = live.clone();
        move |error: cpal::StreamError| live.lock().unwrap().failure = Some(error.to_string())
    };
    let (capture, playback) = (live.clone(), live.clone());
    let input_stream = input
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| capture.lock().unwrap().capture(data),
            fail(&live),
            None,
        )
        .map_err(|e| format!("{}: {e}", name(&input)))?;
    let output_stream = output
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| playback.lock().unwrap().play(data),
            fail(&live),
            None,
        )
        .map_err(|e| format!("{}: {e}", name(&output)))?;
    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;
    report(
        "priming",
        Some(&live.lock().unwrap().engine),
        &[("rate", opts.rate.to_string()), ("delay_ms", opts.delay_ms.to_string()), ("mode", quoted("live"))],
    );
    // Runs until Ctrl-C ends the process or the audio fails.
    loop {
        std::thread::sleep(Duration::from_secs(2));
        let mut live = live.lock().unwrap();
        if let Some(error) = live.failure.take() {
            return Err(error);
        }
        let line = status_line(
            progress(&live.engine),
            Some(&live.engine),
            &[
                ("input_peak", live.input_peak.to_string()),
                ("underrun_frames", (live.underrun_frames / 2).to_string()),
                ("drift_dropped_frames", live.drift_dropped_frames.to_string()),
                ("max_callback_gap_ms", rounded(live.max_callback_gap_ms).to_string()),
            ],
        );
        live.max_callback_gap_ms = 0.0;
        live.input_peak = 0;
        // Print only after releasing the lock: a slow terminal must never stall the audio callbacks.
        drop(live);
        eprintln!("{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("devices") => run_devices(),
        Some(mode @ ("wav" | "live")) => parse(mode, &args[1..]).and_then(|opts| match mode {
            "wav" => run_wav(&opts),
            _ => run_live(&opts),
        }),
        _ => Err(USAGE.into()),
    };
    if let Err(error) = result {
        report::<Worker>("failed", None, &[("error", quoted(&error))]);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::testing::{engine, tone};
    use std::cell::{Cell, RefCell};

    #[test]
    fn paced_wav_does_not_read_ahead_or_change_length() {
        let mut engine = engine();
        let total = engine.block * 100 + 71;
        let source = tone(total);
        let now = Cell::new(10.0);
        let (read_at, written_at) = (RefCell::new(Vec::new()), RefCell::new(Vec::new()));
        let mut position = 0;
        let mut written = Vec::new();
        let mut read = |count: usize| {
            read_at.borrow_mut().push(now.get());
            position += count * 2;
            Ok(source[position - count * 2..position].to_vec())
        };
        let mut write = |pcm: &[i16]| {
            written_at.borrow_mut().push(now.get());
            written.extend_from_slice(pcm);
            Ok(())
        };
        let mut wait = |deadline: f64| {
            assert!(deadline > now.get());
            now.set(deadline);
        };
        let timing = pace_wav(total, &mut read, &mut write, &mut engine, &|| now.get(), &mut wait).unwrap();
        assert_eq!(written, source);
        assert_eq!(timing.first_release_ms, Some(400.0));
        assert_eq!(read_at.borrow().len(), 101);
        for (released, read) in written_at.borrow().iter().zip(read_at.borrow().iter()) {
            assert!((released - read - 0.4).abs() < 1e-7);
        }
        let jobs = &engine.worker.jobs;
        let last = jobs.last().unwrap();
        assert_eq!(jobs[0].end, 16640);
        assert_eq!(last.end, total as i64);
        assert!(last.last);
        assert_eq!(last.blocks.iter().map(|b| b.len()).sum::<usize>(), total * 2);
    }

    #[test]
    fn options_are_validated() {
        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(parse("wav", &args(&["--model", "m", "a.wav", "b.wav"])).is_ok());
        assert!(parse("wav", &args(&["--model", "m", "a.wav"])).is_err());
        assert!(parse("wav", &args(&["a.wav", "b.wav"])).unwrap_err().contains("--model"));
        assert!(parse("live", &args(&["--model", "m", "--delay-ms", "1010"])).is_err());
        assert!(parse("wav", &args(&["--model", "m", "--rate", "48000", "a", "b"])).is_err());
    }
}
