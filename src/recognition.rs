//! Local Zipformer adapter (non-streaming GigaSpeech model; LibriSpeech-trained
//! models transcribe profanity as clean look-alikes). Only matching words and
//! timestamps leave this module.
use crate::engine::Word;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::path::Path;

/// Byte ranges of words: letters, optionally joined by single apostrophes ("don't").
fn word_spans(text: &[u8]) -> Vec<(usize, usize)> {
    let letter = |i: usize| text.get(i).is_some_and(u8::is_ascii_alphabetic);
    let mut spans = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if !letter(i) {
            i += 1;
            continue;
        }
        let start = i;
        loop {
            while letter(i) {
                i += 1;
            }
            if text.get(i) == Some(&b'\'') && letter(i + 1) {
                i += 1;
            } else {
                break;
            }
        }
        spans.push((start, i));
    }
    spans
}

/// Parse a word list: one English word per line, `#` starts a comment.
pub fn vocabulary(name: &str, text: &str) -> Result<HashSet<String>, String> {
    let mut entries = HashSet::new();
    for (number, line) in text.lines().enumerate() {
        let entry = line.split('#').next().unwrap_or("").trim().to_ascii_lowercase();
        if entry.is_empty() {
            continue;
        }
        if word_spans(entry.as_bytes()) != [(0, entry.len())] {
            return Err(format!("{name}:{} must contain one English word", number + 1));
        }
        entries.insert(entry);
    }
    if entries.is_empty() {
        return Err("Vocabulary is empty".into());
    }
    Ok(entries)
}

pub struct Token {
    pub text: String,
    pub start: i64,
    pub end: i64,
}

/// Match reconstructed words, including words split across BPE tokens.
pub fn matched_words(tokens: &[Token], entries: &HashSet<String>) -> Vec<Word> {
    let mut contents = Vec::new();
    let mut timing = Vec::new();
    for token in tokens {
        contents.extend_from_slice(token.text.as_bytes());
        timing.resize(contents.len(), (token.start, token.end));
    }
    let mut words = Vec::new();
    for (from, to) in word_spans(&contents) {
        let term = String::from_utf8_lossy(&contents[from..to]).to_ascii_lowercase();
        let bounds = &timing[from..to];
        if !entries.contains(&term) || bounds.iter().any(|&(start, end)| start < 0 || end < start) {
            continue;
        }
        let start = bounds.iter().map(|b| b.0).min().unwrap();
        let end = bounds.iter().map(|b| b.1).max().unwrap();
        if start < end {
            words.push(Word { term, start, end });
        }
    }
    words
}

const KERNEL_RADIUS: i64 = 40;

fn decimation_kernel(rate: usize) -> Vec<f32> {
    let cutoff = 7200.0 / rate as f64;
    let weights: Vec<f64> = (-KERNEL_RADIUS..=KERNEL_RADIUS)
        .map(|i| {
            let x = i as f64;
            let sinc = if i == 0 { 2.0 * cutoff } else { (2.0 * PI * cutoff * x).sin() / (PI * x) };
            sinc * (0.54 + 0.46 * (PI * x / KERNEL_RADIUS as f64).cos())
        })
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| (w / total) as f32).collect()
}

pub struct Zipformer {
    recognizer: OfflineRecognizer,
    entries: HashSet<String>,
    rate: usize,
    stride: usize,
    kernel: Vec<f32>,
}

impl Zipformer {
    pub fn new(model: &Path, entries: HashSet<String>, rate: usize, threads: i32) -> Result<Self, String> {
        let file = |name: &str| -> Result<Option<String>, String> {
            let path = model.join(name);
            if !path.is_file() {
                return Err(format!("Local model file missing: {}", path.display()));
            }
            Ok(Some(path.to_string_lossy().into_owned()))
        };
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer = OfflineTransducerModelConfig {
            encoder: file("encoder-epoch-30-avg-1.int8.onnx")?,
            decoder: file("decoder-epoch-30-avg-1.int8.onnx")?,
            joiner: file("joiner-epoch-30-avg-1.int8.onnx")?,
        };
        config.model_config.tokens = file("tokens.txt")?;
        config.model_config.num_threads = threads;
        config.model_config.provider = Some("cpu".into());
        config.decoding_method = Some("greedy_search".into());
        let recognizer = OfflineRecognizer::create(&config).ok_or("Local model could not be loaded")?;
        Ok(Zipformer { recognizer, entries, rate, stride: rate / 16000, kernel: decimation_kernel(rate) })
    }

    /// Average the stereo pair and low-pass decimate to 16 kHz, scaled to ±1.
    fn mono16k(&self, pcm: &[i16]) -> Vec<f32> {
        let mono: Vec<f32> =
            pcm.chunks_exact(2).map(|f| (f[0] as f32 + f[1] as f32) / 2.0 / 32768.0).collect();
        if self.stride == 1 {
            return mono;
        }
        (0..mono.len())
            .step_by(self.stride)
            .map(|center| {
                self.kernel
                    .iter()
                    .enumerate()
                    .filter_map(|(k, weight)| {
                        let i = center as i64 + KERNEL_RADIUS - k as i64;
                        mono.get(usize::try_from(i).ok()?).map(|sample| weight * sample)
                    })
                    .sum()
            })
            .collect()
    }

    pub fn decode(&self, pcm: &[i16]) -> Result<Vec<Word>, String> {
        let frames = (pcm.len() / 2) as i64;
        let rate = self.rate as i64;
        if frames < rate / 10 {
            return Ok(Vec::new());
        }
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16000, &self.mono16k(pcm));
        self.recognizer.decode(&stream);
        let result = stream.get_result().ok_or("Recognition returned no result")?;
        let starts: Vec<i64> = result
            .timestamps
            .unwrap_or_default()
            .iter()
            .map(|&t| (t as f64 * rate as f64).round_ties_even() as i64)
            .collect();
        if starts.len() != result.tokens.len() {
            return Err("Recognition returned inconsistent token timestamps".into());
        }
        let tokens: Vec<Token> = result
            .tokens
            .iter()
            .enumerate()
            .map(|(i, text)| {
                // The model reports onsets. Bound the last-piece estimate when the next
                // token is absent, coincident, or separated by silence.
                let mut end = (starts[i] + rate * 180 / 1000).min(frames);
                if let Some(&next) = starts.get(i + 1).filter(|&&next| next > starts[i]) {
                    end = end.min(next);
                }
                Token { text: text.replace('▁', " "), start: starts[i], end }
            })
            .collect();
        Ok(matched_words(&tokens, &self.entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_matches_whole_word_across_tokens() {
        let token = |text: &str, start, end| Token { text: text.into(), start, end };
        let tokens = [
            token(" Sh", 10, 20),
            token("it", 20, 35),
            token("!", 35, 36),
            token(" shitake", 40, 90),
            token(" shit", -1, -1),
        ];
        let entries = HashSet::from(["shit".to_string()]);
        assert_eq!(matched_words(&tokens, &entries), [Word::new("shit", 10, 35)]);
    }

    #[test]
    fn built_in_word_list_loads() {
        assert!(!vocabulary("words.txt", include_str!("../words.txt")).unwrap().is_empty());
    }

    #[test]
    fn word_list_rejects_lines_that_are_not_one_word() {
        assert!(vocabulary("w", "two words").unwrap_err().contains("w:1"));
        assert_eq!(vocabulary("w", "Don't # ok\n\n").unwrap(), HashSet::from(["don't".to_string()]));
    }
}
