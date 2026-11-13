//! Frame-clocked buffering and filtering. Capture and render have a single owner.
//! Positions are stereo frame counts; PCM is interleaved signed 16-bit stereo.
use std::collections::VecDeque;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

pub const BLOCK_MS: usize = 20;
pub const HOP_MS: usize = 320;
pub const WINDOW_MS: usize = 3200;
pub const MIN_WINDOW_MS: usize = 1040;

pub type Block = Arc<[i16]>;

/// Seconds on the one monotonic clock shared by deadlines and recognition results.
pub fn now() -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub term: String,
    pub start: i64,
    pub end: i64,
}

#[cfg(test)]
impl Word {
    pub fn new(term: &str, start: i64, end: i64) -> Self {
        Word { term: term.to_string(), start, end }
    }
}

pub struct Job {
    pub start: i64,
    pub end: i64,
    pub blocks: Vec<Block>,
    pub last: bool,
}

pub struct Outcome {
    pub start: i64,
    pub end: i64,
    pub last: bool,
    pub words: Vec<Word>,
    pub completed: f64,
    pub seconds: f64,
}

pub trait Recognition {
    fn submit(&mut self, job: Job);
    fn poll(&mut self) -> Option<Outcome>;
    fn error(&self) -> Option<String>;
    fn dropped(&self) -> u64;
}

#[derive(Default)]
struct Shared {
    pending: Option<Job>,
    stopping: bool,
    dropped: u64,
    error: Option<String>,
}

type Link = Arc<(Mutex<Shared>, Condvar)>;

/// Fail closed: a panic inside recognition must surface as an engine error,
/// not as audio that silently passes through unfiltered.
struct PanicGuard(Link);

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let mut shared = self.0 .0.lock().unwrap_or_else(|e| e.into_inner());
            shared.error = Some("Recognition thread panicked".into());
        }
    }
}

/// At most one active decode and one waiting snapshot.
pub struct Worker {
    link: Link,
    results: Receiver<Outcome>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn new<D>(mut decode: D) -> Self
    where
        D: FnMut(&[i16]) -> Result<Vec<Word>, String> + Send + 'static,
    {
        let link: Link = Arc::new((Mutex::new(Shared::default()), Condvar::new()));
        let (sender, results) = sync_channel(8);
        let shared = link.clone();
        let thread = std::thread::Builder::new()
            .name("local-recognition".into())
            .spawn(move || {
                let _guard = PanicGuard(shared.clone());
                loop {
                    let job = {
                        let (lock, ready) = &*shared;
                        let mut state = ready
                            .wait_while(lock.lock().unwrap(), |s| !s.stopping && s.pending.is_none())
                            .unwrap();
                        match state.pending.take() {
                            Some(job) => job,
                            None => return,
                        }
                    };
                    let start = now();
                    let pcm: Vec<i16> = job.blocks.iter().flat_map(|b| b.iter().copied()).collect();
                    let decoded = decode(&pcm);
                    let done = now();
                    let failure = match decoded {
                        Ok(words) => sender
                            .try_send(Outcome {
                                start: job.start,
                                end: job.end,
                                last: job.last,
                                words,
                                completed: done,
                                seconds: done - start,
                            })
                            .err()
                            .map(|_| "Recognition result queue overrun".to_string()),
                        Err(error) => Some(error),
                    };
                    if failure.is_some() {
                        shared.0.lock().unwrap().error = failure;
                        return;
                    }
                }
            })
            .expect("start recognition thread");
        Worker { link, results, thread: Some(thread) }
    }

    /// With `drain`, a waiting snapshot is still decoded before the thread exits.
    pub fn close(&mut self, drain: bool) {
        {
            let mut state = self.link.0.lock().unwrap_or_else(|e| e.into_inner());
            state.stopping = true;
            if !drain {
                state.pending = None;
            }
            self.link.1.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.close(false);
    }
}

impl Recognition for Worker {
    fn submit(&mut self, job: Job) {
        let mut state = self.link.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.pending.is_some() {
            state.dropped += 1;
        }
        state.pending = Some(job);
        self.link.1.notify_all();
    }

    fn poll(&mut self) -> Option<Outcome> {
        self.results.try_recv().ok()
    }

    fn error(&self) -> Option<String> {
        self.link.0.lock().unwrap_or_else(|e| e.into_inner()).error.clone()
    }

    fn dropped(&self) -> u64 {
        self.link.0.lock().unwrap_or_else(|e| e.into_inner()).dropped
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub windows: u64,
    pub max_decode_ms: f64,
    pub hits: i64,
    pub late_hits: i64,
    pub fully_late_hits: i64,
    pub muted_frames: u64,
    pub max_late_core_ms: f64,
    pub max_hit_age_ms: f64,
}

#[derive(Debug)]
pub struct Hit {
    pub word: Word,
    late: bool,
    fully_late: bool,
}

pub struct Engine<R: Recognition> {
    pub rate: usize,
    pub block: usize,
    pub delay_blocks: usize,
    hop_blocks: usize,
    history: VecDeque<Block>,
    window_blocks: usize,
    pub buffer: VecDeque<Block>,
    pub worker: R,
    pub stats: Stats,
    pub captured: i64,
    pub rendered: i64,
    block_count: usize,
    next_job: usize,
    pub hits: Vec<Hit>,
    next_result: Option<Outcome>,
    finished: bool,
    fade: i64,
    pre: i64,
    post: i64,
    edge: i64,
}

impl<R: Recognition> Engine<R> {
    pub fn new(rate: usize, delay_ms: usize, worker: R) -> Self {
        let ms = |milliseconds: usize| (rate * milliseconds / 1000) as i64;
        Engine {
            rate,
            block: rate * BLOCK_MS / 1000,
            delay_blocks: delay_ms / BLOCK_MS,
            hop_blocks: HOP_MS / BLOCK_MS,
            window_blocks: WINDOW_MS / BLOCK_MS,
            history: VecDeque::new(),
            buffer: VecDeque::new(),
            worker,
            stats: Stats::default(),
            captured: 0,
            rendered: 0,
            block_count: 0,
            next_job: MIN_WINDOW_MS / BLOCK_MS,
            hits: Vec::new(),
            next_result: None,
            finished: false,
            fade: ms(8),
            // Tuned on real recordings: drawn-out words start ~0.2 s before the first
            // token timestamp. 70 ms leaked onsets, 250 ms leaked one, 400 ms none.
            pre: ms(400),
            post: ms(90),
            edge: ms(120),
        }
    }

    /// Override the recognition window and the time between checks.
    pub fn with_timing(mut self, window_ms: usize, hop_ms: usize) -> Self {
        self.window_blocks = window_ms / BLOCK_MS;
        self.hop_blocks = hop_ms / BLOCK_MS;
        self
    }

    pub fn silence(&self) -> Vec<i16> {
        vec![0; self.block * 2]
    }

    pub fn capture(&mut self, pcm: &[i16]) -> Result<(), String> {
        let frames = pcm.len() / 2;
        if self.finished || frames == 0 || frames > self.block || pcm.len() % 2 != 0 {
            return Err("Invalid PCM capture block".into());
        }
        if self.captured != (self.block_count * self.block) as i64 {
            return Err("Short capture block is only supported at end of input".into());
        }
        let block: Block = pcm.into();
        if self.history.len() == self.window_blocks {
            self.history.pop_front();
        }
        self.history.push_back(block.clone());
        self.buffer.push_back(block);
        self.captured += frames as i64;
        self.block_count += 1;
        if self.block_count >= self.next_job && frames == self.block {
            self.submit(false);
            self.next_job += self.hop_blocks;
        }
        Ok(())
    }

    fn submit(&mut self, last: bool) {
        let blocks: Vec<Block> = self.history.iter().cloned().collect();
        let start = ((self.block_count - blocks.len()) * self.block) as i64;
        self.worker.submit(Job { start, end: self.captured, blocks, last });
    }

    pub fn finish(&mut self) {
        if !self.finished && self.captured > 0 {
            self.submit(true);
        }
        self.finished = true;
    }

    pub fn collect(&mut self, deadline: f64) -> Result<(), String> {
        if let Some(error) = self.worker.error() {
            return Err(format!("Recognition stopped: {error}"));
        }
        loop {
            if self.next_result.is_none() {
                self.next_result = self.worker.poll();
            }
            match self.next_result.take() {
                Some(result) if result.completed <= deadline => self.apply(result),
                waiting => {
                    self.next_result = waiting;
                    return Ok(());
                }
            }
        }
    }

    pub fn apply(&mut self, result: Outcome) {
        self.stats.windows += 1;
        self.stats.max_decode_ms = self.stats.max_decode_ms.max(result.seconds * 1000.0);
        let horizon = result.start.min(self.rendered);
        let (post, fade) = (self.post, self.fade);
        self.hits.retain(|hit| hit.word.end + post + fade > horizon);
        let length = result.end - result.start;
        for word in result.words {
            if !(0 <= word.start && word.start < word.end && word.end <= length) {
                continue;
            }
            if result.start != 0 && word.start < self.edge {
                continue;
            }
            if !result.last && word.end > length - self.edge {
                continue;
            }
            let word = Word {
                term: word.term,
                start: result.start + word.start,
                end: result.start + word.end,
            };
            self.add_hit(word, result.end);
        }
    }

    pub fn add_hit(&mut self, word: Word, captured_at_job: i64) {
        let (mut overlaps, kept): (Vec<Hit>, Vec<Hit>) = std::mem::take(&mut self.hits)
            .into_iter()
            .partition(|hit| {
                hit.word.term == word.term
                    && hit.word.start < word.end
                    && word.start < hit.word.end
            });
        self.hits = kept;
        overlaps.sort_by_key(|hit| hit.word.start);
        // Account only for newly identified core samples that already left the buffer.
        let late_end = word.end.min(self.rendered);
        let mut uncovered = (late_end - word.start).max(0);
        let mut covered_end = word.start;
        for hit in &overlaps {
            let start = covered_end.max(hit.word.start);
            let end = late_end.min(hit.word.end);
            if end > start {
                uncovered -= end - start;
                covered_end = end;
            }
        }
        let fully_late = word.end <= self.rendered && overlaps.iter().all(|hit| hit.fully_late);
        let late = uncovered > 0 || overlaps.iter().any(|hit| hit.late);
        let count = |flag: fn(&Hit) -> bool| overlaps.iter().filter(|hit| flag(hit)).count() as i64;
        self.stats.late_hits += late as i64 - count(|hit| hit.late);
        self.stats.fully_late_hits += fully_late as i64 - count(|hit| hit.fully_late);
        self.stats.hits += 1 - overlaps.len() as i64;
        let rate = self.rate as f64;
        self.stats.max_late_core_ms =
            self.stats.max_late_core_ms.max(uncovered as f64 * 1000.0 / rate);
        if overlaps.is_empty() {
            self.stats.max_hit_age_ms = self
                .stats
                .max_hit_age_ms
                .max((captured_at_job - word.start) as f64 * 1000.0 / rate);
        }
        let start = overlaps.iter().map(|hit| hit.word.start).fold(word.start, i64::min);
        let end = overlaps.iter().map(|hit| hit.word.end).fold(word.end, i64::max);
        self.hits.push(Hit { word: Word { term: word.term, start, end }, late, fully_late });
    }

    /// Render one due block. Late results cannot rewrite samples already emitted.
    pub fn render(&mut self, deadline: f64) -> Result<Vec<i16>, String> {
        self.collect(deadline)?;
        let pcm = self.buffer.pop_front().ok_or("Render without buffered audio")?;
        let count = (pcm.len() / 2) as i64;
        let start = self.rendered;
        let end = start + count;
        self.rendered = end;
        let fade = self.fade;
        let ranges: Vec<(i64, i64)> = self
            .hits
            .iter()
            .map(|hit| (hit.word.start - self.pre, hit.word.end + self.post))
            .filter(|&(low, high)| low - fade < end && high + fade > start)
            .collect();
        let mut samples = pcm.to_vec();
        if ranges.is_empty() {
            return Ok(samples);
        }
        let mut gains = vec![1.0f64; count as usize];
        for (low, high) in ranges {
            for frame in start.max(low - fade)..end.min(high + fade) {
                let gain = (low - frame).max(frame - high + 1).max(0) as f64 / fade as f64;
                let slot = &mut gains[(frame - start) as usize];
                *slot = slot.min(gain);
            }
        }
        for (frame, gain) in gains.into_iter().enumerate() {
            if gain < 1.0 {
                for sample in &mut samples[frame * 2..frame * 2 + 2] {
                    *sample = (*sample as f64 * gain).round_ties_even() as i16;
                }
                if gain == 0.0 {
                    self.stats.muted_frames += 1;
                }
            }
        }
        Ok(samples)
    }
}

#[cfg(test)]
pub mod testing {
    use super::*;

    #[derive(Default)]
    pub struct ManualWorker {
        pub results: VecDeque<Outcome>,
        pub jobs: Vec<Job>,
        pub error: Option<String>,
    }

    impl Recognition for ManualWorker {
        fn submit(&mut self, job: Job) {
            self.jobs.push(job);
        }
        fn poll(&mut self) -> Option<Outcome> {
            self.results.pop_front()
        }
        fn error(&self) -> Option<String> {
            self.error.clone()
        }
        fn dropped(&self) -> u64 {
            0
        }
    }

    pub fn pcm(frames: usize, left: i16, right: i16) -> Vec<i16> {
        [left, right].repeat(frames)
    }

    pub fn tone(frames: usize) -> Vec<i16> {
        pcm(frames, 12000, -6000)
    }

    pub fn engine() -> Engine<ManualWorker> {
        Engine::new(16000, 400, ManualWorker::default())
    }

    pub fn outcome(start: i64, end: i64, last: bool, words: Vec<Word>, completed: f64) -> Outcome {
        Outcome { start, end, last, words, completed, seconds: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn pcm_is_unchanged_outside_stereo_mute_and_fades() {
        let mut engine = engine();
        for _ in 0..28 {
            engine.capture(&tone(engine.block)).unwrap();
        }
        let captured = engine.captured;
        engine.add_hit(Word::new("bad", 6880, 7360), captured);
        let mut output = Vec::new();
        for _ in 0..28 {
            output.extend(engine.render(0.0).unwrap());
        }
        let frame = |index: usize| (output[2 * index], output[2 * index + 1]);
        assert_eq!(frame(352), (12000, -6000));
        assert_eq!(frame(416), (6000, -3000));
        assert_eq!(frame(480), (0, 0));
        assert_eq!(frame(2080), (0, 0));
        assert_eq!(frame(8799), (0, 0));
        assert_eq!(frame(8863), (6000, -3000));
        assert_eq!(frame(8927), (12000, -6000));
        assert_eq!(engine.stats.muted_frames, 8320);
    }

    #[test]
    fn window_history_is_bounded_and_offsets_are_absolute() {
        let mut engine = engine();
        for _ in 0..300 {
            engine.capture(&tone(engine.block)).unwrap();
            if engine.buffer.len() > engine.delay_blocks {
                engine.render(0.0).unwrap();
            }
        }
        let jobs = &engine.worker.jobs;
        let job = jobs.last().unwrap();
        assert!(job.start > 0);
        assert_eq!(job.end - job.start, 51200);
        assert_eq!(job.blocks.iter().map(|b| b.len()).sum::<usize>(), 51200 * 2);
        assert_eq!(engine.buffer.len(), 20);
        assert!(jobs.windows(2).all(|pair| pair[1].end - pair[0].end == 5120));
    }

    #[test]
    fn recognition_after_deadline_is_late_even_if_already_queued() {
        let mut engine = engine();
        for _ in 0..8 {
            engine.capture(&tone(engine.block)).unwrap();
        }
        let words = vec![Word::new("bad", 200, 600)];
        engine.worker.results.push_back(Outcome {
            seconds: 0.08,
            ..outcome(0, 2560, true, words, 2.01)
        });
        assert_eq!(engine.render(2.00).unwrap(), tone(320));
        assert_eq!(engine.render(2.02).unwrap(), pcm(320, 0, 0));
        assert_eq!(engine.stats.late_hits, 1);
        assert_eq!(engine.stats.fully_late_hits, 0);
        assert_eq!(engine.stats.max_late_core_ms, 7.5);
    }

    #[test]
    fn overlap_jitter_and_edge_filtering() {
        let mut engine = engine();
        engine.apply(outcome(
            3200,
            16000,
            false,
            vec![
                Word::new("bad", 500, 2900),
                Word::new("bad", 11000, 12500),
                Word::new("bad", 2000, 3400),
            ],
            0.0,
        ));
        engine.apply(outcome(4000, 16800, false, vec![Word::new("bad", 2050, 2700)], 0.0));
        assert_eq!(engine.stats.hits, 1);
        assert_eq!(engine.hits[0].word, Word::new("bad", 5200, 6700));
    }

    #[test]
    fn repeated_words_and_duplicate_late_reports() {
        let mut engine = engine();
        engine.rendered = 10000;
        engine.add_hit(Word::new("bad", 1000, 2000), 11000);
        engine.add_hit(Word::new("bad", 1050, 2100), 11500);
        engine.add_hit(Word::new("bad", 2500, 3300), 12000);
        assert_eq!(engine.stats.hits, 2);
        assert_eq!(engine.stats.late_hits, 2);
        assert_eq!(engine.stats.fully_late_hits, 2);
    }

    #[test]
    fn failed_recognition_stops_output_before_pop() {
        let mut engine = engine();
        engine.capture(&tone(engine.block)).unwrap();
        engine.worker.error = Some("test failure".into());
        assert!(engine.render(0.0).unwrap_err().contains("test failure"));
        assert_eq!(engine.rendered, 0);
        assert_eq!(engine.buffer.len(), 1);
    }

    fn job(end: i64, last: bool) -> Job {
        Job { start: 0, end, blocks: vec![vec![1i16, 1].into()], last }
    }

    #[test]
    fn worker_coalesces_backlog_without_growing_the_queue() {
        let (started, wait_started) = channel();
        let (release, wait_release) = channel::<()>();
        let mut worker = Worker::new(move |_| {
            started.send(()).ok();
            wait_release
                .recv_timeout(Duration::from_secs(2))
                .map(|_| Vec::new())
                .or_else(|_| Ok(Vec::new()))
        });
        worker.submit(job(1, false));
        wait_started.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.submit(job(2, false));
        worker.submit(job(3, true));
        release.send(()).unwrap();
        drop(release);
        worker.close(true);
        assert_eq!(worker.error(), None);
        assert_eq!(worker.dropped(), 1);
        assert_eq!(worker.poll().unwrap().end, 1);
        assert_eq!(worker.poll().unwrap().end, 3);
    }

    #[test]
    fn worker_panic_becomes_an_engine_error() {
        let mut worker = Worker::new(|_| panic!("decode blew up"));
        worker.submit(job(1, false));
        worker.close(true);
        assert!(worker.error().unwrap().contains("panicked"));
    }
}
