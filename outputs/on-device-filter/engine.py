"""Frame-clocked buffering and filtering. Capture and render have a single owner."""
from array import array
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Condition, Thread
from time import monotonic

FRAME_BYTES = 4
BLOCK_MS = 20
HOP_MS = 320
WINDOW_MS = 3200
MIN_WINDOW_MS = 1040


@dataclass(frozen=True)
class Word:
    term: bytes
    start: int
    end: int


@dataclass(frozen=True)
class Job:
    start: int
    end: int
    blocks: tuple
    final: bool


@dataclass(frozen=True)
class Result:
    job: Job
    words: tuple
    completed: float
    seconds: float


class Worker:
    """At most one active decode and one waiting snapshot."""
    def __init__(self, decode):
        self.decode = decode
        self.lock = Condition()
        self.pending = None
        self.results = Queue(maxsize=8)
        self.error = None
        self.dropped = 0
        self.stopping = False
        self.thread = Thread(target=self.run, name="local-recognition")
        self.thread.start()

    def submit(self, job):
        with self.lock:
            if self.pending is not None:
                self.dropped += 1
            self.pending = job
            self.lock.notify()

    def run(self):
        try:
            while True:
                with self.lock:
                    self.lock.wait_for(lambda: self.stopping or self.pending is not None)
                    if self.pending is None:
                        return
                    job, self.pending = self.pending, None
                start = monotonic()
                words = self.decode(b"".join(job.blocks))
                done = monotonic()
                self.results.put_nowait(Result(job, words, done, done - start))
        except Full:
            self.error = RuntimeError("Recognition result queue overrun")
        except Exception as error:
            self.error = error

    def close(self, drain=False):
        with self.lock:
            self.stopping = True
            if not drain:
                self.pending = None
            self.lock.notify()
        self.thread.join()


@dataclass
class Stats:
    windows: int = 0
    max_decode_ms: float = 0
    hits: int = 0
    late_hits: int = 0
    fully_late_hits: int = 0
    muted_frames: int = 0
    max_late_core_ms: float = 0
    max_hit_age_ms: float = 0


@dataclass
class Hit:
    word: Word
    late: bool
    fully_late: bool


class Engine:
    def __init__(self, rate, delay_ms, worker, window_ms=WINDOW_MS):
        self.rate = rate
        self.block = rate * BLOCK_MS // 1000
        self.delay_blocks = delay_ms // BLOCK_MS
        self.window_blocks = window_ms // BLOCK_MS
        self.hop_blocks = HOP_MS // BLOCK_MS
        self.history = deque(maxlen=self.window_blocks)
        self.buffer = deque()
        self.silence = bytes(self.block * FRAME_BYTES)
        self.worker = worker
        self.stats = Stats()
        self.captured = 0
        self.rendered = 0
        self.block_count = 0
        self.next_job = MIN_WINDOW_MS // BLOCK_MS
        self.hits = []
        self.next_result = None
        self.finished = False
        self.fade = self.ms(8)
        # Tuned on real recordings: drawn-out words start ~0.2 s before the first
        # token timestamp. 70 ms leaked onsets, 250 ms leaked one, 400 ms none.
        self.pre = self.ms(400)
        self.post = self.ms(90)
        self.edge = self.ms(120)

    def ms(self, milliseconds):
        return self.rate * milliseconds // 1000

    def capture(self, pcm):
        frames = len(pcm) // FRAME_BYTES
        if self.finished or not 0 < frames <= self.block or len(pcm) % FRAME_BYTES:
            raise ValueError("Invalid PCM capture block")
        if self.captured != self.block_count * self.block:
            raise ValueError("Short capture block is only supported at end of input")
        self.history.append(pcm)
        self.buffer.append(pcm)
        self.captured += frames
        self.block_count += 1
        if self.block_count >= self.next_job and frames == self.block:
            self.submit()
            self.next_job += self.hop_blocks

    def submit(self, final=False):
        blocks = tuple(self.history)
        start = max(0, self.block_count - len(blocks)) * self.block
        self.worker.submit(Job(start, self.captured, blocks, final))

    def finish(self):
        if not self.finished and self.captured:
            self.submit(final=True)
        self.finished = True

    def collect(self, deadline):
        if self.worker.error is not None:
            raise RuntimeError(f"Recognition stopped: {self.worker.error}")
        while True:
            if self.next_result is None:
                try:
                    self.next_result = self.worker.results.get_nowait()
                except Empty:
                    break
            if self.next_result.completed > deadline:
                break
            result, self.next_result = self.next_result, None
            self.apply(result)

    def apply(self, result):
        job = result.job
        self.stats.windows += 1
        self.stats.max_decode_ms = max(self.stats.max_decode_ms, result.seconds * 1000)
        self.hits = [
            hit for hit in self.hits
            if hit.word.end + self.post + self.fade > min(job.start, self.rendered)
        ]
        for word in result.words:
            if not 0 <= word.start < word.end <= job.end - job.start:
                continue
            if job.start and word.start < self.edge:
                continue
            if not job.final and word.end > job.end - job.start - self.edge:
                continue
            self.add_hit(Word(word.term, job.start + word.start, job.start + word.end), job.end)

    def add_hit(self, word, captured_at_job):
        overlaps = [
            hit for hit in self.hits if hit.word.term == word.term
            and hit.word.start < word.end and word.start < hit.word.end
        ]
        # Account only for newly identified core samples that already left the buffer.
        late_end = min(word.end, self.rendered)
        uncovered = max(0, late_end - word.start)
        covered_end = word.start
        for hit in sorted(overlaps, key=lambda match: match.word.start):
            start = max(covered_end, hit.word.start)
            end = min(late_end, hit.word.end)
            if end > start:
                uncovered -= end - start
                covered_end = end

        fully_late = word.end <= self.rendered and all(hit.fully_late for hit in overlaps)
        late = uncovered > 0 or any(hit.late for hit in overlaps)
        self.stats.late_hits += int(late) - sum(hit.late for hit in overlaps)
        self.stats.fully_late_hits += int(fully_late) - sum(hit.fully_late for hit in overlaps)
        self.stats.hits += 1 - len(overlaps)
        self.stats.max_late_core_ms = max(
            self.stats.max_late_core_ms, uncovered * 1000 / self.rate
        )
        if not overlaps:
            self.stats.max_hit_age_ms = max(
                self.stats.max_hit_age_ms, (captured_at_job - word.start) * 1000 / self.rate
            )
        start = min([word.start] + [hit.word.start for hit in overlaps])
        end = max([word.end] + [hit.word.end for hit in overlaps])
        for hit in overlaps:
            self.hits.remove(hit)
        self.hits.append(Hit(Word(word.term, start, end), late, fully_late))

    def render(self, deadline):
        """Render one due block. Late results cannot rewrite samples already emitted."""
        self.collect(deadline)
        pcm = self.buffer.popleft()
        count = len(pcm) // FRAME_BYTES
        start = self.rendered
        end = start + count
        self.rendered = end
        ranges = []
        for hit in self.hits:
            low = hit.word.start - self.pre
            high = hit.word.end + self.post
            if low - self.fade < end and high + self.fade > start:
                ranges.append((low, high))
        if not ranges:
            return pcm

        gains = [1.0] * count
        for low, high in ranges:
            for frame in range(max(start, low - self.fade), min(end, high + self.fade)):
                gain = max(low - frame, frame - high + 1, 0) / self.fade
                offset = frame - start
                gains[offset] = min(gains[offset], gain)
        samples = array("h")
        samples.frombytes(pcm)
        for frame, gain in enumerate(gains):
            if gain < 1:
                index = frame * 2
                samples[index] = round(samples[index] * gain)
                samples[index + 1] = round(samples[index + 1] * gain)
                if gain == 0:
                    self.stats.muted_frames += 1
        return samples.tobytes()
