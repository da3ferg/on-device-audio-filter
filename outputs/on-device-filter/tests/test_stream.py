import io
from array import array
from queue import Queue
from threading import Event
import unittest
import wave

from audio_filter import pace_wav
from engine import Engine, Job, Result, Word, Worker
from recognition import matched_words


class ManualWorker:
    def __init__(self):
        self.results = Queue()
        self.jobs = []
        self.error = None
        self.dropped = 0

    def submit(self, job):
        self.jobs.append(job)


def pcm(frames, left=12000, right=-6000):
    return array("h", [left, right] * frames).tobytes()


def wav_source(frames):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((2, 2, 16000, frames, "NONE", "not compressed"))
        wav.writeframes(pcm(frames))
    stream.seek(0)
    return stream


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.worker = ManualWorker()
        self.engine = Engine(16000, 400, self.worker)

    def test_pcm_is_unchanged_outside_stereo_mute_and_fades(self):
        engine = self.engine
        for _ in range(28):
            engine.capture(pcm(engine.block))
        engine.add_hit(Word(b"bad", 6880, 7360), engine.captured)
        output = array("h")
        for _ in range(28):
            output.frombytes(engine.render(0))

        def frame(index):
            return tuple(output[2 * index:2 * index + 2])

        self.assertEqual(frame(352), (12000, -6000))
        self.assertEqual(frame(416), (6000, -3000))
        self.assertEqual(frame(480), (0, 0))
        self.assertEqual(frame(2080), (0, 0))
        self.assertEqual(frame(8799), (0, 0))
        self.assertEqual(frame(8863), (6000, -3000))
        self.assertEqual(frame(8927), (12000, -6000))
        self.assertEqual(engine.stats.muted_frames, 8320)

    def test_paced_wav_does_not_read_ahead_or_change_length(self):
        engine = self.engine
        total = engine.block * 100 + 71
        now = [10.0]
        read = []
        written = []

        def wait(deadline):
            self.assertGreater(deadline, now[0])
            now[0] = deadline

        class Source:
            def __init__(self, wav):
                self.wav = wav

            def getnframes(self):
                return self.wav.getnframes()

            def readframes(self, count):
                read.append(now[0])
                return self.wav.readframes(count)

        class Target:
            def writeframesraw(self, block):
                written.append((now[0], block))

        with wave.open(wav_source(total), "rb") as wav:
            metrics = pace_wav(Source(wav), Target(), engine, lambda: now[0], wait)
        self.assertEqual(b"".join(block for _, block in written), pcm(total))
        self.assertEqual(metrics["first_release_delay_ms"], 400)
        self.assertEqual(len(read), 101)
        for index, (timestamp, _) in enumerate(written):
            self.assertAlmostEqual(timestamp - read[index], 0.4)
        self.assertEqual(self.worker.jobs[0].end, 16640)
        self.assertEqual(self.worker.jobs[-1].end, total)
        self.assertTrue(self.worker.jobs[-1].final)
        self.assertEqual(sum(map(len, self.worker.jobs[-1].blocks)), total * 4)

    def test_window_history_is_bounded_and_offsets_are_absolute(self):
        engine = self.engine
        for _ in range(300):
            engine.capture(pcm(engine.block))
            if len(engine.buffer) > engine.delay_blocks:
                engine.render(0)
        job = self.worker.jobs[-1]
        self.assertGreater(job.start, 0)
        self.assertEqual(job.end - job.start, 51200)
        self.assertEqual(sum(map(len, job.blocks)), 51200 * 4)
        self.assertEqual(len(engine.buffer), 20)
        self.assertTrue(all(b.end - a.end == 5120
                            for a, b in zip(self.worker.jobs, self.worker.jobs[1:])))

    def test_recognition_after_deadline_is_late_even_if_already_queued(self):
        engine = self.engine
        for _ in range(8):
            engine.capture(pcm(engine.block))
        job = Job(0, 2560, (), True)
        self.worker.results.put(Result(job, (Word(b"bad", 200, 600),), 2.01, 0.08))
        self.assertEqual(engine.render(2.00), pcm(320))
        self.assertEqual(engine.render(2.02), pcm(320, 0, 0))
        self.assertEqual(engine.stats.late_hits, 1)
        self.assertEqual(engine.stats.fully_late_hits, 0)
        self.assertEqual(engine.stats.max_late_core_ms, 7.5)

    def test_overlap_jitter_and_edge_filtering(self):
        engine = self.engine
        engine.apply(Result(Job(3200, 16000, (), False),
                            (Word(b"bad", 500, 2900), Word(b"bad", 11000, 12500),
                             Word(b"bad", 2000, 3400)), 0, 0))
        engine.apply(Result(Job(4000, 16800, (), False),
                            (Word(b"bad", 2050, 2700),), 0, 0))
        self.assertEqual(engine.stats.hits, 1)
        self.assertEqual(engine.hits[0].word, Word(b"bad", 5200, 6700))

    def test_repeated_words_and_duplicate_late_reports(self):
        engine = self.engine
        engine.rendered = 10000
        engine.add_hit(Word(b"bad", 1000, 2000), 11000)
        engine.add_hit(Word(b"bad", 1050, 2100), 11500)
        engine.add_hit(Word(b"bad", 2500, 3300), 12000)
        self.assertEqual(engine.stats.hits, 2)
        self.assertEqual(engine.stats.late_hits, 2)
        self.assertEqual(engine.stats.fully_late_hits, 2)

    def test_failed_recognition_stops_output_before_pop(self):
        self.engine.capture(pcm(self.engine.block))
        self.worker.error = RuntimeError("test failure")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            self.engine.render(0)
        self.assertEqual(self.engine.rendered, 0)
        self.assertEqual(len(self.engine.buffer), 1)

    def test_worker_coalesces_backlog_without_growing_the_queue(self):
        started = Event()
        release = Event()

        def decode(data):
            started.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test synchronization timeout")
            return ()

        worker = Worker(decode)
        try:
            worker.submit(Job(0, 1, (b"1",), False))
            self.assertTrue(started.wait(timeout=2))
            worker.submit(Job(0, 2, (b"2",), False))
            worker.submit(Job(0, 3, (b"3",), True))
            release.set()
            worker.close(drain=True)
            self.assertIsNone(worker.error)
            self.assertEqual(worker.dropped, 1)
            self.assertEqual(worker.results.get_nowait().job.end, 1)
            self.assertEqual(worker.results.get_nowait().job.end, 3)
        finally:
            release.set()
            worker.close()

    def test_vocabulary_matches_whole_word_across_tokens(self):
        tokens = [(b" Sh", 10, 20), (b"it", 20, 35), (b"!", 35, 36),
                  (b" shitake", 40, 90), (b" shit", -1, -1)]
        self.assertEqual(list(matched_words(tokens, {b"shit"})),
                         [Word(b"shit", 10, 35)])


if __name__ == "__main__":
    unittest.main()
