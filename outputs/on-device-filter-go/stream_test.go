package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

type manualWorker struct {
	results chan Result
	jobs    []Job
	err     error
}

func (w *manualWorker) Submit(job Job)         { w.jobs = append(w.jobs, job) }
func (w *manualWorker) Results() <-chan Result { return w.results }
func (w *manualWorker) Err() error             { return w.err }
func (w *manualWorker) Dropped() int           { return 0 }

func setup() (*Engine, *manualWorker) {
	worker := &manualWorker{results: make(chan Result, 8)}
	return NewEngine(16000, 400, worker, windowMS), worker
}

func pcm(frames int, left, right int16) []byte {
	out := make([]byte, 0, frames*frameBytes)
	for range frames {
		out = binary.LittleEndian.AppendUint16(out, uint16(left))
		out = binary.LittleEndian.AppendUint16(out, uint16(right))
	}
	return out
}

func tone(frames int) []byte { return pcm(frames, 12000, -6000) }

func equal(t *testing.T, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestPCMIsUnchangedOutsideStereoMuteAndFades(t *testing.T) {
	engine, _ := setup()
	for range 28 {
		engine.Capture(tone(engine.block))
	}
	engine.addHit(Word{"bad", 6880, 7360}, engine.captured)
	var output []byte
	for range 28 {
		block, _ := engine.Render(0)
		output = append(output, block...)
	}
	frame := func(index int) [2]int16 {
		return [2]int16{int16(binary.LittleEndian.Uint16(output[4*index:])),
			int16(binary.LittleEndian.Uint16(output[4*index+2:]))}
	}
	equal(t, frame(352), [2]int16{12000, -6000})
	equal(t, frame(416), [2]int16{6000, -3000})
	equal(t, frame(480), [2]int16{0, 0})
	equal(t, frame(2080), [2]int16{0, 0})
	equal(t, frame(8799), [2]int16{0, 0})
	equal(t, frame(8863), [2]int16{6000, -3000})
	equal(t, frame(8927), [2]int16{12000, -6000})
	equal(t, engine.stats.MutedFrames, 8320)
}

type tracedSource struct {
	wav  *wavReader
	now  *float64
	read *[]float64
}

func (s tracedSource) Frames() int { return s.wav.Frames() }
func (s tracedSource) ReadFrames(count int) ([]byte, error) {
	*s.read = append(*s.read, *s.now)
	return s.wav.ReadFrames(count)
}

type tracedTarget struct {
	now     *float64
	times   *[]float64
	written *bytes.Buffer
}

func (t tracedTarget) Write(block []byte) (int, error) {
	*t.times = append(*t.times, *t.now)
	return t.written.Write(block)
}

func TestPacedWavDoesNotReadAheadOrChangeLength(t *testing.T) {
	engine, worker := setup()
	total := engine.block*100 + 71
	now := 10.0
	var read, times []float64
	var written bytes.Buffer
	wav, err := openWav(bytes.NewReader(append(wavHeader(16000, total), tone(total)...)))
	if err != nil {
		t.Fatal(err)
	}
	wait := func(deadline float64) {
		if deadline <= now {
			t.Fatal("deadline did not advance")
		}
		now = deadline
	}
	metrics, err := paceWav(tracedSource{wav, &now, &read}, tracedTarget{&now, &times, &written},
		engine, func() float64 { return now }, wait)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(written.Bytes(), tone(total)) {
		t.Fatal("output differs from input")
	}
	equal(t, metrics["first_release_delay_ms"], any(400.0))
	equal(t, len(read), 101)
	for index, timestamp := range times {
		if math.Abs(timestamp-read[index]-0.4) > 1e-7 {
			t.Fatalf("block %d released after %v s", index, timestamp-read[index])
		}
	}
	last := worker.jobs[len(worker.jobs)-1]
	equal(t, worker.jobs[0].End, 16640)
	equal(t, last.End, total)
	equal(t, last.Final, true)
	equal(t, len(bytes.Join(last.Blocks, nil)), total*4)
}

func TestWindowHistoryIsBoundedAndOffsetsAreAbsolute(t *testing.T) {
	engine, worker := setup()
	for range 300 {
		engine.Capture(tone(engine.block))
		if len(engine.buffer) > engine.delayBlocks {
			engine.Render(0)
		}
	}
	job := worker.jobs[len(worker.jobs)-1]
	if job.Start <= 0 {
		t.Fatal("window did not slide")
	}
	equal(t, job.End-job.Start, 51200)
	equal(t, len(bytes.Join(job.Blocks, nil)), 51200*4)
	equal(t, len(engine.buffer), 20)
	for i := 1; i < len(worker.jobs); i++ {
		equal(t, worker.jobs[i].End-worker.jobs[i-1].End, 5120)
	}
}

func TestRecognitionAfterDeadlineIsLateEvenIfAlreadyQueued(t *testing.T) {
	engine, worker := setup()
	for range 8 {
		engine.Capture(tone(engine.block))
	}
	worker.results <- Result{Job{0, 2560, nil, true}, []Word{{"bad", 200, 600}}, 2.01, 0.08}
	first, _ := engine.Render(2.00)
	second, _ := engine.Render(2.02)
	equal(t, first, tone(320))
	equal(t, second, pcm(320, 0, 0))
	equal(t, engine.stats.LateHits, 1)
	equal(t, engine.stats.FullyLateHits, 0)
	equal(t, engine.stats.MaxLateCoreMS, 7.5)
}

func TestOverlapJitterAndEdgeFiltering(t *testing.T) {
	engine, _ := setup()
	engine.apply(Result{Job: Job{3200, 16000, nil, false},
		Words: []Word{{"bad", 500, 2900}, {"bad", 11000, 12500}, {"bad", 2000, 3400}}})
	engine.apply(Result{Job: Job{4000, 16800, nil, false}, Words: []Word{{"bad", 2050, 2700}}})
	equal(t, engine.stats.Hits, 1)
	equal(t, engine.hits[0].word, Word{"bad", 5200, 6700})
}

func TestRepeatedWordsAndDuplicateLateReports(t *testing.T) {
	engine, _ := setup()
	engine.rendered = 10000
	engine.addHit(Word{"bad", 1000, 2000}, 11000)
	engine.addHit(Word{"bad", 1050, 2100}, 11500)
	engine.addHit(Word{"bad", 2500, 3300}, 12000)
	equal(t, engine.stats.Hits, 2)
	equal(t, engine.stats.LateHits, 2)
	equal(t, engine.stats.FullyLateHits, 2)
}

func TestFailedRecognitionStopsOutputBeforePop(t *testing.T) {
	engine, worker := setup()
	engine.Capture(tone(engine.block))
	worker.err = errors.New("test failure")
	if _, err := engine.Render(0); err == nil || !strings.Contains(err.Error(), "test failure") {
		t.Fatalf("got %v", err)
	}
	equal(t, engine.rendered, 0)
	equal(t, len(engine.buffer), 1)
}

func TestWorkerCoalescesBacklogWithoutGrowingTheQueue(t *testing.T) {
	started := make(chan struct{}, 3)
	release := make(chan struct{})
	worker := NewWorker(func([]byte) ([]Word, error) {
		started <- struct{}{}
		select {
		case <-release:
			return nil, nil
		case <-time.After(2 * time.Second):
			return nil, errors.New("test synchronization timeout")
		}
	})
	worker.Submit(Job{0, 1, [][]byte{{1}}, false})
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("decode did not start")
	}
	worker.Submit(Job{0, 2, [][]byte{{2}}, false})
	worker.Submit(Job{0, 3, [][]byte{{3}}, true})
	close(release)
	worker.Close(true)
	equal(t, worker.Err(), error(nil))
	equal(t, worker.Dropped(), 1)
	equal(t, (<-worker.results).Job.End, 1)
	equal(t, (<-worker.results).Job.End, 3)
}

func TestVocabularyMatchesWholeWordAcrossTokens(t *testing.T) {
	tokens := []token{{[]byte(" Sh"), 10, 20}, {[]byte("it"), 20, 35}, {[]byte("!"), 35, 36},
		{[]byte(" shitake"), 40, 90}, {[]byte(" shit"), -1, -1}}
	equal(t, matchedWords(tokens, map[string]bool{"shit": true}), []Word{{"shit", 10, 35}})
}

func TestBuiltInWordListLoads(t *testing.T) {
	entries, err := vocabulary("words.txt", defaultWords)
	if err != nil || len(entries) == 0 {
		t.Fatalf("got %d entries, error %v", len(entries), err)
	}
}
