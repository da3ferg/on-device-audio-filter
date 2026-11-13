// Frame-clocked buffering and filtering. Capture and render have a single owner.
package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

const (
	frameBytes  = 4
	blockMS     = 20
	hopMS       = 320
	windowMS    = 3200
	minWindowMS = 1040
)

var processStart = time.Now()

// monotonic returns seconds on the clock every deadline in this program uses.
func monotonic() float64 { return time.Since(processStart).Seconds() }

type Word struct {
	Term       string
	Start, End int
}

type Job struct {
	Start, End int
	Blocks     [][]byte
	Final      bool
}

type Result struct {
	Job       Job
	Words     []Word
	Completed float64
	Seconds   float64
}

// recognition is what the engine needs from a worker; tests use a manual one.
type recognition interface {
	Submit(Job)
	Results() <-chan Result
	Err() error
	Dropped() int
}

// Worker runs at most one active decode and keeps one waiting snapshot.
type Worker struct {
	decode   func([]byte) ([]Word, error)
	mu       sync.Mutex
	cond     *sync.Cond
	pending  *Job
	results  chan Result
	err      error
	dropped  int
	stopping bool
	done     chan struct{}
}

func NewWorker(decode func([]byte) ([]Word, error)) *Worker {
	w := &Worker{decode: decode, results: make(chan Result, 8), done: make(chan struct{})}
	w.cond = sync.NewCond(&w.mu)
	go w.run()
	return w
}

func (w *Worker) Submit(job Job) {
	w.mu.Lock()
	if w.pending != nil {
		w.dropped++
	}
	w.pending = &job
	w.cond.Signal()
	w.mu.Unlock()
}

func (w *Worker) Results() <-chan Result { return w.results }

func (w *Worker) Err() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.err
}

func (w *Worker) Dropped() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.dropped
}

func (w *Worker) run() {
	defer close(w.done)
	for {
		w.mu.Lock()
		for !w.stopping && w.pending == nil {
			w.cond.Wait()
		}
		if w.pending == nil {
			w.mu.Unlock()
			return
		}
		job := *w.pending
		w.pending = nil
		w.mu.Unlock()

		start := monotonic()
		words, err := w.decode(bytes.Join(job.Blocks, nil))
		done := monotonic()
		if err == nil {
			select {
			case w.results <- Result{job, words, done, done - start}:
			default:
				err = errors.New("Recognition result queue overrun")
			}
		}
		if err != nil {
			w.mu.Lock()
			w.err = err
			w.mu.Unlock()
			return
		}
	}
}

// Close stops the worker. With drain, a waiting snapshot is decoded first.
func (w *Worker) Close(drain bool) {
	w.mu.Lock()
	w.stopping = true
	if !drain {
		w.pending = nil
	}
	w.cond.Signal()
	w.mu.Unlock()
	<-w.done
}

type Stats struct {
	Windows       int     `json:"windows"`
	MaxDecodeMS   float64 `json:"max_decode_ms"`
	Hits          int     `json:"hits"`
	LateHits      int     `json:"late_hits"`
	FullyLateHits int     `json:"fully_late_hits"`
	MutedFrames   int     `json:"muted_frames"`
	MaxLateCoreMS float64 `json:"max_late_core_ms"`
	MaxHitAgeMS   float64 `json:"max_hit_age_ms"`
}

type hit struct {
	word            Word
	late, fullyLate bool
}

type Engine struct {
	rate, block                             int
	delayBlocks, windowBlocks, hopBlocks    int
	history, buffer                         [][]byte
	silence                                 []byte
	worker                                  recognition
	stats                                   Stats
	captured, rendered, blockCount, nextJob int
	hits                                    []hit
	nextResult                              *Result
	finished                                bool
	fade, pre, post, edge                   int
}

func NewEngine(rate, delayMS int, worker recognition, windowMS int) *Engine {
	e := &Engine{
		rate:         rate,
		block:        rate * blockMS / 1000,
		delayBlocks:  delayMS / blockMS,
		windowBlocks: windowMS / blockMS,
		hopBlocks:    hopMS / blockMS,
		worker:       worker,
		nextJob:      minWindowMS / blockMS,
	}
	e.silence = make([]byte, e.block*frameBytes)
	e.fade = e.ms(8)
	// Tuned on real recordings: drawn-out words start ~0.2 s before the first
	// token timestamp. 70 ms leaked onsets, 250 ms leaked one, 400 ms none.
	e.pre = e.ms(400)
	e.post = e.ms(90)
	e.edge = e.ms(120)
	return e
}

func (e *Engine) ms(milliseconds int) int { return e.rate * milliseconds / 1000 }

// Capture takes ownership of pcm; the caller must not reuse the slice.
func (e *Engine) Capture(pcm []byte) error {
	frames := len(pcm) / frameBytes
	if e.finished || frames <= 0 || frames > e.block || len(pcm)%frameBytes != 0 {
		return errors.New("Invalid PCM capture block")
	}
	if e.captured != e.blockCount*e.block {
		return errors.New("Short capture block is only supported at end of input")
	}
	e.history = append(e.history, pcm)
	if len(e.history) > e.windowBlocks {
		e.history = e.history[1:]
	}
	e.buffer = append(e.buffer, pcm)
	e.captured += frames
	e.blockCount++
	if e.blockCount >= e.nextJob && frames == e.block {
		e.submit(false)
		e.nextJob += e.hopBlocks
	}
	return nil
}

func (e *Engine) submit(final bool) {
	blocks := append([][]byte(nil), e.history...)
	start := max(0, e.blockCount-len(blocks)) * e.block
	e.worker.Submit(Job{start, e.captured, blocks, final})
}

func (e *Engine) Finish() {
	if !e.finished && e.captured > 0 {
		e.submit(true)
	}
	e.finished = true
}

// Collect applies every recognition result that completed by the deadline.
func (e *Engine) Collect(deadline float64) error {
	if err := e.worker.Err(); err != nil {
		return fmt.Errorf("Recognition stopped: %w", err)
	}
	for {
		if e.nextResult == nil {
			select {
			case result := <-e.worker.Results():
				e.nextResult = &result
			default:
				return nil
			}
		}
		if e.nextResult.Completed > deadline {
			return nil
		}
		result := *e.nextResult
		e.nextResult = nil
		e.apply(result)
	}
}

func (e *Engine) apply(result Result) {
	job := result.Job
	e.stats.Windows++
	e.stats.MaxDecodeMS = max(e.stats.MaxDecodeMS, result.Seconds*1000)
	kept := e.hits[:0]
	for _, h := range e.hits {
		if h.word.End+e.post+e.fade > min(job.Start, e.rendered) {
			kept = append(kept, h)
		}
	}
	e.hits = kept
	length := job.End - job.Start
	for _, word := range result.Words {
		if word.Start < 0 || word.Start >= word.End || word.End > length {
			continue
		}
		if job.Start > 0 && word.Start < e.edge {
			continue
		}
		if !job.Final && word.End > length-e.edge {
			continue
		}
		e.addHit(Word{word.Term, job.Start + word.Start, job.Start + word.End}, job.End)
	}
}

func (e *Engine) addHit(word Word, capturedAtJob int) {
	var overlaps, rest []hit
	for _, h := range e.hits {
		if h.word.Term == word.Term && h.word.Start < word.End && word.Start < h.word.End {
			overlaps = append(overlaps, h)
		} else {
			rest = append(rest, h)
		}
	}
	// Account only for newly identified core samples that already left the buffer.
	lateEnd := min(word.End, e.rendered)
	uncovered := max(0, lateEnd-word.Start)
	coveredEnd := word.Start
	sort.SliceStable(overlaps, func(a, b int) bool { return overlaps[a].word.Start < overlaps[b].word.Start })
	fullyLate := word.End <= e.rendered
	late := false
	wereLate, wereFullyLate := 0, 0
	start, end := word.Start, word.End
	for _, h := range overlaps {
		low := max(coveredEnd, h.word.Start)
		high := min(lateEnd, h.word.End)
		if high > low {
			uncovered -= high - low
			coveredEnd = high
		}
		fullyLate = fullyLate && h.fullyLate
		late = late || h.late
		wereLate += b2i(h.late)
		wereFullyLate += b2i(h.fullyLate)
		start = min(start, h.word.Start)
		end = max(end, h.word.End)
	}
	late = late || uncovered > 0
	e.stats.LateHits += b2i(late) - wereLate
	e.stats.FullyLateHits += b2i(fullyLate) - wereFullyLate
	e.stats.Hits += 1 - len(overlaps)
	e.stats.MaxLateCoreMS = max(e.stats.MaxLateCoreMS, float64(uncovered)*1000/float64(e.rate))
	if len(overlaps) == 0 {
		e.stats.MaxHitAgeMS = max(e.stats.MaxHitAgeMS, float64(capturedAtJob-word.Start)*1000/float64(e.rate))
	}
	e.hits = append(rest, hit{Word{word.Term, start, end}, late, fullyLate})
}

func b2i(value bool) int {
	if value {
		return 1
	}
	return 0
}

// Render emits one due block. Late results cannot rewrite samples already emitted.
func (e *Engine) Render(deadline float64) ([]byte, error) {
	if err := e.Collect(deadline); err != nil {
		return nil, err
	}
	pcm := e.buffer[0]
	e.buffer = e.buffer[1:]
	count := len(pcm) / frameBytes
	start := e.rendered
	end := start + count
	e.rendered = end

	var gains []float64
	for _, h := range e.hits {
		low := h.word.Start - e.pre
		high := h.word.End + e.post
		if low-e.fade >= end || high+e.fade <= start {
			continue
		}
		if gains == nil {
			gains = make([]float64, count)
			for i := range gains {
				gains[i] = 1
			}
		}
		for frame := max(start, low-e.fade); frame < min(end, high+e.fade); frame++ {
			gain := float64(max(low-frame, frame-high+1, 0)) / float64(e.fade)
			gains[frame-start] = min(gains[frame-start], gain)
		}
	}
	if gains == nil {
		return pcm, nil
	}
	out := append([]byte(nil), pcm...)
	for frame, gain := range gains {
		if gain >= 1 {
			continue
		}
		for i := frame * frameBytes; i < (frame+1)*frameBytes; i += 2 {
			sample := float64(int16(binary.LittleEndian.Uint16(out[i:])))
			binary.LittleEndian.PutUint16(out[i:], uint16(int16(math.RoundToEven(sample*gain))))
		}
		if gain == 0 {
			e.stats.MutedFrames++
		}
	}
	return out, nil
}
