// Offline device-output audio filter.
package main

import (
	"context"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gen2brain/malgo"
)

//go:embed words.txt
var defaultWords []byte

const period = float64(blockMS) / 1000

var rates = map[int]bool{16000: true, 32000: true, 48000: true}

type options struct {
	model, words           string
	delayMS, threads       int
	windowMS, rate, hopMS  int
	inputDevice, outDevice string
	input, output          string
}

func report(status string, engine *Engine, extra map[string]any) {
	fields := map[string]any{"status": status}
	if engine != nil {
		raw, _ := json.Marshal(engine.stats)
		json.Unmarshal(raw, &fields)
		fields["captured_frames"] = engine.captured
		fields["rendered_frames"] = engine.rendered
		fields["skipped_windows"] = engine.worker.Dropped()
	}
	for key, value := range extra {
		fields[key] = value
	}
	line, _ := json.Marshal(fields)
	fmt.Fprintln(os.Stderr, string(line))
}

func health(engine *Engine) string {
	if engine.worker.Dropped() > 0 || engine.stats.LateHits > 0 {
		return "degraded"
	}
	return "running"
}

func sleepUntil(deadline float64) {
	if remaining := deadline - monotonic(); remaining > 0 {
		time.Sleep(time.Duration(remaining * float64(time.Second)))
	}
}

func rounded(value float64) float64 { return math.Round(value*1000) / 1000 }

type frameSource interface {
	Frames() int
	ReadFrames(count int) ([]byte, error)
}

// paceWav makes audio available one block per tick. Output keeps the input file's timeline.
func paceWav(source frameSource, target io.Writer, engine *Engine,
	clock func() float64, wait func(float64)) (map[string]any, error) {
	remaining := source.Frames()
	epoch := clock()
	maxSlip := 0.0
	var firstRelease any
	for tick := 1; remaining > 0 || len(engine.buffer) > 0; tick++ {
		deadline := epoch + float64(tick)*period
		wait(deadline)
		if remaining > 0 {
			count := min(remaining, engine.block)
			pcm, err := source.ReadFrames(count)
			if err != nil {
				return nil, err
			}
			if err := engine.Capture(pcm); err != nil {
				return nil, err
			}
			if remaining -= count; remaining == 0 {
				engine.Finish()
			}
		}
		if tick > engine.delayBlocks && len(engine.buffer) > 0 {
			pcm, err := engine.Render(deadline)
			if err != nil {
				return nil, err
			}
			if _, err := target.Write(pcm); err != nil {
				return nil, err
			}
			emitted := clock()
			maxSlip = max(maxSlip, emitted-deadline)
			if firstRelease == nil {
				firstRelease = rounded((emitted - epoch - period) * 1000)
			}
		} else if err := engine.Collect(deadline); err != nil {
			return nil, err
		}
		if tick%250 == 0 {
			status := "priming"
			if engine.rendered > 0 {
				status = health(engine)
			}
			report(status, engine, nil)
		}
	}
	return map[string]any{
		"first_release_delay_ms": firstRelease,
		"max_release_slip_ms":    rounded(maxSlip * 1000),
		"paced_seconds":          rounded(clock() - epoch),
	}, nil
}

type wavReader struct {
	file   io.Reader
	rate   int
	frames int
}

// openWav accepts only stereo signed 16-bit PCM and leaves the reader at the samples.
func openWav(file io.Reader) (*wavReader, error) {
	bad := errors.New("WAV input must be stereo signed 16-bit PCM at 16, 32 or 48 kHz")
	var riff [12]byte
	if _, err := io.ReadFull(file, riff[:]); err != nil || string(riff[:4]) != "RIFF" || string(riff[8:]) != "WAVE" {
		return nil, bad
	}
	rate := 0
	for {
		var head [8]byte
		if _, err := io.ReadFull(file, head[:]); err != nil {
			return nil, bad
		}
		size := int(binary.LittleEndian.Uint32(head[4:]))
		if string(head[:4]) == "data" {
			if !rates[rate] {
				return nil, bad
			}
			return &wavReader{file, rate, size / frameBytes}, nil
		}
		body := make([]byte, size+size%2)
		if _, err := io.ReadFull(file, body); err != nil {
			return nil, bad
		}
		if string(head[:4]) == "fmt " {
			if size < 16 || binary.LittleEndian.Uint16(body) != 1 || binary.LittleEndian.Uint16(body[2:]) != 2 ||
				binary.LittleEndian.Uint16(body[14:]) != 16 {
				return nil, bad
			}
			rate = int(binary.LittleEndian.Uint32(body[4:]))
		}
	}
}

func (w *wavReader) Frames() int { return w.frames }

func (w *wavReader) ReadFrames(count int) ([]byte, error) {
	pcm := make([]byte, count*frameBytes)
	if _, err := io.ReadFull(w.file, pcm); err != nil {
		return nil, errors.New("Input WAV ended before its declared frame count")
	}
	return pcm, nil
}

func wavHeader(rate, frames int) []byte {
	head := make([]byte, 44)
	size := frames * frameBytes
	copy(head, "RIFF")
	binary.LittleEndian.PutUint32(head[4:], uint32(36+size))
	copy(head[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(head[16:], 16)
	binary.LittleEndian.PutUint16(head[20:], 1)
	binary.LittleEndian.PutUint16(head[22:], 2)
	binary.LittleEndian.PutUint32(head[24:], uint32(rate))
	binary.LittleEndian.PutUint32(head[28:], uint32(rate*frameBytes))
	binary.LittleEndian.PutUint16(head[32:], frameBytes)
	binary.LittleEndian.PutUint16(head[34:], 16)
	copy(head[36:], "data")
	binary.LittleEndian.PutUint32(head[40:], uint32(size))
	return head
}

func openModel(opts options, rate int) (*Zipformer, error) {
	report("loading_model", nil, nil)
	name, data := "words.txt", defaultWords
	if opts.words != "" {
		var err error
		name = opts.words
		if data, err = os.ReadFile(name); err != nil {
			return nil, err
		}
	}
	entries, err := vocabulary(name, data)
	if err != nil {
		return nil, err
	}
	return NewZipformer(opts.model, entries, rate, opts.threads)
}

func runWav(opts options) error {
	in, _ := filepath.Abs(opts.input)
	out, _ := filepath.Abs(opts.output)
	if in == out {
		return errors.New("Use different input and output WAV paths")
	}
	file, err := os.Open(opts.input)
	if err != nil {
		return err
	}
	defer file.Close()
	source, err := openWav(file)
	if err != nil {
		return err
	}
	model, err := openModel(opts, source.rate)
	if err != nil {
		return err
	}
	worker := NewWorker(model.Decode)
	defer worker.Close(false)
	engine := NewEngine(source.rate, opts.delayMS, worker, opts.windowMS)
	engine.hopBlocks = opts.hopMS / blockMS
	report("priming", engine, map[string]any{"rate": source.rate, "delay_ms": opts.delayMS, "mode": "wav"})

	temp, err := os.CreateTemp(filepath.Dir(out), "*.wav")
	if err != nil {
		return err
	}
	defer os.Remove(temp.Name()) // no-op once renamed
	defer temp.Close()
	if _, err := temp.Write(wavHeader(source.rate, source.frames)); err != nil {
		return err
	}
	timing, err := paceWav(source, temp, engine, monotonic, sleepUntil)
	if err != nil {
		return err
	}
	worker.Close(true)
	// Count results after the final deadline without modifying the file.
	if err := engine.Collect(monotonic()); err != nil {
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(temp.Name(), out); err != nil {
		return err
	}
	timing["health"] = health(engine)
	report("complete", engine, timing)
	return nil
}

func audioContext() (*malgo.AllocatedContext, error) {
	return malgo.InitContext(nil, malgo.ContextConfig{}, nil)
}

// findDevice resolves a number from `devices` or a name; empty means the system default.
func findDevice(ctx *malgo.AllocatedContext, kind malgo.DeviceType, selector string) (*malgo.DeviceID, error) {
	if selector == "" {
		return nil, nil
	}
	devices, err := ctx.Devices(kind)
	if err != nil {
		return nil, err
	}
	if number, err := strconv.Atoi(selector); err == nil && number >= 0 && number < len(devices) {
		return &devices[number].ID, nil
	}
	for i := range devices {
		if strings.Contains(strings.ToLower(devices[i].Name()), strings.ToLower(selector)) {
			return &devices[i].ID, nil
		}
	}
	return nil, fmt.Errorf("No audio device matches %q", selector)
}

func runDevices() error {
	ctx, err := audioContext()
	if err != nil {
		return err
	}
	defer ctx.Free()
	for _, group := range []struct {
		label string
		kind  malgo.DeviceType
	}{{"input", malgo.Capture}, {"output", malgo.Playback}} {
		devices, err := ctx.Devices(group.kind)
		if err != nil {
			return err
		}
		for number := range devices {
			fmt.Printf("%s %d: %s\n", group.label, number, devices[number].Name())
		}
	}
	return nil
}

func runLive(opts options) error {
	ctx, err := audioContext()
	if err != nil {
		return err
	}
	defer ctx.Free()
	inputID, err := findDevice(ctx, malgo.Capture, opts.inputDevice)
	if err != nil {
		return err
	}
	outputID, err := findDevice(ctx, malgo.Playback, opts.outDevice)
	if err != nil {
		return err
	}
	model, err := openModel(opts, opts.rate)
	if err != nil {
		return err
	}
	worker := NewWorker(model.Decode)
	defer worker.Close(false)
	engine := NewEngine(opts.rate, opts.delayMS, worker, opts.windowMS)
	engine.hopBlocks = opts.hopMS / blockMS

	// The device hands over whatever frame count it likes; the engine wants exact
	// 20 ms blocks, so input and output are re-blocked through two small queues.
	var mu sync.Mutex // guards engine, failure and the queues
	var failure error
	var pendingIn, pendingOut []byte
	underrun := 0  // output frames filled with silence because nothing was ready; should stay 0
	inputPeak := 0 // loudest input sample since the last report; 0 means silence is arriving
	blockBytes := engine.block * frameBytes
	// One block of silence up front: without it the output queue runs dry whenever
	// the device's callback size is not a multiple of a block, which sounds like dropouts.
	pendingOut = append(pendingOut, engine.silence...)
	onData := func(out, in []byte, _ uint32) {
		mu.Lock()
		defer mu.Unlock()
		for i := 0; i+1 < len(in); i += 2 {
			if level := int(int16(binary.LittleEndian.Uint16(in[i:]))); max(level, -level) > inputPeak {
				inputPeak = max(level, -level)
			}
		}
		pendingIn = append(pendingIn, in...)
		for failure == nil && len(pendingIn) >= blockBytes {
			pcm := append([]byte(nil), pendingIn[:blockBytes]...)
			pendingIn = pendingIn[blockBytes:]
			deadline := monotonic()
			block := engine.silence
			err := engine.Capture(pcm)
			if err == nil {
				if len(engine.buffer) > engine.delayBlocks {
					block, err = engine.Render(deadline)
				} else {
					err = engine.Collect(deadline)
				}
			}
			if err != nil {
				failure = err
				break
			}
			pendingOut = append(pendingOut, block...)
		}
		copied := copy(out, pendingOut)
		pendingOut = pendingOut[copied:]
		underrun += (len(out) - copied) / frameBytes
		clear(out[copied:])
	}

	config := malgo.DefaultDeviceConfig(malgo.Duplex)
	config.SampleRate = uint32(opts.rate)
	config.PeriodSizeInMilliseconds = blockMS
	config.Capture.Format, config.Capture.Channels = malgo.FormatS16, 2
	config.Playback.Format, config.Playback.Channels = malgo.FormatS16, 2
	if inputID != nil {
		config.Capture.DeviceID = inputID.Pointer()
	}
	if outputID != nil {
		config.Playback.DeviceID = outputID.Pointer()
	}
	device, err := malgo.InitDevice(ctx.Context, config, malgo.DeviceCallbacks{Data: onData})
	if err != nil {
		return err
	}
	defer device.Uninit()
	if err := device.Start(); err != nil {
		return err
	}
	report("priming", engine, map[string]any{"rate": opts.rate, "delay_ms": opts.delayMS, "mode": "live"})

	stop, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for running := true; running; {
		select {
		case <-stop.Done():
			running = false
		case <-ticker.C:
			mu.Lock()
			err := failure
			if err == nil {
				status := "priming"
				if engine.rendered > 0 {
					status = health(engine)
				}
				report(status, engine, map[string]any{"input_peak": inputPeak, "underrun_frames": underrun})
				inputPeak = 0
			}
			mu.Unlock()
			if err != nil {
				return err
			}
		}
	}
	device.Stop()
	mu.Lock()
	defer mu.Unlock()
	report("stopped", engine, map[string]any{
		"health": health(engine), "discarded_frames": engine.captured - engine.rendered})
	return nil
}

func parseArgs() (string, options, error) {
	usage := errors.New("usage: audiofilter {wav|live|devices} --model DIR [options] [input.wav output.wav]")
	if len(os.Args) < 2 {
		return "", options{}, usage
	}
	mode := os.Args[1]
	var opts options
	flags := flag.NewFlagSet(mode, flag.ContinueOnError)
	flags.StringVar(&opts.model, "model", "", "folder holding the Zipformer model")
	flags.StringVar(&opts.words, "words", "", "word list, one per line (default: built in)")
	flags.IntVar(&opts.delayMS, "delay-ms", 2000, "output delay")
	flags.IntVar(&opts.threads, "threads", 2, "recognition threads")
	flags.IntVar(&opts.windowMS, "window-ms", windowMS, "audio the recogniser sees per check")
	flags.IntVar(&opts.hopMS, "hop-ms", 160, "time between checks; shorter catches words sooner, costs CPU")
	if mode == "live" {
		flags.IntVar(&opts.rate, "rate", 48000, "sample rate: 16000, 32000 or 48000")
		flags.StringVar(&opts.inputDevice, "input-device", "", "input device name or number")
		flags.StringVar(&opts.outDevice, "output-device", "", "output device name or number")
	}
	switch mode {
	case "devices":
		return mode, opts, nil
	case "wav", "live":
	default:
		return "", opts, usage
	}
	if err := flags.Parse(os.Args[2:]); err != nil {
		return "", opts, err
	}
	switch {
	case opts.model == "":
		return "", opts, errors.New("--model is required")
	case opts.delayMS < blockMS || opts.delayMS > 15000 || opts.delayMS%blockMS != 0:
		return "", opts, errors.New("--delay-ms must be a multiple of 20 between 20 and 15000")
	case opts.windowMS < minWindowMS || opts.windowMS > 12000 || opts.windowMS%blockMS != 0:
		return "", opts, errors.New("--window-ms must be a multiple of 20 between 1040 and 12000")
	case opts.hopMS < blockMS || opts.hopMS > 1000 || opts.hopMS%blockMS != 0:
		return "", opts, errors.New("--hop-ms must be a multiple of 20 between 20 and 1000")
	case opts.threads < 1 || opts.threads > 32:
		return "", opts, errors.New("--threads must be between 1 and 32")
	case mode == "live" && !rates[opts.rate]:
		return "", opts, errors.New("--rate must be 16000, 32000 or 48000")
	case mode == "wav" && flags.NArg() != 2:
		return "", opts, usage
	}
	opts.input, opts.output = flags.Arg(0), flags.Arg(1)
	return mode, opts, nil
}

func main() {
	mode, opts, err := parseArgs()
	if err == nil {
		switch mode {
		case "devices":
			err = runDevices()
		case "wav":
			err = runWav(opts)
		default:
			err = runLive(opts)
		}
	}
	if err != nil {
		report("failed", nil, map[string]any{"error": err.Error()})
		os.Exit(1)
	}
}
