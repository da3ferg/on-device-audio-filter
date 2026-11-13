// Local Zipformer adapter (non-streaming GigaSpeech model; LibriSpeech-trained
// models transcribe profanity as clean look-alikes). Only matching words and
// timestamps leave this file.
package main

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

var (
	wordRE     = regexp.MustCompile(`[a-zA-Z]+(?:'[a-zA-Z]+)*`)
	wholeWordR = regexp.MustCompile(`^[a-zA-Z]+(?:'[a-zA-Z]+)*$`)
)

func vocabulary(path string, data []byte) (map[string]bool, error) {
	entries := map[string]bool{}
	for number, line := range strings.Split(string(data), "\n") {
		entry, _, _ := strings.Cut(line, "#")
		entry = strings.ToLower(strings.TrimSpace(entry))
		if entry == "" {
			continue
		}
		if !wholeWordR.MatchString(entry) {
			return nil, fmt.Errorf("%s:%d must contain one English word", path, number+1)
		}
		entries[entry] = true
	}
	if len(entries) == 0 {
		return nil, errors.New("Vocabulary is empty")
	}
	return entries, nil
}

type token struct {
	text       []byte
	start, end int
}

// matchedWords matches reconstructed words, including words split across BPE tokens.
func matchedWords(tokens []token, entries map[string]bool) []Word {
	var contents []byte
	var timing [][2]int
	for _, t := range tokens {
		contents = append(contents, t.text...)
		for range t.text {
			timing = append(timing, [2]int{t.start, t.end})
		}
	}
	var words []Word
next:
	for _, match := range wordRE.FindAllIndex(contents, -1) {
		term := string(bytes.ToLower(contents[match[0]:match[1]]))
		if !entries[term] {
			continue
		}
		start, end := math.MaxInt, math.MinInt
		for _, bounds := range timing[match[0]:match[1]] {
			if bounds[0] < 0 || bounds[1] < bounds[0] {
				continue next
			}
			start = min(start, bounds[0])
			end = max(end, bounds[1])
		}
		if start < end {
			words = append(words, Word{term, start, end})
		}
	}
	return words
}

const kernelRadius = 40

func decimationKernel(rate int) []float32 {
	cutoff := 7200 / float64(rate)
	weights := make([]float64, 0, 2*kernelRadius+1)
	total := 0.0
	for i := -kernelRadius; i <= kernelRadius; i++ {
		sinc := 2 * cutoff
		if i != 0 {
			sinc = math.Sin(2*math.Pi*cutoff*float64(i)) / (math.Pi * float64(i))
		}
		weight := sinc * (0.54 + 0.46*math.Cos(math.Pi*float64(i)/kernelRadius))
		weights = append(weights, weight)
		total += weight
	}
	kernel := make([]float32, len(weights))
	for i, weight := range weights {
		kernel[i] = float32(weight / total)
	}
	return kernel
}

type Zipformer struct {
	recognizer *sherpa.OfflineRecognizer
	entries    map[string]bool
	rate       int
	stride     int
	kernel     []float32
}

func NewZipformer(model string, entries map[string]bool, rate, threads int) (*Zipformer, error) {
	config := sherpa.OfflineRecognizerConfig{DecodingMethod: "greedy_search"}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80}
	config.ModelConfig.NumThreads = threads
	config.ModelConfig.Provider = "cpu"
	config.ModelConfig.Tokens = filepath.Join(model, "tokens.txt")
	transducer := &config.ModelConfig.Transducer
	transducer.Encoder = filepath.Join(model, "encoder-epoch-30-avg-1.int8.onnx")
	transducer.Decoder = filepath.Join(model, "decoder-epoch-30-avg-1.int8.onnx")
	transducer.Joiner = filepath.Join(model, "joiner-epoch-30-avg-1.int8.onnx")
	for _, path := range []string{config.ModelConfig.Tokens, transducer.Encoder, transducer.Decoder, transducer.Joiner} {
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			return nil, fmt.Errorf("Local model file missing: %s", path)
		}
	}
	recognizer := sherpa.NewOfflineRecognizer(&config)
	if recognizer == nil {
		return nil, errors.New("Local model could not be loaded")
	}
	return &Zipformer{recognizer, entries, rate, rate / 16000, decimationKernel(rate)}, nil
}

// mono16k averages the stereo pair and low-pass decimates to 16 kHz, scaled to ±1.
func (z *Zipformer) mono16k(pcm []byte) []float32 {
	frames := len(pcm) / frameBytes
	mono := make([]float32, frames)
	for i := range mono {
		left := int16(uint16(pcm[4*i]) | uint16(pcm[4*i+1])<<8)
		right := int16(uint16(pcm[4*i+2]) | uint16(pcm[4*i+3])<<8)
		mono[i] = (float32(left) + float32(right)) / 2 / 32768
	}
	if z.stride == 1 {
		return mono
	}
	out := make([]float32, (frames+z.stride-1)/z.stride)
	for j := range out {
		center := j * z.stride
		var sum float32
		for k, weight := range z.kernel {
			if i := center + kernelRadius - k; i >= 0 && i < frames {
				sum += weight * mono[i]
			}
		}
		out[j] = sum
	}
	return out
}

func (z *Zipformer) Decode(pcm []byte) ([]Word, error) {
	frames := len(pcm) / frameBytes
	if frames < z.rate/10 {
		return nil, nil
	}
	stream := sherpa.NewOfflineStream(z.recognizer)
	defer sherpa.DeleteOfflineStream(stream)
	stream.AcceptWaveform(16000, z.mono16k(pcm))
	z.recognizer.Decode(stream)
	result := stream.GetResult()
	if len(result.Timestamps) != len(result.Tokens) {
		return nil, errors.New("Recognition returned inconsistent token timestamps")
	}
	starts := make([]int, len(result.Timestamps))
	for i, timestamp := range result.Timestamps {
		starts[i] = int(math.RoundToEven(float64(timestamp) * float64(z.rate)))
	}
	tokens := make([]token, len(starts))
	for i, start := range starts {
		// The model reports onsets. Bound the last-piece estimate when the next
		// token is absent, coincident, or separated by silence.
		end := min(start+z.rate*180/1000, frames)
		if i+1 < len(starts) && starts[i+1] > start {
			end = min(end, starts[i+1])
		}
		tokens[i] = token{[]byte(strings.ReplaceAll(result.Tokens[i], "▁", " ")), start, end}
	}
	return matchedWords(tokens, z.entries), nil
}
