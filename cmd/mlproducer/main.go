package main

import (
	"bufio"
	"encoding/binary"
	"os"
	"path/filepath"
	"strconv"
	"unsafe"

	"github.com/domino14/macondo/config"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// NPlies is the number of plies to look ahead in the game assembler. We will predict
// the game spread after NPlies.
const NPlies = 5

// ─────────────────────────────────────────────────────────────────────────────
// text-based writer  → one line  per vector
// Format:  "0.000 1.000 0.125 …\n"
// ─────────────────────────────────────────────────────────────────────────────
func writeVectorText(w *bufio.Writer, vec []float32) error {
	for i, f := range vec {
		if i > 0 {
			if err := w.WriteByte(' '); err != nil { // space separator
				return err
			}
		}
		// strconv is ~5× faster than fmt for tight loops
		if _, err := w.WriteString(strconv.FormatFloat(float64(f), 'f', -1, 32)); err != nil {
			return err
		}
	}
	return w.WriteByte('\n') // newline terminator
}

// write one float32 slice as [uint32 lenBytes][payload]
func writeVectorBin(w *bufio.Writer, vec []float32) error {
	// Re-interpret the []float32 backing array as []byte
	byteSlice := unsafe.Slice(
		(*byte)(unsafe.Pointer(&vec[0])),
		len(vec)*4,
	)

	// 1) length prefix (little-endian uint32)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(byteSlice))); err != nil {
		return err
	}
	// 2) payload
	_, err := w.Write(byteSlice)
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// main streaming loop
// ─────────────────────────────────────────────────────────────────────────────
func main() {

	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)

	cfg := &config.Config{}
	args := os.Args[1:]
	cfg.Load(args)
	log.Info().Msgf("Loaded config: %v", cfg.AllSettings())
	cfg.AdjustRelativePaths(exPath)

	var logger zerolog.Logger
	if cfg.GetBool("debug") {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
		logger = zerolog.New(os.Stderr).Level(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
		logger = zerolog.New(os.Stderr).Level(zerolog.InfoLevel)
	}
	zerolog.DefaultContextLogger = &logger
	logger.Debug().Msg("Debug logging is on")

	const bufSize = 1 << 20 // 1 MiB buffered stdout
	const flushEvery = 1000 // emit 1 000 vectors → flush

	out := bufio.NewWriterSize(os.Stdout, bufSize)
	emitted := 0 // counter

	engine := NewGameAssembler(NPlies)  // holds live games, emits vectors
	scanner := NewTurnScanner(os.Stdin) // feeds individual turns

	for scanner.Scan() {
		turn := scanner.Turn()
		engine.FeedTurn(turn)

		for engine.Ready() { // ≥1 vectors ready
			vec := engine.PopVector() // []float32
			if err := writeVectorBin(out, vec); err != nil {
				panic(err) // production: handle/propagate
			}
			emitted++
			if emitted%flushEvery == 0 { // ═══ flush here ═══
				if err := out.Flush(); err != nil {
					panic(err)
				}
			}
			if emitted%100000 == 0 {
				log.Info().Msgf("Emitted %d vectors; games in mem: %d", emitted, len(engine.games))
			}
		}
	}
	out.Flush() // flush any buffered lines
}
