package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strconv"
	"sync"
	"unsafe"

	"github.com/cespare/xxhash"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// NPlies is the number of plies to look ahead in the game assembler. We will predict
// the game spread after NPlies.
const NPlies = 50

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

func BinaryWriteMLVector(w *bufio.Writer, vec outputVector) error {

	feat := vec.features

	// Re-interpret the []float32 backing array as []byte
	featByteSlice := unsafe.Slice(
		(*byte)(unsafe.Pointer((&(*feat)[0]))),
		len(*feat)*4,
	)
	predByteSlice := unsafe.Slice(
		(*byte)(unsafe.Pointer(&vec.predictions[0])),
		len(vec.predictions)*4,
	)
	// 1) length prefix (little-endian uint32)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(featByteSlice)+len(predByteSlice))); err != nil {
		return err
	}
	// 2) payload
	_, err := w.Write(featByteSlice)
	if err != nil {

		return err
	}
	_, err = w.Write(predByteSlice) // predictions are already []byte
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// main streaming loop
// ─────────────────────────────────────────────────────────────────────────────
func main() {
	var profile bool
	flag.BoolVar(&profile, "profile", false, "Enable CPU and memory profiling")
	flag.Parse()

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

	var cpuProfFile, memProfFile *os.File
	if profile {
		cpuProfFile, err = os.Create("/tmp/mlproducer-cpu.prof")
		if err != nil {
			log.Fatal().Err(err).Msg("Could not create CPU profile file")
		}
		if err := pprof.StartCPUProfile(cpuProfFile); err != nil {
			log.Fatal().Err(err).Msg("Could not start CPU profiling")
		}
		log.Info().Msg("CPU profiling enabled: /tmp/mlproducer-cpu.prof")
		// Defer stop and mem profile
		defer func() {
			pprof.StopCPUProfile()
			cpuProfFile.Close()
			memProfFile, err = os.Create("/tmp/mlproducer-mem.prof")
			if err != nil {
				log.Error().Err(err).Msg("Could not create memory profile file")
				return
			}
			runtime.GC() // get up-to-date statistics
			if err := pprof.WriteHeapProfile(memProfFile); err != nil {
				log.Error().Err(err).Msg("Could not write memory profile")
			}
			memProfFile.Close()
			log.Info().Msg("Memory profile written: /tmp/mlproducer-mem.prof")
		}()
	}

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

	numWorkers := runtime.NumCPU()
	log.Info().Msgf("Lookahead: %d plies", NPlies)
	log.Info().Msgf("Using %d workers", numWorkers)
	jobChans := make([]chan Turn, numWorkers)
	resultsChan := make(chan outputVector, numWorkers)
	var workersWg sync.WaitGroup
	log.Info().Msgf("Creating %d job channels", numWorkers)
	for i := 0; i < numWorkers; i++ {
		jobChans[i] = make(chan Turn, 128)
		workersWg.Add(1)
		go func(jobChan <-chan Turn) {
			defer workersWg.Done()
			assembler := NewGameAssembler(NPlies)
			for turn := range jobChan {
				vecs := assembler.FeedTurn(turn)
				for _, vec := range vecs {
					resultsChan <- vec
				}
			}
		}(jobChans[i])
	}
	log.Info().Msgf("Started %d worker goroutines", numWorkers)
	// Closer goroutine
	go func() {
		workersWg.Wait()
		close(resultsChan)
	}()

	// Dispatcher goroutine
	go func() {
		scanner := NewTurnScanner(os.Stdin) // feeds individual turns
		for scanner.Scan() {
			turn := scanner.Turn()
			hash := xxhash.Sum64String(turn.GameID)
			workerIndex := hash % uint64(numWorkers)
			jobChans[workerIndex] <- turn
		}
		for _, ch := range jobChans {
			close(ch)
		}
	}()
	log.Info().Msg("Starting to read turns from resultsChan")

	for vec := range resultsChan {
		if err := BinaryWriteMLVector(out, vec); err != nil {
			panic(err) // production: handle/propagate
		}
		emitted++
		if emitted%flushEvery == 0 { // ═══ flush here ═══
			if err := out.Flush(); err != nil {
				panic(err)
			}
		}
		if emitted >= 100 && emitted < 200 {
			// find the exchange
			// if vec[len(vec)-13] == 1.0 {
			log.Info().Msgf("Found a test vector: %d", emitted)
			// Output this vec to a file "/tmp/test-vec.bin" for debugging
			testFile, err := os.Create(fmt.Sprintf("/tmp/test-vec-%d.bin", emitted))
			if err != nil {
				log.Fatal().Err(err).Msg("Failed to create test file")
			}
			testOut := bufio.NewWriterSize(testFile, bufSize)
			if err := BinaryWriteMLVector(testOut, vec); err != nil {
				log.Fatal().Err(err).Msg("Failed to write test vector to file")
			}
			if err := testOut.Flush(); err != nil {
				log.Fatal().Err(err).Msg("Failed to flush test vector to file")
			}
			testFile.Close()
			log.Info().Msgf("Wrote test vector to /tmp/test-vec.bin, length: %d", len(*vec.features)+len(vec.predictions))
		}
		if emitted%100000 == 0 {
			log.Info().Msgf("Emitted %d vectors", emitted)
		}
		game.MLVectorPool.Put(vec.features) // return to pool
	}
	log.Info().Msg("Flushing remaining vectors to output")

	out.Flush() // flush any buffered lines
	log.Info().Msgf("Finished processing turns, emitted %d vectors", emitted)
}
