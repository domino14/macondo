package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strconv"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/cespare/xxhash"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	"github.com/domino14/word-golib/kwg"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const VectorBufferSize = 20000

// NPlies is the number of plies to look ahead in the game assembler. We will predict
// the win percentage after NPlies.
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
	var labeler string
	var plies, rollouts int
	var sample float64
	var labelsOut string
	var perGame bool
	var pickMax, endgamePlies int
	flag.BoolVar(&profile, "profile", false, "Enable CPU and memory profiling")
	flag.StringVar(&labeler, "labeler", "table",
		"table: win% table after NPlies real plies; result: the mover's real game result (and spread to the end); "+
			"rollout: mean of N sampled K-ply rollouts scored by the net (needs Triton)")
	flag.BoolVar(&perGame, "per-game", false,
		"emit one position per game (a turn drawn uniformly from 1..pick-max; games shorter than the draw emit nothing) "+
			"instead of every position; with -labeler rollout this replaces -sample")
	flag.IntVar(&pickMax, "pick-max", 30, "with -per-game: the latest turn that can be drawn")
	flag.IntVar(&endgamePlies, "endgame-plies", 0,
		"label emitted positions whose bag was already empty by a quick endgame search of this many plies "+
			"(greedy playout at the leaves) instead of the logged game's outcome; 0 = off")
	flag.IntVar(&plies, "plies", 2, "rollout labeler: plies per rollout (K)")
	flag.IntVar(&rollouts, "rollouts", 16, "rollout labeler: rollouts per position (N)")
	flag.Float64Var(&sample, "sample", 0.25, "rollout labeler: fraction of positions to label and emit")
	flag.StringVar(&labelsOut, "labels-out", "", "rollout labeler: also write gameID,turn,value,spread per labeled position to this CSV")
	flag.Parse()

	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)

	cfg := &config.Config{}
	args := flag.Args()
	cfg.Load(args)
	log.Info().Msgf("Loaded config: %v", cfg.SanitizedSettings())
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

	var labelsFile *os.File
	var labelsW *bufio.Writer
	if labelsOut != "" {
		labelsFile, err = os.Create(labelsOut)
		if err != nil {
			log.Fatal().Err(err).Msg("creating labels file")
		}
		labelsW = bufio.NewWriterSize(labelsFile, bufSize)
		labelsW.WriteString("gameID,turn,value,spread\n")
	}

	numWorkers := runtime.NumCPU()
	var shared *RolloutShared
	switch labeler {
	case "table":
		log.Info().Msgf("Lookahead: %d plies", NPlies)
	case "result":
		log.Info().Msg("Value target: the mover's real game result")
	case "rollout":
		shared, err = NewRolloutShared(cfg, "NWL23")
		if err != nil {
			log.Fatal().Err(err).Msg("rollout labeler setup")
		}
		log.Info().Msgf("Rollout labeler: %d plies x %d rollouts, sampling %.0f%% of positions, model %s v%s at %s",
			plies, rollouts, sample*100, cfg.GetString(config.ConfigTritonModelName),
			cfg.GetString(config.ConfigTritonModelVersion), cfg.GetString(config.ConfigTritonURL))
	default:
		log.Fatal().Msgf("unknown -labeler %q", labeler)
	}
	if perGame {
		log.Info().Msgf("Emitting one position per game, turn drawn from 1..%d", pickMax)
	}
	var gd *kwg.KWG
	if endgamePlies > 0 {
		gd, err = kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
		if err != nil {
			log.Fatal().Err(err).Msg("loading kwg for endgame search")
		}
		// One table, shared by every worker's searches (its entries are
		// lock-free); a small slice of memory is plenty for 2-ply searches.
		negamax.GlobalTranspositionTable.Reset(0.02, 15)
		log.Info().Msgf("Endgame positions labeled by a %d-ply quick search", endgamePlies)
	}
	log.Info().Msgf("Using %d workers", numWorkers)
	jobChans := make([]chan Turn, numWorkers)
	resultsChan := make(chan outputVector, numWorkers)
	var workersWg sync.WaitGroup
	log.Info().Msgf("Creating %d job channels", numWorkers)
	var totalGames, totalLabeled, totalSolved atomic.Int64
	for i := 0; i < numWorkers; i++ {
		jobChans[i] = make(chan Turn, 128)
		workersWg.Add(1)
		go func(jobChan <-chan Turn) {
			defer workersWg.Done()
			assembler := NewGameAssembler(NPlies, shared, plies, rollouts, sample)
			assembler.valueFromResult = labeler == "result"
			if perGame {
				assembler.pickMax = pickMax
			}
			assembler.endgamePlies = endgamePlies
			assembler.kwg = gd
			for turn := range jobChan {
				vecs := assembler.FeedTurn(turn)
				for _, vec := range vecs {
					resultsChan <- vec
				}
			}
			totalGames.Add(assembler.gamesProcessed)
			totalLabeled.Add(assembler.labeled)
			totalSolved.Add(assembler.solved)
		}(jobChans[i])
	}
	log.Info().Msgf("Started %d worker goroutines", numWorkers)
	// Closer goroutine
	go func() {
		workersWg.Wait()
		close(resultsChan)
	}()

	emitChan := make(chan outputVector, VectorBufferSize) // what the writer will read

	go func() {
		buf := make([]outputVector, 0, VectorBufferSize)

		for vec := range resultsChan { // incoming, possibly correlated
			if len(buf) < VectorBufferSize { // fill the ring first
				buf = append(buf, vec)
				continue
			}

			// pick a random index in [0, VectorBufferSize)
			i := rand.Intn(VectorBufferSize)
			swap := buf[i] // this one will be emitted
			buf[i] = vec   // new vec takes its place

			emitChan <- swap
		}

		// resultsChan is closed → drain what’s left in the buffer
		for _, v := range buf {
			emitChan <- v
		}
		close(emitChan)
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
	log.Info().Msg("Starting to read turns from emitChan")

	for vec := range emitChan {
		if err := BinaryWriteMLVector(out, vec); err != nil {
			panic(err) // production: handle/propagate
		}
		emitted++
		if labelsW != nil && vec.label != nil {
			fmt.Fprintf(labelsW, "%s,%d,%.5f,%.2f\n", vec.gameID, vec.turn, vec.label.value, vec.label.spread)
		}
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
	if labelsW != nil {
		labelsW.Flush()
		labelsFile.Close()
	}
	log.Info().Int64("totalGames", totalGames.Load()).
		Int64("rolloutLabeled", totalLabeled.Load()).
		Int64("endgameSolved", totalSolved.Load()).
		Int64("vectorsEmitted", int64(emitted)).
		Msg("Finished processing turns")
}
