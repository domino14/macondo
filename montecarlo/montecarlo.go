// Package montecarlo implements truncated monte-carlo search
// during the regular game. In other words, "simming".
package montecarlo

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"gopkg.in/yaml.v3"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/stats"
)

/*
	How to simulate:

	For iteration in iterations:
		For play in plays:
			place on the board, keep track of leave
			shuffle bag
			for ply in plies:
				- generate rack for user on turn. tiles should be drawn
				in the same order from the bag and replaced. basically,
				this rack should be the same for every play in plays, so
				constant on a per-iteration basis, to make
				it easier to debug / minimize number of variables.
				- place highest valuation play on board, keep track of leave

			compute stats so far

*/

const MaxHeatMapIterations = 7500

type InferenceMode int

const (
	InferenceOff InferenceMode = iota
	InferenceWeightedRandomTiles
	InferenceWeightedRandomRacks
)

// LogIteration is a struct meant for serializing to a log-file, for debug
// and other purposes.
type LogIteration struct {
	Iteration int       `json:"iteration" yaml:"iteration"`
	Plays     []LogPlay `json:"plays" yaml:"plays"`
	Thread    int       `json:"thread" yaml:"thread"`
}

// LogPlay is a single play.
type LogPlay struct {
	Play       string `json:"play" yaml:"play"`
	Leave      string `json:"leave" yaml:"leave"`
	Rack       string `json:"rack" yaml:"rack"`
	PresetRack string `json:"preset_rack,omitempty" yaml:"preset_rack,omitempty"`
	Pts        int    `json:"pts" yaml:"pts"`
	// Leftover is the equity of the leftover tiles at the end of the sim.
	Leftover float64 `json:"left,omitempty" yaml:"left,omitempty"`
	// Although this is a recursive structure we don't really use it
	// recursively.
	WinRatio float64   `json:"win,omitempty" yaml:"win,omitempty"`
	Plies    []LogPlay `json:"plies,omitempty" yaml:"plies,omitempty,flow"`
	Bingo    bool      `json:"bingo,omitempty" yaml:"bingo,omitempty"`
}

type SimmedPlay struct {
	sync.RWMutex
	play          *move.Move
	scoreStats    []stats.Statistic
	bingoStats    []stats.Statistic
	equityStats   stats.Statistic
	leftoverStats stats.Statistic
	// Actually this is win probability (0 to 1), not percent:
	winPctStats stats.Statistic
	ignore      bool
	unignorable bool
}

func (sp *SimmedPlay) String() string {
	return fmt.Sprintf("<Simmed play: %v (stats: %v %v %v %v %v)>", sp.play.ShortDescription(),
		sp.scoreStats, sp.bingoStats, sp.equityStats, sp.leftoverStats, sp.winPctStats)
}

// ScoreStatsNoLock returns the score stats without locking.
func (sp *SimmedPlay) ScoreStatsNoLock() []stats.Statistic {
	return sp.scoreStats
}

func (sp *SimmedPlay) Ignore() {
	sp.Lock()
	sp.ignore = true
	sp.Unlock()
}

func (sp *SimmedPlay) addScoreStat(play *move.Move, ply int) {
	// log.Debug().Msgf("Adding a stat for %v (pidx %v ply %v)", play, pidx, ply)
	var bingos int
	if play.BingoPlayed() {
		bingos = 1
	}
	sp.Lock()
	defer sp.Unlock()
	sp.scoreStats[ply].Push(float64(play.Score()))
	sp.bingoStats[ply].Push(float64(bingos))
}

func (sp *SimmedPlay) addEquityStat(initialSpread int, spread int, leftover float64) {
	sp.Lock()
	defer sp.Unlock()
	sp.equityStats.Push(float64(spread-initialSpread) + leftover)
	sp.leftoverStats.Push(leftover)
}

func (sp *SimmedPlay) addWinPctStat(spread int, leftover float64, gameover bool, winpcts [][]float32,
	tilesUnseen int, pliesAreEven bool) {
	winPct := float64(0.0)

	if gameover || tilesUnseen == 0 {
		if spread == 0 {
			winPct = 0.5
		} else if spread > 0 {
			winPct = 1.0
		}
	} else {
		if tilesUnseen > 93 {
			// Only for ZOMGWords or similar; this is a bit of a hack.
			tilesUnseen = 93
		}
		// for an even-ply sim, it is our opponent's turn at the end of the sim.
		// the table is calculated from our perspective, so flip the spread.
		// i.e. if we are winning by 20 pts at the end of the sim, and our opponent
		// is on turn, we want to look up -20 as the spread, and then flip the win %
		// as well.
		spreadPlusLeftover := spread + int(math.Round(leftover))
		if pliesAreEven {
			spreadPlusLeftover = -spreadPlusLeftover
		}

		if spreadPlusLeftover > equity.MaxRepresentedWinSpread {
			spreadPlusLeftover = equity.MaxRepresentedWinSpread
		}
		if spreadPlusLeftover < -equity.MaxRepresentedWinSpread {
			spreadPlusLeftover = -equity.MaxRepresentedWinSpread
		}
		// winpcts goes from +MaxRepresentedWinSpread to -MaxRespresentedWinSpread
		// spread = index
		// 200 = 0
		// 199 = 1
		// 99 = 101
		// 0 = 200
		// -1 = 201
		// -101 = 301
		// -200 = 400
		pct := winpcts[equity.MaxRepresentedWinSpread-spreadPlusLeftover][tilesUnseen]

		if pliesAreEven {
			// see the above comment re flipping win pct.
			pct = 1 - pct
		}
		winPct = float64(pct)
	}
	sp.Lock()
	defer sp.Unlock()
	sp.winPctStats.Push(float64(winPct))
}

func (s *SimmedPlay) Move() *move.Move {
	return s.play
}

// WinProb is an actually more accurate name for this statistic.
func (s *SimmedPlay) WinProb() float64 {
	return s.winPctStats.Mean()
}

type SimmedPlays struct {
	sync.RWMutex
	plays []*SimmedPlay
}

// PlaysNoLock returns the plays without locking.
func (s *SimmedPlays) PlaysNoLock() []*SimmedPlay {
	return s.plays
}

func (s *SimmedPlays) trimBottom(n int) {
	s.Lock()
	defer s.Unlock()
	s.plays = s.plays[:len(s.plays)-n]
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	origGame *game.Game

	gameCopies []*game.Game
	// movegens          []movegen.MoveGenerator
	equityCalculators []equity.EquityCalculator
	aiplayers         []aiturnplayer.AITurnPlayer
	leaveValues       equity.Leaves

	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount atomic.Uint64
	nodeCount      atomic.Uint64
	threads        int

	simming      bool
	readyToSim   bool
	simmedPlays  *SimmedPlays
	winPcts      [][]float32
	cfg          *config.Config
	knownOppRack []tilemapping.MachineLetter

	logStream             io.Writer
	collectHeatMap        bool
	tempHeatMapFile       *os.File
	activeHeatMap         []LogIteration
	activeHeatMapFilename string

	// See rangefinder.
	inferences               map[*[]tilemapping.MachineLetter]float64
	inferenceMode            InferenceMode
	tilesToInfer             int
	adjustedBagProbabilities []float64
	unseenToSimmingPlayer    map[tilemapping.MachineLetter]int

	autostopper          *AutoStopper
	stochasticStaticEval bool
}

func (s *Simmer) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	leaves equity.Leaves, cfg *config.Config) {
	s.origGame = game
	s.equityCalculators = eqCalcs
	s.leaveValues = leaves
	s.threads = max(1, runtime.NumCPU())
	s.autostopper = newAutostopper()

	// Hard-code the location of the win-pct file for now.
	// If we want to make some for other lexica in the future we'll
	// have to redo the equity calculator stuff.
	s.cfg = cfg
	if s.cfg != nil {
		// some hardcoded stuff here:
		winpct, err := cache.Load(s.cfg.WGLConfig(), "winpctfile:NWL20:winpct.csv", equity.WinPCTLoadFunc)
		if err != nil {
			panic(err)
		}
		var ok bool
		s.winPcts, ok = winpct.([][]float32)
		if !ok {
			panic("win percentages not correct type")
		}
	}
}

func (s *Simmer) SetStoppingCondition(sc StoppingCondition) {
	s.autostopper.stoppingCondition = sc
}

func (s *Simmer) SetThreads(threads int) {
	s.threads = threads
}

func (s *Simmer) Threads() int {
	return s.threads
}

func (s *Simmer) SetStochasticStaticEval(j bool) {
	log.Debug().Msg("set stochastic static evaluator for this bot")
	s.stochasticStaticEval = j
	for i := range s.aiplayers {
		s.aiplayers[i].MoveGenerator().(*movegen.GordonGenerator).SetRecordNTopPlays(2)
	}
}

func (s *Simmer) CleanupTempFile() {
	if s.tempHeatMapFile != nil {
		err := os.Remove(s.tempHeatMapFile.Name())
		if err != nil {
			log.Err(err).Str("heatMapFile", s.tempHeatMapFile.Name()).Msg("could not remove temporary file")
		} else {
			log.Info().Str("heatMapFile", s.tempHeatMapFile.Name()).Msg("deleted temp file")
		}
	}
}

func (s *Simmer) SetCollectHeatmap(b bool) error {
	if s.logStream != nil && b {
		return errors.New("cannot collect heat map if log stream already used")
	}
	s.collectHeatMap = b

	if b {
		s.CleanupTempFile()
		// Create a temporary file
		tempFile, err := os.CreateTemp("", "heatmap-*.gz")
		if err != nil {
			return fmt.Errorf("could not create temp file: %w", err)
		}
		s.tempHeatMapFile = tempFile
		// Create a gzip writer that wraps the temporary file
		gzipWriter := gzip.NewWriter(tempFile)

		// Set logStream to the gzip writer
		s.logStream = gzipWriter
		log.Info().Str("heatmap-file", tempFile.Name()).Msg("collecting-heatmap")
	}
	return nil
}

func (s *Simmer) CollectHeatmap() bool {
	return s.collectHeatMap
}

func (s *Simmer) closeHeatMap(ctx context.Context) error {
	logger := zerolog.Ctx(ctx)

	if s.logStream == nil {
		return nil
	}
	if !s.collectHeatMap {
		return nil
	}
	logger.Info().Msg("closing heatmap writer")
	// Close the gzip writer (flushes remaining data to temp file)
	if gzWriter, ok := s.logStream.(*gzip.Writer); ok {
		if err := gzWriter.Close(); err != nil {
			return fmt.Errorf("could not close gzip writer: %w", err)
		}
	}

	// Close the temporary file
	if err := s.tempHeatMapFile.Close(); err != nil {
		return fmt.Errorf("could not close temp file: %w", err)
	}

	// Reset logStream
	s.logStream = nil
	s.collectHeatMap = false
	return nil
}

// ReadHeatmap reads the gzipped data from the temporary file
func (s *Simmer) ReadHeatmap() ([]LogIteration, error) {
	if s.tempHeatMapFile == nil {
		return nil, errors.New("no heatmap data to read")
	}
	if s.tempHeatMapFile.Name() == s.activeHeatMapFilename {
		log.Info().Msg("loading-heatmap-from-cache")
		return s.activeHeatMap, nil
	}

	// Reopen the temporary file for reading
	file, err := os.Open(s.tempHeatMapFile.Name())
	if err != nil {
		return nil, fmt.Errorf("could not open temp file for reading: %w", err)
	}
	defer file.Close()

	// Create a gzip reader
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("could not create gzip reader: %w", err)
	}
	defer gzReader.Close()

	// Create a scanner to read the decompressed data line by line
	scanner := bufio.NewScanner(gzReader)

	// Parse each line into a LogIteration
	var logIterations []LogIteration
	for scanner.Scan() {
		line := scanner.Text()
		var logIteration LogIteration
		err := json.Unmarshal([]byte(line), &logIteration)
		if err != nil {
			return nil, fmt.Errorf("could not unmarshal JSON line: %w", err)
		}
		logIterations = append(logIterations, logIteration)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error while scanning file: %w", err)
	}

	s.activeHeatMap = logIterations
	s.activeHeatMapFilename = s.tempHeatMapFile.Name()

	return logIterations, nil
}

func (s *Simmer) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Simmer) SetKnownOppRack(r []tilemapping.MachineLetter) {
	s.knownOppRack = r
}

func (s *Simmer) SetInferences(i map[*[]tilemapping.MachineLetter]float64, t int, mode InferenceMode) {
	s.inferences = i
	s.tilesToInfer = t
	s.inferenceMode = mode
}

func (s *Simmer) makeGameCopies() error {
	log.Debug().Int("threads", s.threads).Msg("makeGameCopies")
	s.gameCopies = []*game.Game{}
	s.aiplayers = []aiturnplayer.AITurnPlayer{}
	// Pre-shuffle bag so we can make identical copies of it with fixedOrder
	s.origGame.Bag().Shuffle()

	for i := 0; i < s.threads; i++ {
		s.gameCopies = append(s.gameCopies, s.origGame.Copy())
		s.gameCopies[i].Bag().SetFixedOrder(true)

		player, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(s.gameCopies[i], s.origGame.Config(), s.equityCalculators)
		if err != nil {
			return err
		}
		// not ideal, but refactor later. The play recorder needs it.
		player.MoveGenerator().(*movegen.GordonGenerator).SetGame(s.gameCopies[i])
		s.aiplayers = append(s.aiplayers, player)
	}
	return nil

}

func (s *Simmer) resetStats(plies int, plays []*move.Move) {
	s.iterationCount.Store(0)
	s.nodeCount.Store(0)
	s.maxPlies = plies
	for _, g := range s.gameCopies {
		g.SetStateStackLength(1)
	}
	s.initialSpread = s.gameCopies[0].CurrentSpread()
	s.initialPlayer = s.gameCopies[0].PlayerOnTurn()
	s.simmedPlays = &SimmedPlays{plays: make([]*SimmedPlay, len(plays))}
	for idx, play := range plays {
		s.simmedPlays.plays[idx] = &SimmedPlay{}
		s.simmedPlays.plays[idx].play = play
		s.simmedPlays.plays[idx].scoreStats = make([]stats.Statistic, plies)
		s.simmedPlays.plays[idx].bingoStats = make([]stats.Statistic, plies)
	}

}

// AvoidPruningMoves sets a list of moves that must not be trimmed / pruned away
// during simulation. this function assumes that `resetStats` has already been called.
func (s *Simmer) AvoidPruningMoves(plays []*move.Move) {
	for _, play := range plays {
		s.simmedPlays.plays = append(s.simmedPlays.plays, &SimmedPlay{
			play:        play,
			scoreStats:  make([]stats.Statistic, s.maxPlies),
			bingoStats:  make([]stats.Statistic, s.maxPlies),
			unignorable: true,
		})
	}
}

func (s *Simmer) IsSimming() bool {
	return s.simming
}

func (s *Simmer) Reset() {
	s.gameCopies = nil
	s.readyToSim = false
}

// PrepareSim resets all the stats before a simulation.
func (s *Simmer) PrepareSim(plies int, plays []*move.Move) error {
	err := s.makeGameCopies()
	if err != nil {
		return err
	}
	s.resetStats(plies, plays)
	s.readyToSim = true
	s.knownOppRack = nil
	s.inferenceMode = InferenceOff
	return nil
}

func (s *Simmer) Ready() bool {
	return s.readyToSim
}

// Simulate sims all the plays. It is a blocking function.
func (s *Simmer) Simulate(ctx context.Context) error {
	logger := zerolog.Ctx(ctx)

	if len(s.simmedPlays.plays) == 0 || len(s.gameCopies) == 0 {
		return errors.New("please prepare the simulation first")
	}

	s.simming = true
	defer func() {
		s.simming = false
		logger.Info().Int("plies", s.maxPlies).Uint64("iterationCt", s.iterationCount.Load()).
			Msg("sim-ended")
	}()

	if !s.collectHeatMap {
		s.tempHeatMapFile = nil
	}

	nodes := s.nodeCount.Load()
	// This should be zero, but I think something is wrong with Lambda.
	logger.Info().Uint64("starting-node-count", nodes).Msg("nodes")

	// use an errgroup here and listen for a ctx done outside this loop, but
	// in another goroutine.
	// protect the simmed play statistics with a mutex.
	logger.Debug().Msgf("Simulating with %v threads", s.threads)
	syncExitChan := make(chan bool, s.threads)

	logChan := make(chan []byte)
	done := make(chan bool)

	ctrl := errgroup.Group{}
	writer := errgroup.Group{}

	if s.inferenceMode == InferenceWeightedRandomTiles {
		s.calculateWeightedProbabilitiesForBag()
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ctrl.Go(func() error {
		defer func() {
			logger.Debug().Msgf("Sim controller thread exiting")
		}()
		for range ctx.Done() {
		}
		logger.Debug().Msgf("Context is done: %v", ctx.Err())
		for t := 0; t < s.threads; t++ {
			syncExitChan <- true
		}
		// Send another exit signal to the stopping condition monitor
		if s.autostopper.stoppingCondition != StopNone {
			syncExitChan <- true
		}
		logger.Debug().Msgf("Sent sync messages to children threads...")

		return ctx.Err()
	})

	if s.logStream != nil {

		writer.Go(func() error {
			defer func() {
				logger.Debug().Msgf("Writer routine exiting")
			}()
			for {
				select {
				case bytes := <-logChan:
					s.logStream.Write(bytes)
				case <-done:
					// Ok, actually quit now.
					logger.Debug().Msgf("Got quit signal...")
					return nil
				}
			}
		})
	}
	tstart := time.Now()
	g := errgroup.Group{}
	s.autostopper.reset()

	for t := 0; t < s.threads; t++ {
		g.Go(func() error {
			defer func() {
				logger.Debug().Msgf("Thread %v exiting sim", t)
			}()
			logger.Debug().Msgf("Thread %v starting sim", t)
			for {
				numIters := s.iterationCount.Add(1)
				err := s.simSingleIteration(ctx, s.maxPlies, t, numIters-1, logChan)
				if err != nil {
					logger.Err(err).Msg("error simming iteration; canceling")
					cancel()
				}
				// check if we need to stop
				if s.autostopper.stoppingCondition != StopNone {
					if numIters%s.autostopper.stopConditionCheckInterval == 0 {
						logger.Debug().Uint64("numIters", numIters).Msg("checking-stopping-condition")
						stop := s.autostopper.shouldStop(numIters, s.simmedPlays, s.maxPlies)
						if stop {
							logger.Info().Uint64("numIters", numIters).Msg("reached stopping condition")
							cancel()
						}
					}
				}

				select {
				case v := <-syncExitChan:
					logger.Debug().Msgf("Thread %v got sync msg %v", t, v)
					return nil
				default:
					// Do nothing
				}
			}
		})
	}

	// Wait for threads in errgroup:
	err := g.Wait()
	logger.Debug().Msgf("errgroup returned err %v", err)
	elapsed := time.Since(tstart) // duration is in nanosecs
	nodes = s.nodeCount.Load()
	nps := float64(nodes) / elapsed.Seconds()
	logger.Info().Msgf("time taken: %v, nps: %f, nodes: %d", elapsed.Seconds(), nps, nodes)

	// Writer thread will exit now:
	if s.logStream != nil {
		close(done)
		writer.Wait()
	}
	if err = s.closeHeatMap(ctx); err != nil {
		logger.Err(err).Msg("close-heat-map")
	}

	ctrlErr := ctrl.Wait()
	logger.Debug().Msgf("ctrl errgroup returned err %v", ctrlErr)
	// sort plays at the end anyway.
	s.sortPlaysByWinRate(false)
	if ctrlErr == context.Canceled || ctrlErr == context.DeadlineExceeded {
		// Not actually an error
		logger.Debug().AnErr("ctrlErr", ctrlErr).Msg("montecarlo-it's ok, not an error")
		return nil
	}
	return ctrlErr
}

func (s *Simmer) Iterations() int {
	return int(s.iterationCount.Load())
}

func (s *Simmer) TrimBottom(totrim int) error {
	if s.simming {
		return errors.New("please stop sim before trimming plays")
	}
	if totrim > len(s.simmedPlays.plays)-1 {
		return errors.New("there are not that many plays to trim away")
	}
	s.simmedPlays.trimBottom(totrim)
	return nil
}

func (s *Simmer) simSingleIteration(ctx context.Context, plies, thread int, iterationCount uint64, logChan chan []byte) error {
	logger := zerolog.Ctx(ctx)

	g := s.gameCopies[thread]

	opp := (s.initialPlayer + 1) % g.NumPlayers()
	rackToSet := s.knownOppRack
	var err error
	if s.inferenceMode != InferenceOff {

		// If we have a very low number of inferred racks, we still want to
		// try to use their info. But we draw more random racks the fewer inferred
		// racks we have.

		minInferences := 2
		maxInferences := 25
		maxProbability := 0.9 // 90%
		minProbability := 0.0 // 0%

		numInferences := len(s.inferences)

		probability := maxProbability
		if numInferences > minInferences {
			if numInferences < maxInferences {
				probability = maxProbability -
					((float64(numInferences-minInferences) / float64(maxInferences-minInferences)) * maxProbability)
			} else {
				probability = minProbability
			}
		}
		if rand.Float64() < probability {
			rackToSet = nil
		} else {
			if s.inferenceMode == InferenceWeightedRandomTiles {
				rackToSet, err = s.weightedInferredDrawTiles()
			} else if s.inferenceMode == InferenceWeightedRandomRacks {
				rackToSet, err = s.weightedInferredDrawRacks()
			}
			if err != nil {
				return err
			}
		}

	}

	_, err = g.SetRandomRack(opp, rackToSet)
	if err != nil {
		return err
	}
	logIter := LogIteration{Iteration: int(iterationCount), Plays: nil, Thread: thread}

	var logPlay LogPlay
	var plyChild LogPlay
	for _, simmedPlay := range s.simmedPlays.plays {
		if simmedPlay.ignore {
			continue
		}
		if s.logStream != nil {
			logPlay = LogPlay{
				Play:  g.Board().MoveDescriptionWithPlaythrough(simmedPlay.play),
				Leave: simmedPlay.play.LeaveString(),
				Rack:  simmedPlay.play.FullRack(),
				Bingo: simmedPlay.play.BingoPlayed(),
				Pts:   simmedPlay.play.Score()}
		}
		// equity of the leftover tiles at the end of the sim
		leftover := float64(0.0)
		// logIter.Plays = append(logIter.Plays)
		// Play the move, and back up the game state.
		// log.Debug().Msgf("Playing move %v", play)'
		// Set the backup mode to simulation mode only to back up the first move:
		g.SetBackupMode(game.SimulationMode)
		g.PlayMove(simmedPlay.play, false, 0)
		s.nodeCount.Add(1)
		g.SetBackupMode(game.NoBackup)
		// Further plies will NOT be backed up.
		for ply := 0; ply < plies; ply++ {
			// Each ply is a player taking a turn
			onTurn := g.PlayerOnTurn()
			if g.Playing() != pb.PlayState_PLAYING {
				break
			}
			// Assume there are exactly two players.

			bestPlay := s.bestStaticTurn(onTurn, thread)
			// log.Debug().Msgf("Ply %v, Best play: %v", ply+1, bestPlay)
			g.PlayMove(bestPlay, false, 0)
			s.nodeCount.Add(1)
			// log.Debug().Msgf("Score is now %v", s.game.Score())

			if s.logStream != nil {
				presetRack := ""
				if len(rackToSet) > 0 && ply == 0 {
					presetRack = tilemapping.MachineWord(rackToSet).UserVisible(g.Alphabet())
				}
				plyChild = LogPlay{
					Play:       g.Board().MoveDescriptionWithPlaythrough(bestPlay),
					PresetRack: presetRack,
					Leave:      bestPlay.LeaveString(),
					Rack:       bestPlay.FullRack(),
					Pts:        bestPlay.Score(),
					Bingo:      bestPlay.BingoPlayed()}
			}
			if ply == plies-2 || ply == plies-1 {
				// It's either OUR last turn or OPP's last turn.
				// Calculate equity of leftover tiles.
				thisLeftover := s.leaveValues.LeaveValue(bestPlay.Leave())
				if s.logStream != nil {
					plyChild.Leftover = thisLeftover
				}

				// log.Debug().Msgf("Calculated leftover %v", plyChild.Leftover)
				if onTurn == s.initialPlayer {
					leftover += thisLeftover
				} else {
					leftover -= thisLeftover
				}
			}
			if s.logStream != nil {
				logPlay.Plies = append(logPlay.Plies, plyChild)
			}
			// Maybe these add{X}Stat functions can instead write them to
			// a channel to avoid mutices
			simmedPlay.addScoreStat(bestPlay, ply)

		}
		// log.Debug().Msgf("Spread for initial player: %v, leftover: %v",
		// 	s.game.SpreadFor(s.initialPlayer), leftover)
		spread := g.SpreadFor(s.initialPlayer)
		simmedPlay.addEquityStat(
			s.initialSpread,
			spread,
			leftover,
		)
		simmedPlay.addWinPctStat(
			spread,
			leftover,
			g.Playing() == pb.PlayState_GAME_OVER,
			s.winPcts,
			// Tiles unseen: number of tiles in the bag + tiles on my opponent's rack:
			g.Bag().TilesRemaining()+
				int(g.RackFor(1-s.initialPlayer).NumTiles()),
			plies%2 == 0,
		)
		g.ResetToFirstState()
		if s.logStream != nil {
			logPlay.WinRatio = simmedPlay.winPctStats.Last()
			logIter.Plays = append(logIter.Plays, logPlay)
		}
	}
	if s.logStream != nil {
		var out []byte
		if s.collectHeatMap && iterationCount < MaxHeatMapIterations {
			// If heat map collection is on, only collect a fixed number of iterations.
			out, err = json.Marshal(logIter)
			if err != nil {
				logger.Error().Err(err).Msg("marshalling log")
				return err
			}
			out = append(out, '\n')
			logChan <- out
		} else if !s.collectHeatMap {
			out, err = yaml.Marshal([]LogIteration{logIter})
			if err != nil {
				logger.Error().Err(err).Msg("marshalling log")
				return err
			}
			logChan <- out
		}

	}
	return nil
}

func (s *Simmer) bestStaticTurn(playerID, thread int) *move.Move {
	if !s.stochasticStaticEval {
		return aiturnplayer.GenBestStaticTurn(s.gameCopies[thread], s.aiplayers[thread], playerID)
	}
	return aiturnplayer.GenStochasticStaticTurn(s.gameCopies[thread], s.aiplayers[thread], playerID)
}

func (s *Simmer) sortPlaysByEquity() {
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	s.simmedPlays.Lock()
	defer s.simmedPlays.Unlock()
	sort.Slice(s.simmedPlays.plays, func(i, j int) bool {
		return s.simmedPlays.plays[i].equityStats.Mean() > s.simmedPlays.plays[j].equityStats.Mean()
	})
}

func (s *Simmer) sortPlaysByWinRate(ignoredAtBottom bool) {
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	s.simmedPlays.Lock()
	defer s.simmedPlays.Unlock()
	sort.Slice(s.simmedPlays.plays, func(i, j int) bool {
		if ignoredAtBottom {
			if s.simmedPlays.plays[i].ignore {
				return false
			}
			if s.simmedPlays.plays[j].ignore {
				return true
			}
		}
		if s.simmedPlays.plays[i].winPctStats.Mean() == s.simmedPlays.plays[j].winPctStats.Mean() {
			return s.simmedPlays.plays[i].equityStats.Mean() > s.simmedPlays.plays[j].equityStats.Mean()
		}
		return s.simmedPlays.plays[i].winPctStats.Mean() > s.simmedPlays.plays[j].winPctStats.Mean()
	})
}

func (s *Simmer) printStats() string {
	return s.EquityStats() + "\n Details per play \n" + s.ScoreDetails()
}

func (s *Simmer) EquityStats() string {
	var ss strings.Builder
	s.sortPlaysByWinRate(false)
	fmt.Fprintf(&ss, "%-20s%-14s%-9s%-16s%-16s\n", "Play", "Leave", "Score", "Win%", "Equity")
	s.simmedPlays.RLock()
	defer s.simmedPlays.RUnlock()
	for _, play := range s.simmedPlays.plays {
		wpStats := fmt.Sprintf("%.2f±%.2f", 100.0*play.winPctStats.Mean(), 100.0*stats.Z99*play.winPctStats.StandardError())
		eqStats := fmt.Sprintf("%.2f±%.2f", play.equityStats.Mean(), stats.Z99*play.equityStats.StandardError())
		ignore := ""
		if play.ignore {
			ignore = "❌"
		}

		fmt.Fprintf(&ss, "%-20s%-14s%-9d%-16s%-16s%s\n",
			s.origGame.Board().MoveDescriptionWithPlaythrough(play.play),
			play.play.LeaveString(),
			play.play.Score(), wpStats, eqStats, ignore)
	}
	fmt.Fprintf(&ss, "Iterations: %d (intervals are 99%% confidence, ❌ marks plays cut off early)\n", s.iterationCount.Load())
	return ss.String()
}

func (s *Simmer) ScoreDetails() string {
	if s.simmedPlays == nil {
		return "No simmed plays."
	}
	stats := ""
	s.sortPlaysByWinRate(false)
	s.simmedPlays.RLock()
	defer s.simmedPlays.RUnlock()

	for ply := 0; ply < s.maxPlies; ply++ {
		who := "You"
		if ply%2 == 0 {
			who = "Opponent"
		}
		stats += fmt.Sprintf("**Ply %d (%s)**\n%-20s%-14s%8s%8s%8s%8s%8s\n%s\n",
			ply+1, who, "Play", "Leave", "Win%", "Mean", "Stdev", "Bingo %", "Iters", strings.Repeat("-", 75))
		for _, play := range s.simmedPlays.plays {
			stats += fmt.Sprintf("%-20s%-14s%8.2f%8.3f%8.3f%8.3f%8d\n",
				s.origGame.Board().MoveDescriptionWithPlaythrough(play.play),
				play.play.LeaveString(),
				100.0*play.winPctStats.Mean(),
				play.scoreStats[ply].Mean(), play.scoreStats[ply].Stdev(),
				100.0*play.bingoStats[ply].Mean(), play.scoreStats[ply].Iterations())
		}
		stats += "\n"
	}
	stats += fmt.Sprintf("Iterations: %d\n", s.iterationCount.Load())
	return stats
}

func (s *Simmer) ShortDetails(nplays int) string {
	var ss strings.Builder

	s.sortPlaysByWinRate(false)

	s.simmedPlays.RLock()
	defer s.simmedPlays.RUnlock()

	plays := s.simmedPlays.plays
	if len(plays) > nplays {
		plays = plays[:nplays]
	}

	for idx, play := range plays {
		fmt.Fprintf(&ss, "%d) %s (%.1f%%)  ", idx+1,
			s.origGame.Board().MoveDescriptionWithPlaythrough(play.play), 100.0*play.winPctStats.Mean())
	}
	fmt.Fprintf(&ss, "; iters = %d", s.iterationCount.Load())
	return ss.String()

}

// SimSingleThread is a fast utility function to sim on a single thread with
// a fixed number of iterations and an optional stopping condition.
func (s *Simmer) SimSingleThread(iters, plies int) {
	ctx := context.Background()
	s.iterationCount.Store(0)

	writer := &errgroup.Group{}
	logChan := make(chan []byte)
	done := make(chan bool)

	if s.logStream != nil {
		// I lied, it's not single thread if we're debug logging.
		writer.Go(func() error {
			for {
				select {
				case bytes := <-logChan:
					s.logStream.Write(bytes)
				case <-done:
					return nil
				}
			}
		})
	}

	for i := range uint64(iters) {

		s.simSingleIteration(ctx, plies, 0, i, logChan)

		// check if we need to stop
		if s.autostopper.stoppingCondition != StopNone {
			if (i+1)%s.autostopper.stopConditionCheckInterval == 0 {
				stop := s.autostopper.shouldStop(i+1, s.simmedPlays, plies)
				if stop {
					log.Debug().Uint64("numIters", i+1).Msg("reached stopping condition")
					break
				}
			}
		}
	}
	if s.logStream != nil {
		close(done)
		err := writer.Wait()
		if err != nil {
			log.Err(err).Msg("simsinglethread-errgroup-error")
		}
	}
	s.iterationCount.Add(uint64(iters))
}

func (s *Simmer) WinningPlay() *SimmedPlay {
	s.sortPlaysByWinRate(true)
	s.simmedPlays.RLock()
	defer s.simmedPlays.RUnlock()
	return s.simmedPlays.plays[0]
}

func (s *Simmer) SetAutostopPPScaling(i int) {
	s.autostopper.perPlyStopScaling = i
}

func (s *Simmer) SetAutostopIterationsCutoff(i int) {
	s.autostopper.iterationsCutoff = i
}

func (s *Simmer) SetAutostopCheckInterval(i uint64) {
	s.autostopper.stopConditionCheckInterval = i
}

// PlaysByWinProb returns the simmedplays structure sorted by win probability.
func (s *Simmer) PlaysByWinProb() *SimmedPlays {
	s.sortPlaysByWinRate(true)
	return s.simmedPlays
}

func bagCount(bag []tilemapping.MachineLetter) map[tilemapping.MachineLetter]int {
	counts := map[tilemapping.MachineLetter]int{}
	for _, t := range bag {
		counts[t]++
	}
	return counts
}

func (s *Simmer) weightedInferredDrawTiles() ([]tilemapping.MachineLetter, error) {
	chosen := make([]tilemapping.MachineLetter, 0, s.tilesToInfer)

	unseenMap := map[tilemapping.MachineLetter]int{}

	for k, v := range s.unseenToSimmingPlayer {
		unseenMap[k] = v
	}

	for i := 0; i < s.tilesToInfer; i++ {
		// Build a list of tiles that still have count > 0
		tiles := []tilemapping.MachineLetter{}
		weights := []float64{}
		for tile, c := range unseenMap {
			if c > 0 {
				w := s.adjustedBagProbabilities[tile]
				// If w <= 0, skip as well
				if w > 0 {
					tiles = append(tiles, tile)
					weights = append(weights, w)
				}
			}
		}
		if len(tiles) == 0 {
			break // no more tiles can be drawn
		}
		// Randomly pick 1 tile from this distribution
		picked, err := weightedChoice(tiles, weights)
		if err != nil {
			return chosen, err
		}
		chosen = append(chosen, picked)
		// Decrement the actual bag count
		unseenMap[picked]--
		if unseenMap[picked] == 0 {
			delete(unseenMap, picked)
		}
	}
	return chosen, nil
}

// weightedChoice selects one element from tiles based on the provided weights
func weightedChoice(tiles []tilemapping.MachineLetter, weights []float64) (tilemapping.MachineLetter, error) {
	if len(tiles) != len(weights) || len(tiles) == 0 {
		return 0, errors.New("tiles and weights must be of same non-zero length")
	}

	// Calculate the cumulative weights
	cumulative := make([]float64, len(weights))
	cumulative[0] = weights[0]
	for i := 1; i < len(weights); i++ {
		cumulative[i] = cumulative[i-1] + weights[i]
	}

	// Generate a random number between 0 and total weight
	r := rand.Float64() * cumulative[len(cumulative)-1]

	// Find the first cumulative weight that is greater than r
	for i, cw := range cumulative {
		if r < cw {
			return tiles[i], nil
		}
	}

	return 0, errors.New("weighted choice failed to select a tile")
}

func (s *Simmer) weightedInferredDrawRacks() ([]tilemapping.MachineLetter, error) {

	picked, err := weightedChoiceRack(s.inferences)
	if err != nil {
		return nil, err
	}
	return picked, nil
}

// weightedChoice selects one element from tiles based on the provided weights
func weightedChoiceRack(choices map[*[]tilemapping.MachineLetter]float64) ([]tilemapping.MachineLetter, error) {
	// turn into two arrays.
	racks := [][]tilemapping.MachineLetter{}
	weights := []float64{}
	for k, v := range choices {
		racks = append(racks, *k)
		weights = append(weights, v)
	}

	// Calculate the cumulative weights
	cumulative := make([]float64, len(weights))

	cumulative[0] = weights[0]
	for i := 1; i < len(weights); i++ {
		cumulative[i] = cumulative[i-1] + weights[i]
	}

	// Generate a random number between 0 and total weight
	r := rand.Float64() * cumulative[len(cumulative)-1]

	// Find the first cumulative weight that is greater than r
	for i, cw := range cumulative {
		if r < cw {
			return racks[i], nil
		}
	}

	return nil, errors.New("weighted choice failed to select a rack")
}

func (s *Simmer) calculateWeightedProbabilitiesForBag() {
	// Calculate weighted probabilities for the bag.
	totalTiles := float64(0)
	tileCounts := map[tilemapping.MachineLetter]float64{}
	for rack, weight := range s.inferences {
		for _, t := range *rack {
			tileCounts[t] += weight
			totalTiles += weight
		}
	}
	tileProbabilities := map[tilemapping.MachineLetter]float64{}
	for t, c := range tileCounts {
		tileProbabilities[t] = float64(c) / float64(totalTiles)
	}
	s.adjustedBagProbabilities = make([]float64, s.origGame.Alphabet().NumLetters())
	for i := range s.origGame.Alphabet().NumLetters() {
		s.adjustedBagProbabilities[i] = tileProbabilities[tilemapping.MachineLetter(i)]
	}
	// calculate unseen to simming player
	s.unseenToSimmingPlayer = map[tilemapping.MachineLetter]int{}
	for _, t := range s.origGame.Bag().Peek() {
		s.unseenToSimmingPlayer[t]++
	}
	// lower allocations later!
	for _, t := range s.origGame.RackFor(1 - s.initialPlayer).TilesOn() {
		s.unseenToSimmingPlayer[t]++
	}

	log.Debug().Interface("bag-probabilities", s.adjustedBagProbabilities).Msg("calculated-weighted-bag")
}
