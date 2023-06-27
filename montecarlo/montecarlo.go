// Package montecarlo implements truncated monte-carlo search
// during the regular game. In other words, "simming".
package montecarlo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"lukechampine.com/frand"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/common"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/stats"
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v2"
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

type StoppingCondition int

const (
	StopNone StoppingCondition = iota
	Stop95
	Stop98
	Stop99
)

type InferenceMode int

const (
	InferenceOff InferenceMode = iota
	InferenceCycle
	InferenceRandom
)

const HugeNumber = 1000000
const MaxNegamaxPlays = 50

// LogIteration is a struct meant for serializing to a log-file, for debug
// and other purposes.
type LogIteration struct {
	Iteration int       `json:"iteration" yaml:"iteration"`
	Plays     []LogPlay `json:"plays" yaml:"plays"`
	Thread    int       `json:"thread" yaml:"thread"`
}

// LogPlay is a single play.
type LogPlay struct {
	Play string `json:"play" yaml:"play"`
	Rack string `json:"rack" yaml:"rack"`
	Pts  int    `json:"pts" yaml:"pts"`
	// Leftover is the equity of the leftover tiles at the end of the sim.
	Leftover float64 `json:"left,omitempty" yaml:"left,omitempty"`
	// Although this is a recursive structure we don't really use it
	// recursively.
	WinRatio float64   `json:"win,omitempty" yaml:"win,omitempty"`
	Plies    []LogPlay `json:"plies,omitempty" yaml:"plies,omitempty,flow"`
}

type SimmedPlay struct {
	sync.RWMutex
	play          *move.Move
	scoreStats    []stats.Statistic
	bingoStats    []stats.Statistic
	equityStats   stats.Statistic
	leftoverStats stats.Statistic
	winPctStats   stats.Statistic
	ignore        bool
}

func (sp *SimmedPlay) String() string {
	return fmt.Sprintf("<Simmed play: %v (stats: %v %v %v %v %v)>", sp.play.ShortDescription(),
		sp.scoreStats, sp.bingoStats, sp.equityStats, sp.leftoverStats, sp.winPctStats)
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

type extraPlayData struct {
	playState   pb.PlayState
	tilesUnseen int
}

// Simmer implements the actual look-ahead search
type Simmer struct {
	origGame *game.Game

	gameCopies        []*game.Game
	movegens          []movegen.MoveGenerator
	equityCalculators []equity.EquityCalculator
	aiplayers         []aiturnplayer.AITurnPlayer
	leaveValues       equity.Leaves

	initialSpread int
	maxPlies      int
	// initialPlayer is the player for whom we are simming.
	initialPlayer  int
	iterationCount int
	threads        int

	negamaxMode bool

	simming      bool
	readyToSim   bool
	plays        []*SimmedPlay
	winPcts      [][]float32
	cfg          *config.Config
	knownOppRack []tilemapping.MachineLetter

	logStream         io.Writer
	stoppingCondition StoppingCondition

	// See rangefinder.
	inferences    [][]tilemapping.MachineLetter
	inferenceMode InferenceMode
}

func (s *Simmer) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	leaves equity.Leaves, cfg *config.Config) {
	s.origGame = game
	s.stoppingCondition = StopNone
	s.equityCalculators = eqCalcs
	s.leaveValues = leaves
	s.threads = int(math.Max(1, float64(runtime.NumCPU()-1)))

	// Hard-code the location of the win-pct file for now.
	// If we want to make some for other lexica in the future we'll
	// have to redo the equity calculator stuff.
	s.cfg = cfg
	if s.cfg != nil {
		// some hardcoded stuff here:
		winpct, err := cache.Load(s.cfg, "winpctfile:CSW:winpct.csv", equity.WinPCTLoadFunc)
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

func (s *Simmer) SetNegamaxMode(n bool) {
	s.negamaxMode = n
}

func (s *Simmer) SetStoppingCondition(sc StoppingCondition) {
	s.stoppingCondition = sc
}

func (s *Simmer) SetThreads(threads int) {
	s.threads = threads
}

func (s *Simmer) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Simmer) SetKnownOppRack(r []tilemapping.MachineLetter) {
	s.knownOppRack = r
}

func (s *Simmer) SetInferences(i [][]tilemapping.MachineLetter, mode InferenceMode) {
	s.inferences = i
	s.inferenceMode = mode
}

func (s *Simmer) makeGameCopies() error {
	log.Debug().Int("threads", s.threads).Msg("makeGameCopies")
	s.gameCopies = []*game.Game{}
	s.movegens = []movegen.MoveGenerator{}
	s.aiplayers = []aiturnplayer.AITurnPlayer{}
	// Pre-shuffle bag so we can make identical copies of it with fixedOrder
	s.origGame.Bag().Shuffle()

	for i := 0; i < s.threads; i++ {
		s.gameCopies = append(s.gameCopies, s.origGame.Copy())
		s.gameCopies[i].Bag().SetFixedOrder(true)
		s.gameCopies[i].SetBackupMode(game.SimulationMode)

		player, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(s.gameCopies[i], s.origGame.Config(), s.equityCalculators)
		if err != nil {
			return err
		}
		s.aiplayers = append(s.aiplayers, player)
		// used by the negamax engine:
		s.movegens = append(s.movegens, movegen.NewGordonGenerator(
			player.MoveGenerator().(*movegen.GordonGenerator).GADDAG(),
			s.gameCopies[i].Board(),
			s.gameCopies[i].Bag().LetterDistribution()))
		s.movegens[i].SetSortingParameter(movegen.SortByScore)
		s.movegens[i].SetGenPass(true)
		s.movegens[i].SetPlayRecorder(movegen.AllPlaysRecorder)

	}
	return nil

}

func (s *Simmer) resetStats(plies int, plays []*move.Move) {
	s.iterationCount = 0
	s.maxPlies = plies
	for _, g := range s.gameCopies {
		g.SetStateStackLength(plies + 1)
	}
	s.initialSpread = s.gameCopies[0].CurrentSpread()
	s.initialPlayer = s.gameCopies[0].PlayerOnTurn()
	s.plays = make([]*SimmedPlay, len(plays))
	for idx, play := range plays {
		s.plays[idx] = &SimmedPlay{}
		s.plays[idx].play = play
		s.plays[idx].scoreStats = make([]stats.Statistic, plies)
		s.plays[idx].bingoStats = make([]stats.Statistic, plies)
	}

}

func (s *Simmer) IsSimming() bool {
	return s.simming
}

func (s *Simmer) Reset() {
	s.plays = nil
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
	if len(s.plays) == 0 || len(s.gameCopies) == 0 {
		return errors.New("please prepare the simulation first")
	}

	s.simming = true
	defer func() {
		s.simming = false
		log.Info().Int("plies", s.maxPlies).Int("iterationCt", s.iterationCount).
			Msg("sim-ended")
	}()

	// use an errgroup here and listen for a ctx done outside this loop, but
	// in another goroutine.
	// protect the simmed play statistics with a mutex.
	log.Debug().Msgf("Simulating with %v threads", s.threads)
	syncExitChan := make(chan bool, s.threads)

	logChan := make(chan []byte)
	done := make(chan bool)

	ctrl := errgroup.Group{}
	writer := errgroup.Group{}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ctrl.Go(func() error {
		defer func() {
			log.Debug().Msgf("Sim controller thread exiting")
		}()
		for range ctx.Done() {
		}
		log.Debug().Msgf("Context is done: %v", ctx.Err())
		for t := 0; t < s.threads; t++ {
			syncExitChan <- true
		}
		// Send another exit signal to the stopping condition monitor
		if s.stoppingCondition != StopNone {
			syncExitChan <- true
		}
		log.Debug().Msgf("Sent sync messages to children threads...")

		return ctx.Err()
	})

	if s.logStream != nil {

		writer.Go(func() error {
			defer func() {
				log.Debug().Msgf("Writer routine exiting")
			}()
			for {
				select {
				case bytes := <-logChan:
					s.logStream.Write(bytes)
				case <-done:
					// Ok, actually quit now.
					log.Debug().Msgf("Got quit signal...")
					return nil
				}
			}
		})
	}

	var iterMutex sync.Mutex
	g := errgroup.Group{}
	for t := 0; t < s.threads; t++ {
		t := t
		g.Go(func() error {
			defer func() {
				log.Debug().Msgf("Thread %v exiting sim", t)
			}()
			log.Debug().Msgf("Thread %v starting sim", t)
			for {

				iterMutex.Lock()
				s.iterationCount++
				iterNum := s.iterationCount
				iterMutex.Unlock()
				if !s.negamaxMode {
					err := s.simSingleIteration(s.maxPlies, t, iterNum, logChan)
					if err != nil {
						log.Err(err).Msg("error simming iteration; canceling")
						cancel()
					}
				} else {
					err := s.simSingleIterationNegamax(s.maxPlies, t, iterNum, logChan)
					if err != nil {
						log.Err(err).Msg("error simming iteration; canceling")
						cancel()
					}
				}

				select {
				case v := <-syncExitChan:
					log.Debug().Msgf("Thread %v got sync msg %v", t, v)
					return nil
				default:
					// Do nothing
				}
			}
		})
	}

	if s.stoppingCondition != StopNone {
		// If there is some sort of programmatic stopping condition, we must
		// monitor the stats to determine when to stop!
		ctrl.Go(func() error {
			defer func() {
				log.Debug().Msg("stoppingcondition monitor exiting")
			}()
			interval := time.Duration(1) * time.Second
			tk := time.NewTicker(interval)
			playSimilarityCache := map[string]bool{}
			for {
				select {
				case <-syncExitChan:
					log.Debug().Msg("stopping condition monitor got exit signal")
					return nil
				case <-tk.C:
					stop := shouldStop(s.plays, s.stoppingCondition, s.iterationCount, playSimilarityCache)
					if stop {
						cancel()
					}
				}
			}

		})
	}

	// Wait for threads in errgroup:
	err := g.Wait()
	log.Debug().Msgf("errgroup returned err %v", err)

	// Writer thread will exit now:
	if s.logStream != nil {
		close(done)
		writer.Wait()
	}

	ctrlErr := ctrl.Wait()
	log.Debug().Msgf("ctrl errgroup returned err %v", ctrlErr)
	// sort plays at the end anyway.
	s.sortPlaysByWinRate()
	if ctrlErr == context.Canceled || ctrlErr == context.DeadlineExceeded {
		// Not actually an error
		log.Debug().AnErr("ctrlErr", ctrlErr).Msg("montecarlo-it's ok, not an error")
		return nil
	}
	return ctrlErr
}

func (s *Simmer) Iterations() int {
	return s.iterationCount
}

func (s *Simmer) TrimBottom(totrim int) error {
	if s.simming {
		return errors.New("please stop sim before trimming plays")
	}
	if totrim > len(s.plays)-1 {
		return errors.New("there are not that many plays to trim away")
	}
	s.plays = s.plays[:len(s.plays)-totrim]
	return nil
}

func (s *Simmer) simSingleIteration(plies, thread, iterationCount int, logChan chan []byte) error {
	// Give opponent a random rack from the bag. Note that this also
	// shuffles the bag!

	g := s.gameCopies[thread]

	opp := (s.initialPlayer + 1) % g.NumPlayers()
	rackToSet := s.knownOppRack
	if s.inferenceMode == InferenceCycle {
		rackToSet = s.inferences[iterationCount%len(s.inferences)]
	} else if s.inferenceMode == InferenceRandom {
		rackToSet = s.inferences[frand.Intn(len(s.inferences))]
	}
	_, err := g.SetRandomRack(opp, rackToSet)
	if err != nil {
		return err
	}
	logIter := LogIteration{Iteration: iterationCount, Plays: nil, Thread: thread}

	var logPlay LogPlay
	var plyChild LogPlay
	for _, simmedPlay := range s.plays {
		if simmedPlay.ignore {
			continue
		}
		if s.logStream != nil {
			logPlay = LogPlay{Play: simmedPlay.play.ShortDescription(),
				Rack: simmedPlay.play.FullRack(),
				Pts:  simmedPlay.play.Score()}
		}
		// equity of the leftover tiles at the end of the sim
		leftover := float64(0.0)
		// logIter.Plays = append(logIter.Plays)
		// Play the move, and back up the game state.
		// log.Debug().Msgf("Playing move %v", play)'
		// Set the backup mode to simulation mode only to back up the first move:
		g.SetBackupMode(game.SimulationMode)
		g.PlayMove(simmedPlay.play, false, 0)
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
			// log.Debug().Msgf("Score is now %v", s.game.Score())
			if s.logStream != nil {
				plyChild = LogPlay{Play: bestPlay.ShortDescription(), Rack: bestPlay.FullRack(), Pts: bestPlay.Score()}
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
		out, err := yaml.Marshal([]LogIteration{logIter})
		if err != nil {
			log.Error().Err(err).Msg("marshalling log")
			return err
		}
		logChan <- out
	}
	return nil
}

func (s *Simmer) simSingleIterationNegamax(plies, thread, iterationCount int, logChan chan []byte) error {
	// Give opponent a random rack from the bag. Note that this also
	// shuffles the bag!

	g := s.gameCopies[thread]

	opp := (s.initialPlayer + 1) % g.NumPlayers()
	rackToSet := s.knownOppRack
	if s.inferenceMode == InferenceCycle {
		rackToSet = s.inferences[iterationCount%len(s.inferences)]
	} else if s.inferenceMode == InferenceRandom {
		rackToSet = s.inferences[frand.Intn(len(s.inferences))]
	}
	_, err := g.SetRandomRack(opp, rackToSet)
	if err != nil {
		return err
	}
	logIter := LogIteration{Iteration: iterationCount, Plays: nil, Thread: thread}
	// var logPlay LogPlay
	// var plyChild LogPlay
	for _, simmedPlay := range s.plays {
		pv := common.PVLine{}
		g.PlayMove(simmedPlay.play, false, 0)
		// common parlance for sim plies usually refers to sim plies + 1.
		// i.e. a 2-ply sim is our move, their move, our move.
		val := s.negamax(plies, -HugeNumber, HugeNumber, thread, &pv)
		// equity of the leftover tiles at the end of the sim
		leftover := float64(0.0)
		var lastMove, penultimateMove *move.Move
		var lmi, pmi int
		if len(pv.Moves) == 0 {

		} else if len(pv.Moves) == 1 {
			lastMove = pv.Moves[0]
			lmi = 0
		} else {
			lastMove = pv.Moves[len(pv.Moves)-1]
			penultimateMove = pv.Moves[len(pv.Moves)-2]
			lmi = len(pv.Moves) - 1
			pmi = len(pv.Moves) - 2
		}
		// In the sim we make a play, the PV has a list of all plays afterwards.
		// Therefore, even indexes (0, 2, etc) is our opponents, odds are us.
		if lastMove != nil {
			thisLeftover := s.leaveValues.LeaveValue(lastMove.Leave())
			if lmi%2 != 0 { // odds, our leftover
				leftover += thisLeftover
			} else {
				leftover -= thisLeftover
			}
		}
		if penultimateMove != nil {
			thisLeftover := s.leaveValues.LeaveValue(penultimateMove.Leave())
			if pmi%2 != 0 { // odds, our leftover
				leftover += thisLeftover
			} else {
				leftover -= thisLeftover
			}
		}
		spread := int(-val) - s.initialSpread
		simmedPlay.addEquityStat(
			s.initialSpread,
			spread,
			leftover,
		)
		var extraData *extraPlayData
		var ok bool
		if lastMove != nil {
			extraData, ok = lastMove.UglyBagOfData().(*extraPlayData)
			if !ok {
				return errors.New("not found ugly bag of data")
			}
		}
		simmedPlay.addWinPctStat(
			spread,
			leftover,
			extraData.playState == pb.PlayState_GAME_OVER,
			s.winPcts,
			extraData.tilesUnseen,
			plies%2 == 0,
		)
		g.UnplayLastMove() // go back to beginning state

	}

	if s.logStream != nil {
		out, err := yaml.Marshal([]LogIteration{logIter})
		if err != nil {
			log.Error().Err(err).Msg("marshalling log")
			return err
		}
		logChan <- out
	}
	return nil
}

func (s *Simmer) generateSTMPlays(thread int) []*move.Move {
	// STM means side-to-move
	g := s.gameCopies[thread]
	mg := s.movegens[thread]

	var genPlays []*move.Move

	stmRack := g.RackFor(g.PlayerOnTurn())
	plays := mg.GenAll(stmRack, g.Bag().TilesRemaining() >= game.ExchangeLimit)
	// movegen owns the plays array. Make a copy of these.
	if len(plays) > MaxNegamaxPlays {
		plays = plays[:MaxNegamaxPlays]
	}
	genPlays = make([]*move.Move, len(plays))
	for idx := range plays {
		genPlays[idx] = &move.Move{}
		genPlays[idx].CopyFrom(plays[idx])
	}
	// Plays are already sorted by score by the movegen.
	return genPlays
}

func (s *Simmer) negamax(depth int, α, β float32, thread int, pv *common.PVLine) float32 {
	g := s.gameCopies[thread]

	if depth == 0 || g.Playing() != pb.PlayState_PLAYING {
		// Evaluate the state.
		// A very simple evaluation function for now. Just the current spread,
		// even if the game is not over yet.
		spreadNow := g.SpreadFor(g.PlayerOnTurn())
		return float32(spreadNow)
	}
	childPV := common.PVLine{}
	children := s.generateSTMPlays(thread)

	bestValue := float32(-HugeNumber)
	for _, child := range children {
		err := g.PlayMove(child, false, 0)
		playing := g.Playing()
		tilesUnseen := g.Bag().TilesRemaining() +
			int(g.RackFor(1-s.initialPlayer).NumTiles())
		if err != nil {
			log.Err(err).Msg("error-playing-move")
			return 0
		}

		value := s.negamax(depth-1, -β, -α, thread, &childPV)

		g.UnplayLastMove()
		if -value > bestValue {
			bestValue = -value
			child.SetUglyBagOfData(&extraPlayData{
				playState:   playing,
				tilesUnseen: tilesUnseen,
			})
			pv.Update(child, childPV, int16(bestValue))
		}
		if bestValue > α {
			α = bestValue
		}
		if bestValue >= β {
			break // beta cut-off
		}
		childPV.Clear()
	}
	return bestValue

}

func (s *Simmer) bestStaticTurn(playerID, thread int) *move.Move {
	return aiturnplayer.GenBestStaticTurn(s.gameCopies[thread], s.aiplayers[thread], playerID)
}

func (s *Simmer) sortPlaysByEquity() {
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].equityStats.Mean() > s.plays[j].equityStats.Mean()
	})
}

func (s *Simmer) sortPlaysByWinRate() {
	// log.Debug().Msgf("Sorting plays: %v", s.plays)
	sort.Slice(s.plays, func(i, j int) bool {
		if s.plays[i].winPctStats.Mean() == s.plays[j].winPctStats.Mean() {
			return s.plays[i].equityStats.Mean() > s.plays[j].equityStats.Mean()
		}
		return s.plays[i].winPctStats.Mean() > s.plays[j].winPctStats.Mean()
	})
}

func (s *Simmer) printStats() string {
	return s.EquityStats() + "\n Details per play \n" + s.ScoreDetails()
}

func (s *Simmer) EquityStats() string {
	var ss strings.Builder
	s.sortPlaysByWinRate()
	fmt.Fprintf(&ss, "%-20s%-9s%-16s%-16s\n", "Play", "Score", "Win%", "Equity")

	for _, play := range s.plays {
		wpStats := fmt.Sprintf("%.3f±%.3f", 100.0*play.winPctStats.Mean(), 100.0*play.winPctStats.StandardError(stats.Z99))
		eqStats := fmt.Sprintf("%.3f±%.3f", play.equityStats.Mean(), play.equityStats.StandardError(stats.Z99))
		ignore := ""
		if play.ignore {
			ignore = "❌"
		}

		fmt.Fprintf(&ss, "%-20s%-9d%-16s%-16s%s\n", play.play.ShortDescription(),
			play.play.Score(), wpStats, eqStats, ignore)
	}
	fmt.Fprintf(&ss, "Iterations: %d (intervals are 99%% confidence, ❌ marks plays cut off early)\n", s.iterationCount)
	return ss.String()
}

func (s *Simmer) ScoreDetails() string {
	stats := ""
	s.sortPlaysByWinRate()
	for ply := 0; ply < s.maxPlies; ply++ {
		who := "You"
		if ply%2 == 0 {
			who = "Opponent"
		}
		stats += fmt.Sprintf("**Ply %d (%s)**\n%-20s%8s%8s%8s%8s%8s\n%s\n",
			ply+1, who, "Play", "Win%", "Mean", "Stdev", "Bingo %", "Iters", strings.Repeat("-", 60))
		for _, play := range s.plays {
			stats += fmt.Sprintf("%-20s%8.2f%8.3f%8.3f%8.3f%8d\n",
				play.play.ShortDescription(), 100.0*play.winPctStats.Mean(),
				play.scoreStats[ply].Mean(), play.scoreStats[ply].Stdev(),
				100.0*play.bingoStats[ply].Mean(), play.scoreStats[ply].Iterations())
		}
		stats += "\n"
	}
	stats += fmt.Sprintf("Iterations: %d\n", s.iterationCount)
	return stats
}

func (s *Simmer) WinningPlays() []*SimmedPlay {
	s.sortPlaysByWinRate()
	return s.plays
}
