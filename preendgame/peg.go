package preendgame

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/zobrist"
)

var ErrCanceledEarly = errors.New("canceled early")

const InBagMaxLimit = 6
const TieBreakerPlays = 20

type PEGOutcome int

const (
	PEGNotInitialized PEGOutcome = 0
	PEGWin            PEGOutcome = 1
	PEGDraw           PEGOutcome = 2
	PEGLoss           PEGOutcome = 3
)

func (p PEGOutcome) String() string {
	switch p {
	case PEGNotInitialized:
		return "not initialized"
	case PEGWin:
		return "win"
	case PEGDraw:
		return "draw"
	case PEGLoss:
		return "loss"
	}
	return "?"
}

type Outcome struct {
	tiles     []tilemapping.MachineLetter
	ct        int
	outcome   PEGOutcome
	Finalized bool
}

// Equal tells whether a and b contain the same elements.
// A nil argument is equivalent to an empty slice.
func Equal(a, b []tilemapping.MachineLetter) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

type PreEndgamePlay struct {
	sync.RWMutex
	Play          *move.Move
	Points        float32
	FoundLosses   float32
	Spread        int
	spreadSet     bool
	outcomesArray []Outcome
	Ignore        bool
}

func (p *PreEndgamePlay) Copy() *PreEndgamePlay {
	// don't copy the mutex.
	return &PreEndgamePlay{
		Play:          p.Play, // shallow copy
		Points:        p.Points,
		FoundLosses:   p.FoundLosses,
		Spread:        p.Spread,
		spreadSet:     p.spreadSet,
		Ignore:        p.Ignore,
		outcomesArray: p.outcomesArray, // shallow copy
	}
}

func (p *PreEndgamePlay) stopAnalyzing() {
	p.Lock()
	defer p.Unlock()
	p.Ignore = true
}

func (p *PreEndgamePlay) outcomeIndex(tiles []tilemapping.MachineLetter) int {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		found = len(p.outcomesArray)
		p.outcomesArray = append(p.outcomesArray, Outcome{
			tiles: tiles,
		})
	}
	return found
}

func (p *PreEndgamePlay) addWinPctStat(result PEGOutcome, ct int, tiles []tilemapping.MachineLetter) {
	p.Lock()
	defer p.Unlock()
	found := p.outcomeIndex(tiles)
	p.outcomesArray[found].outcome = result
	p.outcomesArray[found].ct += ct
	switch result {
	case PEGWin:
		p.Points += float32(ct)
	case PEGDraw:
		p.Points += float32(ct) / 2
		p.FoundLosses += float32(ct) / 2
	case PEGLoss:
		// no wins
		p.FoundLosses += float32(ct)
	}
	p.outcomesArray[found].Finalized = true
}

func (p *PreEndgamePlay) addSpreadStat(spread, ct int) {
	p.Lock()
	defer p.Unlock()
	p.spreadSet = true
	p.Spread += (spread * ct)
}

func (p *PreEndgamePlay) finalize() {
	p.Lock()
	defer p.Unlock()
	for idx := range p.outcomesArray {
		if p.outcomesArray[idx].Finalized {
			continue
		}
		// otherwise assign points/losses accordingly
		ct := p.outcomesArray[idx].ct
		switch p.outcomesArray[idx].outcome {
		case PEGWin:
			p.Points += float32(ct)
		case PEGDraw:
			p.Points += float32(ct) / 2
			p.FoundLosses += float32(ct) / 2
		case PEGLoss:
			// no wins
			p.FoundLosses += float32(ct)
		}
		p.outcomesArray[idx].Finalized = true
	}

}

func (p *PreEndgamePlay) setUnfinalizedWinPctStat(unfinalizedResult PEGOutcome, ct int, tiles []tilemapping.MachineLetter) {
	p.Lock()
	defer p.Unlock()
	found := p.outcomeIndex(tiles)
	if p.outcomesArray[found].outcome == PEGNotInitialized {
		p.outcomesArray[found].ct = ct
	}

	// If any draw is found for a combination of tiles, that whole
	// combination gets classified as a draw at best, and a loss at worst.
	// If any loss is found for a combination of tiles, that whole
	// combination is a loss.
	// If any win is found for a combination of tiles, we must make
	// sure that they're ALL wins before calling it a win.

	switch unfinalizedResult {
	case PEGWin:
		if p.outcomesArray[found].outcome != PEGDraw &&
			p.outcomesArray[found].outcome != PEGLoss {

			p.outcomesArray[found].outcome = PEGWin
		}
	case PEGDraw:
		if p.outcomesArray[found].outcome != PEGLoss {
			p.outcomesArray[found].outcome = PEGDraw
		}
	case PEGLoss:
		p.outcomesArray[found].outcome = PEGLoss
	}

}

func (p *PreEndgamePlay) HasLoss(tiles []tilemapping.MachineLetter) bool {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		return false
	}
	p.RLock()
	defer p.RUnlock()
	return p.outcomesArray[found].outcome == PEGLoss
}

func (p *PreEndgamePlay) HasFinalizedOutcome(tiles []tilemapping.MachineLetter) bool {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		return false
	}
	p.RLock()
	defer p.RUnlock()
	return p.outcomesArray[found].Finalized
}

func (p *PreEndgamePlay) AllHaveLoss(tiles [][]tilemapping.MachineLetter) bool {
	p.RLock()
	defer p.RUnlock()
	for _, tileset := range tiles {
		found := -1
		for idx, outcome := range p.outcomesArray {
			if Equal(outcome.tiles, tileset) {
				found = idx
				if p.outcomesArray[idx].outcome != PEGLoss {
					return false
				}
				break
			}
		}
		if found == -1 {
			// We have no results for this tileset, so caller must search.
			return false
		}
	}
	return true
}

func (p *PreEndgamePlay) OutcomeFor(tiles []tilemapping.MachineLetter) PEGOutcome {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		return PEGNotInitialized
	}
	p.RLock()
	defer p.RUnlock()
	return p.outcomesArray[found].outcome
}

func (p *PreEndgamePlay) String() string {
	p.RLock()
	defer p.RUnlock()
	return fmt.Sprintf("<play %v, wins %f>", p.Play.ShortDescription(), p.Points)
}

type jobLog struct {
	PEGPlay              string         `yaml:"peg_play"`
	FoundLosses          int            `yaml:"found_losses"`
	MinPotentialLosses   int            `yaml:"min_potential_losses"`
	CutoffAtStart        bool           `yaml:"cutoff_at_start"`
	CutoffWhileIterating bool           `yaml:"cutoff_while_iterating"`
	PEGPlayEmptiesBag    bool           `yaml:"peg_play_empties_bag"`
	Options              []jobOptionLog `yaml:"options"`
	EndgamePlies         int            `yaml:"endgame_plies"`
}

type jobOptionLog struct {
	PermutationInBag              string `yaml:"perm_in_bag"`
	PermutationCount              int    `yaml:"perm_ct"`
	OppRack                       string `yaml:"opp_rack"`
	OurRack                       string `yaml:"our_rack"`
	CutoffBecauseAlreadyLoss      bool   `yaml:"cutoff_already_loss"`
	CutoffBecauseFinalizedOutcome bool   `yaml:"cutoff_finalized_outcome"`
	FinalSpread                   int    `yaml:"final_spread"`
	OppPerspective                bool   `yaml:"opp_perspective"`
	EndgameMoves                  string `yaml:"endgame_moves"`
	GameEnded                     bool   `yaml:"game_ended"`
	TimeToSolveMs                 int64  `yaml:"time_to_solve_ms"`
	// Options                  []jobOptionLog `yaml:"options"`
}

type Solver struct {
	endgameSolvers []*negamax.Solver

	movegen          movegen.MoveGenerator
	game             *game.Game
	gaddag           *kwg.KWG
	ttable           *negamax.TranspositionTable
	curEndgamePlies  int
	maxEndgamePlies  int
	initialSpread    int
	threads          int
	numinbag         int
	plays            []*PreEndgamePlay
	winnerSoFar      *PreEndgamePlay
	knownOppRack     []tilemapping.MachineLetter
	busy             bool
	solvingForPlayer int
	logStream        io.Writer
	solveOnlyMoves   []*move.Move

	earlyCutoffOptim     bool
	skipNonEmptyingOptim bool
	skipTiebreaker       bool
	skipLossOptim        bool
	iterativeDeepening   bool

	numEndgamesSolved    atomic.Uint64
	numCutoffs           atomic.Uint64
	potentialWinnerMutex sync.RWMutex
	minPotentialLosses   float32

	threadLogs []jobLog
}

// Init initializes the solver. It creates all the parallel endgame solvers.
func (s *Solver) Init(g *game.Game, gd *kwg.KWG) error {
	s.ttable = negamax.GlobalTranspositionTable
	s.threads = max(1, runtime.NumCPU())
	s.threadLogs = make([]jobLog, s.threads)
	s.ttable.SetMultiThreadedMode()
	s.game = g.Copy()
	s.game.SetBackupMode(game.SimulationMode)
	s.curEndgamePlies = 4
	s.maxEndgamePlies = s.curEndgamePlies
	s.iterativeDeepening = true
	s.gaddag = gd
	s.earlyCutoffOptim = true
	s.skipNonEmptyingOptim = false
	s.skipTiebreaker = false
	return nil
}

func (s *Solver) SetLogStream(l io.Writer) {
	s.logStream = l
}

func (s *Solver) Solve(ctx context.Context) ([]*PreEndgamePlay, error) {
	s.busy = true
	defer func() {
		s.busy = false
	}()
	ts := time.Now()
	log.Info().
		Int("endgame-plies", s.maxEndgamePlies).
		Bool("early-cutoff-optim", s.earlyCutoffOptim).
		Bool("skip-non-emptying-optim", s.skipNonEmptyingOptim).
		Bool("skip-tiebreaker-optim", s.skipTiebreaker).
		Bool("skip-loss-optim", s.skipLossOptim).
		Bool("iterative-deepening", s.iterativeDeepening).
		Int("threads", s.threads).
		Msg("preendgame-solve-called")

	if s.iterativeDeepening {
		s.curEndgamePlies = 1
	} else {
		s.curEndgamePlies = s.maxEndgamePlies
	}

	if s.game.RackFor(s.solvingForPlayer).NumTiles() < game.RackTileLimit {
		return nil, errors.New("the rack of the player being solved for must be fully specified")
	}

	s.numEndgamesSolved.Store(0)
	s.numCutoffs.Store(0)

	var winners []*PreEndgamePlay
	var err error

	s.movegen = movegen.NewGordonGenerator(s.gaddag, s.game.Board(), s.game.Bag().LetterDistribution())
	s.movegen.SetGenPass(true)
	// Don't allow pre-endgame opponent to use more than 7 tiles.
	s.movegen.SetMaxTileUsage(7)
	// Examine high equity plays first.
	var moves []*move.Move
	if len(s.solveOnlyMoves) != 0 {
		moves = s.solveOnlyMoves
	} else {
		moves = s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()), false)
	}
	c, err := equity.NewCombinedStaticCalculator(
		s.game.LexiconName(), s.game.Config(), "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	for _, m := range moves {
		m.SetEquity(c.Equity(m, s.game.Board(), s.game.Bag(), nil))
	}
	sort.Slice(moves, func(i, j int) bool {
		return moves[i].Equity() > moves[j].Equity()
	})
	s.ttable.Reset(s.game.Config().GetFloat64(config.ConfigTtableMemFraction), s.game.Board().Dim())
	var lastWinners []*PreEndgamePlay
	s.solvingForPlayer = s.game.PlayerOnTurn()

	writer := errgroup.Group{}
	logChan := make(chan []byte)
	done := make(chan bool)

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

	for s.curEndgamePlies <= s.maxEndgamePlies {
		if s.iterativeDeepening {
			log.Info().Int("endgame-plies", s.curEndgamePlies).Msg("iterative-deepening")
			if len(winners) > 0 {
				// sort moves by the last iteration's winners.
				moves = make([]*move.Move, len(winners))
				for widx, w := range winners {
					moves[widx] = w.Play
				}
				log.Info().Str("move", moves[0].ShortDescription()).Msg("last-iteration-winner")
				lastWinners = winners
			}
		}
		s.minPotentialLosses = 100000.0

		// Fill opponent's rack for now. Ignore the "known opp rack", if any. That
		// is handled properly later.
		if s.game.RackFor(1-s.solvingForPlayer).NumTiles() < game.RackTileLimit {
			_, err := s.game.SetRandomRack(1-s.solvingForPlayer, nil)
			if err != nil {
				return nil, err
			}
		}

		if s.game.Bag().TilesRemaining() > InBagMaxLimit {
			return nil, fmt.Errorf("bag has too many tiles remaining; limit is %d", InBagMaxLimit)
		} else if s.game.Bag().TilesRemaining() == 0 {
			return nil, errors.New("bag is empty; use endgame solver instead")
		}
		s.numinbag = s.game.Bag().TilesRemaining()
		s.endgameSolvers = make([]*negamax.Solver, s.threads)
		s.initialSpread = s.game.CurrentSpread()

		for idx := range s.endgameSolvers {
			es := &negamax.Solver{}
			// share the same transposition table and zobrist params across all endgames.
			// Copy the game so each endgame solver can manipulate it independently.
			g := s.game.Copy()
			g.SetBackupMode(game.SimulationMode)
			// we need to set the state stack length now to account for the PEG move.
			// there can also be passes etc. just add a hacky number.
			// XXX: state stack length should probably just be a fixed large number.
			g.SetStateStackLength(s.curEndgamePlies + 5)
			g.SetEndgameMode(true)
			// Set a fixed order for the bag. This makes it easy for us to control
			// what tiles we draw after making a move.
			g.Bag().SetFixedOrder(true)
			mg := movegen.NewGordonGenerator(s.gaddag, g.Board(), g.Bag().LetterDistribution())
			err := es.Init(mg, g)
			if err != nil {
				return nil, err
			}
			// Endgame itself should be single-threaded; we are solving many individual
			// endgames in parallel.
			es.SetThreads(1)

			// Endgame should quit early if it finds any win.
			es.SetFirstWinOptim(true)
			// Even though the endgame search window is already tiny, this still seems to help
			// for some reason:
			es.SetNegascoutOptim(true)
			s.endgameSolvers[idx] = es
			if s.logStream != nil {
				// Also create endgame loggers
				esLogFile, err := os.Create(fmt.Sprintf("./macondo-endgame-log-%d", idx))
				if err != nil {
					return nil, err
				}
				es.SetLogStream(esLogFile)
			}
		}

		log.Info().Int("nmoves", len(moves)).Int("nthreads", s.threads).Msg("peg-generated-moves")

		winners, err = s.multithreadSolveGeneric(ctx, moves, logChan)
		if err != nil {
			if err == ErrCanceledEarly {
				return lastWinners, nil
			}
			return winners, err
		}
		s.curEndgamePlies++
	}

	if s.logStream != nil {
		close(done)
		writer.Wait()
	}
	log.Info().Str("ttable-stats", s.ttable.Stats()).
		Float64("time-elapsed-sec", time.Since(ts).Seconds()).
		Msg("solve-returning")
	return winners, err
}

type job struct {
	ourMove         *PreEndgamePlay
	fullSolve       bool
	maybeInBagTiles []int
}

func moveIsPossible(mtiles []tilemapping.MachineLetter, partialRack []tilemapping.MachineLetter) bool {
	// check whether m is a possible move given the total pool of tiles to choose from
	// (unseenRack) and the known partial rack.
	// Note: assumes that m can be made from unseenRack.
	// For example, the following state is impossible:
	// play: COOKIE
	// partial: KLL
	// bag: CEIKLLOO

	partialCopy := make([]int, zobrist.MaxLetters)
	pcount := 0
	for _, t := range partialRack {
		partialCopy[t]++
		pcount++
	}

	for _, t := range mtiles {
		if t == 0 {
			continue
		}
		t = t.IntrinsicTileIdx()
		if partialCopy[t] > 0 {
			partialCopy[t]--
			pcount--
		}
	}
	// unseen: LL
	// partial: LL
	// Try to re-add the letters to partialRack.
	for _, t := range mtiles {
		if t == 0 {
			continue
		}
		t = t.IntrinsicTileIdx()
		partialCopy[t]++
		pcount++
	}
	return pcount <= game.RackTileLimit
}

func (s *Solver) SetEndgamePlies(p int) {
	s.maxEndgamePlies = p
}

func (s *Solver) SetThreads(t int) {
	s.threads = t
	s.threadLogs = make([]jobLog, t)
}

func moveTilesToBeginning(order []tilemapping.MachineLetter, bag *tilemapping.Bag) {
	// move tiles to the beginning of the bag. The tiles should be in the order given.
	// (i.e. order[0] should be at the beginning of the bag)

	bagContents := bag.Tiles()
	lastPlacedIdx := 0

	for oidx := lastPlacedIdx; oidx < len(order); oidx++ {
		// We want to place the tile at order[oidx] in spot oidx in the bag.
		desiredTile := order[oidx]

		for idx := lastPlacedIdx; idx < len(bagContents); idx++ {
			if bagContents[idx] == desiredTile {
				bag.SwapTile(idx, lastPlacedIdx)
				lastPlacedIdx++
				break
			}
		}
	}
}

func possibleTilesInBag(unseenTiles []tilemapping.MachineLetter, moveTiles []tilemapping.MachineLetter,
	knownPlayerTiles []tilemapping.MachineLetter) []tilemapping.MachineLetter {
	// given a set of unseenTiles and a move, determine which of the unseen tiles
	// could be in the bag and still allow for move to be played.
	// Return a deduplicated set, essentially.

	ucTally := make([]tilemapping.MachineLetter, zobrist.MaxLetters)
	for _, u := range unseenTiles {
		ucTally[u]++
	}
	kpTally := make([]tilemapping.MachineLetter, zobrist.MaxLetters)
	for _, t := range knownPlayerTiles {
		kpTally[t]++
	}

	// ucTally contains tiles that are unseen and not in the move. However,
	// if we know that some of these tiles are in our hand, they cannot
	// possibly be in the bag.
	for _, t := range moveTiles {
		if t == 0 {
			continue // not a played tile
		}
		t = t.IntrinsicTileIdx()
		for _, u := range unseenTiles {
			if t == u {
				// First take it away from the known player tiles, if there
				if kpTally[u] > 0 {
					kpTally[u]--
				}
				ucTally[u]--
				break
			}
		}
	}
	for idx := range kpTally {
		ucTally[idx] -= kpTally[idx]
	}

	pt := []tilemapping.MachineLetter{}
	for idx, u := range ucTally {
		if u > 0 {
			pt = append(pt, tilemapping.MachineLetter(idx))
		}
	}
	return pt
}

func (s *Solver) SolutionStats(maxMoves int) string {
	// Assume plays are already sorted.
	var ss strings.Builder
	fmt.Fprintf(&ss, "%-20s%-8s%-8s%-9s%-32s%-2s\n", "Play", "Wins", "%Win", "Spread", "Outcomes", "")

	for _, play := range s.plays[:maxMoves] {
		fmt.Fprint(&ss, s.SingleSolutionStats(play, false))
	}
	fmt.Fprintf(&ss, "❌ marks plays cut off early\n")
	fmt.Fprintf(&ss, "[] brackets indicate order does not matter\n")

	return ss.String()
}

func (s *Solver) SingleSolutionStats(play *PreEndgamePlay, pctOnly bool) string {
	noutcomes := 0
	for _, el := range play.outcomesArray {
		noutcomes += el.ct
	}

	ignore := ""
	wpStats := "---"
	pts := "---"
	spdStats := ""
	if play.Ignore {
		ignore = "❌"
	} else {
		wpStats = fmt.Sprintf("%.2f", 100.0*play.Points/float32(noutcomes))
		pts = fmt.Sprintf("%.1f", play.Points)
		if play.spreadSet {
			spdStats = fmt.Sprintf("%.2f", float32(play.Spread)/float32(noutcomes))
		}
	}
	var wins, draws, losses []string
	var outcomeStr string
	for _, outcome := range play.outcomesArray {
		// uv := tilemapping.MachineWord(outcome.tiles).UserVisible(s.game.Alphabet())
		uf := toUserFriendly(outcome.tiles, s.game.Alphabet(), play.Play)
		if uf != "" {
			switch outcome.outcome {
			case PEGWin:
				wins = append(wins, uf)
			case PEGDraw:
				draws = append(draws, uf)
			case PEGLoss:
				losses = append(losses, uf)
			}
		}
	}
	slices.Sort(wins)
	slices.Sort(draws)
	slices.Sort(losses)

	if len(wins) > 0 {
		outcomeStr += fmt.Sprintf("👍: %s", strings.Join(wins, " "))
	}
	if len(draws) > 0 {
		outcomeStr += fmt.Sprintf(" 🤝: %s", strings.Join(draws, " "))
	}
	if len(losses) > 0 {
		outcomeStr += fmt.Sprintf(" 👎: %s", strings.Join(losses, " "))
	}
	if pctOnly {
		return fmt.Sprintf("%-20s%-8s%%%-32s", s.game.Board().MoveDescriptionWithPlaythrough(play.Play),
			wpStats, outcomeStr)
	}
	return fmt.Sprintf("%-20s%-8s%-8s%-9s%-32s%-2s\n",
		s.game.Board().MoveDescriptionWithPlaythrough(play.Play),
		pts, wpStats, spdStats, outcomeStr, ignore)
}

func (s *Solver) ShortDetails() string {
	return s.SingleSolutionStats(s.plays[0], true)
}

func (s *Solver) SetEarlyCutoffOptim(o bool) {
	s.earlyCutoffOptim = o
}

func (s *Solver) SetSkipNonEmptyingOptim(o bool) {
	s.skipNonEmptyingOptim = o
}

func (s *Solver) SetSkipLossOptim(o bool) {
	s.skipLossOptim = o
}

func (s *Solver) SetKnownOppRack(rack tilemapping.MachineWord) {
	s.knownOppRack = rack
}

func (s *Solver) SetSkipTiebreaker(o bool) {
	s.skipTiebreaker = o
}

func (s *Solver) SetIterativeDeepening(d bool) {
	s.iterativeDeepening = d
}

func (s *Solver) IsSolving() bool {
	return s.busy
}

func (s *Solver) SetSolveOnly(m []*move.Move) {
	s.solveOnlyMoves = m
}

func toUserFriendly(tileset []tilemapping.MachineLetter, alphabet *tilemapping.TileMapping, pegPlay *move.Move) string {

	// If our pre-endgame play plays off more than 1 tile, and the tileset we
	// are drawing is at least as long as the number of tiles played, order
	// doesn't matter for the tiles that we draw for this move.

	orderMatters := true
	tp := pegPlay.TilesPlayed()
	if tp > 1 {
		// order doesn't matter for the tp tiles played.
		// if order doesn't matter for this sub-tileset, we need to only
		// display one of the orders; default to the alphabetical one.
		begin := len(tileset) - tp
		if begin < 0 {
			begin = 0
		}
		if !slices.IsSortedFunc(tileset[begin:],
			func(a, b tilemapping.MachineLetter) int {
				return int(b) - int(a)
			}) {
			return ""
		}
		//
		orderMatters = false
	}

	tsCopy := make([]tilemapping.MachineLetter, len(tileset))
	copy(tsCopy, tileset)
	// Internally, the tilesets are represented in "right-to-left" ordering;
	// i.e. the order in which they are in the bag is right-to-left.
	// We wish to represent it to the user in reverse order; left-to-right,
	// as it is more intuitive.
	slices.Reverse(tsCopy)
	var ss strings.Builder
	wl := min(len(tsCopy), tp)

	if !orderMatters {
		ss.WriteString("[")
		ss.WriteString(tilemapping.MachineWord(tsCopy[:wl]).UserVisible(alphabet))
		ss.WriteString("]")
	} else {
		ss.WriteString(tilemapping.MachineWord(tsCopy[:wl]).UserVisible(alphabet))
	}
	if wl < len(tsCopy) {
		ss.WriteString(tilemapping.MachineWord(tsCopy[wl:]).UserVisible(alphabet))
	}

	return ss.String()
}
