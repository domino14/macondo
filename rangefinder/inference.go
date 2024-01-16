package rangefinder

import (
	"context"
	"errors"
	"io"
	"runtime"
	"sort"
	"sync"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/proto"
	"gopkg.in/yaml.v2"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var ErrMoveTypeNotSupported = errors.New("opponent move type not suitable for inference")
var ErrNoEvents = errors.New("no events")
var ErrBagEmpty = errors.New("bag is empty")
var ErrNoInformation = errors.New("not enough information to infer")

const (
	// If the player found a play within this limit, then count the rack
	// for inferences.
	InferenceEquityLimit = 3
)

type LogIteration struct {
	Iteration     int     `json:"iteration" yaml:"iteration"`
	Thread        int     `json:"thread" yaml:"thread"`
	Rack          string  `json:"rack" yaml:"rack"`
	TopMoveEquity float64 `json:"topMoveEquity" yaml:"topMoveEquity"`
	TopMove       string  `json:"topMove" yaml:"topMove"`
	// InferredMoveEquity is the equity of the move we are inferring, given
	// that they drew "Rack"
	InferredMoveEquity float64 `json:"inferredMoveEquity" yaml:"inferredMoveEquity"`
	PossibleRack       bool    `json:"possibleRack" yaml:"possibleRack"`
}

type RangeFinder struct {
	origGame          *game.Game
	gameCopies        []*game.Game
	equityCalculators []equity.EquityCalculator
	aiplayers         []*aiturnplayer.AIStaticTurnPlayer
	iterationCount    int
	threads           int

	working      bool
	readyToInfer bool

	inferenceBagMap []uint8
	cfg             *config.Config
	lastOppMove     *move.Move
	// tiles used by the last opponent's move, from their rack:
	lastOppMoveRackTiles []tilemapping.MachineLetter
	inferences           [][]tilemapping.MachineLetter

	logStream io.Writer
}

func (r *RangeFinder) Init(game *game.Game, eqCalcs []equity.EquityCalculator,
	cfg *config.Config) {

	r.origGame = game
	r.equityCalculators = eqCalcs
	r.threads = max(1, runtime.NumCPU())
	r.cfg = cfg
}

func (r *RangeFinder) SetThreads(t int) {
	r.threads = t
}

func (r *RangeFinder) SetLogStream(l io.Writer) {
	r.logStream = l
}

func (r *RangeFinder) PrepareFinder(myRack []tilemapping.MachineLetter) error {
	r.inferences = [][]tilemapping.MachineLetter{}
	evts := r.origGame.History().Events[:r.origGame.Turn()]
	if len(evts) == 0 {
		return ErrNoEvents
	}
	if r.origGame.Bag().TilesRemaining() == 0 {
		return ErrBagEmpty
	}

	oppEvtIdx := len(evts) - 1
	oppIdx := evts[oppEvtIdx].PlayerIndex
	var oppEvt *macondo.GameEvent
	foundOppEvent := false
	for oppEvtIdx >= 0 {
		oppEvt = evts[oppEvtIdx]
		if oppEvt.PlayerIndex != oppIdx {
			break
		}
		if oppEvt.Type == macondo.GameEvent_CHALLENGE_BONUS {
			oppEvtIdx--
			continue
		}
		if oppEvt.Type == macondo.GameEvent_EXCHANGE || oppEvt.Type == macondo.GameEvent_TILE_PLACEMENT_MOVE {
			foundOppEvent = true
			break
		}
		oppEvtIdx--
	}
	if !foundOppEvent {
		return ErrMoveTypeNotSupported
	}

	// We must reset the game back to what it looked like before the opp's move.
	var gameCopy *game.Game
	var err error

	history := proto.Clone(r.origGame.History()).(*macondo.GameHistory)
	history.Events = history.Events[:oppEvtIdx]

	if r.origGame.History().StartingCgp != "" {

		parsedCGP, err := cgp.ParseCGP(r.cfg, r.origGame.History().StartingCgp)
		if err != nil {
			return err
		}
		gameCopy = parsedCGP.Game
		gameCopy.History().Events = history.Events

		for t := 0; t < len(history.Events); t++ {
			err = gameCopy.PlayTurn(t)
			if err != nil {
				return err
			}
		}
		gameCopy.SetPlayerOnTurn(int(oppIdx))
		gameCopy.RecalculateBoard()
	} else {
		gameCopy, err = game.NewFromHistory(history, r.origGame.Rules(), len(history.Events))
		if err != nil {
			return err
		}
	}

	// create rack from the last move.
	r.lastOppMove, err = game.MoveFromEvent(oppEvt, r.origGame.Alphabet(), gameCopy.Board())
	if err != nil {
		return err
	}

	if r.lastOppMove.TilesPlayed() == game.RackTileLimit {
		return ErrNoInformation
	}
	r.lastOppMoveRackTiles = []tilemapping.MachineLetter{}
	for _, t := range r.lastOppMove.Tiles() {
		if t == 0 {
			// 0 is the played-through marker when part of a played move.
			continue
		}
		ml := t.IntrinsicTileIdx()
		r.lastOppMoveRackTiles = append(r.lastOppMoveRackTiles, ml)
	}

	// Sort to make it easy to compare to other plays:
	sort.Slice(r.lastOppMoveRackTiles, func(i, j int) bool {
		return r.lastOppMoveRackTiles[i] < r.lastOppMoveRackTiles[j]
	})
	gameCopy.ThrowRacksIn()

	if len(myRack) > 0 {
		// Assign my rack first, so that the inferencer doesn't try to
		// assign letters from my rack.
		rack := tilemapping.NewRack(r.origGame.Alphabet())
		rack.Set(myRack)
		err = gameCopy.SetRackForOnly(1-gameCopy.PlayerOnTurn(), rack)
		if err != nil {
			return err
		}
	}

	r.inferenceBagMap = gameCopy.Bag().PeekMap()
	if oppEvt.Type == macondo.GameEvent_TILE_PLACEMENT_MOVE {

		_, err = gameCopy.SetRandomRack(gameCopy.PlayerOnTurn(), r.lastOppMoveRackTiles)
		if err != nil {
			return err
		}
		// Save the state of the bag after we assign the random rack. Remove only
		// lastOppMove rack but nothing else.
		for _, ml := range r.lastOppMoveRackTiles {
			r.inferenceBagMap[ml]--
		}
	} else if oppEvt.Type == macondo.GameEvent_EXCHANGE {
		// If this is an exchange move, lastOppMove etc is just a guess.
		// Set any random rack, and don't remove anything from the bagMap.
		// The bagMap already contains the tiles we're about to assign to
		// the user here:
		gameCopy.SetRandomRack(gameCopy.PlayerOnTurn(), nil)
	}
	r.gameCopies = []*game.Game{}
	r.aiplayers = []*aiturnplayer.AIStaticTurnPlayer{}

	for i := 0; i < r.threads; i++ {
		r.gameCopies = append(r.gameCopies, gameCopy.Copy())

		player, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(r.gameCopies[i], r.origGame.Config(), r.equityCalculators)
		if err != nil {
			return err
		}
		r.aiplayers = append(r.aiplayers, player)
	}

	r.readyToInfer = true
	r.iterationCount = 0
	return nil
}

func (r *RangeFinder) Infer(ctx context.Context) error {
	if !r.readyToInfer {
		return errors.New("not ready")
	}
	r.working = true
	defer func() {
		r.working = false
		log.Info().Msg("inference engine quitting")
	}()

	logChan := make(chan []byte)
	syncExitChan := make(chan bool, r.threads)
	logDone := make(chan bool)

	ctrl := errgroup.Group{}
	writer := errgroup.Group{}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ctrl.Go(func() error {
		defer func() {
			log.Debug().Msgf("Inference engine controller thread exiting")
		}()
		for range ctx.Done() {
		}
		log.Debug().Msgf("Context is done: %v", ctx.Err())
		for t := 0; t < r.threads; t++ {
			syncExitChan <- true
		}
		log.Debug().Msgf("Sent sync-exit messages to children threads...")

		return ctx.Err()
	})

	if r.logStream != nil {
		writer.Go(func() error {
			defer func() {
				log.Debug().Msgf("Writer routine exiting")
			}()
			for {
				select {
				case bytes := <-logChan:
					r.logStream.Write(bytes)
				case <-logDone:
					log.Debug().Msgf("Got quit signal...")
					return nil
				}
			}
		})
	}

	g := errgroup.Group{}
	var iterMutex sync.Mutex
	for t := 0; t < r.threads; t++ {
		t := t
		g.Go(func() error {
			defer func() {
				log.Debug().Msgf("Thread %v exiting inferrer", t)
			}()
			log.Debug().Msgf("Thread %v starting inferrer", t)
			for {
				iterMutex.Lock()
				r.iterationCount++
				iterNum := r.iterationCount
				iterMutex.Unlock()
				inference, err := r.inferSingle(t, iterNum, logChan)
				if err != nil {
					log.Err(err).Msg("infer-single-error")
					cancel()
				}
				if len(inference) > 0 {
					iterMutex.Lock()
					r.inferences = append(r.inferences, inference...)
					iterMutex.Unlock()
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

	err := g.Wait()
	log.Debug().Msgf("errgroup returned err %v", err)

	if r.logStream != nil {
		close(logDone)
		writer.Wait()
	}

	ctrlErr := ctrl.Wait()
	log.Debug().Msgf("ctrl errgroup returned err %v", ctrlErr)

	if ctrlErr == context.Canceled || ctrlErr == context.DeadlineExceeded {
		// Not actually an error
		log.Debug().AnErr("ctrlErr", ctrlErr).Msg("inferencer-it's ok, not an error")
		return nil
	}
	return ctrlErr

}

func (r *RangeFinder) inferSingle(thread, iterNum int, logChan chan []byte) ([][]tilemapping.MachineLetter, error) {
	g := r.gameCopies[thread]
	// Since we took back the last move, the player on turn should be our opponent
	// (the person whose rack we are inferring)
	opp := g.PlayerOnTurn()
	var extraDrawn []tilemapping.MachineLetter
	var err error
	isExchange := r.lastOppMove.Action() == move.MoveTypeExchange
	// otherwise, it's a tile placement play.

	if isExchange {
		return r.inferSingleExchange(thread, iterNum, logChan)
	}

	extraDrawn, err = g.SetRandomRack(opp, r.lastOppMoveRackTiles)
	if err != nil {
		return nil, err
	}

	logIter := LogIteration{Iteration: iterNum, Thread: thread, Rack: g.RackLettersFor(opp)}
	log.Trace().Interface("extra-drawn", extraDrawn).Msg("extra-drawn")

	bestMoves := r.aiplayers[thread].GenerateMoves(20)
	winningEquity := bestMoves[0].Equity()
	if r.logStream != nil {
		logIter.TopMove = bestMoves[0].ShortDescription()
		logIter.TopMoveEquity = winningEquity
	}

	var inferences [][]tilemapping.MachineLetter
	for _, m := range bestMoves {
		if m.Equity()+InferenceEquityLimit >= winningEquity {
			// consider this move
			if movesAreKindaTheSame(m, r.lastOppMove, r.lastOppMoveRackTiles, g.Board()) {
				// copy extraDrawn, as setRandomRack does not allocate for it.
				tiles := make([]tilemapping.MachineLetter, len(extraDrawn))
				copy(tiles, extraDrawn)

				if r.logStream != nil {
					logIter.InferredMoveEquity = m.Equity()
					logIter.PossibleRack = true
					out, err := yaml.Marshal([]LogIteration{logIter})
					if err != nil {
						log.Err(err).Msg("marshalling log")
						return nil, err
					}
					logChan <- out
				}
				return [][]tilemapping.MachineLetter{tiles}, nil
			}
		}
	}

	// if not found, calculate equity anyway; it might be a phony play
	// with high equity and we may want to infer anyway.
	m := new(move.Move)
	m.CopyFrom(r.lastOppMove)
	leave, err := tilemapping.Leave(g.RackFor(opp).TilesOn(), m.Tiles(), false)
	if err != nil {
		return nil, err
	}
	m.SetLeave(leave)
	r.aiplayers[thread].AssignEquity([]*move.Move{m}, g.Board(), g.Bag(), g.RackFor(1-opp))
	if m.Equity()+InferenceEquityLimit >= winningEquity {
		tiles := make([]tilemapping.MachineLetter, len(extraDrawn))
		copy(tiles, extraDrawn)
		inferences = append(inferences, tiles)
		logIter.PossibleRack = true
	}
	if r.logStream != nil {
		logIter.InferredMoveEquity = m.Equity()
		out, err := yaml.Marshal([]LogIteration{logIter})
		if err != nil {
			log.Err(err).Msg("marshalling log")
			return nil, err
		}
		logChan <- out
	}
	return inferences, nil
}

func (r *RangeFinder) inferSingleExchange(thread, iterNum int, logChan chan []byte) ([][]tilemapping.MachineLetter, error) {
	g := r.gameCopies[thread]
	// Since we took back the last move, the player on turn should be our opponent
	// (the person whose rack we are inferring)
	opp := g.PlayerOnTurn()
	g.SetRandomRack(opp, nil)
	logIter := LogIteration{Iteration: iterNum, Thread: thread, Rack: g.RackLettersFor(opp)}

	bestMoves := r.aiplayers[thread].GenerateMoves(20)
	winningEquity := bestMoves[0].Equity()
	if r.logStream != nil {
		logIter.TopMove = bestMoves[0].ShortDescription()
		logIter.TopMoveEquity = winningEquity
	}
	var ret [][]tilemapping.MachineLetter
	var tiles []tilemapping.MachineLetter
	// Infer more than one move if possible, since "movesAreSame" returns
	// true for exchanges with the same number of tiles.
	for _, m := range bestMoves {
		if m.Equity()+InferenceEquityLimit >= winningEquity {
			// consider this move
			if m.TilesPlayed() == r.lastOppMove.TilesPlayed() {
				// We just want to copy the new leave
				tiles = make([]tilemapping.MachineLetter, len(m.Leave()))
				copy(tiles, m.Leave())
				if r.logStream != nil {
					logIter.InferredMoveEquity = m.Equity()
					logIter.PossibleRack = true
					out, err := yaml.Marshal([]LogIteration{logIter})
					if err != nil {
						log.Err(err).Msg("marshalling log")
						return nil, err
					}
					logChan <- out
				}
				ret = append(ret, tiles)
			}
		}
	}
	return ret, nil
}

func (r *RangeFinder) Inferences() [][]tilemapping.MachineLetter {
	return r.inferences
}

func (r *RangeFinder) Reset() {
	r.inferences = [][]tilemapping.MachineLetter{}
	r.readyToInfer = false
}

func (r *RangeFinder) IsBusy() bool {
	return r.working
}

func movesAreKindaTheSame(m1 *move.Move, m2 *move.Move, m2tiles []tilemapping.MachineLetter,
	g *board.GameBoard) bool {
	// This is a bit of a fuzzy equality function. We want to see if two
	// tile-play moves are "materially" the same.
	// If they're tile play moves, and they use the same tiles, we will
	// call them the same, even if the plays were in different places.
	// This is because the person we're inferring for may have missed
	// a play using the same tiles in a better spot.

	checkTransposition := false
	if g.IsEmpty() {
		checkTransposition = true
	}
	ignoreLeave := true
	if m1.Equals(m2, checkTransposition, ignoreLeave) {
		return true
	}
	// Otherwise check if it's a single-tile move.
	if m1.TilesPlayed() == 1 && m2.TilesPlayed() == 1 &&
		uniqueSingleTileKey(m1) == uniqueSingleTileKey(m2) {
		return true
	}

	// Otherwise, check if they use the same tiles.
	m1tiles := []tilemapping.MachineLetter{}
	for _, t := range m1.Tiles() {
		if t == 0 {
			continue
		}
		ml := t.IntrinsicTileIdx()
		m1tiles = append(m1tiles, ml)
	}
	if len(m1tiles) != len(m2tiles) {
		return false
	}
	sort.Slice(m1tiles, func(i, j int) bool { return m1tiles[i] < m1tiles[j] })
	// m2tiles is already sorted.
	allEqual := true
	for i := range m1tiles {
		// Take into account the blank. If the player played a blank and
		// missed a better play with a blank being another letter, it's still
		// "the same play". This is handled by the calls to "IntrinsicTileIdx" above.
		if m1tiles[i] != m2tiles[i] {
			allEqual = false
			break
		}
	}
	return allEqual
}

func uniqueSingleTileKey(m *move.Move) int {
	// Find the tile.
	var idx int
	var tile tilemapping.MachineLetter
	for idx, tile = range m.Tiles() {
		if tile != 0 {
			break
		}
	}
	row, col, vert := m.CoordsAndVertical()
	// We want to get the coordinate of the tile that is on the board itself.
	if vert {
		row += idx
	} else {
		col += idx
	}
	// A unique, fast to compute key for this play.
	return row + tilemapping.MaxAlphabetSize*col +
		tilemapping.MaxAlphabetSize*tilemapping.MaxAlphabetSize*int(tile)
}
