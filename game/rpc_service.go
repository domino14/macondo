package game

import (
	"context"
	crypto_rand "crypto/rand"
	"encoding/binary"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/movegen"
	pb "github.com/domino14/macondo/rpc/api/proto"
)

// Note that we will have some singleton data structures here, even though
// this is a server. It is not meant to have multiple games loaded in memory
// at once. If we want to make this a part of a general Crossword PvP tool,
// we should only import the relevant packages, but handle all the database
// stuff in the main PvP tool.

// AnnotationService will be the main API that the front end will talk to
// to annotate, simulate, etc. a game. Note that the variables here are singletons
// for the whole service, as in the comment above.
// All of the methods of this service should be very thin wrappers so that the
// meat of the code is as reusable as possible.
type AnnotationService struct {
	game *Game
	cfg  *config.Config
	// Simmer:
	// simmer        *montecarlo.Simmer
	simCtx        context.Context
	simCancel     context.CancelFunc
	simTicker     *time.Ticker
	simTickerDone chan bool
	simLogFile    *os.File

	gen movegen.MoveGenerator
	// endgameSolver *alphabeta.Solver
}

// gamerules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type gamerules struct {
	board  *board.GameBoard
	dist   *alphabet.LetterDistribution
	gaddag *gaddag.SimpleGaddag
}

func (g gamerules) Board() *board.GameBoard {
	return g.board
}

func (g gamerules) LetterDistribution() *alphabet.LetterDistribution {
	return g.dist
}

func (g gamerules) Gaddag() *gaddag.SimpleGaddag {
	return g.gaddag
}

func seededRandSource() (int64, *rand.Rand) {
	var b [8]byte
	_, err := crypto_rand.Read(b[:])
	if err != nil {
		panic("cannot seed math/rand package with cryptographically secure random number generator")
	}

	randSeed := int64(binary.LittleEndian.Uint64(b[:]))
	randSource := rand.New(rand.NewSource(randSeed))

	return randSeed, randSource
}

func NewGameRules(cfg *config.Config, boardLayout []string, lexicon string,
	letterDistributionName string) (*gamerules, error) {

	board := board.MakeBoard(boardLayout)
	gdFilename := filepath.Join(cfg.LexiconPath, "gaddag", lexicon+".gaddag")
	gd, err := gaddag.LoadGaddag(gdFilename)
	if err != nil {
		return nil, err
	}
	dist := alphabet.NamedLetterDistribution(letterDistributionName, gd.GetAlphabet())
	// bag := dist.MakeBag(randSource)
	return &gamerules{board, dist, gd}, nil
}

func NewAnnotationService(cfg *config.Config) *AnnotationService {
	return &AnnotationService{cfg: cfg}
}

func (a *AnnotationService) NewGame(ctx context.Context, gameReq *pb.NewGameRequest) (
	*pb.GameHistory, error) {
	// log := zerolog.Ctx(ctx)

	var err error
	rules, err := NewGameRules(a.cfg, gameReq.BoardLayout, gameReq.Lexicon,
		gameReq.LetterDistribution)
	if err != nil {
		return nil, err
	}

	a.game, err = NewGame(rules, gameReq.Players)
	if err != nil {
		return nil, err
	}
	return a.game.history, nil
}
