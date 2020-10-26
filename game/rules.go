package game

import (
	"errors"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/lexicon"
)

// GameRules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type GameRules struct {
	cfg         *config.Config
	board       *board.GameBoard
	dist        *alphabet.LetterDistribution
	lexicon     lexicon.Lexicon
	crossSetGen cross_set.Generator
}

func (g GameRules) Config() *config.Config {
	return g.cfg
}

func (g GameRules) Board() *board.GameBoard {
	return g.board
}

func (g GameRules) LetterDistribution() *alphabet.LetterDistribution {
	return g.dist
}

func (g GameRules) Lexicon() lexicon.Lexicon {
	return g.lexicon
}

func (g GameRules) LexiconName() string {
	return g.lexicon.Name()
}

func (g GameRules) CrossSetGen() cross_set.Generator {
	return g.crossSetGen
}

func NewBasicGameRules(cfg *config.Config, boardLayout []string,
	letterDistributionName string) (*GameRules, error) {

	dist, err := cache.Load(cfg, "letterdist:"+letterDistributionName,
		alphabet.CacheLoadFunc)
	if err != nil {
		return nil, err
	}
	distLD, ok := dist.(*alphabet.LetterDistribution)
	if !ok {
		return nil, errors.New("type-assertion failed (letterDistribution)")
	}

	rules := &GameRules{
		cfg:         cfg,
		dist:        distLD,
		board:       board.MakeBoard(boardLayout),
		lexicon:     lexicon.AcceptAll{Alph: distLD.Alphabet()},
		crossSetGen: cross_set.CrossScoreOnlyGenerator{Dist: distLD},
	}
	return rules, nil
}

func NewGameRules(cfg *config.Config, dist *alphabet.LetterDistribution,
	board *board.GameBoard, lex lexicon.Lexicon, cset cross_set.Generator) *GameRules {
	return &GameRules{
		cfg:         cfg,
		dist:        dist,
		board:       board,
		lexicon:     lex,
		crossSetGen: cset,
	}
}
