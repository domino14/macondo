package game

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
	"github.com/rs/zerolog/log"
)

// gamerules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type gamerules struct {
	cfg         *config.Config
	board       *board.GameBoard
	dist        *alphabet.LetterDistribution
	gaddag      gaddag.GenericDawg
	lexicon     lexicon.Lexicon
	crossSetGen cross_set.Generator
}

func (g gamerules) Board() *board.GameBoard {
	return g.board
}

func (g gamerules) LetterDistribution() *alphabet.LetterDistribution {
	return g.dist
}

func (g gamerules) Gaddag() gaddag.GenericDawg {
	return g.gaddag
}

func (g gamerules) Lexicon() lexicon.Lexicon {
	return g.lexicon
}

func (g gamerules) CrossSetGen() cross_set.Generator {
	return g.crossSetGen
}

func (g *gamerules) LoadRule(lexicon, letterDistributionName string) error {
	gd, err := gaddag.LoadFromCache(g.cfg, lexicon)
	if err != nil {
		// Since a gaddag is not a hard requirement for a game (think of the
		// case where it's a player-vs-player game like a GCG) then
		// we don't necessarily exit if we can't load the gaddag.
		log.Err(err).Interface("gd", gd).Msg("unable to load gaddag; using default gaddag")
		gd, err = gaddag.GenericDawgCache.Get(g.cfg, g.cfg.DefaultLexicon)
		if err != nil {
			return err
		}
	}
	dist, err := alphabet.LoadLetterDistribution(g.cfg, letterDistributionName)
	if err != nil {
		return err
	}
	g.gaddag = gd
	g.dist = dist
	return nil
}

func NewGameRules(cfg *config.Config, boardLayout []string, lexicon string,
	letterDistributionName string) (*gamerules, error) {

	rules := &gamerules{cfg: cfg}

	board := board.MakeBoard(boardLayout)
	err := rules.LoadRule(lexicon, letterDistributionName)
	if err != nil {
		return nil, err
	}
	rules.board = board
	gd := rules.gaddag
	cset := &cross_set.GaddagCrossSetGenerator{
		Board:  board.Copy(),
		Gaddag: gd,
		Dist:   rules.dist,
	}
	lex := &gaddag.Lexicon{gd}
	rules.crossSetGen = cset
	rules.lexicon = lex
	return rules, nil
}
