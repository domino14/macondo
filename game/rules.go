package game

import (
	"path/filepath"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
)

// gamerules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type gamerules struct {
	cfg    *config.Config
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

func (g *gamerules) LoadRule(lexicon, letterDistributionName string) error {
	gdFilename := filepath.Join(g.cfg.LexiconPath, "gaddag", lexicon+".gaddag")
	gd, err := gaddag.LoadGaddag(gdFilename)
	if err != nil {
		return err
	}
	dist := alphabet.NamedLetterDistribution(letterDistributionName, gd.GetAlphabet())
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
	return rules, nil
}
