package game

import (
	"path/filepath"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddag"
	"github.com/rs/zerolog/log"
)

// gamerules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type gamerules struct {
	cfg    *config.Config
	board  *board.GameBoard
	dist   *alphabet.LetterDistribution
	gaddag gaddag.GenericDawg
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

func (g *gamerules) LoadRule(lexicon, letterDistributionName string) error {
	gdFilename := filepath.Join(g.cfg.LexiconPath, "gaddag", lexicon+".gaddag")
	gd, err := gaddag.LoadGaddag(gdFilename)
	if err != nil {
		// Since a gaddag is not a hard requirement for a game (think of the
		// case where it's a player-vs-player game like a GCG) then
		// we don't necessarily exit if we can't load the gaddag.
		log.Err(err).Msg("unable to load gaddag; operation may be unideal")
	}
	dist := alphabet.NamedLetterDistribution(g.cfg, g.letterDistributionName, gd.GetAlphabet())
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
