package game

import (
	"errors"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/crosses"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/lexicon"
)

type Variant string

const (
	VarClassic  Variant = "classic"
	VarWordSmog Variant = "wordsmog"
)

const (
	CrossScoreOnly   = "cs"
	CrossScoreAndSet = "css"
)

// GameRules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type GameRules struct {
	cfg         *config.Config
	board       *board.GameBoard
	dist        *alphabet.LetterDistribution
	lexicon     lexicon.Lexicon
	crossSetGen crosses.Generator
	variant     Variant
	boardname   string
	distname    string
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

func (g GameRules) BoardName() string {
	return g.boardname
}

func (g GameRules) LetterDistributionName() string {
	return g.distname
}

func (g GameRules) CrossSetGen() crosses.Generator {
	return g.crossSetGen
}

func (g GameRules) Variant() Variant {
	return g.variant
}

func NewBasicGameRules(cfg *config.Config,
	lexiconName, boardLayoutName, letterDistributionName, csetGenName string,
	variant Variant) (*GameRules, error) {

	dist, err := alphabet.Get(cfg, letterDistributionName)
	if err != nil {
		return nil, err
	}

	var bd []string
	switch boardLayoutName {
	case board.CrosswordGameLayout, "":
		bd = board.CrosswordGameBoard
	default:
		return nil, errors.New("unsupported board layout")
	}

	var lex lexicon.Lexicon
	var csgen crosses.Generator
	switch csetGenName {
	case CrossScoreOnly:
		// just use dawg
		if lexiconName == "" {
			lex = &lexicon.AcceptAll{Alph: dist.Alphabet()}
		} else {
			dawg, err := gaddag.GetDawg(cfg, lexiconName)
			if err != nil {
				return nil, err
			}
			lex = &gaddag.Lexicon{GenericDawg: dawg}
		}
		csgen = crosses.CrossScoreOnlyGenerator{Dist: dist}
	case CrossScoreAndSet:
		if lexiconName == "" {
			return nil, errors.New("lexicon name is required for this cross-set option")
		} else {
			gd, err := gaddag.Get(cfg, lexiconName)
			if err != nil {
				return nil, err
			}
			lex = &gaddag.Lexicon{GenericDawg: gd}
			// We don't instantiate the cross set generator here, because
			// we need to use other additional state.
			csgen = nil // cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
		}
	}

	rules := &GameRules{
		cfg:         cfg,
		dist:        dist,
		distname:    letterDistributionName,
		board:       board.MakeBoard(bd),
		boardname:   boardLayoutName,
		lexicon:     lex,
		crossSetGen: csgen,
		variant:     variant,
	}
	return rules, nil
}
