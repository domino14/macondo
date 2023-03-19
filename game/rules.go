package game

import (
	"errors"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/tilemapping"
)

type Variant string

const (
	VarClassic  Variant = "classic"
	VarWordSmog         = "wordsmog"
	// Redundant information, but we are deciding to treat different board
	// layouts as different variants.
	VarClassicSuper  = "classic_super"
	VarWordSmogSuper = "wordsmog_super"
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
	dist        *tilemapping.LetterDistribution
	lexicon     lexicon.Lexicon
	crossSetGen cross_set.Generator
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

func (g GameRules) LetterDistribution() *tilemapping.LetterDistribution {
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

func (g GameRules) CrossSetGen() cross_set.Generator {
	return g.crossSetGen
}

func (g GameRules) Variant() Variant {
	return g.variant
}

func NewBasicGameRules(cfg *config.Config,
	lexiconName, boardLayoutName, letterDistributionName, csetGenName string,
	variant Variant) (*GameRules, error) {

	dist, err := tilemapping.GetDistribution(cfg, letterDistributionName)
	if err != nil {
		return nil, err
	}

	var bd []string
	switch boardLayoutName {
	case board.CrosswordGameLayout, "":
		bd = board.CrosswordGameBoard
	case board.SuperCrosswordGameLayout:
		bd = board.SuperCrosswordGameBoard
	default:
		return nil, errors.New("unsupported board layout")
	}

	var lex lexicon.Lexicon
	var csgen cross_set.Generator
	switch csetGenName {
	case CrossScoreOnly:
		if lexiconName == "" {
			lex = &lexicon.AcceptAll{Alph: dist.TileMapping()}
		} else {
			k, err := kwg.Get(cfg, lexiconName)
			if err != nil {
				return nil, err
			}
			lex = &kwg.Lexicon{KWG: *k}
		}
		csgen = &cross_set.CrossScoreOnlyGenerator{Dist: dist}
	case CrossScoreAndSet:
		if lexiconName == "" {
			return nil, errors.New("lexicon name is required for this cross-set option")
		} else {
			k, err := kwg.Get(cfg, lexiconName)
			if err != nil {
				return nil, err
			}
			lex = &kwg.Lexicon{KWG: *k}
			csgen = &cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: k}
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
