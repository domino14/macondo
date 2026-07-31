package game

import (
	"errors"
	"fmt"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/alphadawg"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/lexicon"
)

type Variant string

const (
	VarClassic  Variant = "classic"
	VarWordSmog Variant = "wordsmog"
	// Redundant information, but we are deciding to treat different board
	// layouts as different variants.
	VarClassicSuper  Variant = "classic_super"
	VarWordSmogSuper Variant = "wordsmog_super"
)

// IsWordSmog reports whether v is one of the WordSmog variants, where any
// anagram of a valid word is playable.
func IsWordSmog(v Variant) bool {
	return v == VarWordSmog || v == VarWordSmogSuper
}

const (
	CrossScoreOnly   = "cs"
	CrossScoreAndSet = "css"
)

const DefaultExchangeLimit = 7

// GameRules is a simple struct that encapsulates the instantiated objects
// needed to actually play a game.
type GameRules struct {
	cfg           *config.Config
	board         *board.GameBoard
	dist          *tilemapping.LetterDistribution
	lexicon       lexicon.Lexicon
	crossSetGen   cross_set.Generator
	variant       Variant
	alphaDawg     *kwg.KWG
	boardname     string
	distname      string
	exchangeLimit int
}

// AlphaDawg returns the alpha dawg backing a WordSmog game, or nil for classic
// variants. Move generators need it to generate anagram plays.
func (g GameRules) AlphaDawg() *kwg.KWG {
	return g.alphaDawg
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

func (g *GameRules) SetExchangeLimit(l int) {
	g.exchangeLimit = l
}

func (g GameRules) ExchangeLimit() int {
	return g.exchangeLimit
}

func NewBasicGameRules(cfg *config.Config,
	lexiconName, boardLayoutName, letterDistributionName, csetGenName string,
	variant Variant) (*GameRules, error) {

	dist, err := tilemapping.GetDistribution(cfg.WGLConfig(), letterDistributionName)
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

	// WordSmog plays out of an alpha dawg -- a word graph of alphagrams -- so
	// both word validation and cross-set generation use it instead of the
	// gaddag. Leaves are the classic ones for the lexicon.
	wordSmog := IsWordSmog(variant)
	var kad *kwg.KWG
	if wordSmog && lexiconName != "" {
		if err := alphadawg.EnsureKAD(lexiconName, cfg.WGLConfig()); err != nil {
			log.Info().Err(err).Str("lexicon", lexiconName).
				Msg("could not download alpha dawg; will try to load from disk")
		}
		kad, err = alphadawg.Get(cfg.WGLConfig(), lexiconName, letterDistributionName)
		if err != nil {
			return nil, fmt.Errorf("WordSmog needs an alpha dawg for %s: %w", lexiconName, err)
		}
	}

	var lex lexicon.Lexicon
	var csgen cross_set.Generator
	switch csetGenName {
	case CrossScoreOnly:
		if lexiconName == "" {
			lex = &lexicon.AcceptAll{Alph: dist.TileMapping()}
		} else if wordSmog {
			lex = &alphadawg.Lexicon{KWG: kad}
		} else {
			k, err := kwg.GetKWG(cfg.WGLConfig(), lexiconName, kwg.WithDistribution(letterDistributionName))
			if err != nil {
				return nil, err
			}
			lex = &kwg.Lexicon{KWG: *k}
		}
		csgen = &cross_set.CrossScoreOnlyGenerator{Dist: dist}
	case CrossScoreAndSet:
		if lexiconName == "" {
			return nil, errors.New("lexicon name is required for this cross-set option")
		} else if wordSmog {
			lex = &alphadawg.Lexicon{KWG: kad}
			csgen = &cross_set.WordSmogCrossSetGenerator{Dist: dist, AlphaDawg: kad}
		} else {
			k, err := kwg.GetKWG(cfg.WGLConfig(), lexiconName, kwg.WithDistribution(letterDistributionName))
			if err != nil {
				return nil, err
			}
			lex = &kwg.Lexicon{KWG: *k}
			csgen = &cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: k}
		}
	}

	exchLimit := DefaultExchangeLimit
	if lexicon.IsSpanish(lexiconName) {
		// XXX: It's a little bit ghetto, for sure.
		exchLimit = 1
	}

	rules := &GameRules{
		cfg:           cfg,
		dist:          dist,
		distname:      letterDistributionName,
		board:         board.MakeBoard(bd),
		boardname:     boardLayoutName,
		lexicon:       lex,
		crossSetGen:   csgen,
		variant:       variant,
		alphaDawg:     kad,
		exchangeLimit: exchLimit,
	}
	return rules, nil
}

func MaxCanExchange(inbag, exchLimit int) int {
	if inbag < exchLimit {
		return 0
	}
	return min(inbag, RackTileLimit)
}
