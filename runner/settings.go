package runner

import (
	"errors"
	"fmt"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/rs/zerolog/log"
)

type Lexicon struct {
	Name         string
	Distribution string
}

func (lex *Lexicon) ToDisplayString() string {
	return fmt.Sprintf("%s [%s]", lex.Name, lex.Distribution)
}

type GameOptions struct {
	Lexicon         *Lexicon
	ChallengeRule   pb.ChallengeRule
	FirstIsAssigned bool
	GoesFirst       int
	BoardLayoutName string
}

func (opts *GameOptions) SetDefaults(config *config.Config) {
	if opts.Lexicon == nil {
		opts.Lexicon = &Lexicon{config.DefaultLexicon, config.DefaultLetterDistribution}
		log.Info().Msgf("using default lexicon %v", opts.Lexicon)
	}
	if opts.BoardLayoutName == "" {
		opts.BoardLayoutName = board.CrosswordGameLayout
		log.Info().Msgf("using default board layout %v", opts.BoardLayoutName)
	}
}

func (opts *GameOptions) SetLexicon(fields []string) error {
	lexname := ""
	letdist := "english"
	if len(fields) == 1 {
		lexname = fields[0]
	} else if len(fields) == 2 {
		lexname, letdist = fields[0], fields[1]
	} else {
		msg := "Valid formats are 'lexicon' and 'lexicon alphabet'"
		return errors.New(msg)
	}
	opts.Lexicon = &Lexicon{Name: lexname, Distribution: letdist}
	return nil
}

func (opts *GameOptions) SetChallenge(rule string) error {
	val, err := ParseChallengeRule(rule)
	if err != nil {
		return err
	}
	opts.ChallengeRule = val
	return nil
}

func (opts *GameOptions) SetBoardLayoutName(name string) error {
	switch name {
	case board.CrosswordGameLayout, board.SuperCrosswordGameLayout:
		opts.BoardLayoutName = name
	default:
		return fmt.Errorf("%v is not a supported board layout", name)
	}
	return nil
}
