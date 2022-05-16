package runner

import (
	"errors"
	"fmt"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"strings"
	"unicode"
)

func ParseChallengeRule(rule string) (pb.ChallengeRule, error) {
	var challRule pb.ChallengeRule
	switch rule {
	case "void":
		challRule = pb.ChallengeRule_VOID
	case "single":
		challRule = pb.ChallengeRule_SINGLE
	case "double":
		challRule = pb.ChallengeRule_DOUBLE
	case "triple":
		challRule = pb.ChallengeRule_TRIPLE
	case "5pt":
		challRule = pb.ChallengeRule_FIVE_POINT
	case "10pt":
		challRule = pb.ChallengeRule_TEN_POINT
	default:
		msg := "Valid options: 'void', 'single', 'double', '5pt', '10pt'"
		return pb.ChallengeRule_VOID, errors.New(msg)
	}
	return challRule, nil
}

func flipCharCase(r rune) rune {
	if !unicode.IsLetter(r) {
		return r
	}
	if unicode.IsUpper(r) {
		return unicode.ToLower(r)
	} else if unicode.IsLower(r) {
		return unicode.ToUpper(r)
	} else {
		return r
	}
}

func flipCase(s string) string {
	letters := []rune{}
	for _, r := range s {
		letters = append(letters, flipCharCase(r))
	}
	return string(letters)
}

func (g *GameRunner) ParseMove(playerid int, lowercase bool, fields []string) (*move.Move, error) {
	if len(fields) == 1 {
		if fields[0] == "pass" {
			return g.NewPassMove(playerid)
		}
	} else if len(fields) == 2 {
		coords, word := fields[0], fields[1]
		if lowercase {
			word = flipCase(word)
		}
		if coords == "exchange" {
			return g.NewExchangeMove(playerid, word)
		} else {
			return g.NewPlacementMove(playerid, coords, word)
		}
	}
	msg := fmt.Sprintf("unrecognized move: %s", strings.Join(fields, " "))
	return nil, errors.New(msg)
}
