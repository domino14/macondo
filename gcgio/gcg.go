// Package gcgio implements a GCG parser. It might also implement
// other io methods.
package gcgio

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"

	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"
)

var GCGRegexes map[Token]*regexp.Regexp

const (
	PlayerRegex         = `#player(?P<p_number>[1-2])\s+(?P<nick>\S+)\s+(?P<real_name>.+)`
	MoveRegex           = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+(?P<pos>\w+)\s+(?P<play>[\w\\.]+)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
	NoteRegex           = `#note (?P<note>.+)`
	LexiconRegex        = `#lexicon (?P<lexicon>.+)`
	LostChallengeRegex  = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+--\s+-(?P<lost_score>\d+)\s+(?P<cumul>\d+)`
	PassRegex           = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-\s+\+0\s+(?P<cumul>\d+)`
	ChallengeBonusRegex = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+\(challenge\)\s+\+(?P<bonus>\d+)\s+(?P<cumul>\d+)`
	ExchangeRegex       = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-(?P<exchanged>\S+)\s+\+0\s+(?P<cumul>\d+)`
	EndRackPointsRegex  = `>(?P<nick>\S+):\s+\((?P<rack>\S+)\)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
)

// init initializes the regexp map.
func init() {
	GCGRegexes = map[Token]*regexp.Regexp{
		PlayerToken:         regexp.MustCompile(PlayerRegex),
		MoveToken:           regexp.MustCompile(MoveRegex),
		NoteToken:           regexp.MustCompile(NoteRegex),
		LexiconToken:        regexp.MustCompile(LexiconRegex),
		LostChallengeToken:  regexp.MustCompile(LostChallengeRegex),
		PassToken:           regexp.MustCompile(PassRegex),
		ChallengeBonusToken: regexp.MustCompile(ChallengeBonusRegex),
		ExchangeToken:       regexp.MustCompile(ExchangeRegex),
		EndRackPointsToken:  regexp.MustCompile(EndRackPointsRegex),
	}
}

func addTurn(token Token, match []string, gameRepr *mechanics.GameRepr) {
	switch token {
	case PlayerToken:
		pn, _ := strconv.Atoi(match[1])
		gameRepr.Players = append(gameRepr.Players, &mechanics.Player{
			Nickname:     match[2],
			RealName:     match[3],
			PlayerNumber: uint8(pn),
		})
		return
	case MoveToken:
		turn := &mechanics.TilePlacementTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		turn.Position = match[3]
		turn.Play = match[4]
		turn.Score, _ = strconv.Atoi(match[5])
		turn.Cumulative, _ = strconv.Atoi(match[6])
		row, col, vertical := move.FromBoardGameCoords(turn.Position)
		if vertical {
			turn.Direction = "v"
		} else {
			turn.Direction = "h"
		}
		turn.Row = uint8(row)
		turn.Column = uint8(col)
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case NoteToken:
		lastTurnIdx := len(gameRepr.Turns) - 1
		gameRepr.Turns[lastTurnIdx].AppendNote(match[1])
		return
	case LexiconToken:
		gameRepr.Lexicon = match[1]
		return
	case LostChallengeToken:
		turn := &mechanics.ScoreSubtractionTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		score, _ := strconv.Atoi(match[3])
		turn.LostScore = score
		turn.Cumulative, _ = strconv.Atoi(match[4])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case PassToken:
		turn := &mechanics.PassingTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		turn.Cumulative, _ = strconv.Atoi(match[3])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case ChallengeBonusToken, EndRackPointsToken:
		turn := &mechanics.ScoreAdditionTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		if token == ChallengeBonusToken {
			turn.Bonus, _ = strconv.Atoi(match[3])
		} else {
			turn.EndRackPoints, _ = strconv.Atoi(match[3])
		}
		turn.Cumulative, _ = strconv.Atoi(match[4])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case ExchangeToken:
		turn := &mechanics.PassingTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		turn.Exchanged = match[3]
		turn.Cumulative, _ = strconv.Atoi(match[4])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	}
}

func parseLine(line string, gameRepr *mechanics.GameRepr, lastToken Token) (
	Token, error) {

	foundMatch := false

	for token, rx := range GCGRegexes {
		match := rx.FindStringSubmatch(line)
		if match != nil {
			foundMatch = true
			addTurn(token, match, gameRepr)
			lastToken = token
			break
		}
	}
	if !foundMatch {
		log.Debug().Msgf("Found no match for line '%v'", line)

		// maybe it's a multi-line note.
		if lastToken == NoteToken {
			lastTurnIdx := len(gameRepr.Turns) - 1
			gameRepr.Turns[lastTurnIdx].AppendNote("\n" + line)
			return lastToken, nil
		}
		// ignore empty lines
		if strings.TrimSpace(line) == "" {
			return lastToken, nil
		}
		return lastToken, fmt.Errorf("no match found for line '%v'", line)
	}
	return lastToken, nil
}

func parseString(gcg string) (*mechanics.GameRepr, error) {
	lines := strings.Split(gcg, "\n")
	grep := &mechanics.GameRepr{Turns: []mechanics.Turn{}, Players: []*mechanics.Player{},
		Version: 1, OriginalGCG: strings.TrimSpace(gcg)}
	var lastToken Token
	var err error
	for _, line := range lines {
		lastToken, err = parseLine(line, grep, lastToken)
		if err != nil {
			return nil, err
		}
	}
	return grep, nil
}

// ParseGCG parses a GCG file into a GameRepr.
func ParseGCG(filename string) (*mechanics.GameRepr, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return parseString(string(data))
}
