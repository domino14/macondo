// Package io implements a GCG parser. It might also implement
// other io methods.
package io

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"

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

type Turn interface {
	AppendNote(string)
}

type BaseTurn struct {
	Nickname   string `json:"nick"`
	Note       string `json:"note"`
	Rack       string `json:"rack"`
	Type       string `json:"type"`
	Cumulative int    `json:"cumul"`
}

func (bt *BaseTurn) AppendNote(note string) {
	bt.Note = bt.Note + note
}

type TilePlacementTurn struct {
	BaseTurn
	Row       uint8  `json:"row"`
	Column    uint8  `json:"col"`
	Direction string `json:"dir,omitempty"`
	Position  string `json:"pos,omitempty"`
	Play      string `json:"play,omitempty"`
	Score     int    `json:"score"`
}

type PassingTurn struct {
	BaseTurn
	Exchanged string `json:"exchanged,omitempty"`
}

type ScoreAdditionTurn struct {
	BaseTurn
	Bonus         int `json:"bonus,omitempty"`
	EndRackPoints int `json:"score"`
}

type ScoreSubtractionTurn struct {
	BaseTurn
	LostScore int `json:"lost_score"`
}

type Player struct {
	Nickname     string `json:"nick"`
	RealName     string `json:"real_name"`
	PlayerNumber uint8  `json:"p_number"`
}

// A GameRepr is a representation of the GCG that is compatible with Macondo.
type GameRepr struct {
	Turns       []Turn   `json:"turns"`
	Players     []Player `json:"players"`
	Version     int      `json:"version"`
	OriginalGCG string   `json:"originalGCG"`
	Lexicon     string   `json:"lexicon,omitempty"`
	lastToken   Token
}

func addTurn(token Token, match []string, gameRepr *GameRepr) {
	switch token {
	case PlayerToken:
		pn, _ := strconv.Atoi(match[1])
		gameRepr.Players = append(gameRepr.Players, Player{
			Nickname:     match[2],
			RealName:     match[3],
			PlayerNumber: uint8(pn),
		})
		return
	case MoveToken:
		turn := &TilePlacementTurn{}
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
		turn := &ScoreSubtractionTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		score, _ := strconv.Atoi(match[3])
		turn.LostScore = score
		turn.Cumulative, _ = strconv.Atoi(match[4])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case PassToken:
		turn := &PassingTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		turn.Cumulative, _ = strconv.Atoi(match[3])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case ChallengeBonusToken, EndRackPointsToken:
		turn := &ScoreAdditionTurn{}
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
		turn := &PassingTurn{}
		turn.Nickname = match[1]
		turn.Rack = match[2]
		turn.Exchanged = match[3]
		turn.Cumulative, _ = strconv.Atoi(match[4])
		turn.Type = token.String()
		gameRepr.Turns = append(gameRepr.Turns, turn)

	}
}

func parseLine(line string, gameRepr *GameRepr) error {
	foundMatch := false
	for token, rx := range GCGRegexes {
		match := rx.FindStringSubmatch(line)
		if match != nil {
			foundMatch = true
			addTurn(token, match, gameRepr)
			gameRepr.lastToken = token
			break
		}
	}
	if !foundMatch {
		log.Debug().Msgf("Found no match for line '%v'", line)

		// maybe it's a multi-line note.
		if gameRepr.lastToken == NoteToken {
			lastTurnIdx := len(gameRepr.Turns) - 1
			gameRepr.Turns[lastTurnIdx].AppendNote("\n" + line)
			return nil
		}
		// ignore empty lines
		if strings.TrimSpace(line) == "" {
			return nil
		}
		return fmt.Errorf("no match found for line '%v'", line)
	}
	return nil
}

func parseString(gcg string) (*GameRepr, error) {
	lines := strings.Split(gcg, "\n")
	grep := &GameRepr{Turns: []Turn{}, Players: []Player{}, Version: 1,
		OriginalGCG: strings.TrimSpace(gcg)}
	for _, line := range lines {
		err := parseLine(line, grep)
		if err != nil {
			return nil, err
		}
	}
	return grep, nil
}

// Parse a GCG file.
func ParseGCG(filename string) (*GameRepr, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return parseString(string(data))
}
