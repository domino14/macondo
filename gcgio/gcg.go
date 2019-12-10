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

type gcgdatum struct {
	token Token
	regex *regexp.Regexp
}

var GCGRegexes []gcgdatum

const (
	PlayerRegex             = `#player(?P<p_number>[1-2])\s+(?P<nick>\S+)\s+(?P<real_name>.+)`
	MoveRegex               = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+(?P<pos>\w+)\s+(?P<play>[\w\\.]+)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
	NoteRegex               = `#note (?P<note>.+)`
	LexiconRegex            = `#lexicon (?P<lexicon>.+)`
	LostChallengeRegex      = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+--\s+-(?P<lost_score>\d+)\s+(?P<cumul>\d+)`
	PassRegex               = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-\s+\+0\s+(?P<cumul>\d+)`
	ChallengeBonusRegex     = `>(?P<nick>\S+):\s+(?P<rack>\S*)\s+\(challenge\)\s+\+(?P<bonus>\d+)\s+(?P<cumul>\d+)`
	ExchangeRegex           = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-(?P<exchanged>\S+)\s+\+0\s+(?P<cumul>\d+)`
	EndRackPointsRegex      = `>(?P<nick>\S+):\s+\((?P<rack>\S+)\)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
	TimePenaltyRegex        = `>(?P<nick>\S+):\s+(?P<rack>\S*)\s+\(time\)\s+\-(?P<penalty>\d+)\s+(?P<cumul>\d+)`
	PtsLostForLastRackRegex = `>(?P<nick>\S+):\s+\(?P<rack>\S+)\s+\((?P<rack>\S+)\)\s+\-(?P<penalty>\d+)\s+(?P<cumul>\d+)`
)

type parser struct {
	lastToken Token
}

// init initializes the regexp list.
func init() {
	// Important note: ChallengeBonusRegex is defined BEFORE EndRackPointsRegex.
	// That is because a line like  `>frentz:  (challenge) +5 534`  matches
	// both regexes. This can probably be avoided by being more strict about
	// what type of characters the rack can be, etc.

	GCGRegexes = []gcgdatum{
		gcgdatum{PlayerToken, regexp.MustCompile(PlayerRegex)},
		gcgdatum{MoveToken, regexp.MustCompile(MoveRegex)},
		gcgdatum{NoteToken, regexp.MustCompile(NoteRegex)},
		gcgdatum{LexiconToken, regexp.MustCompile(LexiconRegex)},
		gcgdatum{LostChallengeToken, regexp.MustCompile(LostChallengeRegex)},
		gcgdatum{PassToken, regexp.MustCompile(PassRegex)},
		gcgdatum{ChallengeBonusToken, regexp.MustCompile(ChallengeBonusRegex)},
		gcgdatum{ExchangeToken, regexp.MustCompile(ExchangeRegex)},
		gcgdatum{EndRackPointsToken, regexp.MustCompile(EndRackPointsRegex)},
		gcgdatum{TimePenaltyToken, regexp.MustCompile(TimePenaltyRegex)},
		gcgdatum{LastRackPenaltyToken, regexp.MustCompile(PtsLostForLastRackRegex)},
	}
}

func (p *parser) addEventOrPragma(token Token, match []string, gameRepr *mechanics.GameRepr) error {
	var err error

	switch token {
	case PlayerToken:
		pn, err := strconv.Atoi(match[1])
		if err != nil {
			return err
		}
		gameRepr.Players = append(gameRepr.Players, &mechanics.Player{
			Nickname:     match[2],
			RealName:     match[3],
			PlayerNumber: uint8(pn),
		})
		return nil
	case MoveToken:
		evt := &mechanics.TilePlacementEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		evt.Position = match[3]
		evt.Play = match[4]
		evt.Score, err = strconv.Atoi(match[5])
		if err != nil {
			return err
		}
		evt.Cumulative, err = strconv.Atoi(match[6])
		if err != nil {
			return err
		}
		row, col, vertical := move.FromBoardGameCoords(evt.Position)
		if vertical {
			evt.Direction = "v"
		} else {
			evt.Direction = "h"
		}
		evt.Row = uint8(row)
		evt.Column = uint8(col)
		evt.Type = token.String()
		turn := []mechanics.Event{}
		turn = append(turn, evt)

		gameRepr.Turns = append(gameRepr.Turns, turn)

	case NoteToken:
		lastTurnIdx := len(gameRepr.Turns) - 1
		lastEvtIdx := len(gameRepr.Turns[lastTurnIdx]) - 1
		gameRepr.Turns[lastTurnIdx][lastEvtIdx].AppendNote(match[1])
		return nil
	case LexiconToken:
		gameRepr.Lexicon = match[1]
		return nil
	case LostChallengeToken:
		evt := &mechanics.ScoreSubtractionEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		score, err := strconv.Atoi(match[3])
		if err != nil {
			return err
		}
		evt.LostScore = score
		evt.Cumulative, err = strconv.Atoi(match[4])
		if err != nil {
			return err
		}
		evt.Type = token.String()
		// This can not be a stand-alone turn; it must be added to the last
		// turn.
		lastTurnIdx := len(gameRepr.Turns) - 1
		gameRepr.Turns[lastTurnIdx] = append(gameRepr.Turns[lastTurnIdx], evt)
	case TimePenaltyToken:
		evt := &mechanics.ScoreSubtractionEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		score, err := strconv.Atoi(match[3])
		if err != nil {
			return err
		}
		evt.LostScore = score
		evt.Cumulative, err = strconv.Atoi(match[4])
		if err != nil {
			return err
		}
		evt.Type = token.String()
		lastTurnIdx := len(gameRepr.Turns) - 1
		gameRepr.Turns[lastTurnIdx] = append(gameRepr.Turns[lastTurnIdx], evt)

	case LastRackPenaltyToken:
		evt := &mechanics.ScoreSubtractionEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		if evt.Rack != match[3] {
			return fmt.Errorf("last rack penalty event malformed")
		}
		score, err := strconv.Atoi(match[4])
		if err != nil {
			return err
		}
		evt.LostScore = score
		evt.Cumulative, err = strconv.Atoi(match[5])
		if err != nil {
			return err
		}
		evt.Type = token.String()
		lastTurnIdx := len(gameRepr.Turns) - 1
		gameRepr.Turns[lastTurnIdx] = append(gameRepr.Turns[lastTurnIdx], evt)

	case PassToken:
		evt := &mechanics.PassingEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		evt.Cumulative, err = strconv.Atoi(match[3])
		if err != nil {
			return err
		}
		evt.Type = token.String()
		turn := []mechanics.Event{evt}
		gameRepr.Turns = append(gameRepr.Turns, turn)

	case ChallengeBonusToken, EndRackPointsToken:
		evt := &mechanics.ScoreAdditionEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		if token == ChallengeBonusToken {
			evt.Bonus, err = strconv.Atoi(match[3])
		} else {
			evt.EndRackPoints, err = strconv.Atoi(match[3])
		}
		if err != nil {
			return err
		}
		evt.Cumulative, err = strconv.Atoi(match[4])
		if err != nil {
			return err
		}
		evt.Type = token.String()
		lastTurnIdx := len(gameRepr.Turns) - 1
		gameRepr.Turns[lastTurnIdx] = append(gameRepr.Turns[lastTurnIdx], evt)

	case ExchangeToken:
		evt := &mechanics.PassingEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		evt.Exchanged = match[3]
		evt.Cumulative, err = strconv.Atoi(match[4])
		if err != nil {
			return err
		}
		evt.Type = token.String()
		turn := []mechanics.Event{evt}
		gameRepr.Turns = append(gameRepr.Turns, turn)

	}
	return nil
}

func (p *parser) parseLine(line string, gameRepr *mechanics.GameRepr) error {

	foundMatch := false

	for _, datum := range GCGRegexes {
		match := datum.regex.FindStringSubmatch(line)
		if match != nil {
			foundMatch = true
			err := p.addEventOrPragma(datum.token, match, gameRepr)
			if err != nil {
				return err
			}
			p.lastToken = datum.token
			break
		}
	}
	if !foundMatch {
		log.Debug().Msgf("Found no match for line '%v'", line)

		// maybe it's a multi-line note.
		if p.lastToken == NoteToken {
			lastTurnIdx := len(gameRepr.Turns) - 1
			lastEventIdx := len(gameRepr.Turns[lastTurnIdx]) - 1
			gameRepr.Turns[lastTurnIdx][lastEventIdx].AppendNote("\n" + line)
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

func parseString(gcg string) (*mechanics.GameRepr, error) {
	parser := &parser{}

	lines := strings.Split(gcg, "\n")
	grep := &mechanics.GameRepr{Turns: []mechanics.Turn{}, Players: []*mechanics.Player{},
		Version: 1, OriginalGCG: strings.TrimSpace(gcg)}
	var err error
	for _, line := range lines {
		err = parser.parseLine(line, grep)
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
