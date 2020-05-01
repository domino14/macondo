// Package gcgio implements a GCG parser. It might also implement
// other io methods.
package gcgio

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/domino14/macondo/game"

	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/transform"

	pb "github.com/domino14/macondo/rpc/api/proto"
	"github.com/rs/zerolog/log"
)

// A Token is an event in a GCG file.
type Token uint8

const (
	UndefinedToken Token = iota
	PlayerToken
	TitleToken
	DescriptionToken
	Rack1Token
	Rack2Token
	EncodingToken
	MoveToken
	NoteToken
	LexiconToken
	PhonyTilesReturnedToken
	PassToken
	ChallengeBonusToken
	ExchangeToken
	EndRackPointsToken
	TimePenaltyToken
	LastRackPenaltyToken
)

type gcgdatum struct {
	token Token
	regex *regexp.Regexp
}

var GCGRegexes []gcgdatum

const (
	PlayerRegex             = `#player(?P<p_number>[1-2])\s+(?P<nick>\S+)\s+(?P<real_name>.+)`
	TitleRegex              = `#title\s*(?P<title>.*)`
	DescriptionRegex        = `#description\s*(?P<description>.*)`
	Rack1Regex              = `#rack1 (?P<rack>\S+)`
	Rack2Regex              = `#rack2 (?P<rack>\S+)`
	MoveRegex               = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+(?P<pos>\w+)\s+(?P<play>[\w\\.]+)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
	NoteRegex               = `#note (?P<note>.+)`
	LexiconRegex            = `#lexicon (?P<lexicon>.+)`
	CharacterEncodingRegex  = `#character-encoding (?P<encoding>[[:graph:]]+)`
	PhonyTilesReturnedRegex = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+--\s+-(?P<lost_score>\d+)\s+(?P<cumul>\d+)`
	PassRegex               = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-\s+\+0\s+(?P<cumul>\d+)`
	ChallengeBonusRegex     = `>(?P<nick>\S+):\s+(?P<rack>\S*)\s+\(challenge\)\s+\+(?P<bonus>\d+)\s+(?P<cumul>\d+)`
	ExchangeRegex           = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-(?P<exchanged>\S+)\s+\+0\s+(?P<cumul>\d+)`
	EndRackPointsRegex      = `>(?P<nick>\S+):\s+\((?P<rack>\S+)\)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
	TimePenaltyRegex        = `>(?P<nick>\S+):\s+(?P<rack>\S*)\s+\(time\)\s+\-(?P<penalty>\d+)\s+(?P<cumul>\d+)`
	PtsLostForLastRackRegex = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+\((?P<rack>\S+)\)\s+\-(?P<penalty>\d+)\s+(?P<cumul>\d+)`
)

var compiledEncodingRegexp *regexp.Regexp

type parser struct {
	lastToken Token
}

// init initializes the regexp list.
func init() {
	// Important note: ChallengeBonusRegex is defined BEFORE EndRackPointsRegex.
	// That is because a line like  `>frentz:  (challenge) +5 534`  matches
	// both regexes. This can probably be avoided by being more strict about
	// what type of characters the rack can be, etc.

	compiledEncodingRegexp = regexp.MustCompile(CharacterEncodingRegex)

	GCGRegexes = []gcgdatum{
		gcgdatum{PlayerToken, regexp.MustCompile(PlayerRegex)},
		gcgdatum{TitleToken, regexp.MustCompile(TitleRegex)},
		gcgdatum{DescriptionToken, regexp.MustCompile(DescriptionRegex)},
		gcgdatum{Rack1Token, regexp.MustCompile(Rack1Regex)},
		gcgdatum{Rack2Token, regexp.MustCompile(Rack2Regex)},
		gcgdatum{EncodingToken, compiledEncodingRegexp},
		gcgdatum{MoveToken, regexp.MustCompile(MoveRegex)},
		gcgdatum{NoteToken, regexp.MustCompile(NoteRegex)},
		gcgdatum{LexiconToken, regexp.MustCompile(LexiconRegex)},
		gcgdatum{PhonyTilesReturnedToken, regexp.MustCompile(PhonyTilesReturnedRegex)},
		gcgdatum{PassToken, regexp.MustCompile(PassRegex)},
		gcgdatum{ChallengeBonusToken, regexp.MustCompile(ChallengeBonusRegex)},
		gcgdatum{ExchangeToken, regexp.MustCompile(ExchangeRegex)},
		gcgdatum{EndRackPointsToken, regexp.MustCompile(EndRackPointsRegex)},
		gcgdatum{TimePenaltyToken, regexp.MustCompile(TimePenaltyRegex)},
		gcgdatum{LastRackPenaltyToken, regexp.MustCompile(PtsLostForLastRackRegex)},
	}
}

func matchToInt32(str string) (int32, error) {
	x, err := strconv.ParseInt(str, 10, 32)
	if err != nil {
		return 0, err
	}
	return int32(x), nil
}

func (p *parser) addEventOrPragma(token Token, match []string, gameHistory *pb.GameHistory) error {
	var err error

	switch token {
	case PlayerToken:
		pn, err := strconv.Atoi(match[1])
		if err != nil {
			return err
		}
		gameHistory.Players = append(gameHistory.Players, &pb.PlayerInfo{
			Nickname: match[2],
			RealName: match[3],
			Number:   int32(pn),
		})

		return nil
	case TitleToken:
		gameHistory.Title = match[1]
		return nil
	case DescriptionToken:
		gameHistory.Description = match[1]
	// Assume Rack1Token always comes before Rack2Token in a well-formed gcg:
	case Rack1Token:
		gameHistory.LastKnownRacks = []string{match[1]}
	case Rack2Token:
		gameHistory.LastKnownRacks = append(gameHistory.LastKnownRacks, match[1])
	case EncodingToken:
		return errors.New("encoding line must be first line in file if present")
	case MoveToken:
		evt := &pb.GameEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		evt.Position = match[3]
		evt.PlayedTiles = match[4]
		evt.Score, err = matchToInt32(match[5])
		if err != nil {
			return err
		}
		evt.Cumulative, err = matchToInt32(match[6])
		if err != nil {
			return err
		}
		game.CalculateCoordsFromStringPosition(evt)
		evt.Type = pb.GameEvent_TILE_PLACEMENT_MOVE
		evts := []*pb.GameEvent{evt}
		turn := &pb.GameTurn{Events: evts}
		gameHistory.Turns = append(gameHistory.Turns, turn)

	case NoteToken:
		lastTurnIdx := len(gameHistory.Turns) - 1
		lastEvtIdx := len(gameHistory.Turns[lastTurnIdx].Events) - 1
		gameHistory.Turns[lastTurnIdx].Events[lastEvtIdx].Note += match[1]
		return nil
	case LexiconToken:
		gameHistory.Lexicon = match[1]
		return nil
	case PhonyTilesReturnedToken, TimePenaltyToken:
		evt := &pb.GameEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]

		score, err := matchToInt32(match[3])
		if err != nil {
			return err
		}
		evt.LostScore = score
		evt.Cumulative, err = matchToInt32(match[4])
		if err != nil {
			return err
		}
		// This can not be a stand-alone turn; it must be added to the last
		// turn.
		lastTurnIdx := len(gameHistory.Turns) - 1
		gameHistory.Turns[lastTurnIdx].Events = append(gameHistory.Turns[lastTurnIdx].Events, evt)
		if token == PhonyTilesReturnedToken {
			evt.Type = pb.GameEvent_PHONY_TILES_RETURNED
		} else if token == TimePenaltyToken {
			evt.Type = pb.GameEvent_TIME_PENALTY
		}

	case LastRackPenaltyToken:
		evt := &pb.GameEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		if evt.Rack != match[3] {
			return fmt.Errorf("last rack penalty event malformed")
		}
		score, err := matchToInt32(match[4])
		if err != nil {
			return err
		}
		evt.LostScore = score
		evt.Cumulative, err = matchToInt32(match[5])
		if err != nil {
			return err
		}
		evt.Type = pb.GameEvent_END_RACK_PENALTY
		lastTurnIdx := len(gameHistory.Turns) - 1
		gameHistory.Turns[lastTurnIdx].Events = append(gameHistory.Turns[lastTurnIdx].Events, evt)

	case PassToken:
		evt := &pb.GameEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		evt.Cumulative, err = matchToInt32(match[3])
		if err != nil {
			return err
		}
		evt.Type = pb.GameEvent_PASS
		evts := []*pb.GameEvent{evt}
		turn := &pb.GameTurn{Events: evts}
		gameHistory.Turns = append(gameHistory.Turns, turn)

	case ChallengeBonusToken, EndRackPointsToken:
		evt := &pb.GameEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		if token == ChallengeBonusToken {
			evt.Bonus, err = matchToInt32(match[3])
		} else {
			evt.EndRackPoints, err = matchToInt32(match[3])
		}
		if err != nil {
			return err
		}
		evt.Cumulative, err = matchToInt32(match[4])
		if err != nil {
			return err
		}
		if token == ChallengeBonusToken {
			evt.Type = pb.GameEvent_CHALLENGE_BONUS
		} else if token == EndRackPointsToken {
			evt.Type = pb.GameEvent_END_RACK_PTS
		}
		lastTurnIdx := len(gameHistory.Turns) - 1
		gameHistory.Turns[lastTurnIdx].Events = append(gameHistory.Turns[lastTurnIdx].Events, evt)

	case ExchangeToken:
		evt := &pb.GameEvent{}
		evt.Nickname = match[1]
		evt.Rack = match[2]
		evt.Exchanged = match[3]
		evt.Cumulative, err = matchToInt32(match[4])
		if err != nil {
			return err
		}
		evt.Type = pb.GameEvent_EXCHANGE
		evts := []*pb.GameEvent{evt}
		turn := &pb.GameTurn{Events: evts}
		gameHistory.Turns = append(gameHistory.Turns, turn)

	}
	return nil
}

func (p *parser) parseLine(line string, history *pb.GameHistory) error {

	foundMatch := false

	for _, datum := range GCGRegexes {
		match := datum.regex.FindStringSubmatch(line)
		if match != nil {
			foundMatch = true
			err := p.addEventOrPragma(datum.token, match, history)
			if err != nil {
				return err
			}
			p.lastToken = datum.token
			break
		}
	}
	if !foundMatch {
		// maybe it's a multi-line note.
		if p.lastToken == NoteToken {
			lastTurnIdx := len(history.Turns) - 1
			lastEventIdx := len(history.Turns[lastTurnIdx].Events) - 1
			history.Turns[lastTurnIdx].Events[lastEventIdx].Note += ("\n" + line)
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

func encodingOrFirstLine(reader io.Reader) (string, string, error) {
	// Read either the encoding of the file, or the first line,
	// whichever is available.
	const BufSize = 128
	buf := make([]byte, BufSize)
	n := 0
	for {
		// non buffered byte-by-byte
		if _, err := reader.Read(buf[n : n+1]); err != nil {
			return "", "", err
		}
		if buf[n] == 0xa || n == BufSize { // reached CR or size limit
			firstLine := buf[:n]
			match := compiledEncodingRegexp.FindStringSubmatch(string(firstLine))
			if match != nil {
				enc := strings.ToLower(match[1])
				if enc != "utf-8" && enc != "utf8" {
					return "", "", errors.New("unhandled character encoding " + enc)
				}
				// Otherwise, switch to utf8 mode; which means we require no transform
				// since Go does UTF-8 by default.
				return "utf8", "", nil
			}
			// Not an encoding line. We should ocnvert the raw bytes into the default
			// GCG encoding, which is ISO 8859-1.
			decoder := charmap.ISO8859_1.NewDecoder()
			result, _, err := transform.Bytes(decoder, firstLine)
			if err != nil {
				return "", "", err
			}
			// We can stringify the result now, as the transformed bytes will
			// be UTF-8
			return "", string(result), nil
		}
		n++

	}
}

func ParseGCGFromReader(reader io.Reader) (*pb.GameHistory, error) {

	history := &pb.GameHistory{
		Turns:   []*pb.GameTurn{},
		Players: []*pb.PlayerInfo{},
		Version: 1}
	var err error
	parser := &parser{}
	originalGCG := ""

	// Determine encoding from first line
	// Try to match to an encoding pragma line. If it doesn't exist,
	// the encoding is ISO 8859-1 per spec.
	enc, firstLine, err := encodingOrFirstLine(reader)
	if err != nil {
		return nil, err
	}
	var scanner *bufio.Scanner
	if enc != "utf8" {
		gcgEncoding := charmap.ISO8859_1
		r := transform.NewReader(reader, gcgEncoding.NewDecoder())
		scanner = bufio.NewScanner(r)
	} else {
		scanner = bufio.NewScanner(reader)
	}
	if firstLine != "" {
		err = parser.parseLine(firstLine, history)
		if err != nil {
			return nil, err
		}
		originalGCG += firstLine + "\n"
	}

	for scanner.Scan() {
		line := scanner.Text()
		err = parser.parseLine(line, history)
		if err != nil {
			return nil, err
		}
		originalGCG += line + "\n"
	}
	history.OriginalGcg = strings.TrimSpace(originalGCG)
	return history, nil
}

// ParseGCG parses a GCG file into a GameHistory.
func ParseGCG(filename string) (*pb.GameHistory, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	return ParseGCGFromReader(f)
}

func writeGCGHeader(s *strings.Builder) {
	s.WriteString("#character-encoding UTF-8\n")
	log.Debug().Msg("wrote encoding")
}

func writePlayer(s *strings.Builder, p *pb.PlayerInfo) {
	fmt.Fprintf(s, "#player%d %v %v\n", p.Number, p.Nickname, p.RealName)
}

func writeEvent(s *strings.Builder, evt *pb.GameEvent) {

	nick := evt.GetNickname()
	rack := evt.GetRack()
	evtType := evt.GetType()

	// XXX HANDLE MORE TYPES (e.g. time penalty at some point, end rack
	// penalty for international rules)
	switch evtType {
	case pb.GameEvent_TILE_PLACEMENT_MOVE:
		fmt.Fprintf(s, ">%v: %v %v %v +%d %d\n",
			nick, rack, evt.Position, evt.PlayedTiles, evt.Score, evt.Cumulative,
		)
	case pb.GameEvent_PHONY_TILES_RETURNED:
		// >emely: DEIILTZ -- -24 55
		fmt.Fprintf(s, ">%v: %v -- -%d %d\n",
			nick, rack, evt.LostScore, evt.Cumulative)

	case pb.GameEvent_PASS:
		// >Randy: U - +0 380
		fmt.Fprintf(s, ">%v: (%v) - +0 %d\n", nick, rack, evt.Cumulative)
	case pb.GameEvent_CHALLENGE_BONUS:
		// >Joel: DROWNUG (challenge) +5 289
		fmt.Fprintf(s, ">%v: %v (challenge) +%d %d\n",
			nick, rack, evt.Bonus, evt.Cumulative)

	case pb.GameEvent_END_RACK_PTS:
		// >Dave: (G) +4 539
		fmt.Fprintf(s, ">%v: (%v) +%d %d\n",
			nick, rack, evt.EndRackPoints, evt.Cumulative)

	case pb.GameEvent_EXCHANGE:
		// >Marlon: SEQSPO? -QO +0 268
		fmt.Fprintf(s, ">%v: %v -%v +0 %d\n",
			nick, rack, evt.Exchanged, evt.Cumulative)

	}

}

func writeTurn(s *strings.Builder, t *pb.GameTurn) {
	for _, evt := range t.Events {
		writeEvent(s, evt)
	}
}

// GameHistoryToGCG returns a string GCG representation of the GameHistory.
func GameHistoryToGCG(h *pb.GameHistory) string {

	var str strings.Builder
	writeGCGHeader(&str)
	for _, player := range h.Players {
		writePlayer(&str, player)
	}

	for _, turn := range h.Turns {
		writeTurn(&str, turn)
	}

	return str.String()
}
