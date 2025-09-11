// Package gcgio implements a GCG parser. It might also implement
// other io methods.
package gcgio

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/transform"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var (
	errDuplicateNames     = errors.New("two players with same nickname not supported")
	errPragmaPrecedeEvent = errors.New("non-note pragmata should appear before event lines")
	errEncodingWrongPlace = errors.New("encoding line must be first line in file if present")
	errPlayerNotSupported = errors.New("player number not supported")
	errPlayerDoesNotExist = errors.New("player does not exist")
)

// A Token is an event in a GCG file.
type Token uint8

const (
	UndefinedToken Token = iota
	PlayerToken
	TitleToken
	DescriptionToken
	IDToken
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
	GameTypeToken
	TileSetToken
	GameBoardToken
	BoardLayoutToken
	TileDistributionNameToken
	ContinuationToken
	IncompleteToken
	TileDeclarationToken
)

type gcgdatum struct {
	token Token
	regex *regexp.Regexp
}

var GCGRegexes []gcgdatum

const (
	PlayerRegex               = `^#player(?P<p_number>[1-2])\s+(?P<nick>\S+)\s+(?P<real_name>.+)`
	TitleRegex                = `^#title\s+(?P<title>.*)`
	DescriptionRegex          = `^#description\s+(?P<description>.*)`
	IDRegex                   = `^#id\s+(?P<id_authority>\S+)\s+(?P<id>\S+)`
	Rack1Regex                = `^#rack1\s+(?P<rack>\S+)`
	Rack2Regex                = `^#rack2\s+(?P<rack>\S+)`
	MoveRegex                 = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+(?P<pos>\w+)\s+(?P<play>\S+)\s+\+(?P<score>\d+)\s+(?P<cumul>\d+)`
	NoteRegex                 = `^#note\s+(?P<note>.*)`
	LexiconRegex              = `^#lexicon\s+(?P<lexicon>.+)`
	CharacterEncodingRegex    = `^#character-encoding\s+(?P<encoding>[[:graph:]]+)`
	GameTypeRegex             = `^#game-type\s+(?P<gameType>.*)`
	TileSetRegex              = `^#tile-set\s+(?P<tileSet>.*)`
	GameBoardRegex            = `^#game-board\s+(?P<gameBoard>.*)`
	BoardLayoutRegex          = `^#board-layout\s+(?P<boardLayoutName>.*)`
	TileDistributionNameRegex = `^#tile-distribution\s+(?P<tileDistributionName>.*)`
	ContinuationRegex         = `^#- (?P<continuation>.*)`
	PhonyTilesReturnedRegex   = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+--\s+-(?P<lost_score>\d+)\s+(?P<cumul>\d+)`
	PassRegex                 = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-\s+\+0\s+(?P<cumul>\d+)`
	ChallengeBonusRegex       = `>(?P<nick>\S+):\s+(?P<rack>\S*)\s+\(challenge\)\s+\+(?P<bonus>\d+)\s+(?P<cumul>\d+)`
	ExchangeRegex             = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+-(?P<exchanged>\S+)\s+\+0\s+(?P<cumul>\d+)`
	EndRackPointsRegex        = `>(?P<nick>\S+):\s+\((?P<rack>\S+)\)\s+\+(?P<score>\d+)\s+(?P<cumul>-?\d+)`
	TimePenaltyRegex          = `>(?P<nick>\S+):(?:\s+(?P<rack>\S*))?\s+\(time\)\s+\+?\-(?P<penalty>\d+)\s+(?P<cumul>-?\d+)`
	PtsLostForLastRackRegex   = `>(?P<nick>\S+):\s+(?P<rack>\S+)\s+\((?P<rack>\S+)\)\s+\-(?P<penalty>\d+)\s+(?P<cumul>-?\d+)`
	IncompleteRegex           = "^#incomplete.*"
	TileDeclarationRegex      = `^#tile (?P<uppercase>\S+)\s+(?P<lowercase>\S+)`
)

var compiledEncodingRegexp *regexp.Regexp

type parser struct {
	lastToken Token

	history *pb.GameHistory
	game    *game.Game
}

// init initializes the regexp list.
func init() {
	// Important note: ChallengeBonusRegex is defined BEFORE EndRackPointsRegex.
	// That is because a line like  `>frentz:  (challenge) +5 534`  matches
	// both regexes. This can probably be avoided by being more strict about
	// what type of characters the rack can be, etc.

	compiledEncodingRegexp = regexp.MustCompile(CharacterEncodingRegex)

	GCGRegexes = []gcgdatum{
		{PlayerToken, regexp.MustCompile(PlayerRegex)},
		{TitleToken, regexp.MustCompile(TitleRegex)},
		{DescriptionToken, regexp.MustCompile(DescriptionRegex)},
		{IDToken, regexp.MustCompile(IDRegex)},
		{Rack1Token, regexp.MustCompile(Rack1Regex)},
		{Rack2Token, regexp.MustCompile(Rack2Regex)},
		{EncodingToken, compiledEncodingRegexp},
		{MoveToken, regexp.MustCompile(MoveRegex)},
		{NoteToken, regexp.MustCompile(NoteRegex)},
		{LexiconToken, regexp.MustCompile(LexiconRegex)},
		{PhonyTilesReturnedToken, regexp.MustCompile(PhonyTilesReturnedRegex)},
		{PassToken, regexp.MustCompile(PassRegex)},
		{ChallengeBonusToken, regexp.MustCompile(ChallengeBonusRegex)},
		{ExchangeToken, regexp.MustCompile(ExchangeRegex)},
		{EndRackPointsToken, regexp.MustCompile(EndRackPointsRegex)},
		{TimePenaltyToken, regexp.MustCompile(TimePenaltyRegex)},
		{LastRackPenaltyToken, regexp.MustCompile(PtsLostForLastRackRegex)},
		{GameTypeToken, regexp.MustCompile(GameTypeRegex)},
		{TileSetToken, regexp.MustCompile(TileSetRegex)},
		{GameBoardToken, regexp.MustCompile(GameBoardRegex)},
		{ContinuationToken, regexp.MustCompile(ContinuationRegex)},
		{BoardLayoutToken, regexp.MustCompile(BoardLayoutRegex)},
		{TileDistributionNameToken, regexp.MustCompile(TileDistributionNameRegex)},
		{IncompleteToken, regexp.MustCompile(IncompleteRegex)},
		{TileDeclarationToken, regexp.MustCompile(TileDeclarationRegex)},
	}
}

func matchToInt32(str string) (int32, error) {
	x, err := strconv.ParseInt(str, 10, 32)
	if err != nil {
		return 0, err
	}
	return int32(x), nil
}

func nickToPIndex(nick string, players []*pb.PlayerInfo) (uint32, error) {
	for i, p := range players {
		if nick == p.Nickname {
			return uint32(i), nil
		}
	}
	return 0, errPlayerDoesNotExist
}

func (p *parser) addEventOrPragma(cfg *config.Config, token Token, match []string) error {
	var err error

	if token == MoveToken || token == PassToken || token == ExchangeToken ||
		token == Rack1Token || token == Rack2Token {
		// Start the game if we haven't already.
		if len(p.history.Players) != 2 {
			return errors.New("wrong number of players defined")
		}
		if p.game == nil {
			if p.history.Lexicon == "" {
				p.history.Lexicon = cfg.GetString(config.ConfigDefaultLexicon)
			}
			boardLayout, letterDistributionName, variant := game.HistoryToVariant(p.history)

			log.Info().Str("boardLayout", boardLayout).
				Str("letterDistributionName", letterDistributionName).
				Str("lexicon", p.history.Lexicon).
				Str("variant", string(variant)).Msg("creating game")

			// We have both players. Initialize a new game.
			// Don't pass in lexicon to new basic game rules. We don't want GCG
			// parsing to have to load in an actual lexicon to verify any plays.
			rules, err := game.NewBasicGameRules(
				cfg, "",
				boardLayout, letterDistributionName, game.CrossScoreOnly, variant)
			if err != nil {
				return err
			}
			p.game, err = game.NewGame(rules, p.history.Players)
			if err != nil {
				return err
			}
			p.game.StartGame()
			p.game.SetBackupMode(game.InteractiveGameplayMode)
			p.game.SetStateStackLength(1)
			// And set the history to the gcg's history.
			p.game.SetHistory(p.history)
			p.history.PlayState = pb.PlayState_PLAYING
		}
	}

	switch token {
	case PlayerToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		pn, err := strconv.Atoi(match[1])
		if err != nil {
			return err
		}
		if pn != 1 && pn != 2 {
			return errPlayerNotSupported
		}
		if pn == 2 {
			if match[2] == p.history.Players[0].Nickname {
				return errDuplicateNames
			}
		}

		p.history.Players = append(p.history.Players, &pb.PlayerInfo{
			Nickname: match[2],
			RealName: match[3],
		})

		return nil
	case TitleToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.Title = match[1]
		return nil
	case DescriptionToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.Description = match[1]
	case IDToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.IdAuth = match[1]
		p.history.Uid = match[2]
	case Rack1Token:
		// assume if there is a rack2 token, that rack1 will come before it.
		if p.history.LastKnownRacks == nil {
			p.history.LastKnownRacks = []string{match[1], ""}
		}
	case Rack2Token:
		if p.history.LastKnownRacks == nil {
			p.history.LastKnownRacks = []string{"", match[1]}
		} else {
			// There is already a rack1 at the [0] position.
			p.history.LastKnownRacks[1] = match[1]
		}
	case EncodingToken:
		return errEncodingWrongPlace
	case MoveToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}

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

		tp := 0
		ld := p.game.Bag().LetterDistribution()
		tiles, err := tilemapping.ToMachineLetters(evt.PlayedTiles, ld.TileMapping())
		if err != nil {
			return err
		}
		for _, t := range tiles {
			if t != 0 {
				tp++
			}
		}

		evt.IsBingo = tp == game.RackTileLimit
		p.history.Events = append(p.history.Events, evt)
		// Try playing the move
		log.Debug().Msg("PLAYING LATEST EVENT for MoveToken")

		return p.game.PlayLatestEvent()

	case NoteToken:
		lastEvtIdx := len(p.history.Events) - 1
		if lastEvtIdx < 0 {
			log.Warn().Msg("note pragma may not precede events")
		} else {
			p.history.Events[lastEvtIdx].Note += strings.TrimSpace(match[1])
		}
		return nil
	case LexiconToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.Lexicon = match[1]
		return nil
	case BoardLayoutToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.BoardLayout = match[1]
		return nil
	case TileDistributionNameToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.LetterDistribution = match[1]
		return nil
	case GameTypeToken:
		if len(p.history.Events) > 0 {
			return errPragmaPrecedeEvent
		}
		p.history.Variant = match[1]
		return nil
	// need to handle continuation as well as the actual tileSet or gameBoard pragmas.
	case PhonyTilesReturnedToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}
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
		// The PlayedTiles attribute should be set to the LAST event's played tiles
		if len(p.history.Events) == 0 {
			return errors.New("malformed gcg; phony tiles returned without play")
		}
		evt.PlayedTiles = p.history.Events[len(p.history.Events)-1].PlayedTiles
		evt.Type = pb.GameEvent_PHONY_TILES_RETURNED
		p.history.Events = append(p.history.Events, evt)
		log.Debug().Msg("PLAYING LATEST EVENT for PhonytilesReturned")
		return p.game.PlayLatestEvent()

	case TimePenaltyToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}

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
		// Treat this as a stand-alone turn; it should not be attached to
		// the previous event because it can occur after the wrong player
		// (i.e. player2 goes out, and then time penalty is applied to player1)

		evt.Type = pb.GameEvent_TIME_PENALTY
		p.history.Events = append(p.history.Events, evt)
		p.game.SetPlaying(pb.PlayState_GAME_OVER)

		err = p.game.PlayLatestEvent()
		if err != nil {
			return err
		}

	case LastRackPenaltyToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}
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
		p.history.Events = append(p.history.Events, evt)
		err = p.game.PlayLatestEvent()
		// End the game.
		p.game.SetPlaying(pb.PlayState_GAME_OVER)
		if err != nil {
			return err
		}

	case PassToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}
		evt.Rack = match[2]
		evt.Cumulative, err = matchToInt32(match[3])
		if err != nil {
			return err
		}
		evt.Type = pb.GameEvent_PASS
		p.history.Events = append(p.history.Events, evt)
		return p.game.PlayLatestEvent()

	case ChallengeBonusToken, EndRackPointsToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}
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
			// End the game.
			p.game.SetPlaying(pb.PlayState_GAME_OVER)
		}
		p.history.Events = append(p.history.Events, evt)
		return p.game.PlayLatestEvent()

	case ExchangeToken:
		evt := &pb.GameEvent{}
		evt.PlayerIndex, err = nickToPIndex(match[1], p.history.Players)
		if err != nil {
			return errPlayerDoesNotExist
		}
		evt.Rack = match[2]

		nexch, err := matchToInt32(match[3])
		if err != nil {
			evt.Exchanged = match[3]
		} else {
			if int(nexch) > len(match[2]) {
				return errors.New("error in line `" + match[0] + "`: exchanged more tiles than are on rack")
			}
			evt.Exchanged = match[2][:nexch]
		}

		evt.Cumulative, err = matchToInt32(match[4])
		if err != nil {
			return err
		}
		evt.Type = pb.GameEvent_EXCHANGE
		p.history.Events = append(p.history.Events, evt)
		return p.game.PlayLatestEvent()

	case TileDeclarationToken:
		// for now, just ignore this token. We're going to go by the letter
		// distribution to parse the gcg.
	default:
		log.Info().Int("token", int(token)).Interface("match", match).Msg("ignoring-token")
	}
	return nil
}

func (p *parser) parseLine(cfg *config.Config, line string) error {
	line = strings.TrimSpace(line)
	foundMatch := false

	for _, datum := range GCGRegexes {
		match := datum.regex.FindStringSubmatch(line)
		if match != nil {
			foundMatch = true
			err := p.addEventOrPragma(cfg, datum.token, match)
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
			lastEventIdx := len(p.history.Events) - 1
			p.history.Events[lastEventIdx].Note += ("\n" + line)
			return nil
		}
		// ignore empty lines
		if line == "" {
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
		if buf[n] == 0xa || n == BufSize-1 { // reached CR or size limit
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

func ParseGCGFromReader(cfg *config.Config, reader io.Reader) (*pb.GameHistory, error) {
	var err error
	parser := &parser{
		history: &pb.GameHistory{
			Events:  []*pb.GameEvent{},
			Players: []*pb.PlayerInfo{},
			// We are making the challenge rule anything but VOID, which would
			// check the validity of every play.
			ChallengeRule: pb.ChallengeRule_SINGLE,
			Version:       game.CurrentGameHistoryVersion},
	}
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
		err = parser.parseLine(cfg, firstLine)
		if err != nil {
			return nil, err
		}
		originalGCG += firstLine + "\n"
	}

	for scanner.Scan() {
		line := scanner.Text()
		err = parser.parseLine(cfg, line)
		if err != nil {
			return nil, err
		}
		originalGCG += line + "\n"
	}
	parser.history.OriginalGcg = strings.TrimSpace(originalGCG)

	// Determine if the game ended.
	if parser.game.Playing() == pb.PlayState_GAME_OVER {
		parser.history.PlayState = pb.PlayState_GAME_OVER
		parser.game.AddFinalScoresToHistory()
	}
	// Set challenge rule back to void since we don't know or care what it is.
	parser.history.ChallengeRule = pb.ChallengeRule_VOID
	return parser.history, nil
}

// ParseGCG parses a GCG file into a GameHistory.
func ParseGCG(cfg *config.Config, filename string) (*pb.GameHistory, error) {
	f, _, err := cache.Open(filename)
	if err != nil {
		return nil, err
	}
	return ParseGCGFromReader(cfg, f)
}

func writeGCGHeader(s *strings.Builder, h *pb.GameHistory, addlInfo bool) {
	s.WriteString("#character-encoding UTF-8\n")
	if addlInfo {
		if h.Title != "" {
			s.WriteString("#title " + h.Title + "\n")
		}
		if h.Description != "" {
			s.WriteString("#description " + h.Description + "\n")
		}
		if h.IdAuth != "" && h.Uid != "" {
			s.WriteString("#id " + h.IdAuth + " " + h.Uid + "\n")
		}
		if h.Lexicon != "" {
			s.WriteString("#lexicon " + h.Lexicon + "\n")
		}
		if h.Variant != "" {
			s.WriteString("#game-type " + h.Variant + "\n")
		}
		if h.BoardLayout != "" && h.BoardLayout != board.CrosswordGameLayout {
			s.WriteString("#board-layout " + h.BoardLayout + "\n")
		}
		if h.LetterDistribution != "" && h.LetterDistribution != "english" {
			s.WriteString("#tile-distribution " + h.LetterDistribution + "\n")
			// Write out multi-tile pragmata
			cfg := config.DefaultConfig()
			tm, err := tilemapping.GetDistribution(cfg.WGLConfig(), h.LetterDistribution)
			if err != nil {
				// Log the error
				log.Err(err).Str("dist", h.LetterDistribution).Msg("cannot-get-distribution")
			}
			for idx := uint8(0); idx < tm.TileMapping().NumLetters(); idx++ {
				letter := tm.TileMapping().Letter(tilemapping.MachineLetter(idx))
				if len([]rune(letter)) > 1 {
					s.WriteString("#tile " + letter + " " + strings.ToLower(letter) + "\n")
				}
			}
		}
	}
	log.Debug().Msg("wrote header")
}

func writeEvent(s *strings.Builder, h *pb.GameHistory, evt *pb.GameEvent) error {

	nick := h.Players[evt.GetPlayerIndex()].Nickname
	rack := evt.GetRack()
	evtType := evt.GetType()
	note := evt.GetNote()

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
		fmt.Fprintf(s, ">%v: %v - +0 %d\n", nick, rack, evt.Cumulative)
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

	case pb.GameEvent_END_RACK_PENALTY:
		// >Pakorn: FWLI (FWLI) -10 426
		fmt.Fprintf(s, ">%v: %v (%v) -%d %d\n",
			nick, rack, rack, evt.LostScore, evt.Cumulative)
	case pb.GameEvent_TIME_PENALTY:
		// >Pakorn: ISBALI (time) -10 409
		fmt.Fprintf(s, ">%v: %v (time) -%d %d\n",
			nick, rack, evt.LostScore, evt.Cumulative)
	case pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS:
		// Treat exactly like a pass, but append a note. The GCG format
		// does not distinguish between these two cases.
		fmt.Fprintf(s, ">%v: %v - +0 %d\n", nick, rack, evt.Cumulative)
		fmt.Fprint(s, "#note #unsuccessful-challenge\n")

	default:
		return fmt.Errorf("event type %v not supported", evtType)

	}
	if note != "" {
		// Note that the note can have line breaks within it ...
		fmt.Fprintf(s, "#note %v\n", note)
	}
	return nil

}

func writePlayer(s *strings.Builder, pn int, p *pb.PlayerInfo) {
	realname := p.RealName
	if realname == "" {
		realname = p.Nickname
	}
	fmt.Fprintf(s, "#player%d %v %v\n", pn, p.Nickname, realname)
}

func writePlayers(s *strings.Builder, players []*pb.PlayerInfo) {
	writePlayer(s, 1, players[0])
	writePlayer(s, 2, players[1])
}

func isPassBeforeEndRackPoints(h *pb.GameHistory, i int) bool {
	return len(h.Events) > i+1 &&
		(h.Events[i].Type == pb.GameEvent_PASS ||
			h.Events[i].Type == pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS) &&
		h.Events[i+1].Type == pb.GameEvent_END_RACK_PTS
}

// GameHistoryToGCG returns a string GCG representation of the GameHistory.
func GameHistoryToGCG(h *pb.GameHistory, addlHeaderInfo bool) (string, error) {
	if h.StartingCgp != "" {
		return "", errors.New("cannot turn a game history with a starting CGP into a GCG file")
	}
	var str strings.Builder
	writeGCGHeader(&str, h, addlHeaderInfo)
	writePlayers(&str, h.Players)

	for i, evt := range h.Events {
		if !isPassBeforeEndRackPoints(h, i) {
			err := writeEvent(&str, h, evt)
			if err != nil {
				return "", err
			}
		}
	}

	return str.String(), nil
}
