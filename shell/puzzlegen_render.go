package shell

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// pgShow selects which sections of a matched puzzle get printed. The default
// is the one-line summary alone, so a survey of a few hundred games still
// scrolls; boards and CGP are opt-in via -show.
type pgShow struct {
	line   bool
	board  bool
	cgp    bool
	answer bool
	stats  bool
}

func parseShow(spec string) (pgShow, error) {
	if spec == "" {
		return pgShow{line: true}, nil
	}
	var s pgShow
	for _, part := range strings.Split(spec, ",") {
		switch strings.TrimSpace(strings.ToLower(part)) {
		case "":
		case "line":
			s.line = true
		case "board":
			s.board = true
		case "cgp":
			s.cgp = true
		case "answer":
			s.answer = true
		case "stats":
			s.stats = true
		case "all":
			s = pgShow{line: true, board: true, cgp: true, answer: true, stats: true}
		default:
			return pgShow{}, fmt.Errorf("unknown -show section %q; use line, board, cgp, answer, stats, or all", part)
		}
	}
	return s, nil
}

func (s pgShow) any() bool {
	return s.line || s.board || s.cgp || s.answer || s.stats
}

// pgPosition is everything about a puzzle that has to be read off the game
// rather than off the PuzzleCreationResponse: the board as `s` draws it and the
// position as CGP.
type pgPosition struct {
	board string
	cgp   string
}

// pgSeekTo rewinds the game to a puzzle's position and renders it.
//
// CreatePuzzlesFromGame leaves the game wherever it stopped examining, so the
// seek is necessary; do it only after a game's puzzles have all been collected,
// or generation is looking at the wrong board.
func pgSeekTo(g *game.Game, turn int32) (pgPosition, error) {
	if err := g.PlayToTurn(int(turn)); err != nil {
		return pgPosition{}, err
	}
	return pgPosition{
		board: g.ToDisplayText(),
		// formatForBot blanks the opponent's rack, which is exactly what a
		// puzzle should show: the solver sees their own tiles and the board.
		cgp: g.ToCGP(true),
	}, nil
}

// pgFormatFull renders the sections -show asked for.
func pgFormatFull(pz *pb.PuzzleCreationResponse, pos pgPosition, show pgShow) string {
	var b strings.Builder
	if show.line {
		b.WriteString(pgFormatPuzzle(pz))
		b.WriteString("\n")
	}
	if show.board {
		b.WriteString(pos.board)
		b.WriteString("\n")
	}
	if show.cgp {
		b.WriteString("CGP: " + pos.cgp + "\n")
	}
	if show.answer {
		b.WriteString("Answer: " + pgAnswerString(pz.GetAnswer()) + "\n")
	}
	if show.stats {
		if j, err := protojson.Marshal(pz.GetStats()); err == nil {
			b.WriteString("Stats: " + string(j) + "\n")
		}
	}
	return b.String()
}

// pgAnswerString renders an answer event the way the shell writes a play.
func pgAnswerString(evt *pb.GameEvent) string {
	if evt == nil {
		return ""
	}
	switch evt.Type {
	case pb.GameEvent_EXCHANGE:
		return fmt.Sprintf("(exch %s)", evt.Exchanged)
	case pb.GameEvent_PASS:
		return "(Pass)"
	}
	return fmt.Sprintf("%s %s (%d, rack %s)", evt.Position, evt.PlayedTiles, evt.Score, evt.Rack)
}

// pgRecord is one line of the -out file. The answer and stats keep their proto
// shape via protojson rather than being flattened by hand, so a consumer can
// decode them straight back into GameEvent and PuzzleStats.
type pgRecord struct {
	GameID             string          `json:"game_id"`
	Turn               int32           `json:"turn"`
	Seed               string          `json:"seed,omitempty"`
	Lexicon            string          `json:"lexicon"`
	LetterDistribution string          `json:"letter_distribution,omitempty"`
	CGP                string          `json:"cgp"`
	Board              string          `json:"board"`
	Answer             json.RawMessage `json:"answer,omitempty"`
	Tags               []string        `json:"tags"`
	Stats              json.RawMessage `json:"stats,omitempty"`
}

func pgBuildRecord(g *game.Game, pz *pb.PuzzleCreationResponse, pos pgPosition, seed string) (*pgRecord, error) {
	rec := &pgRecord{
		GameID:  pz.GetGameId(),
		Turn:    pz.GetTurnNumber(),
		Seed:    seed,
		Lexicon: g.LexiconName(),
		CGP:     pos.cgp,
		Board:   pos.board,
		Tags:    pgTagNames(pz.GetTags()),
	}
	if h := g.History(); h != nil {
		rec.LetterDistribution = h.LetterDistribution
	}
	if pz.GetAnswer() != nil {
		j, err := protojson.Marshal(pz.GetAnswer())
		if err != nil {
			return nil, err
		}
		rec.Answer = j
	}
	if pz.GetStats() != nil {
		j, err := protojson.Marshal(pz.GetStats())
		if err != nil {
			return nil, err
		}
		rec.Stats = j
	}
	return rec, nil
}

// pgWriteGCG writes a puzzle's whole game so the position can be reloaded:
// `load <file>` followed by `turn <N>` with the puzzle's turn number.
func pgWriteGCG(dir string, g *game.Game) (string, error) {
	contents, err := gcgio.GameHistoryToGCG(g.History(), true)
	if err != nil {
		return "", err
	}
	name := pgSafeFilename(g.Uid())
	if name == "" {
		name = "game"
	}
	path := filepath.Join(dir, name+".gcg")
	return path, os.WriteFile(path, []byte(contents), 0644)
}

// pgSafeFilename strips anything a filename should not carry. A seeded game's
// UID is "seed:" followed by URL-safe base64, so the colon is the only thing
// that actually needs replacing, but a UID from elsewhere could hold anything.
func pgSafeFilename(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return r
		case r == '.' || r == '_' || r == '-':
			return r
		}
		return '-'
	}, s)
}

// pgDeriveSeed turns a user-supplied base seed into a per-game one, so that a
// single -seed reproduces a whole run rather than just its first game.
func pgDeriveSeed(base [32]byte, gidx int) [32]byte {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(gidx))
	return sha256.Sum256(append(base[:], buf[:]...))
}

func pgParseSeed(s string) ([32]byte, error) {
	var out [32]byte
	// Accept any hex string; a short one is a perfectly good seed, it just
	// leaves the rest of the block zero.
	b, err := hex.DecodeString(strings.TrimPrefix(s, "0x"))
	if err != nil {
		// Not hex -- take the bytes of whatever the user typed. A memorable
		// word makes a better label for a run than 64 hex digits.
		b = []byte(s)
	}
	if len(b) > len(out) {
		return sha256.Sum256(b), nil
	}
	copy(out[:], b)
	return out, nil
}
