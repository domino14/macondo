package shell

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// puzzle browses a puzzle file written by `puzzlegen -out`. Each puzzle is
// loaded as the current game, so every ordinary command -- s, gen, sim -- works
// on it; the answer stays hidden until asked for.
//
// See "help puzzle".
func (sc *ShellController) puzzle(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) == 0 {
		return nil, fmt.Errorf("usage: puzzle <open|next|prev|goto|keep|unkeep|kept|answer|info|list> [args...]")
	}

	verb := strings.ToLower(cmd.args[0])
	args := cmd.args[1:]

	if verb == "open" {
		if len(args) == 0 {
			return nil, fmt.Errorf("usage: puzzle open <file>")
		}
		return sc.puzzleOpen(args[0])
	}

	// A bare number is a shorthand for goto.
	if n, err := strconv.Atoi(verb); err == nil {
		return sc.puzzleGoto(n)
	}

	if len(sc.puzzleSet) == 0 {
		return nil, fmt.Errorf("no puzzles are open; use `puzzle open <file>` first")
	}

	switch verb {
	case "next", "n":
		if sc.puzzleIdx+1 >= len(sc.puzzleSet) {
			return nil, fmt.Errorf("already at the last puzzle (%d of %d)",
				sc.puzzleIdx+1, len(sc.puzzleSet))
		}
		return sc.puzzleGoto(sc.puzzleIdx + 2) // goto is 1-based

	case "prev", "p":
		if sc.puzzleIdx == 0 {
			return nil, fmt.Errorf("already at the first puzzle")
		}
		return sc.puzzleGoto(sc.puzzleIdx)

	case "goto":
		if len(args) == 0 {
			return nil, fmt.Errorf("usage: puzzle goto <N>")
		}
		n, err := strconv.Atoi(args[0])
		if err != nil {
			return nil, fmt.Errorf("invalid puzzle number %q", args[0])
		}
		return sc.puzzleGoto(n)

	case "answer":
		return msg(sc.puzzleAnswer(sc.puzzleSet[sc.puzzleIdx])), nil

	case "info":
		return msg(sc.puzzleInfo(sc.puzzleSet[sc.puzzleIdx])), nil

	case "list":
		return msg(sc.puzzleList()), nil

	case "keep":
		dest := ""
		if len(args) > 0 {
			dest = args[0]
		}
		return sc.puzzleKeep(dest)

	case "unkeep":
		return sc.puzzleUnkeep()

	case "kept":
		return msg(sc.puzzleKeptSummary()), nil
	}

	return nil, fmt.Errorf("unknown puzzle command %q; use open, next, prev, goto, keep, unkeep, kept, answer, info, or list", verb)
}

// puzzleOpen reads a JSON Lines puzzle file and shows the first puzzle.
func (sc *ShellController) puzzleOpen(path string) (*Response, error) {
	f, err := os.Open(expandHomePath(path))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var recs []*pgRecord
	scanner := bufio.NewScanner(f)
	// Records carry a rendered board and a GCG, so the default 64KB line limit
	// is not generous enough.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for line := 1; scanner.Scan(); line++ {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		var rec pgRecord
		if err := json.Unmarshal([]byte(text), &rec); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, line, err)
		}
		recs = append(recs, &rec)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(recs) == 0 {
		return nil, fmt.Errorf("%s has no puzzles in it", path)
	}

	sc.puzzleSet = recs
	sc.puzzleFile = path
	sc.showMessage(fmt.Sprintf("Opened %s: %d puzzles.", path, len(recs)))
	return sc.puzzleGoto(1)
}

// puzzleGoto loads the Nth puzzle (1-based) as the current game.
func (sc *ShellController) puzzleGoto(n int) (*Response, error) {
	if len(sc.puzzleSet) == 0 {
		return nil, fmt.Errorf("no puzzles are open; use `puzzle open <file>` first")
	}
	if n < 1 || n > len(sc.puzzleSet) {
		return nil, fmt.Errorf("puzzle %d is out of range; the file has %d", n, len(sc.puzzleSet))
	}
	rec := sc.puzzleSet[n-1]
	if err := sc.loadCGP(rec.CGP); err != nil {
		return nil, err
	}
	sc.puzzleIdx = n - 1
	sc.curTurnNum = 0
	sc.gameSource = "" // a puzzle position has no persistent game identity
	sc.initializeVariationTree()

	return msg(sc.puzzleMeta(rec) + "\n\n" + sc.game.ToDisplayText()), nil
}

// pgShapeTags name tags that describe the shape of the answer -- how many tiles
// it uses, which tiles, what vocabulary it comes from. Printing those above the
// board hands the solver most of the puzzle: BINGO_NINE_OR_ABOVE says the answer
// is a nine, NON_BINGO rules out a whole class of play, POWER_TILE says a Q or a
// Z goes down. They are kept for `puzzle info`, which is the deliberate act of
// asking.
//
// EQUITY and POINTS stay visible. They say what kind of question is being asked
// -- is there one clear best play, is it the top scorer -- rather than anything
// about the answer's form.
var pgShapeTags = map[string]bool{
	"BINGO":               true,
	"ONLY_BINGO":          true,
	"BLANK_BINGO":         true,
	"BINGO_NINE_OR_ABOVE": true,
	"NON_BINGO":           true,
	"POWER_TILE":          true,
	"CEL_ONLY":            true,
	"CEL_PLUS_TWOS":       true,
}

// pgVisibleTags splits a puzzle's tags into the ones safe to show beside the
// board and a count of the ones held back.
func pgVisibleTags(tags []string) (shown []string, hidden int) {
	for _, t := range tags {
		if pgShapeTags[t] {
			hidden++
			continue
		}
		shown = append(shown, t)
	}
	return shown, hidden
}

// puzzleMeta is the block shown above a puzzle's board: what kind of question it
// is and where it came from. It deliberately carries neither the answer nor
// anything describing it -- not the shape tags, and not the stats, which spell
// out the tile count and the squares covered. `puzzle info` has all of it.
func (sc *ShellController) puzzleMeta(rec *pgRecord) string {
	var b strings.Builder

	shown, hidden := pgVisibleTags(rec.Tags)
	fmt.Fprintf(&b, "Puzzle %d of %d", sc.puzzleIdx+1, len(sc.puzzleSet))
	if len(shown) > 0 {
		fmt.Fprintf(&b, "   %s", strings.Join(shown, " "))
	}
	if hidden > 0 {
		fmt.Fprintf(&b, "   (+%d tag(s) in `puzzle info`)", hidden)
	}
	b.WriteString("\n")

	provenance := []string{rec.Lexicon}
	if rec.LetterDistribution != "" {
		provenance = append(provenance, rec.LetterDistribution)
	}
	provenance = append(provenance, fmt.Sprintf("%s turn %d", pgShortID(rec.GameID), rec.Turn))
	if rec.Seed != "" {
		provenance = append(provenance, "seed "+rec.Seed)
	}
	fmt.Fprintf(&b, "  %s", strings.Join(provenance, "  ·  "))

	return b.String()
}

// pgShortID abbreviates a seeded game's UID, which is 43 characters of base64
// and would otherwise take the whole line. `puzzle info` prints it in full.
func pgShortID(id string) string {
	const keep = 12
	if len(id) <= keep {
		return id
	}
	return id[:keep] + "…"
}

// puzzleAnswer reveals the answer to a puzzle.
func (sc *ShellController) puzzleAnswer(rec *pgRecord) string {
	evt := rec.answerEvent()
	if evt == nil {
		return "This puzzle has no recorded answer."
	}
	out := "Answer: " + pgAnswerString(evt)
	if words := evt.GetWordsFormed(); len(words) > 0 {
		out += "\n  words: " + strings.Join(words, " ")
	}
	return out
}

// puzzleInfo tells you everything the record holds except the answer itself:
// every tag including the ones held back from the board header, the full game
// ID, the source file, the CGP, and every stat. Asking for it is a deliberate
// act, which is why the spoilers live here.
func (sc *ShellController) puzzleInfo(rec *pgRecord) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Puzzle %d of %d from %s\n", sc.puzzleIdx+1, len(sc.puzzleSet), sc.puzzleFile)
	fmt.Fprintf(&b, "  tags     %s\n", strings.Join(rec.Tags, " "))
	fmt.Fprintf(&b, "  game     %s turn %d\n", rec.GameID, rec.Turn)
	if rec.Seed != "" {
		fmt.Fprintf(&b, "  seed     %s\n", rec.Seed)
	}
	fmt.Fprintf(&b, "  lexicon  %s", rec.Lexicon)
	if rec.LetterDistribution != "" {
		fmt.Fprintf(&b, " (%s)", rec.LetterDistribution)
	}
	fmt.Fprintf(&b, "\n  cgp      %s\n", rec.CGP)
	if len(rec.Stats) > 0 {
		fmt.Fprintf(&b, "  stats    %s", pgStatsAll(rec.Stats))
	}
	return strings.TrimRight(b.String(), "\n")
}

// puzzleList indexes the open file, marking where you are. It hides the same
// things the board header does -- an index that prints every answer's score and
// shape spoils the whole file at once, which is exactly what someone about to
// solve them would run first.
func (sc *ShellController) puzzleList() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %d puzzles", sc.puzzleFile, len(sc.puzzleSet))
	if sc.puzzleKeepFile != "" {
		fmt.Fprintf(&b, "   (* kept in %s)", sc.puzzleKeepFile)
	}
	b.WriteString("\n")
	for i, rec := range sc.puzzleSet {
		here := "  "
		if i == sc.puzzleIdx {
			here = "->"
		}
		mark := " "
		if sc.puzzleKept[pgKeepKey(rec)] {
			mark = "*"
		}
		shown, hidden := pgVisibleTags(rec.Tags)
		line := strings.Join(shown, " ")
		if hidden > 0 {
			if line != "" {
				line += " "
			}
			line += fmt.Sprintf("(+%d)", hidden)
		}
		fmt.Fprintf(&b, "%s%s %3d  turn %2d  %s\n", here, mark, i+1, rec.Turn, line)
	}
	return strings.TrimRight(b.String(), "\n")
}

// answerEvent decodes the record's answer, which is stored as protojson so that
// a consumer outside macondo can read it as an ordinary object.
func (rec *pgRecord) answerEvent() *pb.GameEvent {
	if len(rec.Answer) == 0 {
		return nil
	}
	var evt pb.GameEvent
	if err := protojson.Unmarshal(rec.Answer, &evt); err != nil {
		return nil
	}
	return &evt
}

// pgStatsAll renders every stat that is set. It reads the record's JSON back as
// a plain map rather than walking the proto field by field, so a stat added to
// PuzzleStats shows up here without anyone remembering to add it -- and under
// the same name `-filter` expects, since the record is written with proto names.
func pgStatsAll(raw json.RawMessage) string {
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		return ""
	}
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	sort.Strings(names)

	var parts []string
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s=%v", name, fields[name]))
	}
	return strings.Join(parts, "  ")
}
