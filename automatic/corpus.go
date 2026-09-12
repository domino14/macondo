package automatic

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/turnplayer"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
)

// Replaying a finished run's positions.
//
// A completed autoplay run is a corpus of graded inference problems: the
// per-turn log holds every rack, every play, and -- for the inferring bot --
// the leave the opponent really kept. Re-running inference over those same
// positions is how a change to the engine gets tested, and it is the only way
// to test one honestly. Two configurations replayed over one corpus see
// identical boards, identical racks and identical truths, so the comparison is
// paired: the noise that dominates a fresh autoplay run cancels.
//
// It is also the difference between an afternoon and a week. The run these
// positions came from took 43 hours; replaying a few hundred of its positions
// takes minutes.

// CorpusPosition is one inference problem lifted out of a run.
type CorpusPosition struct {
	// GameID and Half say which game and which seating it came from; Turn is
	// the play the inference was reading, 1-based as the log numbers them.
	GameID string
	Half   int
	Turn   int

	// Game is replayed up to and including the opponent's play, so the
	// position is exactly what the inferring bot faced -- including the
	// inferring player's own rack, which decides the unseen pool and so the
	// prior every inference is graded against.
	Game *game.Game

	// MyRack is the inferring player's rack at this position, taken from the
	// row where they play next. Replaying alone cannot recover it: the tiles
	// they drew came off a bag this replay never had in the same order.
	MyRack string

	// TrueLeave is what the opponent actually kept -- the answer. OppRack and
	// OppPlay are what produced it.
	TrueLeave string
	OppRack   string
	OppPlay   string
	OppScore  int

	// TilesRemaining is the bag count after the play, and LeaveLen the number
	// of tiles being inferred; both are what the run's own breakdowns cut on.
	TilesRemaining int
	LeaveLen       int

	// LoggedLiftBits is what the original run scored this inference, when it
	// recorded one. A replay that reproduces the run should land near it.
	LoggedLiftBits  float64
	LoggedMeasured  bool
	LoggedRank      int
	LoggedLeaves    int
	HasLoggedResult bool
}

// CorpusFilter narrows which positions to lift out.
type CorpusFilter struct {
	// LeaveLen, when non-zero, keeps only inferences of that many tiles. Six
	// is the one-tile-play bucket, where the read goes wrong.
	LeaveLen int
	// MaxBag and MinBag bound the phase; MaxBag 0 means no upper bound.
	MinBag, MaxBag int
	// WorseThan keeps only inferences the run scored below this many bits.
	// Leave at zero and pass AllLifts to take everything.
	WorseThan float64
	AllLifts  bool
	// Limit caps how many positions are returned, 0 for all.
	Limit int
}

// LoadCorpus replays a run's per-turn log and returns the positions matching
// the filter, each rebuilt to the moment the inference ran. Games that will
// not replay are skipped and counted in Skipped; a corpus that comes back
// empty because every game failed is reported as an error rather than as an
// empty result, since the two look identical to a caller and mean opposite
// things.
func LoadCorpus(cfg *config.Config, turnFile, lexicon, letterdist, boardlayout string,
	filter CorpusFilter) ([]*CorpusPosition, error) {
	pos, _, err := LoadCorpusVerbose(cfg, turnFile, lexicon, letterdist, boardlayout, filter)
	return pos, err
}

// LoadCorpusVerbose is LoadCorpus, also returning how many games were skipped
// because they would not replay.
func LoadCorpusVerbose(cfg *config.Config, turnFile, lexicon, letterdist, boardlayout string,
	filter CorpusFilter) ([]*CorpusPosition, int, error) {

	if letterdist == "" {
		letterdist = "english"
	}
	if boardlayout == "" {
		boardlayout = board.CrosswordGameLayout
	}
	if lexicon == "" {
		lexicon = "NWL23"
	}

	file, _, err := cache.Open(turnFile)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	r := csv.NewReader(file)
	var cols map[string]int
	var order []string
	byGame := map[string][][]string{}
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, err
		}
		if record[0] == "playerID" {
			cols = map[string]int{}
			for i, name := range record {
				cols[name] = i
			}
			if _, ok := cols["trueLeave"]; !ok {
				return nil, 0, fmt.Errorf(
					"%s has no inference columns; it was not produced by a run with an inferring bot",
					turnFile)
			}
			continue
		}
		if cols == nil {
			continue
		}
		gid := record[1]
		if _, seen := byGame[gid]; !seen {
			order = append(order, gid)
		}
		byGame[gid] = append(byGame[gid], record)
	}
	if cols == nil {
		return nil, 0, fmt.Errorf("%s has no header row", turnFile)
	}

	// CrossScoreAndSet, not CrossScoreOnly: these positions get handed to move
	// generation, and without cross-sets a generator produces the wrong moves.
	// The GCG exporter can do without them because it only replays for
	// notation; anything that has to think about the position cannot.
	rules, err := game.NewBasicGameRules(cfg, lexicon, boardlayout,
		letterdist, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		return nil, 0, err
	}

	var out []*CorpusPosition
	skipped, wanted := 0, 0
	var firstErr error
	for _, gid := range order {
		for h, half := range splitTurnHalves(byGame[gid]) {
			picks := wantedTurns(half, cols, filter)
			if len(picks) == 0 {
				continue
			}
			wanted++
			pos, err := replayHalf(rules, gid, h+1, half, cols, picks)
			if err != nil {
				// One broken game is not a reason to abandon the corpus, but a
				// corpus where they are all broken is a bug worth surfacing.
				skipped++
				if firstErr == nil {
					firstErr = fmt.Errorf("game %s half %d: %w", gid, h+1, err)
				}
				continue
			}
			out = append(out, pos...)
			if filter.Limit > 0 && len(out) >= filter.Limit {
				return out[:filter.Limit], skipped, nil
			}
		}
	}
	if len(out) == 0 && wanted > 0 {
		return nil, skipped, fmt.Errorf(
			"every one of the %d matching games failed to replay; first failure: %w",
			wanted, firstErr)
	}
	return out, skipped, nil
}

// wantedTurns returns the indices within a half whose *following* row carries
// an inference matching the filter. The inference at row i+1 reads the play at
// row i, so the position to rebuild ends with row i.
func wantedTurns(half [][]string, cols map[string]int, f CorpusFilter) map[int]int {
	get := func(row []string, name string) string {
		i, ok := cols[name]
		if !ok || i >= len(row) {
			return ""
		}
		return row[i]
	}
	picks := map[int]int{}
	for i := 0; i+1 < len(half); i++ {
		next := half[i+1]
		if get(next, "inferCount") == "" || get(next, "truePost") == "" {
			continue
		}
		truth := get(next, "trueLeave")
		if truth == "" || truth != get(half[i], "leave") {
			continue
		}
		if f.LeaveLen > 0 && len([]rune(truth)) != f.LeaveLen {
			continue
		}
		bag, err := strconv.Atoi(get(half[i], "tilesremaining"))
		if err != nil {
			continue
		}
		if bag < f.MinBag || (f.MaxBag > 0 && bag > f.MaxBag) {
			continue
		}
		if !f.AllLifts {
			lift, err := strconv.ParseFloat(get(next, "liftBits"), 64)
			// A ruled-out leave has no ratio and logs as NaN or blank. It is
			// the worst outcome there is, so it belongs in a "worse than" cut.
			if err != nil {
				if get(next, "truePost") != "0" {
					continue
				}
			} else if lift >= f.WorseThan {
				continue
			}
		}
		picks[i] = i + 1
	}
	return picks
}

// replayHalf plays a half's rows in order, snapshotting the game at each
// wanted turn. Each snapshot is a deep copy, so the caller gets independent
// positions rather than views of one mutating game.
func replayHalf(rules *game.GameRules, gid string, halfNo int, half [][]string,
	cols map[string]int, picks map[int]int) ([]*CorpusPosition, error) {

	players := []*pb.PlayerInfo{
		{Nickname: half[0][0], RealName: half[0][0]},
		{Nickname: half[1][0], RealName: half[1][0]},
	}
	g, err := turnplayer.BaseTurnPlayerFromRules(&turnplayer.GameOptions{
		BoardLayoutName: board.CrosswordGameLayout,
		Variant:         game.VarClassic,
	}, players, rules)
	if err != nil {
		return nil, err
	}
	g.StartGame()

	get := func(row []string, name string) string {
		i, ok := cols[name]
		if !ok || i >= len(row) {
			return ""
		}
		return row[i]
	}

	var out []*CorpusPosition
	for i, row := range half {
		infIdx, want := picks[i]

		// Set both racks at once on a turn we are going to snapshot. The
		// inferring player's rack decides the unseen pool, and so the prior
		// every inference is graded against, but replaying a log cannot
		// recover it: the tiles they drew came off a bag this replay never had
		// in the same order. It has to be read off the row where they play
		// next -- and it has to be set together with the opponent's, because
		// SetRackFor hands the other player a fresh random rack, which would
		// throw away whatever we had just arranged.
		myRack := ""
		if want {
			myRack = get(half[infIdx], "rack")
		}
		if myRack != "" {
			me := 0
			if g.History().Players[1].Nickname == half[infIdx][0] {
				me = 1
			}
			racks := make([]*tilemapping.Rack, 2)
			racks[me] = tilemapping.RackFromString(myRack, g.Alphabet())
			racks[1-me] = tilemapping.RackFromString(row[3], g.Alphabet())
			if err := g.Game.SetRacksForBoth(racks); err != nil {
				return nil, fmt.Errorf(
					"turn %d: seating %s with %q and %s with %q: %w",
					i+1, half[infIdx][0], myRack, row[0], row[3], err)
			}
			if err := playLoggedRow(g, row, false); err != nil {
				return nil, err
			}
		} else if err := playLoggedRow(g, row, true); err != nil {
			return nil, err
		}
		if !want {
			continue
		}
		inf := half[infIdx]
		// With history: inference reads the opponent's last play out of it.
		snap := g.Game.CopyWithHistory()

		bag, _ := strconv.Atoi(get(row, "tilesremaining"))
		score, _ := strconv.Atoi(get(row, "score"))
		truth := get(row, "leave")
		p := &CorpusPosition{
			GameID: gid, Half: halfNo, Turn: i + 1,
			Game:           snap,
			TrueLeave:      truth,
			OppRack:        get(row, "rack"),
			OppPlay:        get(row, "play"),
			OppScore:       score,
			MyRack:         myRack,
			TilesRemaining: bag,
			LeaveLen:       len([]rune(truth)),
		}
		if lift, err := strconv.ParseFloat(get(inf, "liftBits"), 64); err == nil {
			p.LoggedLiftBits = lift
			p.HasLoggedResult = true
		}
		p.LoggedMeasured = get(inf, "trueMeasured") == "true"
		p.LoggedRank, _ = strconv.Atoi(get(inf, "trueRank"))
		p.LoggedLeaves, _ = strconv.Atoi(get(inf, "inferLeaves"))
		out = append(out, p)
	}
	return out, nil
}

// playLoggedRow applies one per-turn log row to a game in progress. setRack
// false says the caller has already seated both players and the racks must be
// left alone.
func playLoggedRow(g *turnplayer.BaseTurnPlayer, row []string, setRack bool) error {
	pidx := 0
	if g.History().Players[1].Nickname == row[0] {
		pidx = 1
	}
	if setRack {
		if err := g.SetRackFor(pidx, tilemapping.RackFromString(row[3], g.Alphabet())); err != nil {
			return err
		}
	}
	switch {
	case strings.HasPrefix(row[4], "(exch"):
		cmd := strings.Split(row[4], " ")
		m, err := g.NewExchangeMove(pidx, strings.TrimSuffix(cmd[1], ")"))
		if err != nil {
			return err
		}
		return g.PlayMove(m, true, 0)
	case row[4] == "(Pass)":
		m, err := g.NewPassMove(pidx)
		if err != nil {
			return err
		}
		return g.PlayMove(m, true, 0)
	default:
		play := strings.Split(strings.TrimSpace(row[4]), " ")
		if len(play) < 2 {
			return fmt.Errorf("cannot parse play %q", row[4])
		}
		m, err := g.NewPlacementMove(pidx, play[0], play[1], false)
		if err != nil {
			return err
		}
		return g.PlayMove(m, true, 0)
	}
}
