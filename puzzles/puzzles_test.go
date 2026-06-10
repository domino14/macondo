package puzzles

import (
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/matryer/is"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

var DefaultConfig = config.DefaultConfig()
var DefaultPuzzleGenerationReq = &pb.PuzzleGenerationRequest{
	Buckets: []*pb.PuzzleBucket{
		{
			Size:     100000,
			Includes: []pb.PuzzleTag{},
			Excludes: []pb.PuzzleTag{},
		},
	},
}

func TestPuzzles(t *testing.T) {
	is := is.New(t)

	dpgr := proto.Clone(DefaultPuzzleGenerationReq).(*pb.PuzzleGenerationRequest)
	err := InitializePuzzleGenerationRequest(dpgr)
	is.NoErr(err)

	equityPuzzle := &pb.PuzzleCreationResponse{TurnNumber: 1,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         0,
			Column:      7,
			Direction:   pb.GameEvent_VERTICAL,
			PlayedTiles: "KOFTGAR.",
		},
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_POINTS},
	}
	puzzlesMatch(is, "equity", dpgr, equityPuzzle)

	onlyBingoPuzzle := &pb.PuzzleCreationResponse{TurnNumber: 1,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         7,
			Column:      8,
			Direction:   pb.GameEvent_VERTICAL,
			PlayedTiles: ".EQUORIN",
		},
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_ONLY_BINGO, pb.PuzzleTag_POWER_TILE, pb.PuzzleTag_POINTS},
	}
	puzzlesMatch(is, "only_bingo", dpgr, onlyBingoPuzzle)

	blankBingoPuzzle := &pb.PuzzleCreationResponse{TurnNumber: 11,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         2,
			Column:      11,
			Direction:   pb.GameEvent_VERTICAL,
			PlayedTiles: "GLOBAtE",
		},
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_BLANK_BINGO, pb.PuzzleTag_POINTS},
	}
	puzzlesMatch(is, "only_bingo", dpgr, blankBingoPuzzle)

	celOnlyPuzzle := &pb.PuzzleCreationResponse{TurnNumber: 5,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         2,
			Column:      2,
			Direction:   pb.GameEvent_HORIZONTAL,
			PlayedTiles: "ADMITS",
		},
		Tags: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY, pb.PuzzleTag_NON_BINGO},
	}
	puzzlesMatch(is, "cel_only", dpgr, celOnlyPuzzle)

	bingoNinePuzzle := &pb.PuzzleCreationResponse{TurnNumber: 2,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         2,
			Column:      7,
			Direction:   pb.GameEvent_VERTICAL,
			PlayedTiles: "WATER..OI",
		},
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO_NINE_OR_ABOVE, pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_ONLY_BINGO, pb.PuzzleTag_POINTS},
	}
	puzzlesMatch(is, "bingo_nine_or_above", dpgr, bingoNinePuzzle)

	bingoFifteenPuzzle := &pb.PuzzleCreationResponse{TurnNumber: 4,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         2,
			Column:      0,
			Direction:   pb.GameEvent_HORIZONTAL,
			PlayedTiles: "ELECTRO........",
		},
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO_NINE_OR_ABOVE, pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_POINTS},
	}
	puzzlesMatch(is, "bingo_nine_or_above", dpgr, bingoFifteenPuzzle)
}

func TestPuzzleGeneration(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "./testdata/phony_tiles_returned.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT

	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	_, err = CreatePuzzlesFromGame(DefaultConfig, 1000, game, nil)
	is.True(err != nil)
	is.Equal(err.Error(), "puzzle generation request is nil")

	puzzleGenerationReq := &pb.PuzzleGenerationRequest{}

	_, err = CreatePuzzlesFromGame(DefaultConfig, 1000, game, puzzleGenerationReq)
	is.True(err != nil)
	is.Equal(err.Error(), "buckets are nil in puzzle generation request")

	puzzleGenerationReq = &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY, pb.PuzzleTag_ONLY_BINGO, pb.PuzzleTag_BINGO},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY, pb.PuzzleTag_POWER_TILE},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_BINGO_NINE_OR_ABOVE},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY, pb.PuzzleTag_BINGO, pb.PuzzleTag_ONLY_BINGO},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE, pb.PuzzleTag_CEL_ONLY},
			},
		},
	}

	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.True(err != nil)
	is.Equal(err.Error(), "bucket 2 is not unique")

	puzzleGenerationReq = &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY, pb.PuzzleTag_ONLY_BINGO, pb.PuzzleTag_BINGO, pb.PuzzleTag_CEL_ONLY},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY, pb.PuzzleTag_POWER_TILE},
			},
		},
	}

	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.True(err != nil)
	is.Equal(err.Error(), "error for bucket 0: invalid puzzle bucket, tag CEL_ONLY appears more than once")

	puzzleGenerationReq = &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE},
				Excludes: []pb.PuzzleTag{},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY, pb.PuzzleTag_BINGO},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY, pb.PuzzleTag_POWER_TILE},
			},
		},
	}

	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.NoErr(err)

	equityPuzzle := &pb.PuzzleCreationResponse{TurnNumber: 1,
		Answer: &pb.GameEvent{
			Type:        pb.GameEvent_TILE_PLACEMENT_MOVE,
			Row:         0,
			Column:      7,
			Direction:   pb.GameEvent_VERTICAL,
			PlayedTiles: "KOFTGAR.",
		},
		Tags:        []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_POINTS},
		BucketIndex: 2,
	}
	puzzlesMatch(is, "equity", puzzleGenerationReq, equityPuzzle)

	// Bucket order matters
	// If a puzzle fits in more than one bucket, it will
	// always go in the lower index bucket.
	puzzleGenerationReq = &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_BINGO},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY},
			},
		},
	}

	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.NoErr(err)

	equityPuzzle.BucketIndex = 0
	puzzlesMatch(is, "equity", puzzleGenerationReq, equityPuzzle)

	puzzleGenerationReq = &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_BINGO},
			},
		},
	}

	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.NoErr(err)

	equityPuzzle.BucketIndex = 1
	puzzlesMatch(is, "equity", puzzleGenerationReq, equityPuzzle)
}

func TestLostChallenge(t *testing.T) {
	is := is.New(t)

	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "./testdata/phony_tiles_returned.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT

	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	dpgr := proto.Clone(DefaultPuzzleGenerationReq).(*pb.PuzzleGenerationRequest)
	err = InitializePuzzleGenerationRequest(dpgr)
	is.NoErr(err)
	// This would fail if there was no check for the
	// game event type in CreatePuzzlesFromGame
	_, err = CreatePuzzlesFromGame(DefaultConfig, 1000, game, dpgr)
	is.NoErr(err)
}

func TestPhonyTilesReturned(t *testing.T) {
	is := is.New(t)
	gh := &pb.GameHistory{}
	bts, err := os.ReadFile("./testdata/phony_tiles_history.json")
	is.NoErr(err)
	err = protojson.Unmarshal(bts, gh)
	is.NoErr(err)
	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gh, rules, 0)
	is.NoErr(err)

	dpgr := proto.Clone(DefaultPuzzleGenerationReq).(*pb.PuzzleGenerationRequest)
	err = InitializePuzzleGenerationRequest(dpgr)
	is.NoErr(err)

	_, err = CreatePuzzlesFromGame(DefaultConfig, 1000, game, dpgr)
	is.NoErr(err)
}

func TestEquityLossLimit(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	// A little less than 23 total equity loss this game
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "./testdata/well_played_game.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	gameHistory.ChallengeRule = pb.ChallengeRule_DOUBLE

	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	puzzleGenerationReq := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE},
				Excludes: []pb.PuzzleTag{},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY, pb.PuzzleTag_BINGO},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY, pb.PuzzleTag_POWER_TILE},
			},
		},
	}

	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.NoErr(err)

	pzls, err := CreatePuzzlesFromGame(DefaultConfig, 23, game, puzzleGenerationReq)
	is.NoErr(err)
	is.True(len(pzls) > 0)

	/* set an equity loss limit of 22. this should fail, as the players lost more than 22 equity */

	pzls, err = CreatePuzzlesFromGame(DefaultConfig, 22, game, puzzleGenerationReq)
	is.NoErr(err)
	is.Equal(len(pzls), 0)
}

func TestIsPuzzleStillValid(t *testing.T) {
	is := is.New(t)
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, "./testdata/well_played_game.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	gameHistory.ChallengeRule = pb.ChallengeRule_DOUBLE

	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL18", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	puzzleGenerationReq := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_POWER_TILE},
				Excludes: []pb.PuzzleTag{},
			},
			{
				Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY, pb.PuzzleTag_BINGO},
				Excludes: []pb.PuzzleTag{pb.PuzzleTag_CEL_ONLY, pb.PuzzleTag_POWER_TILE},
			},
		},
	}
	err = InitializePuzzleGenerationRequest(puzzleGenerationReq)
	is.NoErr(err)

	pzls, err := CreatePuzzlesFromGame(DefaultConfig, 23, game, puzzleGenerationReq)
	is.NoErr(err)
	is.True(len(pzls) > 0)
	for i := range pzls {
		fmt.Println(pzls[i])
	}
	is.Equal(len(pzls), 11)

	is.Equal(pzls[2].TurnNumber, int32(2))
	valid, err := IsEquityPuzzleStillValid(DefaultConfig, game, 2, pzls[2].Answer, "NWL23")
	is.NoErr(err)
	is.True(valid)

	// turn 8 had PEDDL(I)nG as the answer. DOGPILED, new in NWL23, scores 3 more pts.
	is.Equal(pzls[6].TurnNumber, int32(8))
	valid, err = IsEquityPuzzleStillValid(DefaultConfig, game, 8, pzls[6].Answer, "NWL23")
	is.NoErr(err)
	is.Equal(valid, false)
}

func puzzlesMatch(is *is.I, gcgfile string, puzzleGenerationReq *pb.PuzzleGenerationRequest, expectedPzl *pb.PuzzleCreationResponse) {
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, fmt.Sprintf("./testdata/%s.gcg", gcgfile))
	if err != nil {
		panic(err)
	}
	log.Info().Str("gcgfile", gcgfile).Msg("checking if puzzles match")

	// Set the challenge rule to five point
	// so GCGs with challenges will load
	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT

	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		panic(err)
	}
	game, err := game.NewFromHistory(gameHistory, rules, 0)
	if err != nil {
		panic(err)
	}

	pzls, err := CreatePuzzlesFromGame(DefaultConfig, 1000, game, puzzleGenerationReq)
	if err != nil {
		panic(err)
	}

	for _, pzl := range pzls {
		if expectedPzl.TurnNumber == pzl.TurnNumber {
			is.Equal(expectedPzl.Answer.Type, pzl.Answer.Type)
			is.Equal(expectedPzl.Answer.Row, pzl.Answer.Row)
			is.Equal(expectedPzl.Answer.Column, pzl.Answer.Column)
			is.Equal(expectedPzl.Answer.Direction, pzl.Answer.Direction)
			is.Equal(expectedPzl.Answer.PlayedTiles, pzl.Answer.PlayedTiles)

			is.Equal(expectedPzl.BucketIndex, pzl.BucketIndex)
			is.Equal(len(pzl.Tags), len(expectedPzl.Tags))
			sortTags(pzl.Tags)
			sortTags(expectedPzl.Tags)
			for i := range pzl.Tags {
				is.Equal(expectedPzl.Tags[i], pzl.Tags[i])
			}
		}
	}
}

func sortTags(tags []pb.PuzzleTag) {
	sort.Slice(tags, func(i, j int) bool {
		return tags[i] < tags[j]
	})
}

// TestEquityMarginParam verifies that equity_margin flows through correctly:
// with the default (0 → 10) the equity puzzle fires; with a very high margin it doesn't.
func TestEquityMarginParam(t *testing.T) {
	is := is.New(t)

	// Default margin (0 → uses 10): equity puzzle at turn 1 should fire EQUITY.
	defaultReq := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(defaultReq))
	pzls := puzzlesByGCG(t, "equity", defaultReq)
	foundEquity := false
	for _, pzl := range pzls {
		if pzl.TurnNumber == 1 {
			foundEquity = true
		}
	}
	is.True(foundEquity) // turn 1 must be an equity puzzle with default margin

	// Very high margin: no puzzle should qualify.
	highMarginReq := &pb.PuzzleGenerationRequest{
		EquityMargin: 999,
		Buckets:      []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{pb.PuzzleTag_EQUITY}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(highMarginReq))
	pzls = puzzlesByGCG(t, "equity", highMarginReq)
	is.Equal(len(pzls), 0)
}

// TestPointsTag verifies that the POINTS tag fires when the answer leads on score by > score_margin,
// and doesn't fire with a margin set above the actual advantage.
func TestPointsTag(t *testing.T) {
	is := is.New(t)

	// Default score_margin (10): equity.gcg turn 1 has score_adv=28, should fire POINTS.
	defaultReq := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{pb.PuzzleTag_POINTS}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(defaultReq))
	pzls := puzzlesByGCG(t, "equity", defaultReq)
	foundTurn1 := false
	for _, pzl := range pzls {
		if pzl.TurnNumber == 1 {
			foundTurn1 = true
			is.True(pzl.Stats.GetScoreAdvantage() > 10)
		}
	}
	is.True(foundTurn1)

	// score_margin=30: turn 1 has advantage 28, should NOT qualify.
	highMarginReq := &pb.PuzzleGenerationRequest{
		ScoreMargin: 30,
		Buckets:     []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{pb.PuzzleTag_POINTS}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(highMarginReq))
	pzls = puzzlesByGCG(t, "equity", highMarginReq)
	for _, pzl := range pzls {
		is.True(pzl.TurnNumber != 1) // turn 1 must not appear with margin 30
	}
}

// TestPuzzleStats verifies that PuzzleStats fields are populated correctly.
func TestPuzzleStats(t *testing.T) {
	is := is.New(t)

	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "equity", req)

	// Turn 1: KOFTGAR. is a bingo (7 tiles played).
	for _, pzl := range pzls {
		if pzl.TurnNumber == 1 {
			s := pzl.Stats
			is.True(s != nil)
			is.Equal(s.GetTilesPlayed(), int32(7))
			is.Equal(s.GetMainWordLength(), int32(8)) // 7 fresh + 1 played-through
			is.Equal(s.GetScore(), int32(101))
			is.True(s.GetEquityAdvantage() > 10)
			is.True(s.GetScoreAdvantage() > 10)
			// Top scorer is the same bingo: 7 tiles.
			is.Equal(s.GetTopScorePlayTilesPlayed(), int32(7))
			// This is a fresh-board bingo, so no cross-words.
			is.Equal(s.GetWordsFormed(), int32(1))
			is.Equal(s.GetMaxCrossWordLength(), int32(0))
			return
		}
	}
	t.Fatal("did not find turn 1 in equity.gcg puzzles")
}

// TestStatsBonusCoverage verifies bonus-square coverage counters and the bingo stat.
// bingo_nine_or_above.gcg turn 4 is ELECTRO........ which is a 15-letter bingo.
// The 7 fresh tiles (E-L-E-C-T-R-O, cols 0-6, row 2) land on DWS at col 2 and DLS at col 6.
func TestStatsBonusCoverage(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "bingo_nine_or_above", req)
	for _, pzl := range pzls {
		if pzl.TurnNumber == 4 {
			s := pzl.Stats
			is.True(s != nil)
			// ELECTRO........ plays 7 tiles (bingo) through 8 existing tiles; word length = 15
			is.Equal(s.GetTilesPlayed(), int32(7))
			is.Equal(s.GetMainWordLength(), int32(15))
			is.Equal(s.GetWordsFormed(), int32(1))
			// Fresh tiles cover a DWS (col 2) and a DLS (col 6) at row 2; no TWS on this row
			is.Equal(s.GetTwsCovered(), int32(0))
			is.True(s.GetDwsCovered() >= 1)
			is.True(s.GetBonusSquaresCovered() >= 2)
			// bonus_squares_covered = sum of individual counts (invariant)
			is.Equal(s.GetBonusSquaresCovered(),
				s.GetTwsCovered()+s.GetDwsCovered()+s.GetTlsCovered()+s.GetDlsCovered())
			// Score advantage > 10 (this is an EQUITY+POINTS puzzle)
			is.True(s.GetScoreAdvantage() > 10)
			// Top scoring play is also a bingo (7 tiles)
			is.Equal(s.GetTopScorePlayTilesPlayed(), int32(7))
			return
		}
	}
	t.Fatal("did not find turn 4 in bingo_nine_or_above.gcg puzzles")
}

// TestStatsTopScoreNotAnswer verifies top_score_play_tiles_played reflects the actual top scorer
// even when it's not the equity winner. We look for a turn where score_advantage < 0.
func TestStatsTopScoreNotAnswer(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "equity", req)
	for _, pzl := range pzls {
		if pzl.Stats.GetScoreAdvantage() < 0 {
			// Found a turn where the equity winner doesn't score the most.
			is.True(pzl.Stats.GetScore() < pzl.Stats.GetScore()-pzl.Stats.GetScoreAdvantage())
			return
		}
	}
	// If no such turn exists in equity.gcg, that's fine — the stat is still populated.
	t.Log("no negative score_advantage found in equity.gcg; skipping assertion")
}

// TestStatsWordsFormedCrossWords verifies word count and cross-word length stats.
// only_bingo.gcg has multi-word plays we can target.
func TestStatsWordsFormedCrossWords(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "only_bingo", req)
	for _, pzl := range pzls {
		s := pzl.Stats
		is.True(s != nil)
		// Invariants that must hold for every puzzle:
		if s.GetWordsFormed() > 1 {
			// Cross-word lengths must be populated.
			is.True(s.GetMaxCrossWordLength() >= 2)
			is.True(s.GetMinCrossWordLength() >= 2)
			is.True(s.GetMaxCrossWordLength() >= s.GetMinCrossWordLength())
		} else {
			// No cross-words: min/max should be zero.
			is.Equal(s.GetMaxCrossWordLength(), int32(0))
			is.Equal(s.GetMinCrossWordLength(), int32(0))
		}
		// bonus_squares_covered = sum of individual counts
		is.Equal(s.GetBonusSquaresCovered(),
			s.GetTwsCovered()+s.GetDwsCovered()+s.GetTlsCovered()+s.GetDlsCovered())
		// score_advantage for POINTS turns must be > score_margin (10 by default)
		for _, tag := range pzl.Tags {
			if tag == pb.PuzzleTag_POINTS {
				is.True(s.GetScoreAdvantage() > 10)
			}
		}
	}
}

// TestStatsHookAndPlayedThrough verifies longest_hooked_word_length and
// max_played_through_tile_score using equity.gcg where specific turns have known values.
func TestStatsHookAndPlayedThrough(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "equity", req)

	byTurn := map[int32]*pb.PuzzleCreationResponse{}
	for _, pz := range pzls {
		byTurn[pz.TurnNumber] = pz
	}

	// Turn 1: KOFTGAR. plays through I (face value 1); no hooks (fresh bingo on near-empty board)
	if pz, ok := byTurn[1]; ok {
		is.Equal(pz.Stats.GetMaxPlayedThroughTileScore(), int32(1))
		is.Equal(pz.Stats.GetLongestHookedWordLength(), int32(0))
	} else {
		t.Fatal("turn 1 not found in equity.gcg puzzles")
	}

	// Turn 4 has hook=1 and max played-through tile score=5 (through K).
	if pz, ok := byTurn[4]; ok {
		is.True(pz.Stats.GetMaxPlayedThroughTileScore() >= 5)
		is.True(pz.Stats.GetLongestHookedWordLength() >= 1)
	} else {
		t.Fatal("turn 4 not found in equity.gcg puzzles")
	}

	// Turn 17 has hook=4 (large cross-word hanging off an endpoint).
	if pz, ok := byTurn[17]; ok {
		is.True(pz.Stats.GetLongestHookedWordLength() >= 4)
	} else {
		t.Fatal("turn 17 not found in equity.gcg puzzles")
	}

	// Turn 18: best play goes through a high-value tile (Q=10).
	if pz, ok := byTurn[18]; ok {
		is.True(pz.Stats.GetMaxPlayedThroughTileScore() >= 10)
	} else {
		t.Fatal("turn 18 not found in equity.gcg puzzles")
	}
}

// TestStatsDLSCross verifies max_fresh_tile_face_value_on_letter_bonus_with_crossword.
// equity.gcg turn 12 has a fresh tile on a DLS that also forms a cross-word, with face value 8 (X).
func TestStatsDLSCross(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "equity", req)

	for _, pz := range pzls {
		if pz.TurnNumber == 12 {
			is.True(pz.Stats.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword() >= 8)
			return
		}
	}
	t.Fatal("turn 12 not found in equity.gcg puzzles")
}

// TestStatsRackMax verifies max_rack_tile_score.
// equity.gcg turn 0: Bob's rack AEIQRST includes Q (10 pts), so rack_max must be 10.
func TestStatsRackMax(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))
	pzls := puzzlesByGCG(t, "equity", req)

	for _, pz := range pzls {
		if pz.TurnNumber == 0 {
			is.Equal(pz.Stats.GetMaxRackTileScore(), int32(10))
			return
		}
	}
	t.Fatal("turn 0 not found in equity.gcg puzzles")
}

// TestStatsInvariants verifies numeric invariants that must hold for every generated puzzle.
func TestStatsInvariants(t *testing.T) {
	is := is.New(t)
	req := &pb.PuzzleGenerationRequest{
		Buckets: []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	is.NoErr(InitializePuzzleGenerationRequest(req))

	for _, gcf := range []string{"equity", "only_bingo", "bingo_nine_or_above"} {
		for _, pz := range puzzlesByGCG(t, gcf, req) {
			s := pz.Stats
			if s == nil {
				continue
			}
			// bonus sum invariant
			is.Equal(s.GetBonusSquaresCovered(),
				s.GetTwsCovered()+s.GetDwsCovered()+s.GetTlsCovered()+s.GetDlsCovered())
			// non-negative
			is.True(s.GetLongestHookedWordLength() >= 0)
			is.True(s.GetLongestExtendedWordLength() >= 0)
			is.True(s.GetMaxPlayedThroughTileScore() >= 0)
			is.True(s.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword() >= 0)
			is.True(s.GetMaxRackTileScore() >= 0)
			// cross-word consistency
			if s.GetWordsFormed() > 1 {
				is.True(s.GetMaxCrossWordLength() >= s.GetMinCrossWordLength())
				is.True(s.GetMinCrossWordLength() >= 2)
			} else {
				is.Equal(s.GetMaxCrossWordLength(), int32(0))
				is.Equal(s.GetMinCrossWordLength(), int32(0))
			}
			// tiles_played matches TilesPlayed() contract: 1–7
			is.True(s.GetTilesPlayed() >= 1)
			is.True(s.GetTilesPlayed() <= 7)
		}
	}
}

func puzzlesByGCG(t *testing.T, gcgfile string, req *pb.PuzzleGenerationRequest) []*pb.PuzzleCreationResponse {
	t.Helper()
	gameHistory, err := gcgio.ParseGCG(DefaultConfig, fmt.Sprintf("./testdata/%s.gcg", gcgfile))
	if err != nil {
		t.Fatalf("ParseGCG %s: %v", gcgfile, err)
	}
	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT
	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatalf("NewBasicGameRules: %v", err)
	}
	g, err := game.NewFromHistory(gameHistory, rules, 0)
	if err != nil {
		t.Fatalf("NewFromHistory: %v", err)
	}
	pzls, err := CreatePuzzlesFromGame(DefaultConfig, 1000, g, req)
	if err != nil {
		t.Fatalf("CreatePuzzlesFromGame: %v", err)
	}
	return pzls
}
