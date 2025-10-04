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
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY},
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
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_ONLY_BINGO, pb.PuzzleTag_POWER_TILE},
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
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_BLANK_BINGO},
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
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO_NINE_OR_ABOVE, pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY, pb.PuzzleTag_ONLY_BINGO},
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
		Tags: []pb.PuzzleTag{pb.PuzzleTag_BINGO_NINE_OR_ABOVE, pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY},
	}
	puzzlesMatch(is, "bingo_nine_or_above", dpgr, bingoFifteenPuzzle)
}

func TestPuzzleGeneration(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	game, err := gcgio.ParseGCG(DefaultConfig, "./testdata/phony_tiles_returned.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	game.SetChallengeRule(pb.ChallengeRule_FIVE_POINT)

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
		Tags:        []pb.PuzzleTag{pb.PuzzleTag_BINGO, pb.PuzzleTag_EQUITY},
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

	game, err := gcgio.ParseGCG(DefaultConfig, "./testdata/phony_tiles_returned.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	game.SetChallengeRule(pb.ChallengeRule_FIVE_POINT)

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
	game, err := gcgio.ParseGCG(DefaultConfig, "./testdata/well_played_game.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	game.SetChallengeRule(pb.ChallengeRule_DOUBLE)

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
	game, err := gcgio.ParseGCG(DefaultConfig, "./testdata/well_played_game.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	game.SetChallengeRule(pb.ChallengeRule_DOUBLE)

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
		log.Info().Interface("pzl", pzls[i]).Msg("puzzle")
	}

	is.Equal(pzls[2].TurnNumber, int32(2))
	valid, err := IsEquityPuzzleStillValid(DefaultConfig, game, 2, pzls[2].Answer, "NWL23")
	is.NoErr(err)
	is.True(valid)

	// turn 8 had PEDDL(I)nG as the answer. DOGPILED, new in NWL23, scores 3 more pts.
	is.Equal(pzls[5].TurnNumber, int32(8))
	valid, err = IsEquityPuzzleStillValid(DefaultConfig, game, 8, pzls[5].Answer, "NWL23")
	is.NoErr(err)
	is.Equal(valid, false)
}

func puzzlesMatch(is *is.I, gcgfile string, puzzleGenerationReq *pb.PuzzleGenerationRequest, expectedPzl *pb.PuzzleCreationResponse) {
	game, err := gcgio.ParseGCG(DefaultConfig, fmt.Sprintf("./testdata/%s.gcg", gcgfile))
	if err != nil {
		panic(err)
	}
	log.Info().Str("gcgfile", gcgfile).Msg("checking if puzzles match")

	// Set the challenge rule to five point
	// so GCGs with challenges will load
	game.SetChallengeRule(pb.ChallengeRule_FIVE_POINT)

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
