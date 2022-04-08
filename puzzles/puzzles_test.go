package puzzles

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/matryer/is"
	"github.com/rs/zerolog"
	"google.golang.org/protobuf/encoding/protojson"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"CSW21"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func TestPuzzles(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.Disabled)
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
	puzzlesMatch(is, "equity", equityPuzzle)

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
	puzzlesMatch(is, "only_bingo", onlyBingoPuzzle)

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
	puzzlesMatch(is, "only_bingo", blankBingoPuzzle)

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
	puzzlesMatch(is, "cel_only", celOnlyPuzzle)

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
	puzzlesMatch(is, "bingo_nine_or_above", bingoNinePuzzle)

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
	puzzlesMatch(is, "bingo_nine_or_above", bingoFifteenPuzzle)
}

func TestLostChallenge(t *testing.T) {
	is := is.New(t)

	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, "./testdata/phony_tiles_returned.gcg")
	is.NoErr(err)

	// Set the correct challenge rule
	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT

	rules, err := game.NewBasicGameRules(&DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gameHistory, rules, 0)
	is.NoErr(err)

	// This would fail if there was no check for the
	// game event type in CreatePuzzlesFromGame
	_, err = CreatePuzzlesFromGame(&DefaultConfig, game)
	is.NoErr(err)
}

func TestPhonyTilesReturned(t *testing.T) {
	is := is.New(t)
	gh := &pb.GameHistory{}
	bts, err := os.ReadFile("./testdata/phony_tiles_history.json")
	is.NoErr(err)
	err = protojson.Unmarshal(bts, gh)
	is.NoErr(err)
	rules, err := game.NewBasicGameRules(&DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	game, err := game.NewFromHistory(gh, rules, 0)
	is.NoErr(err)

	_, err = CreatePuzzlesFromGame(&DefaultConfig, game)
	is.NoErr(err)
}

func puzzlesMatch(is *is.I, gcgfile string, expectedPzl *pb.PuzzleCreationResponse) {
	gameHistory, err := gcgio.ParseGCG(&DefaultConfig, fmt.Sprintf("./testdata/%s.gcg", gcgfile))
	if err != nil {
		panic(err)
	}

	// Set the challenge rule to five point
	// so GCGs with challenges will load
	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT

	rules, err := game.NewBasicGameRules(&DefaultConfig, "CSW21", board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		panic(err)
	}
	game, err := game.NewFromHistory(gameHistory, rules, 0)
	if err != nil {
		panic(err)
	}

	pzls, err := CreatePuzzlesFromGame(&DefaultConfig, game)
	if err != nil {
		panic(err)
	}

	for _, pzl := range pzls {
		if expectedPzl.TurnNumber == pzl.TurnNumber {
			fmt.Printf("%v\n", pzl.Tags)
			fmt.Printf("%v\n", pzl.Answer)
			is.Equal(expectedPzl.Answer.Type, pzl.Answer.Type)
			is.Equal(expectedPzl.Answer.Row, pzl.Answer.Row)
			is.Equal(expectedPzl.Answer.Column, pzl.Answer.Column)
			is.Equal(expectedPzl.Answer.Direction, pzl.Answer.Direction)
			is.Equal(expectedPzl.Answer.PlayedTiles, pzl.Answer.PlayedTiles)

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
