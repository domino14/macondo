package runner

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"America", "NWL18"} {
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

func compareCrossScores(t *testing.T, b1 *board.GameBoard, b2 *board.GameBoard) {
	dim := b1.Dim()
	dirs := []board.BoardDirection{board.HorizontalDirection, board.VerticalDirection}
	var d board.BoardDirection

	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			for _, d = range dirs {
				cs1 := b1.GetCrossScore(r, c, d)
				cs2 := b2.GetCrossScore(r, c, d)
				assert.Equal(t, cs1, cs2)
			}
		}
	}
}

type testMove struct {
	coords string
	word   string
	rack   string
	score  int
}

func TestCompareGameMove(t *testing.T) {
	opts := &GameOptions{
		FirstIsAssigned: true,
		GoesFirst:       0,
		ChallengeRule:   pb.ChallengeRule_SINGLE,
	}
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}

	// gen1 := cross_set.GaddagCrossSetGenerator{Dist: dist, Gaddag: gd}
	// gen2 := cross_set.CrossScoreOnlyGenerator{Dist: dist}

	rules1, err := game.NewBasicGameRules(
		&DefaultConfig, "America", board.CrosswordGameLayout, "english",
		game.CrossScoreAndSet, "")
	rules2, err := game.NewBasicGameRules(
		&DefaultConfig, "America", board.CrosswordGameLayout, "english",
		game.CrossScoreOnly, "")

	var testCases = []testMove{
		{"8D", "QWERTY", "QWERTYU", 62},
		{"H8", "TAEL", "TAELABC", 4},
		{"D7", "EQUALITY", "EUALITY", 90},
		{"E10", "MINE", "MINEFHI", 24},
		{"C13", "AB", "ABIIOOO", 21},
	}

	game1, err := NewGameRunnerFromRules(opts, players, rules1)
	if err != nil {
		t.Error(err)
	}
	game2, err := NewGameRunnerFromRules(opts, players, rules2)
	if err != nil {
		t.Error(err)
	}
	// create a move.
	for _, tc := range testCases {
		err = game1.SetCurrentRack(tc.rack)
		if err != nil {
			t.Error(err)
		}
		err = game2.SetCurrentRack(tc.rack)
		if err != nil {
			t.Error(err)
		}
		m1, err := game1.NewPlacementMove(game1.PlayerOnTurn(), tc.coords, tc.word)
		if err != nil {
			t.Error(err)
		}
		m2, err := game2.NewPlacementMove(game2.PlayerOnTurn(), tc.coords, tc.word)
		if err != nil {
			t.Error(err)
		}
		game1.PlayMove(m1, true, 0)
		game2.PlayMove(m2, true, 0)
		compareCrossScores(t, game1.Board(), game2.Board())
		assert.Equal(t, tc.score, m1.Score())
		assert.Equal(t, tc.score, m2.Score())
	}
}
