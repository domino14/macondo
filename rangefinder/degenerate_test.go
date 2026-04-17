package rangefinder

// Degenerate-case tests for the inference pipeline. These verify correctness
// properties that should hold for all positions, and catch regressions in edge
// cases the normal inference tests don't exercise.

import (
	"context"
	"sort"
	"testing"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

// TestBingoLeaveIsNoInformation verifies that when the opponent plays a
// 7-tile bingo, PrepareFinder returns ErrNoInformation: there is nothing to
// infer because the opp used their entire rack.
func TestBingoLeaveIsNoInformation(t *testing.T) {
	is := is.New(t)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)

	// Give player 0 a rack of exactly 7 tiles, play them all (a bingo).
	g.SetRackFor(0, tilemapping.RackFromString("LATHING", g.Alphabet()))
	_, err = g.PlayScoringMove("H8", "LATHING", true)
	is.NoErr(err)
	is.Equal(g.PlayerOnTurn(), 1) // now player 1's turn

	calcs := defaultSimCalculators(lex)
	rf := &RangeFinder{}
	rf.Init(g, calcs, DefaultConfig)

	err = rf.PrepareFinder(nil)
	is.Equal(err, ErrNoInformation)
}

// TestExchangeRackLength verifies that after an exchange of K tiles,
// PrepareFinder sets RackLength = 7 − K and the posterior racks all have
// that length.
func TestExchangeRackLength(t *testing.T) {
	is := is.New(t)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)

	// Player 0 holds UVWXYZ?, exchanges UVW (3 tiles), keeping XYZ? (4 tiles).
	g.SetRackFor(0, tilemapping.RackFromString("UVWXYZ?", g.Alphabet()))

	// Build exchange move: exchange UVW.
	uvw, err := tilemapping.ToMachineLetters("UVW", g.Alphabet())
	is.NoErr(err)
	xyzBlank, err := tilemapping.ToMachineLetters("XYZ?", g.Alphabet())
	is.NoErr(err)
	m := move.NewExchangeMove(uvw, xyzBlank, g.Alphabet())
	err = g.PlayMove(m, true, 0)
	is.NoErr(err)
	is.Equal(g.PlayerOnTurn(), 1)

	calcs := defaultSimCalculators(lex)
	rf := &RangeFinder{}
	rf.Init(g, calcs, DefaultConfig)

	// Player 1's rack is whatever they were dealt.
	err = rf.PrepareFinder(nil)
	is.NoErr(err)
	is.Equal(rf.inference.RackLength, 4) // kept XYZ? (4 tiles)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	// Every inferred rack must have exactly 4 tiles.
	for _, ir := range rf.inference.InferredRacks {
		is.Equal(len(ir.Leave), 4)
	}
}

// TestAllPosteriorRacksBagConsistent verifies that every rack in the posterior
// is a multiset-subset of the inference bag map, for a normal mid-game position.
func TestAllPosteriorRacksBagConsistent(t *testing.T) {
	is := is.New(t)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)

	// Play a 4-tile word, keeping 3 tiles — not a bingo.
	g.SetRackFor(0, tilemapping.RackFromString("HELPXYZ", g.Alphabet()))
	_, err = g.PlayScoringMove("H8", "HELP", true)
	is.NoErr(err)
	is.Equal(g.PlayerOnTurn(), 1)

	calcs := defaultSimCalculators(lex)
	rf := &RangeFinder{}
	rf.Init(g, calcs, DefaultConfig)
	is.NoErr(rf.PrepareFinder(nil))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	bagMap := rf.BagMap()
	for _, ir := range rf.inference.InferredRacks {
		// Count tiles in this leave.
		counts := make([]int, len(bagMap))
		for _, ml := range ir.Leave {
			counts[int(ml)]++
		}
		// Each tile in the leave must be available in the bag map.
		for i, cnt := range counts {
			if cnt > 0 {
				is.True(int(bagMap[i]) >= cnt)
			}
		}
	}
}

// TestTilePlayRackLengthConsistency verifies that after a normal tile play
// of K tiles, RackLength == 7 − K and all inferred racks have that length.
func TestTilePlayRackLengthConsistency(t *testing.T) {
	is := is.New(t)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)

	// Player 0 plays 4 tiles (HELP), keeping 3.
	g.SetRackFor(0, tilemapping.RackFromString("HELPABC", g.Alphabet()))
	_, err = g.PlayScoringMove("H8", "HELP", true)
	is.NoErr(err)
	is.Equal(g.PlayerOnTurn(), 1)

	calcs := defaultSimCalculators(lex)
	rf := &RangeFinder{}
	rf.Init(g, calcs, DefaultConfig)
	is.NoErr(rf.PrepareFinder(nil))
	is.Equal(rf.inference.RackLength, 3) // kept ABC

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	for _, ir := range rf.inference.InferredRacks {
		is.Equal(len(ir.Leave), 3)
	}
}

// TestPosteriorWeightsPositive verifies all weights in the posterior are > 0.
func TestPosteriorWeightsPositive(t *testing.T) {
	is := is.New(t)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)

	g.SetRackFor(0, tilemapping.RackFromString("CODING?", g.Alphabet()))
	_, err = g.PlayScoringMove("H8", "CODING", true)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)
	rf := &RangeFinder{}
	rf.Init(g, calcs, DefaultConfig)
	is.NoErr(rf.PrepareFinder(nil))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	is.True(len(rf.inference.InferredRacks) > 0)
	for _, ir := range rf.inference.InferredRacks {
		is.True(ir.Weight > 0)
	}
}

// TestNoDuplicateLeaves verifies the deduplication invariant: no two racks in
// the posterior share the same leave (as a sorted tile sequence).
func TestNoDuplicateLeaves(t *testing.T) {
	is := is.New(t)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)

	g.SetRackFor(0, tilemapping.RackFromString("FASTEST", g.Alphabet()))
	_, err = g.PlayScoringMove("H8", "FAST", true)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)
	rf := &RangeFinder{}
	rf.Init(g, calcs, DefaultConfig)
	is.NoErr(rf.PrepareFinder(nil))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	is.NoErr(rf.Infer(ctx))

	seen := make(map[string]bool)
	for _, ir := range rf.inference.InferredRacks {
		sorted := make([]tilemapping.MachineLetter, len(ir.Leave))
		copy(sorted, ir.Leave)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		key := leaveKey(sorted)
		is.True(!seen[key]) // no duplicate
		seen[key] = true
	}
}
