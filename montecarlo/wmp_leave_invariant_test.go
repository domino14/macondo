// wmp_leave_invariant_test.go: root-causes and reproduces a montecarlo
// crash found while building the annowatch shell command. game.go's
// playMove assumes, for every move a move generator hands it, that
//
//	move.TilesPlayed() + len(move.Leave()) == <mover's rack size>
//
// (it draws TilesPlayed() replacement tiles into a 7-capacity placeholder
// rack, then appends Leave() and slices to drew+len(Leave())).
//
// ROOT CAUSE (confirmed): this was an application bug in the new annowatch
// code (shell/annowatch.go's runOneAnnoAnalysis), not a bug in montecarlo,
// movegen, or wmp. cgp.ParseCGP (via game.NewFromSnapshot) only places
// letters on the board grid — it never computes cross-sets or anchors.
// Every other entry point into the analyzer builds its game via
// game.NewFromHistory + PlayToTurn, which incrementally maintains correct
// cross-sets move-by-move via crossSetGen.UpdateForMove, so this is never an
// issue there. Every *other* CGP-loading path in the codebase (the
// interactive `load cgp` shell command, and the pre-existing WMP
// equivalence tests below in this package) calls game.RecalculateBoard()
// immediately after ParseCGP for exactly this reason — runOneAnnoAnalysis
// was the one place that didn't. With stale/default cross-sets and anchors,
// the WMP shadow generator (movegen/wordmap_gen.go, movegen/shadow.go,
// wmp/movegen.go) can accept board windows that don't structurally match
// what its own anchor bookkeeping assumes, occasionally producing a Move
// whose TilesPlayed() doesn't match the number of tiles actually taken from
// the rack. This is confirmed by TestWMPBugReproducesWithoutRecalculateBoard
// below, and by live annowatch testing: crashed in ~half of several live
// attempts before the fix (shell/annowatch.go now calls g.RecalculateBoard()
// right after ParseCGP), zero crashes in repeated live attempts after it.
//
// The tests in this file that DO call RecalculateBoard (matching every
// established CGP-loading path) never reproduced the crash across several
// hundred thousand combined simulated iterations — consistent with the
// conclusion above, not contradicting it.
//
// simSingleIterationRecover (montecarlo.go) — the panic-guard added
// alongside this investigation — stays regardless of this root cause: an
// unrecovered panic inside an errgroup-spawned goroutine takes down the
// entire process (errgroup deliberately does not recover panics; see its
// doc comment), so any future bug of this shape in a background/unattended
// caller (annowatch, volunteer mode, batch analysis) would otherwise crash
// the whole process rather than surface as a normal error.
package montecarlo

import (
	"context"
	"testing"
	"time"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	macondogame "github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	wmppkg "github.com/domino14/macondo/wmp"
)

// A real position pulled from a live annotated Woogles game
// (TpRRRkHXRKFpkWnpEVSobr) that triggered the montecarlo panic during
// annowatch testing (before shell/annowatch.go called RecalculateBoard).
const wmpBugCGP = "15/15/15/15/10S4/10P4/10I4/6GOIER4/2BEGRIMS1A4/10T4/10E4/10D4/15/15/15 " +
	"NEATATN/T?MHEYC 94/95 0 lex CSW24; ld english;"

// buildWMPSimmerWithRack builds a fresh Simmer (Init'd, WMP wired via
// SetWMP), its candidate play list, and a synthetic pass move for the mover
// (mirroring AnalyzePosition's stand-in "played move") against wmpBugCGP,
// using whatever racks are baked into the CGP (no normalization — see the
// bestbot-matching comment below). The caller still needs to call
// SetThreads and PrepareSim before simulating. If recalculateBoard is
// false, it reproduces the original annowatch bug: cross-sets/anchors are
// left at their post-ParseCGP (uncomputed) state.
func buildWMPSimmerWithRack(t *testing.T, recalculateBoard bool) (*Simmer, []*move.Move, *move.Move) {
	t.Helper()
	requireWMPSetup(t)
	// Use the process-wide cache (same as production's TryLoadWMP ->
	// wmp.EnsureWMP), so the *same* WMP object instance is reused across
	// every call to buildWMPSimmerWithRack within this test binary —
	// exactly like a long-lived shell process reuses one cached WMP
	// across many annowatch polls.
	w, err := wmppkg.EnsureWMP(DefaultConfig.WGLConfig(), "CSW24")
	if err != nil {
		t.Fatal(err)
	}

	game, err := cgp.ParseCGP(DefaultConfig, wmpBugCGP)
	if err != nil {
		t.Fatal(err)
	}
	// Match cmd/lambda's bestbot CGP-loading sequence exactly (proven —
	// serves thousands of games): SetBackupMode + SetStateStackLength,
	// RecalculateBoard, and NO rack normalization — bestbot uses both CGP
	// racks exactly as given, with no SetRackFor/redraw step. An earlier
	// version of this harness added a SetRackFor "normalization" step
	// that bestbot does not do; removed to stay faithful to the proven
	// path.
	game.SetBackupMode(macondogame.InteractiveGameplayMode)
	game.SetStateStackLength(1)
	if recalculateBoard {
		game.RecalculateBoard()
	}
	playerIdx := game.PlayerOnTurn()

	calc, err := equity.NewCombinedStaticCalculator(
		"CSW24", DefaultConfig, "", equity.PEGAdjustmentFilename)
	if err != nil {
		t.Fatal(err)
	}

	// Replicate bot.BotTurnPlayer.GenerateMoves(40) exactly (can't import
	// ai/bot here: it imports montecarlo, which would cycle). Production's
	// initial 40 candidates are the top 40 BY EQUITY (leave-adjusted via
	// the HASTY_BOT 4-calculator stack: exhaustive leave + opening +
	// pre-endgame + endgame adjustments), not top-40-by-raw-score.
	leaveCalc, err := equity.NewExhaustiveLeaveCalculator(game.LexiconName(), DefaultConfig, "")
	if err != nil {
		t.Fatal(err)
	}
	pegCalc, err := equity.NewPreEndgameAdjustmentCalculator(DefaultConfig, game.LexiconName(), "")
	if err != nil {
		t.Fatal(err)
	}
	botCalcs := []equity.EquityCalculator{
		leaveCalc, &equity.OpeningAdjustmentCalculator{}, pegCalc, &equity.EndgameAdjustmentCalculator{},
	}

	aiPlayer, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(game.Game, DefaultConfig, botCalcs)
	if err != nil {
		t.Fatal(err)
	}
	curRack := aiPlayer.RackFor(aiPlayer.PlayerOnTurn())
	oppRack := aiPlayer.RackFor(aiPlayer.NextPlayer())
	candGen := aiPlayer.MoveGenerator()
	unseen := int(oppRack.NumTiles()) + aiPlayer.Bag().TilesRemaining()
	candGen.SetMaxCanExchange(macondogame.MaxCanExchange(unseen-7, aiPlayer.ExchangeLimit()))
	candGen.GenAll(curRack, unseen-7 >= aiPlayer.ExchangeLimit())
	allPlays := candGen.(*movegen.GordonGenerator).Plays()
	if len(allPlays) == 0 {
		t.Fatal("no plays generated")
	}
	aiPlayer.AssignEquity(allPlays, aiPlayer.Board(), aiPlayer.Bag(), oppRack)
	plays := aiPlayer.TopPlays(allPlays, 40)

	pass := move.NewPassMove(game.RackFor(playerIdx).TilesOn(), game.Alphabet())

	s := &Simmer{}
	s.Init(game.Game, []equity.EquityCalculator{calc}, calc, DefaultConfig)
	s.SetWMP(w)
	return s, plays, pass
}

// TestWMPBugReproducesWithoutRecalculateBoard is the definitive repro: it
// deliberately skips RecalculateBoard after ParseCGP (the exact mistake
// runOneAnnoAnalysis originally made) and drives the real Simulate() path
// until the panic-guard reports an error. This is expected to fail fast.
func TestWMPBugReproducesWithoutRecalculateBoard(t *testing.T) {
	const rounds = 100
	totalIters := 0
	for i := 0; i < rounds; i++ {
		s, plays, pass := buildWMPSimmerWithRack(t, false)
		s.SetThreads(16)
		if err := s.PrepareSim(5, plays); err != nil {
			t.Fatalf("round %d: PrepareSim: %v", i, err)
		}
		s.AvoidPruningMoves([]*move.Move{pass})
		s.SetStoppingCondition(Stop99)

		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := s.Simulate(ctx)
		cancel()
		iters := s.Iterations()
		totalIters += iters
		t.Logf("round %d: %d iterations in %v, err=%v", i, iters, time.Since(start), err)
		if err != nil {
			t.Logf("round %d: reproduced — Simulate errored (panic-guard caught it) after %d iterations: %v",
				i, iters, err)
			return
		}
	}
	t.Errorf("expected the missing-RecalculateBoard bug to reproduce within %d rounds (%d total iterations); "+
		"it didn't — either it's been fixed at a lower level, or this repro is no longer reliable", rounds, totalIters)
}

// TestWMPNoReproWithRecalculateBoard is the control: identical setup, but
// with RecalculateBoard called (the fix now applied in
// shell/annowatch.go's runOneAnnoAnalysis). Should never panic/error.
func TestWMPNoReproWithRecalculateBoard(t *testing.T) {
	const rounds = 20
	for i := 0; i < rounds; i++ {
		s, plays, pass := buildWMPSimmerWithRack(t, true)
		s.SetThreads(16)
		if err := s.PrepareSim(5, plays); err != nil {
			t.Fatalf("round %d: PrepareSim: %v", i, err)
		}
		s.AvoidPruningMoves([]*move.Move{pass})
		s.SetStoppingCondition(Stop99)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := s.Simulate(ctx)
		cancel()
		if err != nil {
			t.Fatalf("round %d: Simulate panicked/errored after %d global iterations "+
				"even with RecalculateBoard called: %v", i, s.Iterations(), err)
		}
	}
	t.Logf("completed %d rounds with RecalculateBoard called, no panic (as expected)", rounds)
}
