package main

import (
	"math"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/triton"
)

// One complete logged game (autoplay-softmax-v-hasty-5.txt), 24 turns.
var sampleGameTurns = []string{
	"p1,685417a6010f675458000001,1,ADEEIOT, 8C IODATE,16,16,6,E,15.884,86,0",
	"p2,685417a6010f675458000001,2,EIIMOSU, E6 OI.IUM,18,18,5,ES,29.190,80,16",
	"p1,685417a6010f675458000001,3,?EEIINO,D11 OI,8,24,2,?EEIN,37.981,75,18",
	"p2,685417a6010f675458000001,4,DELPRSZ, C7 Z.P,27,45,2,DELRS,43.085,73,24",
	"p1,685417a6010f675458000001,5,?AEEINW, 7G INWEAvE,70,94,7,,70.000,71,45",
	"p2,685417a6010f675458000001,6,ADEELRS, N1 LASERED,81,126,7,,81.000,64,94",
	"p1,685417a6010f675458000001,7,CGHLLNS, 6D L.CH,21,115,3,GLNS,25.490,57,126",
	"p2,685417a6010f675458000001,8,ACDENOT, 3G TACNODE.,80,206,7,,80.000,54,115",
	"p1,685417a6010f675458000001,9,GLNNOSW, H1 GN.WN,39,154,4,LOS,46.722,47,206",
	"p2,685417a6010f675458000001,10,DEKMNTY, 1L YE.K,45,251,3,DMNT,37.324,43,154",
	"p1,685417a6010f675458000001,11,ILOOQRS,C12 QI,33,187,2,LOORS,36.510,40,251",
	"p2,685417a6010f675458000001,12,ADEMNST, K6 M.NDATES,72,323,7,,72.000,38,187",
	"p1,685417a6010f675458000001,13,GLOORRS,L10 GOR,17,204,3,LORS,25.232,31,323",
	"p2,685417a6010f675458000001,14,AEEEIPU,M10 EPEE,24,347,4,AIU,12.015,28,204",
	"p1,685417a6010f675458000001,15,LORSSTV,10H LOV...S,21,225,4,RST,31.935,24,347",
	"p2,685417a6010f675458000001,16,AINRTTU, 4J URAT.,22,369,4,INT,24.359,20,225",
	"p1,685417a6010f675458000001,17,BBGLRST, 6M B.G,10,235,2,BLRST,11.210,16,369",
	"p2,685417a6010f675458000001,18,AFIIJNT, 2F JI.,28,397,2,AFINT,30.821,14,235",
	"p1,685417a6010f675458000001,19,BELRSTV, I9 V.LT,11,246,3,BERS,26.111,12,397",
	"p2,685417a6010f675458000001,20,AFINRTU,N13 FUR,17,414,3,AINT,22.921,9,246",
	"p1,685417a6010f675458000001,21,?BEORSX,15J BOXE.S,69,315,5,?R,98.778,6,414",
	"p2,685417a6010f675458000001,22,AFHINRT,F10 HAFT,36,450,4,INR,36.944,1,315",
	"p1,685417a6010f675458000001,23,?AIORUY,H12 AIRY,27,342,4,?OU,13.000,0,450",
	"p2,685417a6010f675458000001,24,AINR,15D RAIN.,9,463,4,,13.000,0,342",
}

func feedGame(t *testing.T, ga *GameAssembler) []outputVector {
	t.Helper()
	hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
	sc := NewTurnScanner(strings.NewReader(hdr + strings.Join(sampleGameTurns, "\n") + "\n"))
	var out []outputVector
	for sc.Scan() {
		out = append(out, ga.FeedTurn(sc.Turn())...)
	}
	if sc.Err() != nil {
		t.Fatal(sc.Err())
	}
	return out
}

// stubScorer stands in for Triton: a fixed value and spread per leaf.
type stubScorer struct{ value, spread float32 }

func (s stubScorer) Infer(planes, scalars []float32, n int) (*triton.ModelOutputs, error) {
	out := &triton.ModelOutputs{Value: make([]float32, n), Spread: make([]float32, n)}
	for i := range n {
		out.Value[i] = s.value
		out.Spread[i] = s.spread
	}
	return out, nil
}

// Rollouts mutate the replay game (random racks, plies played, state
// reset), so every feature vector must come out identical to the plain
// table-mode replay, and every label must be a sane average.
func TestRolloutLabelsLeaveReplayIntact(t *testing.T) {
	table := NewGameAssembler(NPlies, nil, 0, 0, 0)
	want := feedGame(t, table)

	shared := &RolloutShared{
		Client: stubScorer{value: 0.3, spread: 0.1},
		Calcs:  table.eqCalcsForTest(),
	}
	rollout := NewGameAssembler(NPlies, shared, 2, 8, 1.0)
	got := feedGame(t, rollout)

	if len(want) != 23 || len(got) != 23 {
		t.Fatalf("expected 23 vectors from each mode, got table=%d rollout=%d", len(want), len(got))
	}
	for i := range want {
		w, g := *want[i].features, *got[i].features
		if len(w) != len(g) {
			t.Fatalf("vector %d: lengths differ", i)
		}
		for j := range w {
			if w[j] != g[j] {
				t.Fatalf("vector %d: feature %d differs: table %v rollout %v", i, j, w[j], g[j])
			}
		}
		for _, k := range []int{TargetWDL, TargetOppBingo, TargetOppScore} {
			if want[i].predictions[k] != got[i].predictions[k] {
				t.Fatalf("vector %d: target %d differs: table %v rollout %v", i, k, want[i].predictions[k], got[i].predictions[k])
			}
		}
		if got[i].label == nil {
			t.Fatalf("vector %d: no rollout label", i)
		}
		v, s := got[i].label.value, got[i].label.spread
		if v < -1 || v > 1 || math.IsNaN(float64(v)) || math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			t.Fatalf("vector %d: bad label value=%v spread=%v", i, v, s)
		}
		if got[i].predictions[TargetValue] != v {
			t.Fatalf("vector %d: value target %v != label %v", i, got[i].predictions[TargetValue], v)
		}
	}
	if rollout.labeled != 23 {
		t.Fatalf("expected 23 labeled positions, got %d", rollout.labeled)
	}
}

// ── what the rollouts actually do ──────────────────────────────────────────
//
// The tests below use a scorer that records every leaf it is asked to score,
// and check the leaves against what a correct rollout must produce.

const (
	planeSize  = 15 * 15
	blankPlane = 26
	histPlane0 = 83 // the leaf move's tiles
	histPlane1 = 84 // the previous (opponent's) move's tiles
	scalRack   = 0  // 27 rack counts / 7
	scalUnseen = 27 // 27 draw probabilities
	scalBag    = 70 // tiles unseen / 100
	totalTiles = 100
	fullRack   = 7
	safeUnseen = 21 // with this many unseen at the position, no rollout can run the bag out in 2 plies
)

type leafVec struct {
	planes  []float32
	scalars []float32
}

// recordingScorer returns a fixed value/spread and keeps every batch.
type recordingScorer struct {
	value, spread float32
	calls         [][]leafVec
}

func (s *recordingScorer) Infer(planes, scalars []float32, n int) (*triton.ModelOutputs, error) {
	batch := make([]leafVec, n)
	for i := range n {
		batch[i] = leafVec{
			planes:  append([]float32(nil), planes[i*NPlanes:(i+1)*NPlanes]...),
			scalars: append([]float32(nil), scalars[i*NScal:(i+1)*NScal]...),
		}
	}
	s.calls = append(s.calls, batch)
	out := &triton.ModelOutputs{Value: make([]float32, n), Spread: make([]float32, n)}
	for i := range n {
		out.Value[i] = s.value
		out.Spread[i] = s.spread
	}
	return out, nil
}

const (
	NPlanes = 85 * planeSize
	NScal   = 72
)

func sumPlane(planes []float32, p int) int {
	s := float32(0)
	for _, v := range planes[p*planeSize : (p+1)*planeSize] {
		s += v
	}
	return int(math.Round(float64(s)))
}

func tilesOnBoard(planes []float32) int {
	n := 0
	for p := 0; p < blankPlane; p++ {
		n += sumPlane(planes, p)
	}
	return n
}

func rackTiles(scalars []float32) int {
	s := float32(0)
	for _, v := range scalars[scalRack : scalRack+27] {
		s += v
	}
	return int(math.Round(float64(s * fullRack)))
}

func unseenTiles(scalars []float32) int {
	return int(math.Round(float64(scalars[scalBag] * totalTiles)))
}

func unseenProbSum(scalars []float32) float32 {
	s := float32(0)
	for _, v := range scalars[scalUnseen : scalUnseen+27] {
		s += v
	}
	return s
}

func runRollouts(t *testing.T, plies, rollouts int, scorer *recordingScorer) ([]outputVector, []outputVector) {
	t.Helper()
	table := NewGameAssembler(NPlies, nil, 0, 0, 0)
	want := feedGame(t, table)
	shared := &RolloutShared{Client: scorer, Calcs: table.eqCalcsForTest()}
	got := feedGame(t, NewGameAssembler(NPlies, shared, plies, rollouts, 1.0))
	if len(want) != 23 || len(got) != 23 || len(scorer.calls) != 23 {
		t.Fatalf("want 23 positions and 23 scorer calls, got %d %d %d", len(want), len(got), len(scorer.calls))
	}
	return want, got
}

// Every leaf must be a legal training position two plies past the position
// it belongs to: the leaf mover holds only their leave, everything else is
// unseen, and the two history planes are exactly the two rollout moves.
func TestRolloutLeavesAreTwoPliesOn(t *testing.T) {
	scorer := &recordingScorer{value: 0.4}
	want, _ := runRollouts(t, 2, 8, scorer)

	for i, batch := range scorer.calls {
		pos := *want[i].features
		posBoard := tilesOnBoard(pos)
		posUnseen := unseenTiles(pos[NPlanes:])
		if posUnseen >= safeUnseen && len(batch) != 8 {
			t.Fatalf("position %d: %d leaves, want 8 (no rollout can end the game with %d unseen)", i, len(batch), posUnseen)
		}
		if len(batch) > 8 {
			t.Fatalf("position %d: %d leaves > 8 rollouts", i, len(batch))
		}
		for j, leaf := range batch {
			board, rack, unseen := tilesOnBoard(leaf.planes), rackTiles(leaf.scalars), unseenTiles(leaf.scalars)
			if board+rack+unseen != totalTiles {
				t.Fatalf("position %d leaf %d: board %d + rack %d + unseen %d != %d", i, j, board, rack, unseen, totalTiles)
			}
			if unseen > 0 && math.Abs(float64(unseenProbSum(leaf.scalars))-1) > 1e-4 {
				t.Fatalf("position %d leaf %d: unseen probabilities sum to %v", i, j, unseenProbSum(leaf.scalars))
			}
			if rack > fullRack {
				t.Fatalf("position %d leaf %d: rack of %d", i, j, rack)
			}
			h0, h1 := sumPlane(leaf.planes, histPlane0), sumPlane(leaf.planes, histPlane1)
			if h0 > fullRack || h1 > fullRack {
				t.Fatalf("position %d leaf %d: history planes %d, %d", i, j, h0, h1)
			}
			// Two different moves cannot put tiles on the same squares.
			for sq := 0; sq < planeSize; sq++ {
				if leaf.planes[histPlane0*planeSize+sq] > 0.5 && leaf.planes[histPlane1*planeSize+sq] > 0.5 {
					t.Fatalf("position %d leaf %d: both history planes mark square %d: the two rollout moves are the same move", i, j, sq)
				}
			}
			// Two plies were played: the tiles added to the board are exactly
			// the two rollout moves' tiles (exchanges and passes add none).
			if board-posBoard != h0+h1 {
				t.Fatalf("position %d leaf %d: board grew by %d but the two rollout moves show %d + %d tiles",
					i, j, board-posBoard, h0, h1)
			}
			// The leaf mover started the ply with a full rack when the bag
			// allowed it, so leave + tiles played == 7 for a tile play.
			if posUnseen >= safeUnseen && h0 > 0 && rack+h0 != fullRack {
				t.Fatalf("position %d leaf %d: leave %d + played %d != %d", i, j, rack, h0, fullRack)
			}
		}
	}
}

// With a constant leaf value v: two plies land back on the mover, so the
// label is +v; one ply lands on the opponent, so it is -v. With one ply the
// only score change is the opponent's, so the spread label cannot be positive.
func TestRolloutSignByParity(t *testing.T) {
	const v = 0.4
	two := &recordingScorer{value: v}
	want, got2 := runRollouts(t, 2, 8, two)
	one := &recordingScorer{value: v}
	_, got1 := runRollouts(t, 1, 8, one)

	for i := range want {
		posUnseen := unseenTiles((*want[i].features)[NPlanes:])
		if posUnseen < safeUnseen {
			continue // rollouts may end the game and mix in real results
		}
		if d := math.Abs(float64(got2[i].label.value - v)); d > 1e-5 {
			t.Fatalf("position %d, 2 plies: label %v, want +%v", i, got2[i].label.value, v)
		}
		if d := math.Abs(float64(got1[i].label.value + v)); d > 1e-5 {
			t.Fatalf("position %d, 1 ply: label %v, want -%v", i, got1[i].label.value, v)
		}
		if got1[i].label.spread > 0 {
			t.Fatalf("position %d, 1 ply: spread label %v > 0 after only the opponent moved", i, got1[i].label.spread)
		}
		// One ply: the leaf move is the opponent's reply, and the "previous
		// move" plane is the mover's own move from the labeled position.
		posMove := sumPlane(*want[i].features, histPlane0)
		posBoard := tilesOnBoard(*want[i].features)
		for j, leaf := range one.calls[i] {
			h0, h1 := sumPlane(leaf.planes, histPlane0), sumPlane(leaf.planes, histPlane1)
			if h1 != posMove {
				t.Fatalf("position %d leaf %d, 1 ply: previous-move plane has %d tiles, the labeled move had %d", i, j, h1, posMove)
			}
			if tilesOnBoard(leaf.planes)-posBoard != h0 {
				t.Fatalf("position %d leaf %d, 1 ply: board grew by %d, reply shows %d", i, j, tilesOnBoard(leaf.planes)-posBoard, h0)
			}
		}
	}
}

// An exchange is the awkward case: the mover draws before the exchanged
// tiles return to the bag. The rollouts must still produce legal leaves.
func TestRolloutAfterExchange(t *testing.T) {
	scorer := &recordingScorer{value: 0.2}
	table := NewGameAssembler(NPlies, nil, 0, 0, 0)
	shared := &RolloutShared{Client: scorer, Calcs: table.eqCalcsForTest()}
	ga := NewGameAssembler(NPlies, shared, 2, 8, 1.0)

	hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
	turn := "p1,exchgame,1,AAEIOUV,(exch AAEIOU),0,0,6,V,-20.000,86,0\n"
	sc := NewTurnScanner(strings.NewReader(hdr + turn))
	for sc.Scan() {
		ga.FeedTurn(sc.Turn())
	}
	gw := ga.games["exchgame"]
	if gw == nil || len(gw.plies) != 1 || gw.plies[0].label == nil {
		t.Fatal("exchange position was not labeled")
	}
	if len(scorer.calls) != 1 || len(scorer.calls[0]) != 8 {
		t.Fatalf("want one call with 8 leaves, got %d calls", len(scorer.calls))
	}
	for j, leaf := range scorer.calls[0] {
		board, rack, unseen := tilesOnBoard(leaf.planes), rackTiles(leaf.scalars), unseenTiles(leaf.scalars)
		if board+rack+unseen != totalTiles {
			t.Fatalf("leaf %d: board %d + rack %d + unseen %d != %d", j, board, rack, unseen, totalTiles)
		}
		if board != sumPlane(leaf.planes, histPlane0)+sumPlane(leaf.planes, histPlane1) {
			t.Fatalf("leaf %d: board has %d tiles, the two rollout moves show %d",
				j, board, sumPlane(leaf.planes, histPlane0)+sumPlane(leaf.planes, histPlane1))
		}
	}
	// The producer's own state must be intact for the next turn: the mover
	// still holds only the leave, and the bag has everything else.
	g := gw.game.Game
	if got := g.RackFor(0).NumTiles(); got != 1 {
		t.Fatalf("mover's rack after labeling has %d tiles, want the 1-tile leave", got)
	}
	if got := g.Bag().TilesRemaining(); got != totalTiles-1 {
		t.Fatalf("bag has %d tiles after labeling, want %d", got, totalTiles-1)
	}
}

// -labeler result -per-game: one position per game, its value target is
// the mover's real result and its spread target the spread change to the
// end of the game; the features are the table-mode features for that turn.
func TestResultLabelerOnePositionPerGame(t *testing.T) {
	table := NewGameAssembler(NPlies, nil, 0, 0, 0)
	want := feedGame(t, table)
	byTurn := map[int]outputVector{}
	for _, v := range want {
		byTurn[v.turn] = v
	}
	// p2 won the sample game 463-342 (459 + 4 for p1's unplayed ?OU), so
	// the final spread is +121 for p2 and -121 for p1.
	const finalP2 = 121.0

	seen := 0
	for trial := 0; trial < 40; trial++ {
		ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
		ga.valueFromResult = true
		ga.pickMax = 30
		got := feedGame(t, ga)
		if len(got) > 1 {
			t.Fatalf("trial %d: %d vectors from one game, want at most 1", trial, len(got))
		}
		if len(got) == 0 {
			continue // the draw was past the end of the game
		}
		seen++
		v := got[0]
		ref, ok := byTurn[v.turn]
		if !ok {
			t.Fatalf("trial %d: emitted turn %d, which table mode never emits", trial, v.turn)
		}
		for j := range *ref.features {
			if (*ref.features)[j] != (*v.features)[j] {
				t.Fatalf("trial %d turn %d: feature %d differs from table mode", trial, v.turn, j)
			}
		}
		final := float32(finalP2)
		if v.mover == 0 {
			final = -finalP2
		}
		wdl := float32(1)
		if final < 0 {
			wdl = -1
		}
		if v.predictions[TargetValue] != wdl || v.predictions[TargetWDL] != wdl {
			t.Fatalf("trial %d turn %d: value %v wdl %v, want %v", trial, v.turn, v.predictions[TargetValue], v.predictions[TargetWDL], wdl)
		}
		wantSpread := game.NormalizeSpreadForML(final - v.spreadNow)
		if v.predictions[TargetSpread] != wantSpread {
			t.Fatalf("trial %d turn %d: spread %v, want %v (final %v, now %v)", trial, v.turn, v.predictions[TargetSpread], wantSpread, final, v.spreadNow)
		}
		for _, k := range []int{TargetOppBingo, TargetOppScore} {
			if v.predictions[k] != ref.predictions[k] {
				t.Fatalf("trial %d turn %d: target %d differs from table mode", trial, v.turn, k)
			}
		}
	}
	if seen == 0 {
		t.Fatal("no trial emitted a position")
	}
}

// -endgame-plies: a position whose bag was already empty is labeled by a
// quick search from the opponent's reply. In the sample game p1's turn 23
// (AIRY, down 108) leaves p2 to play out RAIN for 9 plus 4 for p1's ?OU,
// so the search must find p2 winning by 13: value -1, spread -13 for p1.
// Turn 22 empties the bag with the move itself, so the mover's new tiles
// are not known and the position keeps its logged-result label. The
// replay must survive the search: features identical to table mode.
func TestEndgameSearchLabels(t *testing.T) {
	table := NewGameAssembler(NPlies, nil, 0, 0, 0)
	want := feedGame(t, table)
	byTurn := map[int]outputVector{}
	for _, v := range want {
		byTurn[v.turn] = v
	}
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	if err != nil {
		t.Fatal(err)
	}
	negamax.GlobalTranspositionTable.Reset(0.01, 15)

	for _, tc := range []struct {
		turn   int
		solved bool
		spread float32 // expected mover's spread change
	}{{23, true, -13}, {22, false, 121 - 135}} {
		ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
		ga.valueFromResult = true
		ga.fixedPick = tc.turn
		ga.endgamePlies = 2
		ga.kwg = gd
		got := feedGame(t, ga)
		if len(got) != 1 || got[0].turn != tc.turn {
			t.Fatalf("turn %d: got %d vectors", tc.turn, len(got))
		}
		v := got[0]
		ref := byTurn[tc.turn]
		for j := range *ref.features {
			if (*ref.features)[j] != (*v.features)[j] {
				t.Fatalf("turn %d: feature %d differs from table mode", tc.turn, j)
			}
		}
		if (v.solved != nil) != tc.solved || int(ga.solved) != map[bool]int{true: 1, false: 0}[tc.solved] {
			t.Fatalf("turn %d: solved=%v (%d solved), want %v", tc.turn, v.solved != nil, ga.solved, tc.solved)
		}
		wantValue := float32(-1) // p1 lost the game either way
		if tc.turn == 22 {
			wantValue = 1 // p2's position: p2 won
		}
		if v.predictions[TargetValue] != wantValue || v.predictions[TargetWDL] != wantValue {
			t.Fatalf("turn %d: value %v wdl %v, want %v", tc.turn, v.predictions[TargetValue], v.predictions[TargetWDL], wantValue)
		}
		if tc.solved && v.solved.spread != tc.spread {
			t.Fatalf("turn %d: solved spread %v, want %v", tc.turn, v.solved.spread, tc.spread)
		}
		if v.predictions[TargetSpread] != game.NormalizeSpreadForML(tc.spread) {
			t.Fatalf("turn %d: spread target %v, want %v", tc.turn, v.predictions[TargetSpread], game.NormalizeSpreadForML(tc.spread))
		}
	}
}
