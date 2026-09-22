package main

import (
	"math"
	"strings"
	"testing"

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
