package game

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func seededRackGame(t *testing.T) *Game {
	t.Helper()
	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := NewBasicGameRules(
		DefaultConfig, "CSW19", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGame(rules, players)
	if err != nil {
		t.Fatal(err)
	}
	g.StartGame()
	return g
}

func rngOf(seed byte) *frand.RNG {
	var s [32]byte
	s[0] = seed
	return frand.NewCustom(s[:], 0, 0)
}

// Two copies of a position whose bags are in different orders, and whose
// opponents hold different racks, must be dealt the same rack from equal
// sources, and must go on to draw the same tiles: that is what a sim's
// common random numbers rest on.
func TestSetRandomRackSeeded(t *testing.T) {
	// A bag hands tiles out at random, or (a sim's) from the end of an
	// order that is shuffled whenever tiles go back.
	for _, fixedOrder := range []bool{false, true} {
		testSetRandomRackSeeded(t, fixedOrder)
	}
}

func testSetRandomRackSeeded(t *testing.T, fixedOrder bool) {
	a := seededRackGame(t)
	a.Bag().SetFixedOrder(fixedOrder)
	b := a.Copy()
	// Different histories: b's opponent has been dealt other racks and its
	// bag shuffled.
	for i := 0; i < 5; i++ {
		if _, err := b.SetRandomRack(1, nil); err != nil {
			t.Fatal(err)
		}
		b.Bag().Shuffle()
	}
	if a.RackLettersFor(1) == b.RackLettersFor(1) {
		t.Fatal("the copies should have started with different racks")
	}
	for seed := byte(1); seed <= 50; seed++ {
		// With nothing known, and with the first one to three tiles of
		// a's opponent's rack known: tiles that are on the opponent's rack
		// in one copy and mostly in the bag in the other.
		for numKnown := 0; numKnown <= 3; numKnown++ {
			known := numKnown
			knownRack := append([]tilemapping.MachineLetter(nil), a.RackFor(1).TilesOn()[:numKnown]...)
			errA := a.SetRandomRackSeeded(1, knownRack, rngOf(seed))
			errB := b.SetRandomRackSeeded(1, knownRack, rngOf(seed))
			if (errA == nil) != (errB == nil) {
				t.Fatalf("seed %d known %d: errors differ: %v, %v", seed, known, errA, errB)
			}
			if errA != nil {
				continue
			}
			if a.RackLettersFor(1) != b.RackLettersFor(1) {
				t.Fatalf("seed %d known %d: racks %s and %s", seed, known, a.RackLettersFor(1), b.RackLettersFor(1))
			}
			if a.Bag().TilesRemaining() != b.Bag().TilesRemaining() {
				t.Fatalf("seed %d: bags of %d and %d tiles", seed, a.Bag().TilesRemaining(), b.Bag().TilesRemaining())
			}
			// The draws that follow agree too. Put them back afterwards.
			da := make([]tilemapping.MachineLetter, 7)
			db := make([]tilemapping.MachineLetter, 7)
			if err := a.Bag().Draw(7, da); err != nil {
				t.Fatal(err)
			}
			if err := b.Bag().Draw(7, db); err != nil {
				t.Fatal(err)
			}
			for i := range da {
				if da[i] != db[i] {
					t.Fatalf("seed %d known %d: later draws differ: %v, %v", seed, known, da, db)
				}
			}
			a.Bag().PutBack(da)
			b.Bag().PutBack(db)
			// Leave the copies out of step again for the next round.
			if _, err := b.SetRandomRack(1, nil); err != nil {
				t.Fatal(err)
			}
			b.Bag().Shuffle()
		}
	}
	// A different source deals a different rack (some seed pair must).
	differs := false
	for seed := byte(1); seed <= 5 && !differs; seed++ {
		a.SetRandomRackSeeded(1, nil, rngOf(seed))
		first := a.RackLettersFor(1)
		a.SetRandomRackSeeded(1, nil, rngOf(seed+100))
		differs = first != a.RackLettersFor(1)
	}
	if !differs {
		t.Fatal("different sources always dealt the same rack")
	}
	// Tiles that are not there are an error, and leave the rack as it was.
	before := a.RackLettersFor(1)
	q := tilemapping.MachineLetter(17)
	if err := a.SetRandomRackSeeded(1, []tilemapping.MachineLetter{q, q, q}, rngOf(1)); err == nil {
		t.Fatal("three Qs were dealt")
	}
	if a.RackLettersFor(1) != before {
		t.Fatalf("the rack changed on an error: %s to %s", before, a.RackLettersFor(1))
	}
}

// Over a cycle's rotations of one source, every unseen tile starts on the
// player's rack at least once and at most twice, and rotation 0 is the
// plain seeded deal.
func TestSetRandomRackCycled(t *testing.T) {
	g := seededRackGame(t)
	g.Bag().SetFixedOrder(true)
	plain := g.Copy()
	if err := plain.SetRandomRackSeeded(1, nil, rngOf(3)); err != nil {
		t.Fatal(err)
	}
	rotations := g.BagCycleRotations(1)
	unseen := g.Bag().TilesRemaining() + int(g.RackFor(1).NumTiles())
	if rotations != (unseen+6)/7 {
		t.Fatalf("%d rotations for %d unseen tiles", rotations, unseen)
	}
	var want, got [64]int
	g.SetRandomRackCycled(1, rngOf(3), 0)
	if g.RackFor(1).String() != plain.RackFor(1).String() {
		t.Fatalf("rotation 0 dealt %s, the seeded deal %s", g.RackFor(1).String(), plain.RackFor(1).String())
	}
	buf := make([]tilemapping.MachineLetter, 7)
	n := g.RackFor(1).NoAllocTilesOn(buf)
	for _, ml := range buf[:n] {
		want[ml]++
	}
	for _, ml := range g.Bag().Tiles() {
		want[ml]++
	}
	for r := 0; r < rotations; r++ {
		if err := g.SetRandomRackCycled(1, rngOf(3), r); err != nil {
			t.Fatal(err)
		}
		n := g.RackFor(1).NoAllocTilesOn(buf)
		if n != 7 {
			t.Fatalf("rotation %d dealt %d tiles", r, n)
		}
		for _, ml := range buf[:n] {
			got[ml]++
		}
	}
	for ml := range want {
		if got[ml] < want[ml] || got[ml] > 2*want[ml] {
			t.Fatalf("letter %d: %d unseen, on the rack %d times over the cycle", ml, want[ml], got[ml])
		}
	}
	total := 0
	for _, c := range got {
		total += c
	}
	if total != 7*rotations {
		t.Fatalf("%d tiles dealt over %d rotations", total, rotations)
	}
}
