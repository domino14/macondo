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
