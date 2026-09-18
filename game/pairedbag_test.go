package game

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

func pairedGame(t *testing.T, seed [32]byte) *Game {
	t.Helper()
	// These tests only exercise the bag, so there is no need for a lexicon.
	rules, err := NewBasicGameRules(
		DefaultConfig, "", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGame(rules, []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "p1"},
		{Nickname: "p2", RealName: "p2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	g.SeedBag(seed)
	g.SetPairedBagMode(true)
	g.StartGame()
	return g
}

func seedFrom(b byte) [32]byte {
	var seed [32]byte
	seed[0] = b
	return seed
}

// drawStream draws n tiles at a time for a seat, count times, and returns the
// tiles in the order they came out of the bag.
func drawStream(g *Game, playerIdx, n, count int) []tilemapping.MachineLetter {
	stream := []tilemapping.MachineLetter{}
	buf := make([]tilemapping.MachineLetter, n)
	for i := 0; i < count; i++ {
		drew := g.drawAtMostFor(playerIdx, n, buf)
		stream = append(stream, buf[:drew]...)
	}
	return stream
}

// TestPairedBagSameSeedSameDraws is the property the whole mode exists for: two
// games built from one seed hand each seat exactly the same tiles.
func TestPairedBagSameSeedSameDraws(t *testing.T) {
	is := is.New(t)
	seed := seedFrom(42)

	g1 := pairedGame(t, seed)
	g2 := pairedGame(t, seed)

	is.Equal(g1.RackLettersFor(0), g2.RackLettersFor(0))
	is.Equal(g1.RackLettersFor(1), g2.RackLettersFor(1))
	// The two seats draw from opposite ends, so they should not have been
	// handed the same tiles as each other.
	is.True(g1.RackLettersFor(0) != g1.RackLettersFor(1))

	for seat := 0; seat < 2; seat++ {
		is.Equal(drawStream(g1, seat, 7, 5), drawStream(g2, seat, 7, 5))
	}
}

// TestPairedBagDifferentSeedsDifferentDraws guards against the mode
// accidentally producing one fixed game regardless of seed.
func TestPairedBagDifferentSeedsDifferentDraws(t *testing.T) {
	is := is.New(t)
	g1 := pairedGame(t, seedFrom(1))
	g2 := pairedGame(t, seedFrom(2))
	is.True(g1.RackLettersFor(0) != g2.RackLettersFor(0))
}

// TestPairedBagSeatsAreIndependent is why the players draw from opposite ends
// rather than the same one: one seat drawing a different number of tiles must
// not disturb what the other seat gets. Draw from the same end and this fails,
// because every draw shifts the tiles behind it.
func TestPairedBagSeatsAreIndependent(t *testing.T) {
	is := is.New(t)
	seed := seedFrom(7)

	// The back-side seat plays tiles at a wildly different rate in each game.
	g1 := pairedGame(t, seed)
	drawStream(g1, 0, 7, 3)
	front1 := drawStream(g1, 1, 7, 4)

	g2 := pairedGame(t, seed)
	drawStream(g2, 0, 1, 2)
	front2 := drawStream(g2, 1, 7, 4)

	is.Equal(front1, front2)
}

// TestPairedBagExchangeKeepsTilesAndOrder checks that an exchange returns the
// right number of tiles, keeps the bag's contents intact, and leaves the fixed
// order almost entirely alone.
func TestPairedBagExchangeKeepsTilesAndOrder(t *testing.T) {
	is := is.New(t)
	g := pairedGame(t, seedFrom(9))

	before := g.bag.Peek()
	remaining := g.bag.TilesRemaining()

	drawn := make([]tilemapping.MachineLetter, 3)
	returned := g.RackFor(0).TilesOn()[:3]
	is.NoErr(g.exchangeFor(0, returned, drawn))

	is.Equal(g.bag.TilesRemaining(), remaining)
	// The bag should have swapped exactly the drawn tiles for the returned ones.
	expected := countTiles(append(append([]tilemapping.MachineLetter{}, before...), returned...))
	for _, ml := range drawn {
		expected[ml]--
		if expected[ml] == 0 {
			delete(expected, ml)
		}
	}
	is.Equal(countTiles(after(g)), expected)

	// Exchanging three tiles moves at most six tiles around (each returned tile
	// displaces one other). Everything else has to stay put, or the paired game
	// stops matching its partner.
	is.True(orderMatchLength(before, after(g)) >= remaining-6)
}

// TestPairedBagExchangeIsReproducible makes sure the random insert position
// comes from the game's seeded RNG and not the global one.
func TestPairedBagExchangeIsReproducible(t *testing.T) {
	is := is.New(t)
	seed := seedFrom(11)

	run := func() []tilemapping.MachineLetter {
		g := pairedGame(t, seed)
		drawn := make([]tilemapping.MachineLetter, 3)
		if err := g.exchangeFor(0, g.RackFor(0).TilesOn()[:3], drawn); err != nil {
			t.Fatal(err)
		}
		return append(drawn, drawStream(g, 0, 7, 3)...)
	}
	is.Equal(run(), run())
}

func after(g *Game) []tilemapping.MachineLetter {
	return g.bag.Peek()
}

func countTiles(tiles []tilemapping.MachineLetter) map[tilemapping.MachineLetter]int {
	counts := map[tilemapping.MachineLetter]int{}
	for _, ml := range tiles {
		counts[ml]++
	}
	return counts
}

// orderMatchLength counts how many tiles sit in the same place in both bags,
// lining them up from the back (the side that does not move when the bag grows
// at the front).
func orderMatchLength(a, b []tilemapping.MachineLetter) int {
	matches := 0
	for i := 1; i <= len(a) && i <= len(b); i++ {
		if a[len(a)-i] == b[len(b)-i] {
			matches++
		}
	}
	return matches
}

// TestPairedBagFullGameReplays drives a paired game through the real move
// machinery -- exchanges included, since those are the one thing that puts
// tiles back and consults the RNG mid-game -- and checks that two runs of the
// same seed stay in lockstep from the opening racks to the end of the bag.
func TestPairedBagFullGameReplays(t *testing.T) {
	is := is.New(t)
	seed := seedFrom(23)

	// Exchange sizes chosen so the two seats consume tiles at different rates,
	// which is the case that would break if they shared an end of the bag.
	exchangeSizes := []int{3, 5, 2, 7, 4, 1, 6, 3, 2, 5}

	replay := func() []string {
		g := pairedGame(t, seed)
		// Exchanging repeatedly is scoreless, and we want the game to keep
		// going rather than end six turns in.
		g.SetMaxScorelessTurns(1000)

		racks := []string{}
		for turn := 0; turn < len(exchangeSizes); turn++ {
			onturn := g.PlayerOnTurn()
			rack := g.RackFor(onturn).TilesOn()
			size := exchangeSizes[turn]
			if size > len(rack) || g.Bag().TilesRemaining() < g.ExchangeLimit() {
				break
			}
			tiles := append(tilemapping.MachineWord{}, rack[:size]...)
			leave := append(tilemapping.MachineWord{}, rack[size:]...)
			if err := g.PlayMove(move.NewExchangeMove(tiles, leave, g.Alphabet()), false, 0); err != nil {
				t.Fatal(err)
			}
			racks = append(racks, g.RackLettersFor(0)+"|"+g.RackLettersFor(1))
		}
		if len(racks) != len(exchangeSizes) {
			t.Fatalf("game ended early after %d of %d turns", len(racks), len(exchangeSizes))
		}
		return racks
	}

	is.Equal(replay(), replay())
}

// TestPairedBagSurvivesSeatSwap is the other half of a game pair: the second
// game swaps which player moves first, the way the autoplay runner does with
// FlipPlayers, and each player then draws the tiles the other one had.
func TestPairedBagSurvivesSeatSwap(t *testing.T) {
	is := is.New(t)
	seed := seedFrom(5)

	first := pairedGame(t, seed)

	rules, err := NewBasicGameRules(
		DefaultConfig, "", board.CrosswordGameLayout, "english",
		CrossScoreOnly, "")
	is.NoErr(err)
	second, err := NewGame(rules, []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "p1"},
		{Nickname: "p2", RealName: "p2"},
	})
	is.NoErr(err)
	second.FlipPlayers()
	second.SeedBag(seed)
	second.SetPairedBagMode(true)
	second.StartGame()

	// Different player is on the clock first...
	is.Equal(first.NickOnTurn(), "p1")
	is.Equal(second.NickOnTurn(), "p2")
	// ...but the seats were dealt the same tiles, so p2 now holds what p1 held
	// and vice versa. That is what cancels the draw luck out of the comparison.
	is.Equal(first.RackLettersFor(0), second.RackLettersFor(0))
	is.Equal(first.RackLettersFor(1), second.RackLettersFor(1))
	is.Equal(drawStream(first, 0, 7, 4), drawStream(second, 0, 7, 4))
	is.Equal(drawStream(first, 1, 7, 4), drawStream(second, 1, 7, 4))
}
