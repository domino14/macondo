package game

import (
	"github.com/domino14/word-golib/tilemapping"
	"lukechampine.com/frand"
)

// Paired-bag mode makes a game's tile draws a pure function of its seed, so
// that two bots can be handed the exact same game twice with the seats
// swapped. It is the Macondo equivalent of MAGPIE's "game pairs" bag.
//
// The bag is shuffled once from the seeded RNG and then left in that fixed
// order for the rest of the game. Each player owns one end of it: the player
// who moves first takes tiles off the back, the player who moves second takes
// them off the front. Nothing else consumes the order.
//
// Owning separate ends is what makes the pairing work. If both players drew
// from the same end, a player's tiles would depend on how many tiles the
// *other* player had already drawn — so the moment one bot played a different
// number of tiles than the other, every subsequent draw for both players would
// shift and the two games would be comparing unrelated luck. Splitting the bag
// decouples the two streams: the nth tile a seat draws depends only on how many
// tiles that seat has drawn. Swap the bots between seats, replay the same seed,
// and each bot sees exactly the tiles the other one saw. Whatever score
// difference is left over is strategy, not draws.
//
// The two ends only meet when the bag runs out, at which point there is nothing
// left to correlate anyway.
const (
	// drawsFromBack is the draw side of the player who moves first.
	drawsFromBack = 0
	// drawsFromFront is the draw side of the player who moves second.
	drawsFromFront = 1
)

// SetPairedBagMode turns two-sided fixed-order draws on or off. It must be
// called before StartGame, which is where the bag order is fixed, and it only
// produces reproducible games in combination with SeedBag: exchanges consult
// the RNG, so an unseeded paired game still drifts apart from its partner.
func (g *Game) SetPairedBagMode(on bool) {
	g.pairedBag = on
}

// PairedBagMode reports whether this game draws tiles from both ends of a
// fixed-order bag.
func (g *Game) PairedBagMode() bool {
	return g.pairedBag
}

// drawSideFor returns the end of the bag a seat draws from. The player who
// started the game owns the back; their opponent owns the front.
func (g *Game) drawSideFor(playerIdx int) int {
	return playerIdx ^ g.startingPlayerIdx
}

// bagIntn returns a random number in [0, n) from the game's seeded RNG, or from
// the global source if this game was never seeded.
func (g *Game) bagIntn(n int) int {
	if g.customRNG != nil {
		return g.customRNG.Intn(n)
	}
	return frand.Intn(n)
}

// drawFor draws n tiles into ml for the given seat. Outside of paired-bag mode
// it is an ordinary randomized draw and the seat is irrelevant.
func (g *Game) drawFor(playerIdx, n int, ml []tilemapping.MachineLetter) error {
	if !g.pairedBag {
		return g.bag.Draw(n, ml)
	}
	if n > 0 && g.drawSideFor(playerIdx) == drawsFromFront {
		// Draw pulls off the back of the bag when the order is fixed, so rotate
		// this player's tiles around to the back first. The tiles left behind
		// keep their relative order, which is the whole point: the back-side
		// player's stream is untouched by anything happening at the front.
		rotateFrontToBack(g.bag.Tiles(), n)
	}
	return g.bag.Draw(n, ml)
}

// drawAtMostFor draws up to n tiles into ml for the given seat, and returns how
// many it actually got.
func (g *Game) drawAtMostFor(playerIdx, n int, ml []tilemapping.MachineLetter) int {
	if !g.pairedBag {
		return g.bag.DrawAtMost(n, ml)
	}
	if remaining := g.bag.TilesRemaining(); n > remaining {
		n = remaining
	}
	if n <= 0 {
		return 0
	}
	if err := g.drawFor(playerIdx, n, ml); err != nil {
		// drawFor only fails when n exceeds the tiles remaining, which we just
		// clamped it to.
		panic(err)
	}
	return n
}

// exchangeFor swaps the given tiles for fresh ones from the seat's end of the
// bag, writing the new tiles into ml.
func (g *Game) exchangeFor(playerIdx int, tiles, ml []tilemapping.MachineLetter) error {
	if !g.pairedBag {
		return g.bag.Exchange(tiles, ml)
	}
	if err := g.drawFor(playerIdx, len(tiles), ml); err != nil {
		return err
	}
	g.putBackFor(playerIdx, tiles)
	return nil
}

// putBackFor returns exchanged tiles to the bag without disturbing the fixed
// order any more than it has to.
func (g *Game) putBackFor(playerIdx int, letters []tilemapping.MachineLetter) {
	if len(letters) == 0 {
		return
	}
	order := append([]tilemapping.MachineLetter(nil), g.bag.Tiles()...)
	for _, ml := range letters {
		order = g.reinsert(playerIdx, order, ml)
	}
	// PutBack reshuffles the whole bag when the order is fixed, which would
	// throw away the order we just worked out. Drop out of fixed order for the
	// call — we only want PutBack for the tile bookkeeping it does — then write
	// our own order over the result.
	g.bag.SetFixedOrder(false)
	g.bag.PutBack(letters)
	g.bag.SetFixedOrder(true)
	copy(g.bag.Tiles(), order)
}

// reinsert puts one exchanged tile back at a uniformly random spot in the bag
// and moves whichever tile it displaces to the exchanging player's own end.
// Only two tiles change place, so the rest of the fixed order survives the
// exchange and the two games of a pair stay in step.
func (g *Game) reinsert(playerIdx int, tiles []tilemapping.MachineLetter,
	ml tilemapping.MachineLetter) []tilemapping.MachineLetter {

	// The tile lands in one of the len(tiles)+1 slots of the grown bag.
	slot := g.bagIntn(len(tiles) + 1)
	if g.drawSideFor(playerIdx) == drawsFromBack {
		// The bag grows at the back, so the displaced tile goes there.
		if slot == len(tiles) {
			return append(tiles, ml)
		}
		tiles = append(tiles, tiles[slot])
		tiles[slot] = ml
		return tiles
	}
	// The bag grows at the front instead, and every existing tile shifts one
	// slot up to make room.
	if slot == 0 {
		return append([]tilemapping.MachineLetter{ml}, tiles...)
	}
	tiles = append([]tilemapping.MachineLetter{tiles[slot-1]}, tiles...)
	tiles[slot] = ml
	return tiles
}

// rotateFrontToBack moves the first n tiles to the back of the slice, leaving
// the relative order of everything else alone.
func rotateFrontToBack(tiles []tilemapping.MachineLetter, n int) {
	if n <= 0 || n >= len(tiles) {
		return
	}
	// A draw is never bigger than a rack, so the scratch space stays on the
	// stack for every call the game itself makes.
	var scratch [RackTileLimit]tilemapping.MachineLetter
	front := scratch[:]
	if n > len(front) {
		front = make([]tilemapping.MachineLetter, n)
	}
	front = front[:n]
	copy(front, tiles[:n])
	copy(tiles, tiles[n:])
	copy(tiles[len(tiles)-n:], front)
}
