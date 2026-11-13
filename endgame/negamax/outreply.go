package negamax

import (
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/tinymove"
)

// The out-reply bound.
//
// A move that does not go out is worth, to the player making it, at most
// what is left if the opponent goes out in reply: the move's score, less the
// reply's score and twice the tiles the move keeps. Any of the opponent's
// going-out plays will do, as long as the move leaves it standing -- and it
// does when the play covers none of the squares the move changes in the
// play's lane (the same record the incremental move lists keep; see
// pathLists.moveHits), since a play's legality and score depend only on its
// lane's squares.
//
// So a node keeps the opponent's best few going-out plays on its own board,
// and a move whose bound against the best one it leaves standing does not
// beat alpha needs no search. Like the keep-one bound this is a fact about
// the game, good wherever the opponent's reply is searched (depth 2 and up).
// The idea is MAGPIE's (out_reply_bound in endgame.c).

// outReplyMaxPlays is how many of the opponent's going-out plays a node
// keeps.
const outReplyMaxPlays = 8

// outReplyBound is what a node needs to bound its moves by the opponent
// going out in reply.
type outReplyBound struct {
	ready bool
	lists *pathLists
	count int
	// The opponent's best going-out plays, best first: score, lane, the
	// lane squares covered, and whether the play places a single tile (and
	// so forms a word in the crossing lane too).
	score  [outReplyMaxPlays]int
	lane   [outReplyMaxPlays]uint8
	span   [outReplyMaxPlays]uint32
	single [outReplyMaxPlays]bool
	// base is the node's spread less twice the rack of the player on turn:
	// what a move adds its score and twice its tiles to.
	base     int
	rackSize int
	dim      int
}

// init finds the opponent's going-out plays on the node's board, from the
// opponent's move list.
func (o *outReplyBound) init(g *game.Game, lists *pathLists, onTurn int, ourSpread int) {
	o.ready = true
	o.lists = lists
	o.count = 0
	ld := g.Bag().LetterDistribution()
	rack := g.RackFor(onTurn)
	o.rackSize = int(rack.NumTiles())
	o.base = ourSpread - 2*rack.ScoreOn(ld)
	o.dim = g.Board().Dim()
	oppTiles := int(g.RackFor(1 - onTurn).NumTiles())
	if oppTiles == 0 {
		return
	}
	if oppTiles == 1 {
		// A lone tile with no square at all, the common case when it is
		// stuck, is known without a move list.
		if _, ok := lists.bestLoneTilePlay(1 - onTurn); !ok {
			return
		}
	}
	l := lists.get(1 - onTurn)
	for i := range l.moves {
		m := &l.moves[i]
		if m.TilesPlayed() != oppTiles {
			continue
		}
		score := m.Score()
		slot := o.count
		if slot == outReplyMaxPlays {
			if score <= o.score[slot-1] {
				continue
			}
			slot--
		} else {
			o.count++
		}
		for slot > 0 && o.score[slot-1] < score {
			o.score[slot] = o.score[slot-1]
			o.lane[slot] = o.lane[slot-1]
			o.span[slot] = o.span[slot-1]
			o.single[slot] = o.single[slot-1]
			slot--
		}
		row, col, vertical := m.CoordsAndVertical()
		lane, start := row, col
		if vertical {
			lane, start = o.dim+col, row
		}
		o.score[slot] = score
		o.lane[slot] = uint8(lane)
		o.span[slot] = (uint32(1)<<uint(m.PlayLength()) - 1) << uint(start)
		o.single[slot] = oppTiles == 1
	}
}

// bound returns an upper bound on the value, to the player on turn, of the
// move m, which must not go out; ok is false when no going-out reply is
// known to stand, or when the bound is above alpha, where it decides
// nothing.
func (o *outReplyBound) bound(m *tinymove.SmallMove, alpha int) (int, bool) {
	if o.count == 0 {
		return 0, false
	}
	ld := o.lists.g.Bag().LetterDistribution()
	afterMove := o.base + m.Score() + 2*playedTilesFaceValue(m, ld)
	if afterMove-o.score[0] > alpha {
		return 0, false
	}
	lv := &o.lists.scratch
	hits := o.lists.moveHits(m, lv)
	for i := 0; i < o.count; i++ {
		value := afterMove - o.score[i]
		if value > alpha {
			break
		}
		lane := int(o.lane[i])
		span := o.span[i]
		if hits&(1<<uint(lane)) != 0 && lv.laneHits[lane]&span != 0 {
			continue
		}
		if o.single[i] {
			// One tile makes a word in the crossing lane as well: the play
			// stands only if its square is unchanged there too.
			pos := 0
			for span>>uint(pos) != 1 {
				pos++
			}
			cross, crossPos := o.dim+pos, lane
			if lane >= o.dim {
				cross, crossPos = pos, lane-o.dim
			}
			if hits&(1<<uint(cross)) != 0 && lv.laneHits[cross]&(1<<uint(crossPos)) != 0 {
				continue
			}
		}
		return value, true
	}
	return 0, false
}
