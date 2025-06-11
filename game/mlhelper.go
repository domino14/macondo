package game

import (
	"errors"
	"fmt"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/word-golib/tilemapping"
)

// MLEvaluateMove evaluates a move using the machine learning model.
// Assume that the board and racks etc are already properly assigned.
func (g *Game) MLEvaluateMove(m *move.Move) (float32, error) {
	if g.backupMode == NoBackup {
		return 0, errors.New("ML evaluation can only be used in backup mode")
	}
	g.backupState()
	switch m.Action() {
	case move.MoveTypePlay:
		g.board.PlayMove(m)
		g.crossSetGen.UpdateForMove(g.board, m)
		score := m.Score()
		g.lastScorelessTurns = g.scorelessTurns
		g.scorelessTurns = 0
		g.players[g.onturn].points += score
		g.players[g.onturn].turns += 1
		if m.TilesPlayed() == RackTileLimit {
			g.players[g.onturn].bingos++
		}
		// Don't draw replacement tiles.
		// don't deal with end of game at this moment.
	case move.MoveTypePass:
		g.lastScorelessTurns = g.scorelessTurns
		g.scorelessTurns++
		g.players[g.onturn].turns += 1

	case move.MoveTypeExchange:
		// don't actually exchange, but we want to track our leave.
		g.bag.PutBack(m.Tiles())
		g.lastScorelessTurns = g.scorelessTurns
		g.scorelessTurns++
		g.players[g.onturn].turns += 1
	}

	vec, err := g.BuildMLVector()
	if err != nil {
		return 0, fmt.Errorf("failed to build ML vector: %w", err)
	}

	g.onturn = (g.onturn + 1) % len(g.players)
	g.UnplayLastMove() // this undoes the g.onturn change above which is why we even do that in the first place.

	// xxx changeme
	return vec[0], nil
}

func NormalizeSpreadForML(spread float32) float32 {
	// Normalize spread to -1 to 1 range.
	// First we clamp it to -300 or 300.
	if spread < -300 {
		spread = -300.0
	} else if spread > 300 {
		spread = 300.0
	}
	return float32(spread) / 300.0
}

// BuildMLVector builds the feature vector for the current game state. Assumes
// the player on turn has just put tiles on a board and tallied up their new
// score but not drawn replacement tiles.
func (g *Game) BuildMLVector() ([]float32, error) {

	tilePlanes := make([]float32, 26*225) // 26 planes for letters A-Z
	isBlankPlane := make([]float32, 225)  // 15x15 board tiles

	// 26 planes for board tiles
	// Each plane is a 15x15 grid, flattened to 225 elements.
	plane := g.Board()
	for j := range 15 {
		for k := range 15 {
			tile := plane.GetLetter(j, k)
			if tile == 0 {
				continue // skip empty squares
			}
			unblanked := tile.Unblank()
			if unblanked != tile {
				// The tile was blanked
				isBlankPlane[j*15+k] = 1.0 // mark as blank
			} else {
				// The tile is a letter. The unblanked tile will range from
				// 1 to 26 for A-Z. (Fix later for other alphabets)
				ll := unblanked - 1 // convert to 0-based index
				if ll >= 26 {
					return nil, fmt.Errorf("Invalid tile index %d for tile %d at position (%d, %d)", ll, tile, j, k)
				}
				// Set the corresponding plane for this letter
				tilePlanes[int(ll)*225+j*15+k] = 1.0 // this tile is in the letter plane
			}
		}
	}

	horCCs := make([]float32, 26*225)
	verCCs := make([]float32, 26*225)

	// 26 planes for horizontal cross-checks
	// 26 planes for vertical cross-checks
	for i := range 15 {
		for j := range 15 {
			hc := g.Board().GetCrossSet(i, j, board.HorizontalDirection)
			vc := g.Board().GetCrossSet(i, j, board.VerticalDirection)
			if hc == board.TrivialCrossSet && vc == board.TrivialCrossSet {
				// We skip trivial cross-checks to make this structure nicer to
				// the neural net. Trivial cross-checks are for empty squares where
				// technically every tile is allowed. But we really care about
				// empty squares that are right next to a tile (anchors, basically);
				// those would always have non-trivial cross-checks.
				continue
			}
			for t := range 26 { // A-Z are 1-26; change for other alphabets in future.
				// For each letter A-Z, we check if it is in the cross-check set.
				letter := tilemapping.MachineLetter(t + 1) // convert to 1-based index
				if hc.Allowed(letter) {
					horCCs[int(t)*225+i*15+j] = 1.0 // this tile is in the horizontal cross-check
				}
				if vc.Allowed(letter) {
					verCCs[int(t)*225+i*15+j] = 1.0 // this tile is in the vertical cross-check
				}
			}
		}
	}
	// Consider a single plane that has all the "anchors" or empty squares that are
	// adjacent to tiles in the future.

	// uncovered bonus square planes
	bonus2LPlane := make([]float32, 225) // 2x letter bonus
	bonus3LPlane := make([]float32, 225) // 3x letter bonus
	bonus2WPlane := make([]float32, 225) // 2x word bonus
	bonus3WPlane := make([]float32, 225) // 3x word bonus
	for i := range 15 {
		for j := range 15 {
			letter := g.Board().GetLetter(i, j)
			if letter != 0 {
				// This is a letter tile, not an empty square.
				continue
			}
			bonus := g.Board().GetBonus(i, j)

			if bonus == board.NoBonus {
				continue // no bonus here
			}
			if bonus == board.Bonus2LS {
				bonus2LPlane[i*15+j] = 1.0 // mark as 2x letter bonus
			} else if bonus == board.Bonus3LS {
				bonus3LPlane[i*15+j] = 1.0 // mark as 3x letter bonus
			} else if bonus == board.Bonus2WS {
				bonus2WPlane[i*15+j] = 1.0 // mark as 2x word bonus
			} else if bonus == board.Bonus3WS {
				bonus3WPlane[i*15+j] = 1.0 // mark as 3x word bonus
			} else {
				return nil, fmt.Errorf("Unknown bonus type %d at position (%d, %d)", bonus, i, j)
			}
		}
	}

	// 1 vector for our rack leave (size = 27 tiles)
	// 1 vector for unseen tiles in the bag (size = 27 tiles)
	// scalar for score diff after making move
	// scalar for num tiles left in bag
	rackVector := make([]float32, 27)   // 26 letters + 1 for unseen
	unseenVector := make([]float32, 27) // 26 letters + 1 for unseen

	rack := g.RackFor(g.PlayerOnTurn())
	oppRack := g.RackFor(1 - g.PlayerOnTurn())
	g.bag.PutBack(oppRack.TilesOn()) // put back opponent's tiles to calculate all unseen tiles
	bag := g.Bag().PeekMap()
	for i := range 27 {
		rackVector[i] = float32(rack.LetArr[i]) / 7.0 // normalize to 0-1 range
		// technically we can even divide by 12 here i think. (number of Es in the bag)
		// divide by 20 for now. make the vectors "cared about" a little bit sooner.
		unseenVector[i] = float32(bag[i]) / 20.0 // normalize to 0-1 range

	}

	// Note the spread is our spread after making this move. The player on turn
	// switched after playing the move, so that's why we do 1 - gw.game.PlayerOnTurn().
	// We temporarily encode the player whose spread this is for since we don't
	// have that data anywhere else.
	// spreadFor := 1 - g.PlayerOnTurn()
	// spread := float32(g.SpreadFor(spreadFor)) // update spread for opponent

	normalizedSpread := NormalizeSpreadForML(float32(g.SpreadFor(g.PlayerOnTurn())))
	// normalize to -1 to 1 range
	tilesRemaining := float32(g.Bag().TilesRemaining()) / 100.0 // normalize to 0-1 range
	// Concatenate all feature planes and vectors into a single flat []float32.
	features := []float32{}
	features = append(features, tilePlanes...)
	features = append(features, isBlankPlane...)
	features = append(features, horCCs...)
	features = append(features, verCCs...)
	features = append(features, bonus2LPlane...)
	features = append(features, bonus3LPlane...)
	features = append(features, bonus2WPlane...)
	features = append(features, bonus3WPlane...)
	features = append(features, rackVector...)
	features = append(features, unseenVector...)
	features = append(features, tilesRemaining, normalizedSpread)

	return features, nil
}
