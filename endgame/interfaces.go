package endgame

import (
	"context"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
)

// Solver is an interface for an endgame solver. The scores don't matter, the
// solver maximizes spread.
type Solver interface {
	Init(game *game.Game) error
	Solve(ctx context.Context, plies int) (float32, []*move.Move, error)
}
