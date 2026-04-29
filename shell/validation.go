package shell

import (
	"fmt"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// validateGameHistory checks that all rack strings in the history can be decoded
// and do not exceed the rack size limit. Returns a descriptive error for the
// first violation found, or nil if the history looks playable.
func validateGameHistory(history *pb.GameHistory, alph *tilemapping.TileMapping) error {
	for i, evt := range history.Events {
		if evt.Rack == "" {
			continue
		}
		tiles, err := tilemapping.ToMachineLetters(evt.Rack, alph)
		if err != nil {
			return fmt.Errorf("turn %d: cannot decode rack %q: %w", i, evt.Rack, err)
		}
		if len(tiles) > game.RackTileLimit {
			return fmt.Errorf("turn %d: rack %q has %d tiles, max is %d",
				i, evt.Rack, len(tiles), game.RackTileLimit)
		}
	}
	for i, r := range history.LastKnownRacks {
		if r == "" {
			continue
		}
		tiles, err := tilemapping.ToMachineLetters(r, alph)
		if err != nil {
			return fmt.Errorf("last-known rack[%d]: cannot decode %q: %w", i, r, err)
		}
		if len(tiles) > game.RackTileLimit {
			return fmt.Errorf("last-known rack[%d] %q has %d tiles, max is %d",
				i, r, len(tiles), game.RackTileLimit)
		}
	}
	return nil
}
