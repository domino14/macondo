package stats

import (
	"testing"

	"github.com/matryer/is"
)

// CalculateLaneStats needs a simulation log to read, so it is exercised end to
// end from the explainer's tests. What can be pinned here is the naming: the
// labels are what the AI explainer is allowed to say out loud about the board,
// so they have to read the way a player would name a lane.
func TestLaneLabel(t *testing.T) {
	is := is.New(t)
	is.Equal(LaneLabel(false, 11), "row 12")  // rows count from 1
	is.Equal(LaneLabel(true, 10), "column K") // columns letter from A
	is.Equal(LaneLabel(true, 0), "column A")
	is.Equal(LaneLabel(false, 0), "row 1")
}
