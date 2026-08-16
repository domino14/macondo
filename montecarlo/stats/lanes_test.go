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
	is.Equal(laneLabel(false, 11), "row 12")  // rows count from 1
	is.Equal(laneLabel(true, 10), "column K") // columns letter from A
	is.Equal(laneLabel(true, 0), "column A")
	is.Equal(laneLabel(false, 0), "row 1")
}
