package explainer

import (
	"strings"
	"testing"

	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/matryer/is"
)

// laneComparison builds one candidate's board dynamics: the lanes its replies
// landed in, and how good those replies were overall.
func laneComparison(play string, best bool, reply ReplyProfile, fp *stats.Footprint,
	lanes ...*stats.LaneStat) *LaneComparison {
	return &LaneComparison{
		Play: play, Best: best, Reply: reply,
		Stats: &stats.LaneStats{
			Play: play, Total: 1000, Placements: 900, Footprint: fp, Lanes: lanes,
		},
	}
}

func lane(label string, vertical bool, index int, pct, mean float64) *stats.LaneStat {
	return &stats.LaneStat{
		Label: label, Vertical: vertical, Index: index, Count: int(pct * 10),
		Pct: pct, MeanScore: mean, MaxScore: int(mean) + 20, BestPlay: "x",
	}
}

// The complaint the whole gate exists for. Column K goes from a tenth of the
// opponent's replies to none at all, which reads like the play shut the board
// down - and the opponent scores exactly as much either way, because their
// replies simply went somewhere else. Nothing at all is said about the board.
func TestLaneThatMovesWhileNothingChangesIsNotAFinding(t *testing.T) {
	is := is.New(t)

	fs := buildBoardFindings([]*LaneComparison{
		laneComparison("8F TOQUE", true,
			ReplyProfile{MeanScore: 31.8, BigPct: 12.1, BingoPct: 5.0, Known: true}, nil,
			lane("row 9", false, 8, 22.0, 30.0)),
		laneComparison("8B OPAQUED", false,
			ReplyProfile{MeanScore: 32.4, BigPct: 13.0, BingoPct: 5.2, Known: true}, nil,
			lane("column K", true, 10, 9.7, 34.0),
			lane("row 9", false, 8, 20.0, 30.0)),
	}, "8B OPAQUED")
	is.Equal(len(fs), 0)
}

// The other half: a lane that goes quiet and takes the opponent's scoring with
// it. One lane can carry the whole explanation when the board-wide figures
// back it, and the finding says so in one sentence.
func TestLaneBackedByTheWholeBoardIsAFinding(t *testing.T) {
	is := is.New(t)

	fs := buildBoardFindings([]*LaneComparison{
		// 2J TOQUE is played in row 2 itself: horizontal, row index 1.
		laneComparison("2J TOQUE", true,
			ReplyProfile{MeanScore: 40.4, BigPct: 21.5, BingoPct: 11.2, Known: true},
			&stats.Footprint{Vertical: false, Index: 1, Start: 9, End: 13},
			lane("row 2", false, 1, 0.3, 20.0)),
		laneComparison("L5 (OPA)QUED", false,
			ReplyProfile{MeanScore: 43.7, BigPct: 33.5, BingoPct: 9.5, Known: true},
			&stats.Footprint{Vertical: true, Index: 11, Start: 7, End: 10},
			lane("row 2", false, 1, 40.8, 47.0)),
	}, "L5 (OPA)QUED")
	is.Equal(len(fs), 1)
	fi := fs[0]
	is.True(fi.Tighter)
	is.Equal(fi.Lane, "row 2")
	// The play sitting in the lane is the one that leaves it quiet, so it took
	// the spot rather than opening it up - the direction the first cut of this
	// got backwards on every lane it described.
	is.Equal(fi.Mechanism, MechanismTakesSpot)
	is.Equal(fi.MechanismPlay, "2J TOQUE")

	text := boardFindingText("2J TOQUE", fi)
	is.True(strings.Contains(text, "holds the opponent to less"))
	is.True(strings.Contains(text, "40.4 against 43.7"))
	is.True(strings.Contains(text, "row 2 that separates them"))
	is.True(strings.Contains(text, "played in row 2 itself, taking the scoring spot"))
}

// The same geometry, the other way round: the lane is busy after the play
// whose tiles run through it, so the replies there are answering those tiles.
func TestBusyLaneOffThePlaysOwnTilesIsCalledOpening(t *testing.T) {
	is := is.New(t)

	fs := buildBoardFindings([]*LaneComparison{
		laneComparison("2J TOQUE", true,
			ReplyProfile{MeanScore: 45.0, BigPct: 33.0, Known: true},
			// Horizontal in row 2, its tiles crossing columns J through N.
			&stats.Footprint{Vertical: false, Index: 1, Start: 9, End: 13},
			lane("column K", true, 10, 24.0, 41.0)),
		laneComparison("L5 (OPA)QUED", false,
			ReplyProfile{MeanScore: 40.0, BigPct: 21.0, Known: true},
			&stats.Footprint{Vertical: true, Index: 11, Start: 7, End: 10},
			lane("column K", true, 10, 1.0, 20.0)),
	}, "L5 (OPA)QUED")
	is.Equal(len(fs), 1)
	fi := fs[0]
	is.True(!fi.Tighter) // the best play is the looser one here
	is.Equal(fi.Lane, "column K")
	is.Equal(fi.Mechanism, MechanismOpens)
	is.Equal(fi.MechanismPlay, "2J TOQUE")

	text := boardFindingText("2J TOQUE", fi)
	is.True(strings.Contains(text, "leaves the opponent better off"))
	is.True(strings.Contains(text, "runs its own tiles through column K"))
}

// A lane can move the wrong way for the finding it would be attached to. The
// board-wide difference is real, but this lane is not where it lives, and
// naming it would hand back the story the finding replaces.
func TestLanePointingTheOtherWayIsLeftOff(t *testing.T) {
	is := is.New(t)

	fs := buildBoardFindings([]*LaneComparison{
		laneComparison("2J TOQUE", true,
			ReplyProfile{MeanScore: 40.4, BigPct: 21.5, Known: true}, nil,
			lane("row 9", false, 8, 19.4, 36.5)),
		laneComparison("L5 (OPA)QUED", false,
			ReplyProfile{MeanScore: 43.7, BigPct: 33.5, Known: true}, nil,
			lane("row 9", false, 8, 2.7, 30.0)),
	}, "L5 (OPA)QUED")
	is.Equal(len(fs), 1)
	is.True(fs[0].Tighter)
	is.Equal(fs[0].Lane, "") // row 9 is busier after the tighter play

	text := boardFindingText("2J TOQUE", fs[0])
	is.True(strings.Contains(text, "holds the opponent to less"))
	is.True(!strings.Contains(text, "row 9"))
}

// When the means are level and it is the big replies that moved, the finding
// has to lead with the big replies. Leading with two means half a point apart
// invites exactly the overclaim the numbers don't support.
func TestFindingLedByWhicheverMeasureMoved(t *testing.T) {
	is := is.New(t)

	fs := buildBoardFindings([]*LaneComparison{
		laneComparison("12K QU(ID)", true,
			ReplyProfile{MeanScore: 33.6, BigPct: 24.2, Known: true},
			&stats.Footprint{Vertical: false, Index: 11, Start: 10, End: 12},
			lane("row 12", false, 11, 2.2, 20.0)),
		laneComparison("11J DAC(HA)", false,
			ReplyProfile{MeanScore: 34.4, BigPct: 33.0, Known: true}, nil,
			lane("row 12", false, 11, 24.0, 56.6)),
	}, "11J DAC(HA)")
	is.Equal(len(fs), 1)
	is.True(fs[0].Tighter) // on the tail alone: the means are level

	text := boardFindingText("12K QU(ID)", fs[0])
	is.True(strings.Contains(text, "fewer big turns"))
	is.True(strings.Contains(text, "mean next turn is much the same"))
	is.True(!strings.Contains(text, "holds the opponent to less"))
}

// Defense is measured on the tail as well as the mean: a play can give up the
// same average reply and still be handing over the occasional huge one.
func TestBigRepliesAloneEstablishADifference(t *testing.T) {
	is := is.New(t)

	tight := ReplyProfile{MeanScore: 30.0, BigPct: 8.0, Known: true}
	loose := ReplyProfile{MeanScore: 31.0, BigPct: 19.0, Known: true}
	is.Equal(CompareDefense(tight, loose), DefenseTighter)
	is.Equal(CompareDefense(loose, tight), DefenseLooser)

	// Neither figure moves enough: the opponent is left where they were.
	same := ReplyProfile{MeanScore: 32.0, BigPct: 12.0, Known: true}
	is.Equal(CompareDefense(same, ReplyProfile{MeanScore: 34.0, BigPct: 15.0, Known: true}),
		DefenseFlat)
	is.Equal(CompareDefense(same, ReplyProfile{}), DefenseUnknown)
}

func TestFootprintTouches(t *testing.T) {
	is := is.New(t)

	// 8B OPAQUED: horizontal, row 8, columns B through K.
	fp := &stats.Footprint{Vertical: false, Index: 7, Start: 1, End: 10}
	is.True(fp.Touches(false, 7))  // its own row
	is.True(!fp.Touches(false, 8)) // the row below is not touched
	is.True(fp.Touches(true, 10))  // column K, its last tile
	is.True(!fp.Touches(true, 11)) // column L is past the end
	is.True(!fp.Touches(true, 0))  // column A is before the start
	is.True(!(*stats.Footprint)(nil).Touches(true, 10))
}
