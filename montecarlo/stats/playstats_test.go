package stats

import (
	"strings"
	"testing"

	"github.com/matryer/is"
)

// blankVariantPlays is a set of sampled follow-ups from a position with
// ?AEIKUW. ZWIEBACK and SAWLIKE can each be made in more than one way, because
// the blank can stand in for different tiles.
//
// The plays carry a leading space on purpose: MoveDescriptionWithPlaythrough
// right-aligns the coordinate ("%3v ") so the table columns line up. That
// padding is presentation - heatmap_test.go asserts the internal values keep
// it, and TestExportedFamiliesMatchTheTable asserts the exported data doesn't.
// One fixture feeds both, so the two layers can't quietly drift apart.
func blankVariantPlays() map[string]*nextPlay {
	npmap := map[string]*nextPlay{}
	for _, p := range []struct {
		play   string
		score  int
		count  int
		ifdraw string
		bingo  bool
	}{
		{" 1H (Z)WIEBAcK", 134, 70, "{B}", false},
		{" 1H (Z)WIEbACK", 125, 68, "{C}", false},
		{" 1H (Z)WIEbAcK", 116, 27, "{?}", false},
		{" 7F W(E)AK", 23, 273, "", false},
		{" I8 SAWlIKE", 81, 82, "{S}", true},
		{" I8 sAWLIKE", 80, 64, "{L}", true},
	} {
		npmap[p.play] = &nextPlay{play: p.play, score: p.score, count: p.count,
			ifdraw: p.ifdraw, bingo: p.bingo}
	}
	return npmap
}

// The structured export and the rendered table have to describe the same set
// of plays: the AI explainer answers questions only about the plays in the
// table, and it looks them up in the export.
func TestExportedFamiliesMatchTheTable(t *testing.T) {
	is := is.New(t)

	npmap := blankVariantPlays()
	total := 500

	fams := sortedFamilies(npmap)
	shown := displayedFamilies(fams, 15, true)
	exported := exportFamilies(shown, total)
	is.Equal(len(exported), 3)

	// A lone play keeps its own notation, and loses the padding the table
	// uses to line coordinates up.
	is.Equal(exported[0].Play, "7F W(E)AK")
	is.Equal(exported[0].Count, 273)
	is.Equal(exported[0].Pct, 54.6)
	is.Equal(exported[0].MinScore, 23)
	is.Equal(exported[0].MaxScore, 23)
	is.True(!exported[0].Grouped())
	is.Equal(exported[0].NeededDraws, []string{""})

	// The three ways of making ZWIEBACK are one opportunity, and the family
	// carries the chance of making it by any route - never one way's share.
	z := exported[1]
	is.Equal(z.Play, "1H (Z)WIEBACK")
	is.Equal(z.Count, 165)
	is.Equal(z.Pct, 33.0)
	is.Equal(z.MinScore, 116)
	is.Equal(z.MaxScore, 134)
	is.True(z.Grouped())
	is.Equal(len(z.Ways), 3)
	is.Equal(z.Ways[0].Play, "1H (Z)WIEBAcK")
	is.Equal(z.Ways[0].NeededDraw, "B")
	is.Equal(z.Ways[0].Pct, 14.0)
	is.Equal(z.NeededDraws, []string{"B", "C", "?"})

	is.Equal(exported[2].Play, "I8 SAWLIKE")
	is.True(exported[2].Bingo)

	// Every exported play appears in the table the model is shown, so a
	// lookup can never succeed for a play the model can't see.
	table := playStatsStr(nil, "", fams, "### Our follow-up play", 15, total, true, false)
	for _, fam := range exported {
		is.True(strings.Contains(table, strings.TrimSpace(fam.Play)))
	}
}

func TestBingoPct(t *testing.T) {
	is := is.New(t)

	npmap := map[string]*nextPlay{
		"A1 BINGOES": {play: "A1 BINGOES", score: 80, count: 40, bingo: true},
		"B2 NOT":     {play: "B2 NOT", score: 12, count: 60},
	}
	// Bingos are counted over every family, not just the displayed ones, and
	// against the total number of sampled plays rather than the family count.
	is.Equal(bingoPct(sortedFamilies(npmap), 200), 20.0)
	is.Equal(bingoPct(sortedFamilies(npmap), 0), 0.0)
}
