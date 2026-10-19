package stats

import (
	"testing"

	"github.com/matryer/is"
)

// This is the grouping itself, on the internal types. The same fixture is run
// through the exported structs by TestExportedFamiliesMatchTheTable, which is
// what the AI explainer reads; the two have to agree about which plays exist,
// and disagree about the leading space, which is table padding rather than
// part of the play.
func TestGroupPlays(t *testing.T) {
	is := is.New(t)

	npmap := blankVariantPlays()

	fams := sortedFamilies(npmap)
	is.Equal(len(fams), 3)

	// most likely family first, by combined count
	is.Equal(fams[0].name(), " 7F W(E)AK") // a lone play keeps its own notation
	is.Equal(fams[0].count, 273)
	is.Equal(fams[0].scoreStr(), "23")
	is.Equal(fams[0].drawStr(), "")

	// the three ways of making ZWIEBACK add up to one 165-count opportunity
	is.Equal(fams[1].name(), " 1H (Z)WIEBACK")
	is.Equal(fams[1].count, 165)
	is.Equal(fams[1].scoreStr(), "116-134")
	is.Equal(fams[1].drawStr(), "{B|C|?}")
	is.Equal(len(fams[1].variants), 3)
	// variants are ordered by how often they came up
	is.Equal(fams[1].variants[0].play, " 1H (Z)WIEBAcK")
	is.Equal(fams[1].variants[2].play, " 1H (Z)WIEbAcK")

	is.Equal(fams[2].name(), " I8 SAWLIKE")
	is.Equal(fams[2].count, 146)
	is.Equal(fams[2].scoreStr(), "80-81")
	is.Equal(fams[2].drawStr(), "{S|L}")

	// by score, the biggest play in a family is what ranks it
	byScore := sortedFamiliesByScore(npmap)
	is.Equal(byScore[0].name(), " 1H (Z)WIEBACK")
	is.Equal(byScore[1].name(), " I8 SAWLIKE")
	is.Equal(byScore[2].name(), " 7F W(E)AK")
}

func TestFamilyDrawStr(t *testing.T) {
	is := is.New(t)

	fam := func(draws ...string) *playFamily {
		f := &playFamily{}
		for _, d := range draws {
			f.variants = append(f.variants, &nextPlay{ifdraw: d})
		}
		return f
	}

	// multi-tile draws stay whole: you need every letter of one alternative
	is.Equal(fam("{EI}", "{E?}", "{I?}").drawStr(), "{EI|E?|I?}")
	// a route that needs nothing is shown as a dash next to the ones that do
	is.Equal(fam("", "{S}").drawStr(), "{-|S}")
	// duplicate alternatives collapse
	is.Equal(fam("{S}", "{S}").drawStr(), "{S}")
	// nothing needed at all: leave the column empty rather than printing {-}
	is.Equal(fam("", "").drawStr(), "")
	// too many alternatives to fit the column: the tail is elided, and the
	// full set is spelled out on the variant lines underneath
	is.Equal(fam("{LET}", "{ETY}", "{LEY}", "{LTY}", "{E?Y}", "{LT?}").drawStr(), "{LET|ETY|...}")
}
