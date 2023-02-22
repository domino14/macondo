package rangefinder

import (
	"fmt"
	"strings"

	"github.com/domino14/macondo/alphabet"
)

func (r *RangeFinder) AnalyzeInferences(detailed bool) string {
	totalCt := 0
	mlcts := map[alphabet.MachineLetter]int{}
	for _, inf := range r.inferences {
		for _, ml := range inf {
			mlcts[ml]++
			totalCt++
		}
	}
	inbag := uint8(0)
	bagmap := r.inferenceBagMap
	for i := range bagmap {
		inbag += bagmap[i]
	}
	if totalCt == 0 || inbag == 0 {
		return "No inference details."
	}

	alph := r.origGame.Alphabet()

	if detailed {
		var ss strings.Builder
		fmt.Fprintf(&ss, "%-5s%-12s%-12s%-10s\n", "Tile", "Found %", "Expected %", "# unseen")

		printLetterStats := func(i int) {
			fmt.Fprintf(&ss, "%-5c%-12.3f%-12.3f%d\n",
				alphabet.MachineLetter(i).UserVisible(alph),
				100.0*float64(mlcts[alphabet.MachineLetter(i)])/float64(totalCt),
				100.0*float64(bagmap[i])/float64(inbag),
				bagmap[i])
		}

		for i := 0; i < int(alph.NumLetters()); i++ {
			printLetterStats(i)
		}
		printLetterStats(alphabet.BlankMachineLetter)
		fmt.Fprintf(&ss, "Considered %d racks, inferred %d racks\n", r.iterationCount, len(r.inferences))

		return ss.String()
	}
	// From likelihood to unlikelihood (index 0 to 7)
	// Index 7 is not found at all.
	bins := [8][]alphabet.MachineLetter{}

	// Otherwise do a very rough statistical analysis.
	for i := 0; i < int(alph.NumLetters()+1); i++ {
		if i == int(alph.NumLetters()) {
			i = alphabet.BlankMachineLetter
		}
		found := float64(mlcts[alphabet.MachineLetter(i)]) / float64(totalCt)
		expected := float64(bagmap[i]) / float64(inbag)
		if expected == 0 {
			bins[7] = append(bins[7], alphabet.MachineLetter(i))
			continue
		}
		ratio := found / expected
		var bin int
		switch {
		case ratio == 0:
			bin = 7
		case ratio < 0.25:
			bin = 6
		case ratio < 0.75:
			bin = 5
		case ratio < 0.9:
			bin = 4
		case ratio < 1.1:
			bin = 3
		case ratio < 1.25:
			bin = 2
		case ratio < 2:
			bin = 1
		default:
			bin = 0
		}
		bins[bin] = append(bins[bin], alphabet.MachineLetter(i))
	}

	var ss strings.Builder

	printTiles := func(tiles []alphabet.MachineLetter) {
		for _, t := range tiles {
			fmt.Fprintf(&ss, " %c ", t.UserVisible(alph))
		}
		fmt.Fprintln(&ss)
		fmt.Fprintln(&ss)
	}

	ss.WriteString("Way more than chance:\n")
	printTiles(bins[0])
	ss.WriteString("More than chance:\n")
	printTiles(bins[1])
	ss.WriteString("Slightly more than chance:\n")
	printTiles(bins[2])
	ss.WriteString("About as expected:\n")
	printTiles(bins[3])
	ss.WriteString("Slightly less than chance:\n")
	printTiles(bins[4])
	ss.WriteString("Less than chance:\n")
	printTiles(bins[5])
	ss.WriteString("Way less than chance:\n")
	printTiles(bins[6])
	ss.WriteString("Unpossible:\n")
	printTiles(bins[7])
	return ss.String()
}
