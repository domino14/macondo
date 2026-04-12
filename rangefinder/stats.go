package rangefinder

import (
	"fmt"
	"sort"
	"strings"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/montecarlo"
)

func (r *RangeFinder) AnalyzeInferences(detailed bool) string {
	totalCt := float64(0)
	mlcts := map[tilemapping.MachineLetter]float64{}
	for _, ir := range r.inference.InferredRacks {
		for _, ml := range ir.Leave {
			mlcts[ml] += ir.Weight
			totalCt += ir.Weight
		}
	}
	inbag := uint8(0)
	bagmap := r.inferenceBagMap
	for i := range bagmap {
		inbag += bagmap[i]
	}
	if totalCt == 0 || inbag == 0 {
		return "No inference details. Could not draw an inference from the last play."
	}

	alph := r.origGame.Alphabet()
	nInferred := len(r.inference.InferredRacks)
	iterations := r.iterationCount
	acceptRate := 0.0
	if iterations > 0 {
		acceptRate = 100.0 * float64(nInferred) / float64(iterations)
	}

	// Compute effective sample size (ESS) = (Σw)² / Σw²
	sumW := 0.0
	sumW2 := 0.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
		sumW2 += ir.Weight * ir.Weight
	}
	ess := 0.0
	if sumW2 > 0 {
		ess = (sumW * sumW) / sumW2
	}

	headerLine := fmt.Sprintf("Inferred %d unique racks from %d iterations (%.1f%% acceptance), tau=%.3f, ESS=%.1f\n",
		nInferred, iterations, acceptRate, r.tau, ess)

	if detailed {
		var ss strings.Builder
		ss.WriteString(headerLine)
		ss.WriteString("\n")

		// Top inferred racks by weight
		ss.WriteString("Top inferred racks (by Bayesian weight):\n")
		ranked := make([]montecarlo.InferredRack, len(r.inference.InferredRacks))
		copy(ranked, r.inference.InferredRacks)
		sort.Slice(ranked, func(i, j int) bool {
			return ranked[i].Weight > ranked[j].Weight
		})

		fmt.Fprintf(&ss, "  %-6s%-12s%-12s%-12s\n", "Rank", "Leave", "Weight", "Wt %")

		showN := min(len(ranked), 15)
		for i := 0; i < showN; i++ {
			ir := ranked[i]
			leaveStr := tilemapping.MachineWord(ir.Leave).UserVisible(alph)
			wtPct := 100.0 * ir.Weight / sumW
			fmt.Fprintf(&ss, "  %-6d%-12s%-12.4f%-12.1f\n", i+1, leaveStr, ir.Weight, wtPct)
		}
		if len(ranked) > showN {
			fmt.Fprintf(&ss, "  ... and %d more\n", len(ranked)-showN)
		}

		// Weight concentration summary
		topN := min(len(ranked), 3)
		topSum := 0.0
		for i := 0; i < topN; i++ {
			topSum += ranked[i].Weight
		}
		topPct := 100.0 * topSum / sumW
		fmt.Fprintf(&ss, "\nWeight concentration: top %d hold %.1f%% of total weight (ESS = %.1f of %d)\n",
			topN, topPct, ess, nInferred)
		if ess < 3 && nInferred >= 5 {
			ss.WriteString("  Note: low ESS means weights are dominated by a few racks.\n")
		}

		ss.WriteString("\n")
		fmt.Fprintf(&ss, "%-5s%-12s%-12s%-10s\n", "Tile", "Found %", "Expected %", "# unseen")

		printLetterStats := func(i int) {
			fmt.Fprintf(&ss, "%-5s%-12.3f%-12.3f%d\n",
				tilemapping.MachineLetter(i).UserVisible(alph, false),
				100.0*float64(mlcts[tilemapping.MachineLetter(i)])/float64(totalCt), // normalize
				100.0*float64(bagmap[i])/float64(inbag),
				bagmap[i])
		}

		for i := 0; i < int(alph.NumLetters()); i++ {
			printLetterStats(i)
		}
		fmt.Fprintf(&ss, "\nSimmed %d times\n", r.simCount.Load())

		return ss.String()
	}

	// Summary (bins) mode
	bins := [8][]tilemapping.MachineLetter{}

	for i := 0; i < int(alph.NumLetters()); i++ {
		found := float64(mlcts[tilemapping.MachineLetter(i)]) / float64(totalCt)
		expected := float64(bagmap[i]) / float64(inbag)
		if expected == 0 {
			bins[7] = append(bins[7], tilemapping.MachineLetter(i))
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
		bins[bin] = append(bins[bin], tilemapping.MachineLetter(i))
	}

	var ss strings.Builder
	ss.WriteString(headerLine)
	ss.WriteString("\n")

	printTiles := func(tiles []tilemapping.MachineLetter) {
		for _, t := range tiles {
			fmt.Fprintf(&ss, " %s ", t.UserVisible(alph, false))
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

