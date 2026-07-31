package rangefinder

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/montecarlo"
)

// AnalyzeLeave reports the posterior details of one specific leave: weight,
// rank, posterior share, hypergeometric prior, the posterior/prior lift, and
// whether the leave was directly measured (with how many evaluations) or
// imputed from marginal lifts.
func (r *RangeFinder) AnalyzeLeave(leaveStr string) (string, error) {
	if r.inference == nil || len(r.inference.InferredRacks) == 0 {
		return "", errors.New("no inference available; run `infer` first")
	}
	alph := r.origGame.Alphabet()
	// Uppercase the input: in tile notation lowercase means a blank
	// designated as that letter, but a kept blank in a leave is just "?",
	// so treat lowercase as the regular tile.
	mls, err := tilemapping.ToMachineLetters(strings.ToUpper(leaveStr), alph)
	if err != nil {
		return "", err
	}
	for _, ml := range mls {
		if int(ml) >= len(r.inferenceBagMap) {
			return "", fmt.Errorf("tile %s cannot appear in a leave",
				ml.UserVisible(alph, false))
		}
	}
	if len(mls) != r.inference.RackLength {
		return "", fmt.Errorf("leave %s has %d tiles; inferred leaves have %d",
			leaveStr, len(mls), r.inference.RackLength)
	}
	key := leaveKey(mls)
	display := tilemapping.MachineWord(mls).UserVisible(alph)

	// Find the leave, its rank, and the total weight.
	sumW := 0.0
	weight := -1.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
		if weight < 0 && leaveKey(ir.Leave) == key {
			weight = ir.Weight
		}
	}
	prior := combinatorialPrior(mls, r.inferenceBagMap)

	if weight < 0 {
		if prior == 0 {
			return fmt.Sprintf("Leave %s is impossible: its tiles are not all available in the unseen pool.\n", display), nil
		}
		if ml, ok := r.measured[key]; ok && ml.count > 0 && ml.sumW <= 0 {
			return fmt.Sprintf("Leave %s: measured %d time(s) with likelihood 0 — the observed play is never chosen with this leave, so it was excluded from the posterior.\n",
				display, ml.count), nil
		}
		return fmt.Sprintf("Leave %s is not in the posterior (weight below the negligible-mass cutoff).\n", display), nil
	}

	rank := 1
	for _, ir := range r.inference.InferredRacks {
		if ir.Weight > weight {
			rank++
		}
	}

	postPct := 100.0 * weight / sumW
	var ss strings.Builder
	// Lead with the leave's provenance so the reader knows up front whether
	// the weight below comes from direct evaluation or the marginal model.
	srcTag := ""
	if r.inference.Complete {
		if ml, ok := r.measured[key]; ok && ml.count > 0 {
			srcTag = fmt.Sprintf(" [MEASURED ×%d]", ml.count)
		} else {
			srcTag = " [IMPUTED]"
		}
	}
	fmt.Fprintf(&ss, "Leave %s%s: weight %.6g, posterior %.6g%% (rank %d of %d)\n",
		display, srcTag, weight, postPct, rank, len(r.inference.InferredRacks))
	if prior > 0 {
		fmt.Fprintf(&ss, "  prior %.6g%%, posterior/prior lift %.6gx\n",
			100.0*prior, (weight/sumW)/prior)
	}
	if ml, ok := r.measured[key]; ok && ml.count > 0 {
		fmt.Fprintf(&ss, "  measured: evaluated %d time(s), mean likelihood %.6g\n",
			ml.count, ml.mean())
		if r.imputeRes != nil && ml.mean() > 0 {
			fmt.Fprintf(&ss, "  weight = prior × mean likelihood ÷ max = %.4g × %.4g ÷ %.4g = %.6g\n",
				prior, ml.mean(), math.Exp(r.imputeRes.maxLogW),
				math.Exp(math.Log(prior)+math.Log(ml.mean())-r.imputeRes.maxLogW))
			fmt.Fprintf(&ss, "  imputation model comparison (not used for this leave's weight):\n")
			if lhat, ok := r.writeSubleaveTable(&ss, mls); ok {
				fmt.Fprintf(&ss, "  measured/imputed likelihood ratio: %.4gx\n",
					ml.mean()/lhat)
			}
		}
	} else if r.inference.Complete {
		fmt.Fprintf(&ss, "  imputed: never directly evaluated; likelihood estimated from marginal lifts (order ≤%d)\n",
			r.imputeRes.marginalOrder)
		if lhat, ok := r.writeSubleaveTable(&ss, mls); ok && prior > 0 {
			fmt.Fprintf(&ss, "  weight = prior × ℓ̂ ÷ max = %.4g × %.4g ÷ %.4g = %.6g\n",
				prior, lhat, math.Exp(r.imputeRes.maxLogW),
				math.Exp(math.Log(prior)+math.Log(lhat)-r.imputeRes.maxLogW))
		}
	}
	return ss.String(), nil
}

// writeSubleaveTable prints every sub-multiset φ term of the leave with its
// underlying containment stats, then Σφ, the calibration constant, and the
// model's imputed likelihood ℓ̂, which it returns. ok is false (nothing
// printed) when there is no imputation model to consult.
func (r *RangeFinder) writeSubleaveTable(ss *strings.Builder,
	mls []tilemapping.MachineLetter) (lhat float64, ok bool) {

	if r.imputeRes == nil || r.imputeRes.model == nil || r.acc == nil ||
		r.acc.n == 0 || r.acc.likTotal <= 0 {
		return 0, false
	}
	alph := r.origGame.Alphabet()
	mod := r.imputeRes.model
	sorted := make([]tilemapping.MachineLetter, len(mls))
	copy(sorted, mls)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	runs := runsOf(sorted, nil)
	terms := mod.subleaveTerms(runs)
	wMean := r.acc.likTotal / r.acc.wtTotal

	// "ess" is the effective sample size behind the term — the plain
	// containment count when every draw was prior-sampled (u = 1), smaller
	// when the draws carried unequal importance weights.
	fmt.Fprintf(ss, "  %-6s%-8s%-13s%-11s%-9s%-11s%-9s\n",
		"sub", "ess", "E[lik|sub]", "lift", "shrink", "φ", "e^φ")
	sumPhi := 0.0
	for _, t := range terms {
		sumPhi += t.phi
		sub := tilemapping.MachineWord(t.tiles).UserVisible(alph)
		e, wt, lik := r.acc.accStats(t.tiles)
		if wt == 0 {
			fmt.Fprintf(ss, "  %-6s%-8s%-13s%-11s%-9s%-11s%-9s (no data)\n",
				sub, "0", "—", "—", "—", "0", "1")
			continue
		}
		condMean := lik / wt
		fmt.Fprintf(ss, "  %-6s%-8.4g%-13.4g%-11.4g%-9.3f%-+11.4g%-9.4g\n",
			sub, e, condMean, condMean/wMean, e/(e+mod.lambda),
			t.phi, math.Exp(t.phi))
	}
	logCalib := r.imputeRes.logCalib
	if r.imputeRes.crossFitted {
		fmt.Fprintf(ss, "  Σφ = %.4g, calibration = %.4g (cross-fit over %d folds; in-sample %.4g, ratio %.3gx)\n",
			sumPhi, logCalib, calibrationFolds, r.imputeRes.logCalibInSample,
			math.Exp(logCalib-r.imputeRes.logCalibInSample))
	} else {
		fmt.Fprintf(ss, "  Σφ = %.4g, calibration = %.4g\n", sumPhi, logCalib)
	}
	lhat = math.Exp(logCalib + sumPhi)
	fmt.Fprintf(ss, "  imputed likelihood ℓ̂ = exp(calibration + Σφ) = %.6g\n", lhat)
	return lhat, true
}

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

	var headerLine string
	if r.inference.Complete && r.imputeRes != nil {
		// Complete-posterior mode (tile placements): every feasible leave
		// has a weight — evaluated leaves keep their measured likelihood,
		// the rest are imputed from marginal lifts.
		src := "MC sampling"
		if r.exhaustiveTotal > 0 {
			src = "enumeration"
		}
		xfit := ""
		if r.imputeRes.crossFitted {
			// How much the in-sample constant was depressed by the lifts
			// fitting their own samples: 1.0x means no detectable overfit.
			xfit = fmt.Sprintf(", xfit=%.2fx",
				math.Exp(r.imputeRes.logCalib-r.imputeRes.logCalibInSample))
		}
		headerLine = fmt.Sprintf(
			"Complete posterior over %d leaves (%s): %d measured holding %.1f%% of mass, %d imputed (marginal order ≤%d), tau=%.3f, ESS=%.1f%s\n",
			nInferred, src, r.imputeRes.measuredLeaves, 100.0*r.imputeRes.measuredMass,
			r.imputeRes.imputedLeaves, r.imputeRes.marginalOrder, r.Tau(), ess, xfit)
	} else if r.exhaustiveTotal > 0 {
		// Enumeration mode: show leaves simmed vs total, and completion %.
		pct := 100.0 * float64(nInferred) / float64(r.exhaustiveTotal)
		complete := "complete"
		if nInferred < r.exhaustiveTotal {
			complete = "incomplete — context deadline"
		}
		headerLine = fmt.Sprintf("Inferred %d of %d leaves (%.1f%%, %s), tau=%.3f, ESS=%.1f\n",
			nInferred, r.exhaustiveTotal, pct, complete, r.Tau(), ess)
	} else {
		// Monte Carlo sampling mode (exchanges, or no posterior was built).
		iterations := r.iterationCount
		acceptRate := 0.0
		if iterations > 0 {
			acceptRate = 100.0 * float64(nInferred) / float64(iterations)
		}
		headerLine = fmt.Sprintf("Inferred %d unique racks from %d iterations (%.1f%% acceptance), tau=%.3f, ESS=%.1f\n",
			nInferred, iterations, acceptRate, r.Tau(), ess)
	}

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

		showSource := r.inference.Complete
		if showSource {
			fmt.Fprintf(&ss, "  %-6s%-12s%-12s%-12s%-14s\n", "Rank", "Leave", "Weight", "Wt %", "Source")
		} else {
			fmt.Fprintf(&ss, "  %-6s%-12s%-12s%-12s\n", "Rank", "Leave", "Weight", "Wt %")
		}

		showN := min(len(ranked), 15)
		for i := 0; i < showN; i++ {
			ir := ranked[i]
			leaveStr := tilemapping.MachineWord(ir.Leave).UserVisible(alph)
			wtPct := 100.0 * ir.Weight / sumW
			if showSource {
				src := "imputed"
				if ml, ok := r.measured[leaveKey(ir.Leave)]; ok && ml.count > 0 {
					src = fmt.Sprintf("measured ×%d", ml.count)
				}
				fmt.Fprintf(&ss, "  %-6d%-12s%-12.4f%-12.1f%-14s\n", i+1, leaveStr, ir.Weight, wtPct, src)
			} else {
				fmt.Fprintf(&ss, "  %-6d%-12s%-12.4f%-12.1f\n", i+1, leaveStr, ir.Weight, wtPct)
			}
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
		simCount := r.simCount.Load()
		elapsed := r.inferElapsed
		simsPerSec := 0.0
		if elapsed.Seconds() > 0 {
			simsPerSec = float64(simCount) / elapsed.Seconds()
		}
		fmt.Fprintf(&ss, "\nSimmed %d times in %.1fs (%.1f sims/sec)\n",
			simCount, elapsed.Seconds(), simsPerSec)

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

