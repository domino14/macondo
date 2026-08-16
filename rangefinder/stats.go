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
			srcTag = fmt.Sprintf(" [MEASURED ×%d, %s]", ml.count, roundLabel(ml.round))
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
		how := "drawn at random from the unseen pool in round 0"
		if ml.round > 0 {
			how = fmt.Sprintf("drawn from the imputed posterior in refine round %d", ml.round)
		}
		fmt.Fprintf(&ss, "  measured: evaluated %d time(s), mean likelihood %.6g (%s)\n",
			ml.count, ml.mean(), how)
		if r.imputeRes != nil && ml.mean() > 0 {
			fmt.Fprintf(&ss, "  weight = prior × mean likelihood ÷ max = %.4g × %.4g ÷ %.4g = %.6g\n",
				prior, ml.mean(), math.Exp(r.imputeRes.maxLogW),
				math.Exp(math.Log(prior)+math.Log(ml.mean())-r.imputeRes.maxLogW))
			if ml.predicted > 0 {
				// The out-of-sample prediction that got this leave selected,
				// made before it was measured. The final model below has
				// since been fit on this very measurement, so its comparison
				// is in-sample; this one is not.
				fmt.Fprintf(&ss, "  the model predicted %.6g before measuring it — measured/predicted = %.4gx\n",
					ml.predicted, ml.mean()/ml.predicted)
			}
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

// roundLabel names the round a leave was measured in: r0 is the prior-sampled
// pass (or exhaustive enumeration), r1 and up are the refinement rounds that
// drew from the imputed posterior.
func roundLabel(round int) string {
	return fmt.Sprintf("r%d", round)
}

// rankedRack is one row of the posterior ranking.
type rankedRack struct {
	rank     int // position in the full posterior, not in a filtered view
	leave    []tilemapping.MachineLetter
	weight   float64
	pct      float64 // share of total posterior weight
	measured bool
	count    int
	round    int
}

func (rr rankedRack) source() string {
	if !rr.measured {
		return "imputed"
	}
	return fmt.Sprintf("measured ×%d %s", rr.count, roundLabel(rr.round))
}

// rankedRacks sorts the whole posterior by weight and tags each leave with
// where its likelihood came from. Ranks are assigned over the full list, so
// they stay meaningful after filtering.
func (r *RangeFinder) rankedRacks() []rankedRack {
	sumW := 0.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
	}
	ranked := make([]montecarlo.InferredRack, len(r.inference.InferredRacks))
	copy(ranked, r.inference.InferredRacks)
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].Weight > ranked[j].Weight })

	out := make([]rankedRack, len(ranked))
	for i, ir := range ranked {
		row := rankedRack{rank: i + 1, leave: ir.Leave, weight: ir.Weight}
		if sumW > 0 {
			row.pct = 100.0 * ir.Weight / sumW
		}
		if ml, ok := r.measured[leaveKey(ir.Leave)]; ok && ml.count > 0 {
			row.measured, row.count, row.round = true, ml.count, ml.round
		}
		out[i] = row
	}
	return out
}

// RankedRacks renders the posterior ranking, optionally filtered to just the
// measured or just the imputed leaves. n ≤ 0 means the default page size.
// Ranks shown are positions in the full posterior, so a filtered view still
// says where its rows sit overall.
func (r *RangeFinder) RankedRacks(filter string, n int) (string, error) {
	switch filter {
	case "", "all", "measured", "imputed":
	default:
		return "", fmt.Errorf("unknown filter %q: use measured, imputed, or nothing for all", filter)
	}
	if len(r.inference.InferredRacks) == 0 {
		return "No inferences. Run `infer` first.", nil
	}
	if n <= 0 {
		n = 15
	}
	alph := r.origGame.Alphabet()
	rows := r.rankedRacks()

	var ss strings.Builder
	label := "leaves"
	if filter != "" && filter != "all" {
		label = filter + " leaves"
	}
	fmt.Fprintf(&ss, "Posterior ranking — top %d %s of %d:\n", n, label, len(rows))
	fmt.Fprintf(&ss, "  %-6s%-12s%-12s%-12s%-14s\n", "Rank", "Leave", "Weight", "Wt %", "Source")

	shown, cumPct := 0, 0.0
	for _, row := range rows {
		if (filter == "measured" && !row.measured) || (filter == "imputed" && row.measured) {
			continue
		}
		fmt.Fprintf(&ss, "  %-6d%-12s%-12.4f%-12.1f%-14s\n", row.rank,
			tilemapping.MachineWord(row.leave).UserVisible(alph),
			row.weight, row.pct, row.source())
		cumPct += row.pct
		shown++
		if shown >= n {
			break
		}
	}
	if shown == 0 {
		fmt.Fprintf(&ss, "  (none)\n")
		return ss.String(), nil
	}
	fmt.Fprintf(&ss, "  these %d hold %.1f%% of the posterior\n", shown, cumPct)
	return ss.String(), nil
}

// imputedSummary finds the highest-weight never-evaluated leave and totals the
// imputed set's share of the posterior. first is nil when every leave was
// directly evaluated.
func imputedSummary(rows []rankedRack) (first *rankedRack, count int, pct float64) {
	for i := range rows {
		if rows[i].measured {
			continue
		}
		if first == nil {
			first = &rows[i]
		}
		pct += rows[i].pct
		count++
	}
	return first, count, pct
}

// firstImputedLine reports where the highest-weight never-evaluated leave sits
// in the ranking, and how much mass the imputed set holds in total. After a
// refine run the top of the table is usually all measured — this says where
// the model is still guessing and how much that guess is worth. Returns "" when
// the space was evaluated exhaustively and nothing was imputed at all.
func (r *RangeFinder) firstImputedLine(rows []rankedRack) string {
	first, count, pct := imputedSummary(rows)
	if first == nil {
		return ""
	}
	return fmt.Sprintf(
		"  highest-weight imputed leave: %s at rank %d of %d (%.2f%% of mass); %d imputed leaves hold %.1f%% in total\n",
		tilemapping.MachineWord(first.leave).UserVisible(r.origGame.Alphabet()),
		first.rank, len(rows), first.pct, count, pct)
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
		refined := ""
		if r.refinedCount > 0 {
			// How the refine loop ended, and how close the model's imputed
			// weights were to the evaluations in its final round.
			last := r.roundLog[len(r.roundLog)-1]
			outcome := "budget"
			if last.converged {
				outcome = "converged"
			}
			// The batch size belongs with the ratio: a round of one or two
			// leaves fits R̂ almost exactly and reports a near-zero standard
			// error, which would otherwise read as a confident verdict rather
			// than the non-result it is.
			verdict := fmt.Sprintf("log R̂=%+.3f ±%.3f over %d",
				last.logRatio, last.seLogRatio, last.evaluated)
			if last.evaluated < refineMinBatch {
				noun := "leaves"
				if last.evaluated == 1 {
					noun = "leaf"
				}
				verdict = fmt.Sprintf("last round only %d %s, too few to test calibration",
					last.evaluated, noun)
			}
			refined = fmt.Sprintf(", %d refined over %d round(s), %s (%s)",
				r.refinedCount, len(r.roundLog), outcome, verdict)
		}
		headerLine = fmt.Sprintf(
			"Complete posterior over %d leaves (%s): %d measured holding %.1f%% of mass, %d imputed (marginal order ≤%d), tau=%.3f, ESS=%.1f%s%s\n",
			nInferred, src, r.imputeRes.measuredLeaves, 100.0*r.imputeRes.measuredMass,
			r.imputeRes.imputedLeaves, r.imputeRes.marginalOrder, r.Tau(), ess, xfit, refined)
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
		ranked := r.rankedRacks()

		showSource := r.inference.Complete
		if showSource {
			fmt.Fprintf(&ss, "  %-6s%-12s%-12s%-12s%-14s\n", "Rank", "Leave", "Weight", "Wt %", "Source")
		} else {
			fmt.Fprintf(&ss, "  %-6s%-12s%-12s%-12s\n", "Rank", "Leave", "Weight", "Wt %")
		}

		if showSource && r.refinedCount > 0 {
			ss.WriteString("  (r0 = prior-sampled exploration; r1+ = drawn from the imputed posterior)\n")
		}

		showN := min(len(ranked), 15)
		for i := 0; i < showN; i++ {
			row := ranked[i]
			leaveStr := tilemapping.MachineWord(row.leave).UserVisible(alph)
			if showSource {
				fmt.Fprintf(&ss, "  %-6d%-12s%-12.4f%-12.1f%-14s\n", row.rank, leaveStr,
					row.weight, row.pct, row.source())
			} else {
				fmt.Fprintf(&ss, "  %-6d%-12s%-12.4f%-12.1f\n", row.rank, leaveStr,
					row.weight, row.pct)
			}
		}
		if len(ranked) > showN {
			fmt.Fprintf(&ss, "  ... and %d more (`infer ranks [n] [measured|imputed]` to browse)\n",
				len(ranked)-showN)
		}
		if showSource {
			// After refinement the visible table is usually all measured, so
			// say where the model is still guessing.
			ss.WriteString(r.firstImputedLine(ranked))
		}

		// Weight concentration summary
		topN := min(len(ranked), 3)
		topSum := 0.0
		for i := 0; i < topN; i++ {
			topSum += ranked[i].weight
		}
		topPct := 100.0 * topSum / sumW
		fmt.Fprintf(&ss, "\nWeight concentration: top %d hold %.1f%% of total weight (ESS = %.1f of %d)\n",
			topN, topPct, ess, nInferred)
		if ess < 3 && nInferred >= 5 {
			ss.WriteString("  Note: low ESS means weights are dominated by a few racks.\n")
		}

		ss.WriteString("\n")
		// "Holds" and "By chance" are per-rack probabilities; "Tile share"
		// and "Pool share" are per-tile shares. Keeping the distinction in
		// the headers matters: a share of the posterior's tiles reads exactly
		// like a probability and is not one. With a 3-tile leave the game's
		// only Z can be 23.6% of the tiles in the posterior and still be in
		// 70.9% of the racks, because each rack it is in holds two other
		// tiles alongside it.
		ss.WriteString("Holds: the chance they are holding at least one, against By chance for a\n" +
			"random rack. Tile share: this tile's share of all the tiles across the posterior's\n" +
			"racks; Pool share: its share of the unseen tiles. The summary view bins the ratio\n" +
			"of those last two.\n\n")
		fmt.Fprintf(&ss, "%-5s%-11s%-12s%-13s%-12s%-10s\n",
			"Tile", "Holds %", "By chance", "Tile share", "Pool share", "# unseen")

		for _, t := range r.tileDeviations() {
			fmt.Fprintf(&ss, "%-5s%-11.2f%-12.2f%-13.3f%-12.3f%d\n",
				t.Tile, t.HoldsPct, t.ChanceHoldsPct, t.FoundPct, t.ExpectedPct, t.Unseen)
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

	// Summary mode: the read as a graph, banded on the chance they hold each
	// tile rather than on its ratio to chance. The ratio hid the size of a
	// read - the game's last Z going from 4% to 71% and a tile going from 1%
	// to 3% both came out as "way more than chance".
	var ss strings.Builder
	ss.WriteString(headerLine)
	ss.WriteString("\n")
	ss.WriteString(HoldingGraph(r.tileDeviations()))
	return ss.String()
}

