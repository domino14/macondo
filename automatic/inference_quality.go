package automatic

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/domino14/word-golib/cache"

	"github.com/domino14/macondo/stats"
)

// Grading inference over a whole run.
//
// The per-turn log records, for every inference, the probability the posterior
// placed on the leave the opponent actually held and the probability the tile
// counts alone would have given it. This reads those columns back and reduces
// them to the numbers worth comparing between configurations.
//
// LiftBits is the one to watch: log2(posterior/prior) per inference, so a mean
// above zero says inference learned something about the opponent's rack beyond
// counting tiles, and a mean at zero says it did not. It is a log score, so
// hedging cannot inflate it -- claiming a flat prior scores exactly zero.

// InferenceQuality summarizes how well a run's inferences matched the truth.
type InferenceQuality struct {
	// Inferences is how many turns produced a graded posterior.
	Inferences int
	// Unscored is how many turns ran inference but had nothing to grade: no
	// history yet, an empty bag, or a bingo that left no tiles to infer.
	Unscored int

	// LiftBits is log2(posterior/prior) per inference. Its mean is the headline.
	LiftBits *stats.Statistic
	// Posterior and Prior are the raw probabilities, kept for context since the
	// lift hides how small both are in a large leave space.
	Posterior, Prior *stats.Statistic
	// RankPct is the true leave's percentile among candidates, 0 being the top.
	// Robust to leave-space size in a way the raw rank is not.
	RankPct *stats.Statistic
	// Leaves is the size of the posterior per inference.
	Leaves *stats.Statistic

	// Better and Worse count inferences that beat and lost to the prior.
	Better, Worse int
	// RuledOut counts inferences that gave the true leave no weight at all --
	// the expensive mistake, since the simmer then never samples the truth.
	RuledOut int
	// Measured counts inferences where the true leave got a real mini-sim rather
	// than an imputed weight.
	Measured int

	// ByLeaveLen breaks the mean lift down by how many tiles were being inferred,
	// which is where inference gets hard.
	ByLeaveLen map[int]*stats.Statistic

	// lifts keeps every lift so the report can quote a median and the tails. A
	// log score has a long left tail -- one badly wrong inference can outweigh a
	// dozen good ones -- so the mean alone hides the typical case.
	lifts []float64
}

// LiftPercentile returns the p-th percentile of the per-inference lift, p in
// [0,100]. NaN when nothing was graded.
func (q *InferenceQuality) LiftPercentile(p float64) float64 {
	if len(q.lifts) == 0 {
		return math.NaN()
	}
	sorted := append([]float64(nil), q.lifts...)
	sort.Float64s(sorted)
	i := int(p / 100 * float64(len(sorted)-1))
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

// AnalyzeInferenceQuality reads a per-turn autoplay log and grades its
// inferences. It wants the per-turn log, not the per-game one.
func AnalyzeInferenceQuality(turnFile string) (*InferenceQuality, error) {
	file, _, err := cache.Open(turnFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	q := &InferenceQuality{
		LiftBits:   &stats.Statistic{},
		Posterior:  &stats.Statistic{},
		Prior:      &stats.Statistic{},
		RankPct:    &stats.Statistic{},
		Leaves:     &stats.Statistic{},
		ByLeaveLen: map[int]*stats.Statistic{},
	}

	r := csv.NewReader(file)
	// Every row of a run with an inferring bot is the same width, the columns
	// being empty on rows the other bot played, so the default check that rows
	// match the header is worth keeping: it would catch the log going ragged.
	cols := map[string]int{}
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if record[0] == "gameID" {
			return nil, fmt.Errorf(
				"this is a per-game log; pass the per-turn log (the one without the games- prefix)")
		}
		if record[0] == "playerID" {
			for i, name := range record {
				cols[name] = i
			}
			if _, ok := cols["liftBits"]; !ok {
				return nil, fmt.Errorf(
					"%s has no inference columns; it was not produced by a run with an inferring bot",
					turnFile)
			}
			continue
		}
		if len(cols) == 0 {
			continue
		}
		get := func(name string) string {
			i, ok := cols[name]
			if !ok || i >= len(record) {
				return ""
			}
			return record[i]
		}
		if get("inferCount") == "" {
			continue // a row from the bot that does not infer
		}
		lift := get("liftBits")
		if lift == "" || get("truePost") == "" {
			q.Unscored++
			continue
		}
		liftV, err1 := strconv.ParseFloat(lift, 64)
		post, err2 := strconv.ParseFloat(get("truePost"), 64)
		prior, err3 := strconv.ParseFloat(get("truePrior"), 64)
		rank, err4 := strconv.Atoi(get("trueRank"))
		leaves, err5 := strconv.Atoi(get("inferLeaves"))
		if err2 != nil || err3 != nil || err4 != nil || err5 != nil || leaves <= 0 {
			q.Unscored++
			continue
		}
		if post == 0 || err1 != nil || math.IsNaN(liftV) {
			// The posterior gave the truth no weight, so there is no ratio to
			// take. Counted, but left out of the averages rather than dragging
			// them to negative infinity.
			q.RuledOut++
			q.Inferences++
			continue
		}

		q.Inferences++
		q.LiftBits.Push(liftV)
		q.lifts = append(q.lifts, liftV)
		q.Posterior.Push(post)
		q.Prior.Push(prior)
		q.Leaves.Push(float64(leaves))
		if leaves > 1 {
			q.RankPct.Push(100 * float64(rank-1) / float64(leaves-1))
		}
		if liftV > 0 {
			q.Better++
		} else if liftV < 0 {
			q.Worse++
		}
		if get("trueMeasured") == "true" {
			q.Measured++
		}
		if n := len([]rune(get("trueLeave"))); n > 0 {
			if _, ok := q.ByLeaveLen[n]; !ok {
				q.ByLeaveLen[n] = &stats.Statistic{}
			}
			q.ByLeaveLen[n].Push(liftV)
		}
	}
	return q, nil
}

// FormatInferenceQuality renders the grading for the shell.
func FormatInferenceQuality(q *InferenceQuality) string {
	var b strings.Builder
	if q.Inferences == 0 {
		return "No graded inferences in this log.\n"
	}
	scored := q.LiftBits.Iterations()

	fmt.Fprintf(&b, "Inferences: %d graded", q.Inferences)
	if q.Unscored > 0 {
		fmt.Fprintf(&b, ", %d with nothing to grade", q.Unscored)
	}
	fmt.Fprintf(&b, "\n\n")

	fmt.Fprintf(&b, "  Information about the true leave: %+.3f bits per inference (± %.3f)\n",
		q.LiftBits.Mean(), stats.Z95*q.LiftBits.StandardError())
	fmt.Fprintf(&b, "    median %+.3f, worst tenth %+.3f, best tenth %+.3f\n",
		q.LiftPercentile(50), q.LiftPercentile(10), q.LiftPercentile(90))
	fmt.Fprintf(&b, "    beat the prior %d of %d times (%.1f%%)\n",
		q.Better, scored, 100*float64(q.Better)/float64(scored))
	// Always printed, including the zero. This is the failure that can put a bot
	// below no-inference at all -- the simmer never samples a leave with no
	// weight -- so "checked, none" is worth saying out loud rather than leaving
	// as a missing line.
	fmt.Fprintf(&b, "    ruled the true leave out entirely: %d of %d\n",
		q.RuledOut, q.Inferences)

	fmt.Fprintf(&b, "\n  P(true leave):  posterior %.5g   prior %.5g   ratio %.2fx\n",
		q.Posterior.Mean(), q.Prior.Mean(), q.Posterior.Mean()/q.Prior.Mean())
	if q.RankPct.Iterations() > 0 {
		fmt.Fprintf(&b, "  True leave's rank: %.1f%% from the top on average, of %.0f candidates\n",
			q.RankPct.Mean(), q.Leaves.Mean())
	}
	fmt.Fprintf(&b, "  Directly measured rather than imputed: %d of %d (%.1f%%)\n",
		q.Measured, scored, 100*float64(q.Measured)/float64(scored))

	if len(q.ByLeaveLen) > 1 {
		lens := make([]int, 0, len(q.ByLeaveLen))
		for n := range q.ByLeaveLen {
			lens = append(lens, n)
		}
		sort.Ints(lens)
		b.WriteString("\n  By number of tiles inferred:\n")
		for _, n := range lens {
			s := q.ByLeaveLen[n]
			fmt.Fprintf(&b, "    %d tile(s): %+.3f bits over %d inference(s)\n",
				n, s.Mean(), s.Iterations())
		}
	}

	b.WriteString("\n  (bits = log2(posterior/prior) for the leave the opponent really held;\n" +
		"   0 means inference did no better than counting the unseen tiles)\n")
	return b.String()
}
