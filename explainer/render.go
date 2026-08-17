package explainer

import (
	"fmt"
	"sort"
	"strings"

	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/rangefinder"
)

// Rendering the fact pack for the model. Everything here is derived from
// PositionFacts, so the tables and the tool answers can never disagree - which
// is exactly what happened when the tools parsed the tables back out of the
// prompt.

// promptTables is what the model gets, as opposed to what `sim playstats`
// shows a person. The opponent's highest-scoring replies are left out
// entirely: they are the tail of the distribution, each one a fraction of a
// percent, and reading them next to real percentages invites the model to
// treat a one-in-five-hundred bingo as a threat. Their frequent replies are
// cut to the handful that actually recur. Our own follow-ups stay long,
// because that is where the plays worth building a turn around live.
var promptTables = stats.RenderOptions{
	OppTopScoring: false,
	OppReplies:    5,
	Followups:     0, // everything computed
}

// Ask is the question the explanation has to answer, which is a different
// question when the reader has a play of their own on the table.
func (f *PositionFacts) Ask() string {
	c := f.Comparison
	if c == nil || c.Rival == nil {
		return "Your explanation for why " + f.Best.Play + " is best:"
	}
	whose := "they asked about"
	if c.FromHistory {
		whose = "they played"
	}
	if c.WasBest {
		return fmt.Sprintf("Your explanation for why %s, the play %s, is the right one here, "+
			"and what it beat:", c.Play, whose)
	}
	return fmt.Sprintf("Your explanation for why %s beats %s, the play %s:",
		f.Best.Play, c.Rival.Play, whose)
}

// Render writes the position, the candidates, and what the simulation saw
// happen next.
func (f *PositionFacts) Render() string {
	var ss strings.Builder
	ss.WriteString(f.renderPosition())
	ss.WriteString("\n")
	ss.WriteString(f.renderCandidates())
	ss.WriteString("\n")
	ss.WriteString(f.renderPlies())
	if s := f.renderFollowupVerdicts(); s != "" {
		ss.WriteString("\n")
		ss.WriteString(s)
	}
	if s := f.renderComparison(); s != "" {
		ss.WriteString("\n")
		ss.WriteString(s)
	}
	if f.PlayStats != nil {
		ss.WriteString("\n### What the simulation saw after " + f.Best.Play + "\n\n")
		ss.WriteString(f.PlayStats.RenderWith(promptTables))
	}
	if c := f.Comparison; c != nil && c.RivalPlayStats != nil {
		ss.WriteString("\n### What the simulation saw after " + c.Rival.Play + "\n\n")
		ss.WriteString(c.RivalPlayStats.RenderWith(promptTables))
	}
	if s := f.renderBoardDynamics(); s != "" {
		ss.WriteString("\n")
		ss.WriteString(s)
	}
	if s := f.renderInference(); s != "" {
		ss.WriteString("\n")
		ss.WriteString(s)
	}
	return ss.String()
}

// renderInference describes the read on the opponent's rack, and what having
// it changed. It writes nothing at all unless both are worth saying - the
// deciding is done in buildInference, against the numbers.
func (f *PositionFacts) renderInference() string {
	inf := f.Inference
	if inf == nil || !inf.Matters {
		return ""
	}
	s := inf.Summary
	var ss strings.Builder
	ss.WriteString("### What the opponent's last play gave away\n")
	fmt.Fprintf(&ss, "Their last play was run through an inference: every rack they could be "+
		"holding, weighted by how likely it is that a good player would have made that play "+
		"from it. %d leaves came out of it", s.NumRacks)
	if s.ESS > 0 {
		fmt.Fprintf(&ss, ", with the top three holding %.0f%% of the weight (effective sample size %.1f)",
			s.TopWeightPct, s.ESS)
	}
	ss.WriteString(".\n")

	if len(s.Racks) > 0 {
		ss.WriteString("\nMost likely racks:\n")
		for _, r := range s.Racks {
			fmt.Fprintf(&ss, "  %-10s %5.1f%% of the posterior\n", dashIfEmpty(r.Leave), r.Pct)
		}
	}

	// A read is sometimes about the makeup of the rack and not about any one
	// tile. When it is, no letter is worth naming and this is the entire
	// finding, so it comes first.
	if sh := inf.ShapeRead; sh != nil {
		fmt.Fprintf(&ss, "\nThe shape of their rack: %.2f consonants and %.2f vowels, where a "+
			"random rack from this pool holds %.2f and %.2f. They are %s by about %s.\n",
			sh.Consonants.Read, sh.Vowels.Read, sh.Consonants.Chance, sh.Vowels.Chance,
			shapeWord(sh), tileCount(abs(sh.Vowels.Diff())))
		if len(sh.VowelCount) > 0 {
			ss.WriteString("  vowels held    with the read   by chance\n")
			for k, pair := range sh.VowelCount {
				if pair.Read < 0.5 && pair.Chance < 0.5 {
					continue
				}
				fmt.Fprintf(&ss, "  %d                    %5.1f%%      %5.1f%%\n",
					k, pair.Read, pair.Chance)
			}
		}
	}

	if len(inf.Outliers) > 0 {
		ss.WriteString("\nWhat they are holding, against what a random rack would hold:\n")
		for _, t := range inf.Outliers {
			word := "more"
			if t.Tiles < 0 {
				word = "fewer"
			}
			fmt.Fprintf(&ss, "  %-4s holding one %.1f%% of the time, against %.1f%% by chance"+
				"  (%+.2f tiles %s than chance; %d unseen)\n",
				t.Tile, t.HoldsPct, t.ChanceHoldsPct, t.Tiles, word, t.Unseen)
		}
	}

	ss.WriteString("\nWhat the read changed:\n")
	ss.WriteString(f.renderReadEffect())
	return ss.String()
}

// shapeWord names which way the read went. Vowels decide it: the blank barely
// moves, so consonants are just vowels read backwards.
func shapeWord(sh *rangefinder.RackShape) string {
	if sh.Vowels.Diff() < 0 {
		return "consonant-heavy"
	}
	return "vowel-heavy"
}

// tileCount says a fraction of a tile the way a person would. "0.38 of a tile"
// is precise and unreadable; a third of a tile is what it means.
func tileCount(tiles float64) string {
	switch {
	case tiles < 0.4:
		return "a third of a tile"
	case tiles < 0.6:
		return "half a tile"
	case tiles < 0.9:
		return "three quarters of a tile"
	case tiles < 1.5:
		return "a whole tile"
	}
	return fmt.Sprintf("%.1f tiles", tiles)
}

// renderReadEffect says what believing the read does to the recommendation.
//
// When the top play changes, the interesting figures are the two plays' standings
// within each simulation - which one is ahead, and by how much - not either play's
// win% across them. The two sims assume different things about the opponent's rack,
// so a play's win% in one is not the same quantity as its win% in the other; the
// ordering within a single sim is what a player acts on.
func (f *PositionFacts) renderReadEffect() string {
	inf := f.Inference
	var ss strings.Builder

	if !inf.ChangedTopPlay || inf.BaselineBest == nil {
		fmt.Fprintf(&ss, "  The recommendation is %s either way.\n", f.Best.Play)
		if b := inf.BaselineOfBest; b != nil {
			fmt.Fprintf(&ss, "  Believing the read moves its win%% from %.2f%% to %.2f%% (%+.2f), "+
				"which is %s.\n", b.WinPct, f.Best.WinPct, inf.WinPctShift,
				establishedPhrase(inf))
		}
		return ss.String()
	}

	fmt.Fprintf(&ss, "  Without the read, %s was the best play. With it, %s is.\n",
		inf.BaselineBest.Play, f.Best.Play)

	// The same two plays as each simulation ranked them.
	withRival := findCandidate(f.Candidates, inf.BaselineBest.Play)
	withoutBest := inf.BaselineOfBest
	if withRival != nil && withoutBest != nil {
		fmt.Fprintf(&ss, "  Ignoring the read: %s %.2f%%, %s %.2f%% - %s ahead by %.2f.\n",
			inf.BaselineBest.Play, inf.BaselineBest.WinPct,
			f.Best.Play, withoutBest.WinPct,
			inf.BaselineBest.Play, inf.BaselineBest.WinPct-withoutBest.WinPct)
		fmt.Fprintf(&ss, "  Believing it:      %s %.2f%%, %s %.2f%% - %s ahead by %.2f.\n",
			f.Best.Play, f.Best.WinPct,
			inf.BaselineBest.Play, withRival.WinPct,
			f.Best.Play, f.Best.WinPct-withRival.WinPct)
		ss.WriteString("  Compare the two plays inside each simulation rather than one play across " +
			"them: the two runs assume different things about the opponent's rack, so the same " +
			"play's win% in each is not quite the same quantity.\n")
	}
	return ss.String()
}

func establishedPhrase(inf *InferenceFacts) string {
	switch {
	case inf.Decisive:
		return "outside both sims' confidence intervals and big enough to matter"
	case inf.Established:
		return "outside the confidence intervals, but small enough that the read is not what " +
			"makes this play right"
	default:
		return "inside the confidence intervals, so the read did not measurably change this " +
			"play's chances"
	}
}

func findCandidate(candidates []montecarlo.CandidateStats, play string) *montecarlo.CandidateStats {
	for i := range candidates {
		if candidates[i].Play == play {
			return &candidates[i]
		}
	}
	return nil
}

// renderComparison is the head-to-head. Everything here is arithmetic the
// model would otherwise have to do across two tables, which is exactly the
// kind of work it gets wrong.
func (f *PositionFacts) renderComparison() string {
	c := f.Comparison
	if c == nil || c.Rival == nil {
		return ""
	}
	var ss strings.Builder

	if c.WasBest {
		fmt.Fprintf(&ss, "### Head to head: %s is the top play\n", c.Play)
		fmt.Fprintf(&ss, "%s is the play %s, and the simulation ranks it first. "+
			"The comparison below is against the runner-up, %s.\n",
			c.Play, playSource(c), c.Rival.Play)
	} else {
		fmt.Fprintf(&ss, "### Head to head: %s versus %s\n", f.Best.Play, c.Rival.Play)
		fmt.Fprintf(&ss, "%s is the play %s. Figures below are %s minus %s, so a "+
			"positive number is the recommended play's advantage.\n",
			c.Rival.Play, playSource(c), f.Best.Play, c.Rival.Play)
	}

	d := c.Deltas
	if d.Established {
		fmt.Fprintf(&ss, "The win%% gap is established: the 99%% confidence intervals do not overlap.\n")
	} else {
		fmt.Fprintf(&ss, "The win%% gap is NOT established: the 99%% confidence intervals overlap, "+
			"so the simulation has not shown one of these plays to be better than the other.\n")
	}

	fmt.Fprintf(&ss, "%-26s %+.2f\n", "Win% difference", d.WinPct)
	fmt.Fprintf(&ss, "%-26s %+.2f\n", "Equity difference", d.Equity)
	fmt.Fprintf(&ss, "%-26s %+d\n", "Score difference", d.Score)
	fmt.Fprintf(&ss, "%-26s %+.2f  (%s vs %s)\n", "Leave value difference", d.LeaveValue,
		dashIfEmpty(f.Best.Leave), dashIfEmpty(c.Rival.Leave))
	fmt.Fprintf(&ss, "%-26s %+.2f mean, %+.2f bingo%%\n", "Opponent's reply", d.OppMeanScore, d.OppBingoPct)
	fmt.Fprintf(&ss, "%-26s %+.2f mean, %+.2f bingo%%\n", "Our next turn", d.OurMeanScore, d.OurBingoPct)
	fmt.Fprintf(&ss, "%-26s %.1f after %s, %.1f after %s\n", "Big follow-up chances",
		d.BestUpside, f.Best.Play, d.RivalUpside, c.Rival.Play)

	writeOpportunities(&ss, "Chances only in the sampled follow-ups after "+f.Best.Play, c.OnlyBest)
	writeOpportunities(&ss, "Chances only in the sampled follow-ups after "+c.Rival.Play, c.OnlyRival)

	// The rival's own chances, so "your play had nothing like this" is a
	// statement about data rather than an absence the model has to infer.
	if s := renderChances("Follow-up chances after "+c.Rival.Play+":",
		c.Rival, c.TypicalNextScore, c.RivalChances); s != "" {
		ss.WriteString(s)
	} else {
		fmt.Fprintf(&ss, "No follow-up after %s clears the bar for a big chance.\n", c.Rival.Play)
	}
	return ss.String()
}

func playSource(c *Comparison) string {
	if c.FromHistory {
		return "the reader actually made"
	}
	return "the reader asked about"
}

func writeOpportunities(ss *strings.Builder, heading string, fs []*FollowupFact) {
	if len(fs) == 0 {
		return
	}
	fmt.Fprintf(ss, "%s:\n", heading)
	for _, fu := range fs {
		kind := "big chance"
		if fu.IsSetup {
			kind = "setup"
		}
		fmt.Fprintf(ss, "  %-22s %8s pts  %5.2f%% of the time  upside %+.1f  (%s)\n",
			fu.Play, scoreRange(fu.MinScore, fu.MaxScore), fu.Pct, fu.Upside, kind)
	}
}

func (f *PositionFacts) renderPosition() string {
	var ss strings.Builder
	ss.WriteString("### Position\n")
	fmt.Fprintf(&ss, "Our rack: %s (%s). Phase: %s.\n", f.Rack, f.Lexicon, f.Phase)
	switch {
	case f.Spread == 0:
		ss.WriteString("The game is tied.\n")
	case f.Spread > 0:
		fmt.Fprintf(&ss, "We are ahead by %d points.\n", f.Spread)
	default:
		fmt.Fprintf(&ss, "We are behind by %d points.\n", -f.Spread)
	}
	fmt.Fprintf(&ss, "%d tiles are unseen to us: %d in the bag and %d on our opponent's rack.\n",
		f.UnseenCount, f.BagCount, f.OppRackSize)
	fmt.Fprintf(&ss, "Unseen tiles: %d vowels, %d consonants, %d blanks.\n",
		f.UnseenVowels, f.UnseenConsonants, f.UnseenBlanks)
	if len(f.UnseenPowerTiles) > 0 {
		fmt.Fprintf(&ss, "Unseen power tiles: %s.\n", strings.Join(f.UnseenPowerTiles, " "))
	}
	return ss.String()
}

func (f *PositionFacts) renderCandidates() string {
	var ss strings.Builder
	fmt.Fprintf(&ss, "### Candidate plays, best win%% first (%d iterations)\n", f.Iterations)
	ss.WriteString("Intervals are 99% confidence. ❌ marks plays the sim cut off early.\n")
	fmt.Fprintf(&ss, "%-20s %-12s %-6s %-15s %s\n", "Play", "Leave", "Score", "Win%", "Equity")
	for i := range f.Candidates {
		c := &f.Candidates[i]
		mark := ""
		if c.Ignored {
			mark = " ❌"
		}
		notes := []string{}
		if c.IsBingo {
			notes = append(notes, "bingo")
		}
		if c.UsesBlank {
			notes = append(notes, "spends a blank")
		}
		note := ""
		if len(notes) > 0 {
			note = "  (" + strings.Join(notes, ", ") + ")"
		}
		fmt.Fprintf(&ss, "%-20s %-12s %-6d %-15s %s%s%s\n",
			c.Play, dashIfEmpty(c.Leave), c.Score,
			fmt.Sprintf("%.2f±%.2f", c.WinPct, c.WinPctCI),
			fmt.Sprintf("%.2f±%.2f", c.Equity, c.EquityCI),
			note, mark)
	}
	if f.Flags["equity_sacrifice"] {
		fmt.Fprintf(&ss, "Note: %s has the highest equity but %s has the highest win%%.\n",
			f.BestByEquity.Play, f.Best.Play)
	}
	return ss.String()
}

// renderPlies shows only the next two plies. The simulation looks further than
// that, and the win% already accounts for it, but nobody can reason usefully
// about the mean score five plies out.
func (f *PositionFacts) renderPlies() string {
	var ss strings.Builder
	ss.WriteString("### Next two plies per candidate\n")
	for ply := 1; ply <= 2; ply++ {
		who := "opponent's reply"
		if ply == 2 {
			who = "our next turn"
		}
		fmt.Fprintf(&ss, "Ply %d (%s)\n", ply, who)
		fmt.Fprintf(&ss, "%-20s %-9s %-9s %s\n", "Play", "Mean", "Stdev", "Bingo%")
		for i := range f.Candidates {
			c := &f.Candidates[i]
			for _, p := range c.Plies {
				if p.Ply != ply {
					continue
				}
				fmt.Fprintf(&ss, "%-20s %-9.2f %-9.2f %.2f\n", c.Play, p.MeanScore, p.Stdev, p.BingoPct)
			}
		}
		ss.WriteString("\n")
	}
	return ss.String()
}

// renderFollowupVerdicts is the part that used to be the model's job: working
// out which follow-up plays are worth building a turn around. Both judgments
// here are ones it does badly - whether a play is a setup comes from replaying
// the position, and whether a chance is big comes from weighing its size
// against how often it happens, which a percentage on its own hides.
func (f *PositionFacts) renderFollowupVerdicts() string {
	return renderChances("### Follow-up chances after "+f.Best.Play+
		" (already checked against the board)", f.Best, f.TypicalNextScore, f.Chances)
}

func renderChances(heading string, play *montecarlo.CandidateStats,
	typical float64, chances []*FollowupCluster) string {

	worth := []*FollowupCluster{}
	for _, ch := range chances {
		if ch.Worthwhile() {
			worth = append(worth, ch)
		}
	}
	if len(worth) == 0 {
		return ""
	}
	// Biggest first: the ordering is itself a hint about what to lead with.
	sort.Slice(worth, func(i, j int) bool { return worth[i].Upside > worth[j].Upside })

	var ss strings.Builder
	ss.WriteString(heading + "\n")
	fmt.Fprintf(&ss, "An ordinary next turn here is worth about %.0f points. Upside is what a "+
		"chance is worth on top of that: how often it comes up times how much bigger it is.\n",
		typical)
	for _, ch := range worth {
		draw := "no draw needed"
		if d := drawList(ch.NeededDraws()); d != "" {
			draw = "needs " + d
		}
		notes := []string{}
		if ch.IsSetup {
			notes = append(notes, "SETUP: does not exist on the board until we play "+play.Play)
		} else {
			// Not a setup in the board sense, but still a consequence of this
			// play: every follow-up here is conditioned on the leave it left.
			notes = append(notes, "the board already allows it; the leave "+
				dashIfEmpty(play.Leave)+" plus the draw is what makes it available")
		}
		if ch.Requirement() == "requires opponent play" {
			notes = append(notes, "the opponent has to play something first for it to be legal")
		}
		fmt.Fprintf(&ss, "%-22s %8s pts  %5.2f%% of the time  upside %+.1f  (%s) - %s\n",
			ch.Label(), scoreRange(ch.MinScore, ch.MaxScore), ch.Pct, ch.Upside, draw,
			strings.Join(notes, "; "))
		// One hook reached four ways is one chance, but a coach still has to
		// name the play. Without this the reader is told about "(QUAD) in row
		// 2" and never learns that the 52-pointer is RUPLY.
		if ch.Grouped() {
			for _, p := range ch.Plays {
				fmt.Fprintf(&ss, "    %-20s %4d pts  %5.2f%% of the time\n",
					p.Play, int(p.AvgScore()+0.5), p.Pct)
			}
		}
	}
	return ss.String()
}

func (f *PositionFacts) renderBoardDynamics() string {
	if !f.Flags["has_lane_data"] {
		return ""
	}
	var ss strings.Builder
	ss.WriteString("### Board dynamics\n")
	ss.WriteString("Where the opponent's sampled replies actually landed, by lane. " +
		"These are the only lanes you may make positional claims about; do not " +
		"work out anything else about the geometry of the board.\n")

	for _, lc := range f.Lanes {
		label := lc.Play
		if lc.Best {
			label += " (the best play)"
		}
		fmt.Fprintf(&ss, "\nAfter %s - %d sampled replies", label, lc.Stats.Total)
		if lc.Stats.Total > 0 {
			fmt.Fprintf(&ss, ", %.0f%% of them one-tile plays",
				float64(lc.Stats.SingleTile*100)/float64(lc.Stats.Total))
		}
		ss.WriteString("\n")
		shown := 0
		for _, l := range lc.Stats.Lanes {
			if l.Pct < laneMinPct || shown >= lanesShown {
				break
			}
			shown++
			// Deliberately no single best reply per lane: the top play in a
			// lane is one sample out of thousands, and quoting it invites
			// treating a one-off as a threat. How often the lane is used and
			// what it pays on average is the part that generalizes.
			extra := ""
			if b := bonusLabel(l.Premiums); b != "" {
				extra = "  covers " + b
			}
			fmt.Fprintf(&ss, "  %-12s %5.1f%%  mean %5.1f%s\n",
				l.Label, l.Pct, l.MeanScore, extra)
		}
		if shown == 0 {
			ss.WriteString("  replies were scattered; no lane stands out\n")
		}
	}

	if diffs := f.laneDifferences(); len(diffs) > 0 {
		ss.WriteString("\nWhat the best play changes about the board:\n")
		for _, d := range diffs {
			ss.WriteString("  " + d + "\n")
		}
	}
	return ss.String()
}

// Lane percentage / mean-score gaps below these are noise, not a difference in
// what the play does to the board.
const (
	lanePctDiffMin  = 8.0
	laneMeanDiffMin = 6.0
)

// laneDifferences compares the best play against the other candidates lane by
// lane. This is the whole point of computing lanes for more than one play: a
// lane that is busy after every candidate isn't something this play opened.
//
// Each lane gets one line, against whichever other candidate it differs from
// most - the top candidates are often near-duplicates of each other, and
// saying the same thing once per near-duplicate is noise.
func (f *PositionFacts) laneDifferences() []string {
	if len(f.Lanes) < 2 || f.Lanes[0].Stats == nil {
		return nil
	}
	best := f.Lanes[0]
	out := []string{}
	for _, l := range notableLanes(f.Lanes) {
		b := best.Stats.Lane(l.Vertical, l.Index)
		bPct, bMean := lanePct(b), laneMean(b)

		// Find the candidate this lane looks least like after our play.
		var rival *LaneComparison
		var rivalPct, rivalMean, widest float64
		for _, other := range f.Lanes[1:] {
			if other.Stats == nil {
				continue
			}
			o := other.Stats.Lane(l.Vertical, l.Index)
			if gap := abs(bPct - lanePct(o)); gap > widest {
				rival, widest = other, gap
				rivalPct, rivalMean = lanePct(o), laneMean(o)
			}
		}
		if rival == nil {
			continue
		}

		switch {
		case bPct-rivalPct >= lanePctDiffMin:
			out = append(out, fmt.Sprintf("%s is busier after %s than after %s (%.1f%% vs %.1f%%)",
				l.Label, best.Play, rival.Play, bPct, rivalPct))
		case rivalPct-bPct >= lanePctDiffMin:
			out = append(out, fmt.Sprintf("%s is quieter after %s than after %s (%.1f%% vs %.1f%%)",
				l.Label, best.Play, rival.Play, bPct, rivalPct))
		case rivalMean-bMean >= laneMeanDiffMin && rivalPct > 0 && bPct > 0:
			out = append(out, fmt.Sprintf("replies in %s score less after %s than after %s (mean %.1f vs %.1f)",
				l.Label, best.Play, rival.Play, bMean, rivalMean))
		}
	}
	sort.Strings(out)
	return out
}

// notableLanes is the union of the lanes worth showing for any candidate, so a
// lane that only one of them opens still gets compared.
func notableLanes(comparisons []*LaneComparison) []*stats.LaneStat {
	out := []*stats.LaneStat{}
	seen := map[string]bool{}
	for _, lc := range comparisons {
		if lc.Stats == nil {
			continue
		}
		for i, l := range lc.Stats.Lanes {
			if i >= lanesShown || l.Pct < laneMinPct {
				break
			}
			if !seen[l.Label] {
				seen[l.Label] = true
				out = append(out, l)
			}
		}
	}
	return out
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

func lanePct(l *stats.LaneStat) float64 {
	if l == nil {
		return 0
	}
	return l.Pct
}

func laneMean(l *stats.LaneStat) float64 {
	if l == nil {
		return 0
	}
	return l.MeanScore
}

func dashIfEmpty(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func scoreRange(lo, hi int) string {
	if lo == hi {
		return fmt.Sprintf("%d", hi)
	}
	return fmt.Sprintf("%d-%d", lo, hi)
}

// drawList renders the alternative draws that unlock a play: "{B} or {C}".
// An empty alternative means one route needs no draw at all.
func drawList(draws []string) string {
	opts := []string{}
	for _, d := range draws {
		if d == "" {
			opts = append(opts, "no draw")
			continue
		}
		opts = append(opts, "{"+d+"}")
	}
	if len(opts) == 0 {
		return ""
	}
	return strings.Join(opts, " or ")
}
