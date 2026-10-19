package stats

import (
	"strings"

	"github.com/aybabtme/uniplot/histogram"
	"github.com/rs/zerolog/log"
)

// This file is the structured face of the follow-up analysis that
// CalculatePlayStats renders as text. Callers that want to reason about the
// numbers - the AI explainer, tests - should use CalculatePlayStatsData and
// read the structs; the fixed-width tables are for humans and for the model's
// eyes only. Nothing should ever parse the tables back into data.

// FollowupWay is one specific way of making a follow-up play. Two ways of the
// same play differ only in which tile is played as the blank, so they score
// differently and need different tiles drawn.
type FollowupWay struct {
	Play string `json:"play"`
	// Score is this way's score. Playthrough tiles are included.
	Score int     `json:"score"`
	Count int     `json:"count"`
	Pct   float64 `json:"pct"`
	// NeededDraw is the tiles we'd have to draw on top of our leave to make
	// this play, with no braces around them. Empty means no draw is needed.
	// Only filled in for our own follow-up plays.
	NeededDraw string `json:"needed_draw,omitempty"`
	Bingo      bool   `json:"bingo"`
}

// FollowupFamily is a play in one of the follow-up tables, together with the
// individual ways of making it when there is more than one. Count and Pct are
// combined across the ways: the chance of making the play by any route.
type FollowupFamily struct {
	// Play is the family's label. A family with a single way keeps that way's
	// exact notation; a grouped one is upper-cased, since no single notation
	// is correct for all of its ways.
	Play     string  `json:"play"`
	Count    int     `json:"count"`
	Pct      float64 `json:"pct"`
	MinScore int     `json:"min_score"`
	MaxScore int     `json:"max_score"`
	// NeededDraws are the alternative draws that unlock this play; any one of
	// them is enough. An empty string among them is a route needing no draw.
	NeededDraws []string `json:"needed_draws,omitempty"`
	Bingo       bool     `json:"bingo"`
	TilePlay    bool     `json:"tile_play"`
	// Ways is every way of making the play, most likely first. A family with
	// one way still lists it.
	Ways []*FollowupWay `json:"ways"`
}

// Grouped reports whether the play can be made in more than one way, i.e.
// whether a blank can stand in for more than one tile.
func (f *FollowupFamily) Grouped() bool {
	return len(f.Ways) > 1
}

// PlayStats is everything the simulation log knows about what happens after
// one root play: what the opponent does next, and what we do the turn after.
type PlayStats struct {
	// Play is the root play, in the same playthrough notation the tables use.
	Play string `json:"play"`
	// Leave is our rack leave after making Play.
	Leave string `json:"leave"`

	TotalOppReplies   int `json:"total_opp_replies"`
	TotalOurFollowups int `json:"total_our_followups"`

	// OppTopScoring is the opponent's highest-scoring sampled replies.
	OppTopScoring []*FollowupFamily `json:"opp_top_scoring"`
	// OppReplies is the opponent's most frequent sampled replies.
	OppReplies []*FollowupFamily `json:"opp_replies"`
	// OurFollowups is our most frequent sampled follow-up plays, with the
	// draws needed to make them. These are the only plays the explainer's
	// follow-up lookup can answer for.
	OurFollowups []*FollowupFamily `json:"our_followups"`

	OppBingoPct float64 `json:"opp_bingo_pct"`
	OurBingoPct float64 `json:"our_bingo_pct"`

	st *SimStats
	// The raw families, kept so Render can produce the text tables through
	// the same code path it always has.
	oppByScore, oppByFreq, ourByFreq []*playFamily
}

const (
	oppTopScoringToDisplay = 10
	followupsToDisplay     = 15
)

// accumulator holds the per-iteration tallies for one root play.
type accumulator struct {
	leave        string
	oppResponses map[string]*nextPlay
	ourNextPlays map[string]*nextPlay
	totalOpp     int
	totalOur     int
	oppScores    []float64
	ourScores    []float64
}

// accumulate walks the simulation log and tallies every sampled continuation
// of the given root play.
func (st *SimStats) accumulate(play string) (*accumulator, error) {
	iters, err := st.simmer.ReadHeatmap()
	if err != nil {
		return nil, err
	}
	log.Debug().Msgf("Read %d log lines", len(iters))
	normalizedPlay := Normalize(play)

	a := &accumulator{
		oppResponses: map[string]*nextPlay{},
		ourNextPlays: map[string]*nextPlay{},
	}
	for i := range iters {
		for j := range iters[i].Plays {
			if normalizedPlay != Normalize(iters[i].Plays[j].Play) {
				continue
			}
			a.leave = iters[i].Plays[j].Leave
			if len(iters[i].Plays[j].Plies) > 0 {
				nextPlay := iters[i].Plays[j].Plies[0]
				addNextPlay(nextPlay.Play, nextPlay.Pts, nextPlay.Bingo, a.oppResponses)
				a.oppScores = append(a.oppScores, float64(nextPlay.Pts))
				a.totalOpp++
			}
			if len(iters[i].Plays[j].Plies) > 1 {
				nextPlay := iters[i].Plays[j].Plies[1]
				addNextPlay(nextPlay.Play, nextPlay.Pts, nextPlay.Bingo, a.ourNextPlays)
				a.ourScores = append(a.ourScores, float64(nextPlay.Pts))
				a.totalOur++
			}
		}
	}
	return a, nil
}

// CalculatePlayStatsData analyzes what follows the given play. The returned
// value renders to the same text tables CalculatePlayStats has always
// produced; read its fields instead of parsing that text.
func (st *SimStats) CalculatePlayStatsData(play string) (*PlayStats, error) {
	a, err := st.accumulate(play)
	if err != nil {
		return nil, err
	}

	ps := &PlayStats{
		Play:              play,
		Leave:             a.leave,
		TotalOppReplies:   a.totalOpp,
		TotalOurFollowups: a.totalOur,
		st:                st,
		oppByScore:        sortedFamiliesByScore(a.oppResponses),
		oppByFreq:         sortedFamilies(a.oppResponses),
		ourByFreq:         sortedFamilies(a.ourNextPlays),
	}

	// The needed-draw column is only meaningful for our own plays, and it is
	// only computed for the families that make it into the table - which are
	// also the only ones the explainer is allowed to answer questions about.
	ourDisplayed := displayedFamilies(ps.ourByFreq, followupsToDisplay, true)
	for _, fam := range ourDisplayed {
		for _, np := range fam.variants {
			np.ifdraw = neededDraw(st, a.leave, np.play)
		}
	}

	ps.OppTopScoring = exportFamilies(displayedFamilies(ps.oppByScore, oppTopScoringToDisplay, false), a.totalOpp)
	ps.OppReplies = exportFamilies(displayedFamilies(ps.oppByFreq, followupsToDisplay, false), a.totalOpp)
	ps.OurFollowups = exportFamilies(ourDisplayed, a.totalOur)
	ps.OppBingoPct = bingoPct(ps.oppByFreq, a.totalOpp)
	ps.OurBingoPct = bingoPct(ps.ourByFreq, a.totalOur)

	st.oppHist = histogram.Hist(15, a.oppScores)
	st.ourHist = histogram.Hist(15, a.ourScores)

	return ps, nil
}

// RenderOptions controls how much of the analysis reaches the text tables. A
// person browsing `sim playstats` wants everything; a model reasoning about
// the position does not, because a long tail of one-in-a-hundred replies is
// noise to weigh against the handful of things that decide the turn.
type RenderOptions struct {
	// OppTopScoring includes the opponent's highest-scoring sampled replies.
	OppTopScoring bool
	// OppReplies and Followups cap the rows in each table. Zero means every
	// row that was computed.
	OppReplies int
	Followups  int
}

// FullRender shows everything, which is what the shell does.
func FullRender() RenderOptions {
	return RenderOptions{OppTopScoring: true}
}

// Render produces the full markdown tables.
func (ps *PlayStats) Render() string {
	return ps.RenderWith(FullRender())
}

// RenderWith produces the markdown tables, trimmed as asked.
func (ps *PlayStats) RenderWith(o RenderOptions) string {
	rows := func(want, computed int) int {
		if want <= 0 || want > computed {
			return computed
		}
		return want
	}

	var ss strings.Builder
	if o.OppTopScoring {
		ss.WriteString(playStatsStr(ps.st, ps.Leave, ps.oppByScore, "### Opponent's highest scoring plays",
			oppTopScoringToDisplay, ps.TotalOppReplies, false, false))
		ss.WriteString("\n\n")
	}
	ss.WriteString(playStatsStr(ps.st, ps.Leave, ps.oppByFreq, "### Opponent's next play",
		rows(o.OppReplies, followupsToDisplay), ps.TotalOppReplies, true, false))
	ss.WriteString("\n")
	ss.WriteString(playStatsStr(ps.st, ps.Leave, ps.ourByFreq, "### Our follow-up play",
		rows(o.Followups, followupsToDisplay), ps.TotalOurFollowups, true, true))
	ss.WriteString("\n")
	return ss.String()
}

// displayedFamilies returns the families playStatsStr would print for the same
// arguments, so that the exported data and the rendered table always describe
// the same set of plays.
func displayedFamilies(families []*playFamily, maxToDisplay int, tilePlaysOnly bool) []*playFamily {
	out := []*playFamily{}
	for i, fam := range families {
		if i >= maxToDisplay {
			break
		}
		if tilePlaysOnly && !fam.isTilePlay() {
			continue
		}
		out = append(out, fam)
	}
	return out
}

func bingoPct(families []*playFamily, total int) float64 {
	if total == 0 {
		return 0
	}
	bingos := 0
	for _, fam := range families {
		for _, np := range fam.variants {
			if np.bingo {
				bingos += np.count
			}
		}
	}
	return float64(bingos*100) / float64(total)
}

func exportFamilies(families []*playFamily, total int) []*FollowupFamily {
	out := make([]*FollowupFamily, 0, len(families))
	for _, fam := range families {
		out = append(out, exportFamily(fam, total))
	}
	return out
}

func exportFamily(f *playFamily, total int) *FollowupFamily {
	pct := func(count int) float64 {
		if total == 0 {
			return 0
		}
		return float64(count*100) / float64(total)
	}
	out := &FollowupFamily{
		// Play strings are right-aligned on the coordinate for the tables
		// ("%3v " in MoveDescriptionWithPlaythrough), which is presentation,
		// not data.
		Play:     strings.TrimSpace(f.name()),
		Count:    f.count,
		Pct:      pct(f.count),
		MinScore: f.minScore,
		MaxScore: f.maxScore,
		Bingo:    f.variants[0].bingo,
		TilePlay: f.isTilePlay(),
	}
	seen := map[string]bool{}
	for _, np := range f.variants {
		draw := strings.Trim(np.ifdraw, "{}")
		out.Ways = append(out.Ways, &FollowupWay{
			Play:       strings.TrimSpace(np.play),
			Score:      np.score,
			Count:      np.count,
			Pct:        pct(np.count),
			NeededDraw: draw,
			Bingo:      np.bingo,
		})
		if !seen[draw] {
			seen[draw] = true
			out.NeededDraws = append(out.NeededDraws, draw)
		}
	}
	return out
}
