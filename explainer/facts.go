package explainer

import (
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/rangefinder"
	"github.com/rs/zerolog/log"
)

// The point of this file is that the model should never have to do arithmetic
// or apply a threshold. Everything the prompt used to ask it to work out -
// "is this a setup?", "is the leave balanced?", "is this a pre-endgame?",
// "does the top play sacrifice equity for win%?" - is decided here, once, in
// Go, and handed over as a fact. What is left for the model is choosing which
// facts matter and saying them well, which is the part it is actually good at.

// Thresholds. These used to live as prose in the prompt ("higher than 10-20%
// chance ... could be a setup"), which meant the model had to compare numbers
// and then be told to call a tool to check its work.
const (
	// Below 7 tiles in the bag exchanging is no longer permitted, which is
	// what properly makes a position a pre-endgame.
	preEndgameBagMax = 6
	// More than 14 tiles in the bag and it is never a pre-endgame, so this is
	// where the late-game card stops being worth sending.
	lateGameBagMax = 14
	// Below this many tiles in the bag we are in the second half of the game,
	// where turnover starts to matter.
	secondHalfBagMax = 30

	// A follow-up play only counts as a setup if our best play is what makes
	// it possible, and it is both likely enough and big enough to be worth
	// setting up for.
	setupMinPct   = 5.0
	setupMinScore = 35

	// What makes a follow-up worth a sentence is not its probability on its
	// own - a flat percentage floor throws away exactly the plays a strong
	// player plays *for*. A 6% shot at 130 points matters enormously; an 11%
	// shot at 23 does not. So weigh the two against each other: a chance is
	// big when it is a real jump over an ordinary turn (the ratio) and comes
	// up often enough for that jump to be worth expected points (the upside).
	//
	// The two gates do complementary work, which is why both are here. The
	// ratio throws out frequent small plays; the upside throws out rare ones
	// that are only slightly above average. On a real position: a 1.61%
	// chance at 44 points clears the ratio narrowly and fails the upside by
	// twenty times over, while a 6% chance at ~127 clears both easily.
	bigChanceScoreRatio = 1.25
	bigChanceMinUpside  = 1.0

	// A bingo chance worth remarking on. The average turn is around 20%.
	ourBingoPctHigh = 35.0
	oppBingoPctHigh = 30.0

	// How far apart the candidates' next-score standard deviations have to be
	// before volatility is a real difference between them rather than noise.
	volatilitySpreadMin = 4.0

	// Lanes are computed for this many candidates, and a lane has to carry at
	// least this share of the sampled replies to be worth naming.
	laneCandidates = 3
	laneMinPct     = 4.0
	lanesShown     = 4

	// However many plays the caller simmed, only the top few are worth
	// contrasting. The shell trims to 5 before it gets here; this bounds the
	// prompt for anyone who doesn't.
	candidatesShown = 8

	// A tile is worth naming in a read when the opponent is likely to be
	// holding this much more or less of it than chance would give them. Same
	// bar the rangefinder's own graph marks tiles at, so the explanation and
	// `infer output` never disagree about which tiles the read is about.
	inferenceOutlierTiles = rangefinder.NotableTiles
	// How far the read has to move the recommended play's win% before it is
	// worth a sentence. Statistical significance isn't enough on its own: two
	// well-converged sims have narrow intervals, so a third of a point can
	// clear them while meaning nothing to a player.
	inferenceMinWinShift = 1.0
	// Win percentages this close to the ceiling or the floor mean the game is
	// decided, and the read isn't going to change that whatever it says.
	inferenceDecidedPct = 5.0
)

// InferredRacksShown is how many of the most likely leaves belong in a prompt.
// The tail is a long list of near-identical racks and says nothing the top few
// don't. Exported because the caller is what asks the rangefinder for them.
const InferredRacksShown = 5

// Phase is where in the game we are. It decides which advice applies.
type Phase string

const (
	PhaseOpening Phase = "opening"
	PhaseMidgame Phase = "midgame"
	// PhaseLateMidgame is the stretch where the endgame is coming but the
	// position is not yet a pre-endgame. Naming it keeps the model from
	// reaching for pre-endgame language too early.
	PhaseLateMidgame Phase = "late midgame (endgame approaching)"
	PhasePreEndgame  Phase = "pre-endgame"
)

// FollowupFact is one of our sampled follow-up plays, with the judgments the
// model used to have to make about it already made.
type FollowupFact struct {
	*stats.FollowupFamily
	// WayRequirements runs parallel to Ways. Each entry is "none", "requires
	// opponent play", or "requires us to play <best play> first". The last is
	// what actually makes a setup a setup: the follow-up doesn't exist unless
	// we make this play.
	WayRequirements []string `json:"way_requirements"`
	// IsSetup means our best play creates this opportunity.
	IsSetup bool `json:"is_setup"`
	// IsBigChance means the play is far bigger than an ordinary next turn and
	// comes up often enough for that to be worth playing for, whether or not
	// our play is what created it.
	IsBigChance bool `json:"is_big_chance"`
	// Upside is expected points above an ordinary next turn: how often this
	// play comes up times how much bigger than average it is.
	Upside float64 `json:"upside"`
}

// AvgScore is the score across the ways of making this play, weighted by how
// often each came up. A grouped play's MaxScore is only its luckiest route,
// so judging the play by it would overstate what it is worth.
func (f *FollowupFact) AvgScore() float64 {
	total, weight := 0.0, 0.0
	for _, w := range f.Ways {
		total += w.Pct * float64(w.Score)
		weight += w.Pct
	}
	if weight == 0 {
		return float64(f.MaxScore)
	}
	return total / weight
}

// FollowupCluster is one opportunity, and every play that takes it. Plays that
// start on the same square and run through the same tiles already on the board
// are the same chance reached by different draws - 2F (QUAD)RUPLE, RUPLY,
// RUPED and RUPOLE are one hook and four ways of filling it.
//
// The distinction is the whole reason this type exists. Judged one at a time
// those four are 4.91%, 4.81%, 3.49% and 2.58% - each individually too rare to
// build a turn around, and each rejected. Together they are 15.79% of the time
// for an average of 48.6 points, which is the single biggest thing about the
// position. A chance split across spellings is still one chance.
type FollowupCluster struct {
	// Anchor is where the opportunity is: the tiles it runs through and the
	// lane they are in, like "(QUAD) in row 2", or a bare square when there is
	// nothing on the board to name it by.
	Anchor string `json:"anchor"`
	// Through is the letters already on the board that every play here runs
	// through, empty when the opportunity is an open square rather than a hook.
	Through string `json:"through"`
	// Plays are the ways of taking it, most likely first.
	Plays []*FollowupFact `json:"plays"`
	// Pct, MinScore and MaxScore cover the cluster as a whole.
	Pct      float64 `json:"pct"`
	AvgScore float64 `json:"avg_score"`
	MinScore int     `json:"min_score"`
	MaxScore int     `json:"max_score"`

	IsSetup     bool    `json:"is_setup"`
	IsBigChance bool    `json:"is_big_chance"`
	Upside      float64 `json:"upside"`
}

// labelWords is how many of a cluster's plays get named in its label before it
// gives up and counts the rest. Every play is listed underneath anyway; this is
// only what a reader sees first.
const labelWords = 3

// Label names the cluster. One play is named by itself. Several sharing a hook
// are named by the hook - "(QUAD) in row 2" says what the chance is. Several
// sharing only a square have no hook to name, and a bare "J9" says nothing at
// all, so those are named by their words: "J9 CAPERED / PEASCOD".
func (c *FollowupCluster) Label() string {
	if len(c.Plays) == 1 {
		return c.Plays[0].Play
	}
	if c.Through != "" {
		return c.Anchor
	}
	words := []string{}
	for _, p := range c.Plays {
		if len(words) == labelWords {
			words = append(words, fmt.Sprintf("+%d more", len(c.Plays)-labelWords))
			break
		}
		if fields := strings.Fields(p.Play); len(fields) > 1 {
			words = append(words, fields[1])
		}
	}
	return c.Anchor + " " + strings.Join(words, " / ")
}

// Grouped reports whether this is several plays rather than one.
func (c *FollowupCluster) Grouped() bool { return len(c.Plays) > 1 }

// Worthwhile reports whether the cluster is worth a sentence.
func (c *FollowupCluster) Worthwhile() bool { return c.IsSetup || c.IsBigChance }

// NeededDraws is every draw that reaches any play in the cluster, in the order
// the plays came in and without repeats.
func (c *FollowupCluster) NeededDraws() []string {
	seen := map[string]bool{}
	out := []string{}
	for _, p := range c.Plays {
		for _, d := range p.NeededDraws {
			if !seen[d] {
				seen[d] = true
				out = append(out, d)
			}
		}
	}
	return out
}

// Requirement is what has to happen before the cluster's likeliest play is
// available.
func (c *FollowupCluster) Requirement() string {
	if len(c.Plays) == 0 {
		return "none"
	}
	return c.Plays[0].Requirement()
}

// Requirement is what has to happen before the play's most likely route is
// available.
func (f *FollowupFact) Requirement() string {
	if len(f.WayRequirements) == 0 {
		return "none"
	}
	return f.WayRequirements[0]
}

// Worthwhile reports whether the follow-up is worth a sentence: either our
// play creates it, or it is big enough and likely enough to matter.
func (f *FollowupFact) Worthwhile() bool {
	return f.IsSetup || f.IsBigChance
}

// LaneComparison is where one candidate's sampled opponent replies land.
type LaneComparison struct {
	Play  string
	Best  bool
	Stats *stats.LaneStats
}

// InferenceFacts is what the opponent's last play gave away, and whether it
// changed anything. Both halves matter: a read that says something startling
// about their rack but leaves the recommendation untouched is a curiosity, not
// a lesson, and neither half is worth prompt on its own.
type InferenceFacts struct {
	Summary *rangefinder.InferenceSummary `json:"summary"`
	// Outliers are the tiles the read actually says something about, ordered
	// by how far they deviate from chance.
	Outliers []rangefinder.TileDeviation `json:"outliers"`

	// Baseline is the same plays simmed without the read, so we can say what
	// the read changed rather than only what it concluded.
	Baseline []montecarlo.CandidateStats `json:"-"`
	// BaselineBest is the play we would have recommended knowing nothing.
	BaselineBest *montecarlo.CandidateStats `json:"baseline_best"`
	// BaselineOfBest is the recommended play's own showing in that sim, which
	// is what WinPctShift is measured against. Nil if it wasn't simmed there.
	BaselineOfBest *montecarlo.CandidateStats `json:"baseline_of_best"`

	// ChangedTopPlay is the strongest thing a read can do: recommend a
	// different play than we would have made without it.
	ChangedTopPlay bool `json:"changed_top_play"`
	// WinPctShift is the recommended play's win% with the read minus without.
	WinPctShift float64 `json:"win_pct_shift"`
	// Established is true when that shift is outside both sims' confidence
	// intervals. The two runs stop independently, so they aren't equally
	// sampled - non-overlapping intervals still means a real difference, but
	// the size of it shouldn't be read too precisely.
	Established bool `json:"established"`
	// Decisive is Established plus a practical significance test. A shift can
	// be statistically real and mean nothing: in a position already won, win
	// probabilities saturate, their intervals shrink to almost nothing, and a
	// swing from 97.9% to 97.0% clears the statistical bar while changing
	// nothing about how to play. Decisive is what gates the subject.
	Decisive bool `json:"decisive"`

	// Informative is true when the posterior deviates from chance enough to
	// be worth a sentence.
	Informative bool `json:"informative"`
	// Matters gates the whole subject. An uninformative read, or one that
	// moves nothing, is not mentioned at all.
	Matters bool `json:"matters"`
}

// Deltas is the head-to-head arithmetic between the best play and the one it
// is being contrasted with. Every figure is best minus rival, so a positive
// number is the best play's advantage.
type Deltas struct {
	WinPct     float64 `json:"win_pct"`
	Equity     float64 `json:"equity"`
	Score      int     `json:"score"`
	LeaveValue float64 `json:"leave_value"`
	// The opponent's reply, one ply out: a negative OppMeanScore means the
	// best play holds them to less.
	OppMeanScore float64 `json:"opp_mean_score"`
	OppStdev     float64 `json:"opp_stdev"`
	OppBingoPct  float64 `json:"opp_bingo_pct"`
	// Our own next turn, two plies out.
	OurMeanScore float64 `json:"our_mean_score"`
	OurBingoPct  float64 `json:"our_bingo_pct"`
	// BestUpside and RivalUpside are the expected points each play's big
	// follow-up chances are worth on top of an ordinary turn. The gap between
	// them is often the whole reason one play wins, and it is invisible in
	// every other figure here.
	BestUpside  float64 `json:"best_upside"`
	RivalUpside float64 `json:"rival_upside"`
	// Established is true when the two win% confidence intervals don't
	// overlap. When it is false the simulation has not actually shown one
	// play to be better, and saying so would be overclaiming.
	Established bool `json:"established"`
}

// Comparison answers "why is this better than the move I made?". The play in
// question is normally the one the player actually made, taken from the game
// history; it can also be named outright.
type Comparison struct {
	// Play is what we were asked to contrast the best play with.
	Play string `json:"play"`
	// FromHistory is true when it is the move the player actually made, as
	// opposed to one they named.
	FromHistory bool `json:"from_history"`
	// WasBest is true when the play we were asked about *is* the top play. It
	// then has nothing to be contrasted with, so Rival below is the runner-up
	// instead and there is still something to learn.
	WasBest bool `json:"was_best"`

	// Rival is the candidate actually contrasted against.
	Rival          *montecarlo.CandidateStats `json:"rival"`
	RivalPlayStats *stats.PlayStats           `json:"-"`
	RivalFollowups []*FollowupFact            `json:"rival_followups"`
	// RivalChances is RivalFollowups gathered into opportunities.
	RivalChances []*FollowupCluster `json:"rival_chances"`
	// TypicalNextScore is what an ordinary next turn is worth after the rival.
	TypicalNextScore float64 `json:"typical_next_score"`

	Deltas Deltas `json:"deltas"`

	// OnlyBest and OnlyRival are the worthwhile follow-up plays each side has
	// that the other doesn't - the concrete "what you gave up" list.
	OnlyBest  []*FollowupFact `json:"only_best"`
	OnlyRival []*FollowupFact `json:"only_rival"`
}

// Flags are the yes/no answers that decide which concept cards get sent. Every
// one of them replaces a paragraph that used to ship on every single call.
type Flags map[string]bool

// knownFlags is every flag a concept card may trigger on. A card whose trigger
// isn't here would silently never ship, and a flag nothing triggers on is work
// we're throwing away, so tests check the cards against this list in both
// directions.
var knownFlags = []string{
	"always",
	"behind_early",
	"bingo_matters",
	"comparison_was_best",
	"equity_sacrifice",
	"has_big_chance",
	"has_comparison",
	"has_grouped_followup",
	"has_lane_data",
	"has_inference",
	"has_needed_draw",
	"has_setup",
	"inference_changed_play",
	"leave_matters",
	"opp_bingo_pct_high",
	"pre_endgame",
	"turnover_relevant",
	"uses_blank",
	"volatility_matters",
}

// PositionFacts is everything we know about the position, computed rather than
// described.
type PositionFacts struct {
	Rack        string
	Lexicon     string
	BagCount    int
	UnseenCount int
	OppRackSize int
	Spread      int
	Phase       Phase

	UnseenVowels     int
	UnseenConsonants int
	UnseenBlanks     int
	UnseenPowerTiles []string

	Iterations int
	Candidates []montecarlo.CandidateStats
	Best       *montecarlo.CandidateStats

	PlayStats *stats.PlayStats
	Followups []*FollowupFact
	// Chances is Followups gathered into opportunities: the plays that take the
	// same hook are one chance, and are judged and reported as one. Ordered by
	// upside, biggest first.
	Chances []*FollowupCluster
	Lanes   []*LaneComparison
	// TypicalNextScore is what an ordinary next turn is worth after the best
	// play. It is the yardstick every follow-up chance is measured against.
	TypicalNextScore float64

	// BestByEquity is the candidate with the highest equity, which is not
	// always the one with the highest win%.
	BestByEquity *montecarlo.CandidateStats

	// Comparison is set when we were asked why the best play beats some other
	// play - usually the one the player actually made.
	Comparison *Comparison

	// Inference is set when the position was analyzed with a read on the
	// opponent's rack. It is nil unless `explain -infer` asked for one.
	Inference *InferenceFacts

	Flags Flags
}

// ComparisonRequest names a play the best one should be contrasted with.
type ComparisonRequest struct {
	Move *move.Move
	// FromHistory distinguishes "the move you actually made", which we found
	// ourselves, from one the user named.
	FromHistory bool
}

// InferenceInput is a read on the opponent's rack, together with the same
// plays simmed without it. Both are needed: the read on its own can't say
// what it changed.
type InferenceInput struct {
	Summary *rangefinder.InferenceSummary
	// Baseline is CandidateStats from a sim run with inference off, best
	// win% first.
	Baseline []montecarlo.CandidateStats
}

// BuildFacts assembles the fact pack for the current position from a finished
// simulation, and keeps it as what the tools answer from. It must be called
// after the sim has stopped. A non-nil req asks for a head-to-head against
// that play; the simulation must already have evaluated it, which is what
// Simmer.AvoidPruningMoves is for. A non-nil inf adds what a read on the
// opponent's rack changed, if anything.
func (a *Analyzer) BuildFacts(sim *montecarlo.Simmer, ss *stats.SimStats,
	req *ComparisonRequest, inf *InferenceInput) (*PositionFacts, error) {
	if a.game == nil {
		return nil, fmt.Errorf("no game set")
	}
	candidates := sim.CandidateStats()
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no simmed plays")
	}
	candidates, compareIdx := trimCandidates(candidates, a.findCandidate(candidates, req))

	f := &PositionFacts{
		Lexicon:    a.game.LexiconName(),
		Iterations: sim.Iterations(),
		Candidates: candidates,
		Best:       &candidates[0],
		Flags:      Flags{},
	}
	a.addPositionFacts(f)

	f.BestByEquity = &candidates[0]
	for i := range candidates {
		if candidates[i].Equity > f.BestByEquity.Equity {
			f.BestByEquity = &candidates[i]
		}
	}

	var err error
	if f.PlayStats, err = ss.CalculatePlayStatsData(f.Best.Play); err != nil {
		return nil, fmt.Errorf("failed to analyze follow-up plays: %w", err)
	}
	f.TypicalNextScore = typicalNextScore(f.Best)
	followups, err := a.analyzeFollowups(f.Best.Play, f.TypicalNextScore, f.PlayStats)
	if err != nil {
		return nil, err
	}
	f.SetFollowups(followups)
	if compareIdx >= 0 {
		if f.Comparison, err = a.buildComparison(f, req, compareIdx, ss); err != nil {
			return nil, err
		}
	}
	f.Lanes = laneComparisons(ss, candidates, f.rivalPlay())
	f.Inference = buildInference(f, inf)

	f.Flags = computeFlags(f)
	a.facts = f
	return f, nil
}

// findCandidate locates the play we were asked to compare against. Matching is
// on the move rather than its notation: on an empty board a vertical opener is
// a transposition of its horizontal twin, and the simulation only ever holds
// one of the two.
func (a *Analyzer) findCandidate(candidates []montecarlo.CandidateStats, req *ComparisonRequest) int {
	if req == nil || req.Move == nil {
		return -1
	}
	checkTrans := a.game.Board().IsEmpty()
	for i := range candidates {
		if candidates[i].Move != nil && candidates[i].Move.Equals(req.Move, checkTrans, true) {
			return i
		}
	}
	// AvoidPruningMoves should have put it in the sim, so this means the
	// caller didn't ask for it, or the play doesn't fit this position.
	log.Warn().Str("play", req.Move.ShortDescription()).
		Msg("play to compare against was not simulated; skipping the comparison")
	return -1
}

// trimCandidates keeps the top few plays for contrast, plus - wherever it
// ranked - the one we were asked to compare against.
func trimCandidates(candidates []montecarlo.CandidateStats, compareIdx int) ([]montecarlo.CandidateStats, int) {
	if len(candidates) <= candidatesShown {
		return candidates, compareIdx
	}
	kept := append([]montecarlo.CandidateStats{}, candidates[:candidatesShown]...)
	if compareIdx >= candidatesShown {
		kept = append(kept, candidates[compareIdx])
		compareIdx = len(kept) - 1
	}
	return kept, compareIdx
}

// rivalPlay is the play the board-dynamics section should cover on top of the
// top candidates, or "" when there is no comparison.
func (f *PositionFacts) rivalPlay() string {
	if f.Comparison == nil || f.Comparison.Rival == nil {
		return ""
	}
	return f.Comparison.Rival.Play
}

// buildComparison works out the head-to-head between the best play and the one
// we were asked about, giving the latter the same follow-up analysis the best
// play gets - otherwise there is no way to say what it gave up.
func (a *Analyzer) buildComparison(f *PositionFacts, req *ComparisonRequest, idx int,
	ss *stats.SimStats) (*Comparison, error) {

	c := &Comparison{
		Play:        f.Candidates[idx].Play,
		FromHistory: req.FromHistory,
		WasBest:     idx == 0,
	}
	rivalIdx := idx
	if c.WasBest {
		// They already found the top play, so contrast with the runner-up
		// instead: there is still something to learn from what it beat.
		if len(f.Candidates) < 2 {
			return c, nil
		}
		rivalIdx = 1
	}
	c.Rival = &f.Candidates[rivalIdx]

	var err error
	if c.RivalPlayStats, err = ss.CalculatePlayStatsData(c.Rival.Play); err != nil {
		return nil, fmt.Errorf("failed to analyze follow-up plays for %s: %w", c.Rival.Play, err)
	}
	// The rival's follow-ups are judged against the rival, and against what an
	// ordinary turn is worth after *it*: a setup is a setup relative to the
	// play that would create it, and a chance is big relative to what else
	// that play leads to.
	c.TypicalNextScore = typicalNextScore(c.Rival)
	if c.RivalFollowups, err = a.analyzeFollowups(c.Rival.Play, c.TypicalNextScore,
		c.RivalPlayStats); err != nil {
		return nil, err
	}
	c.RivalChances = clusterFollowups(c.RivalFollowups, c.Rival.Play, c.TypicalNextScore)
	c.Deltas = a.deltas(f, c.Rival, c.RivalPlayStats)
	c.Deltas.BestUpside = totalUpside(f.Chances)
	c.Deltas.RivalUpside = totalUpside(c.RivalChances)
	c.OnlyBest, c.OnlyRival = followupDiff(f.Followups, c.RivalFollowups)
	return c, nil
}

// buildInference works out whether the read on the opponent's rack is worth
// telling the player about. Two things have to be true. The posterior has to
// deviate from chance by enough to describe - a flat one tells us nothing -
// and it has to have changed the analysis, either by recommending a different
// play or by moving the recommended play's win% outside both sims' intervals.
// A read that is interesting but changes nothing is a curiosity.
func buildInference(f *PositionFacts, in *InferenceInput) *InferenceFacts {
	if in == nil || in.Summary == nil {
		return nil
	}
	inf := &InferenceFacts{Summary: in.Summary}
	for _, t := range in.Summary.Tiles {
		if abs(t.Tiles) >= inferenceOutlierTiles {
			inf.Outliers = append(inf.Outliers, t)
		}
	}
	inf.Informative = len(inf.Outliers) > 0

	inf.Baseline = in.Baseline
	if len(inf.Baseline) > 0 {
		inf.BaselineBest = &inf.Baseline[0]
		inf.ChangedTopPlay = !samePlay(f.Best, inf.BaselineBest)
		for i := range inf.Baseline {
			if samePlay(f.Best, &inf.Baseline[i]) {
				inf.BaselineOfBest = &inf.Baseline[i]
				break
			}
		}
	}
	if b := inf.BaselineOfBest; b != nil {
		inf.WinPctShift = f.Best.WinPct - b.WinPct
		shift := abs(inf.WinPctShift)
		inf.Established = shift > f.Best.WinPctCI+b.WinPctCI
		// A won game is won with or without the read. Once both figures are
		// up against the ceiling (or the floor) what is left of the win
		// probability is jitter in its last decimal places, and its interval
		// has shrunk to match - the same reason the simmer refuses to rank
		// plays by win probability in that regime.
		inf.Decisive = inf.Established && shift >= inferenceMinWinShift &&
			!gameDecided(f.Best.WinPct, b.WinPct)
	}

	inf.Matters = inf.Informative && (inf.ChangedTopPlay || inf.Decisive)
	return inf
}

// gameDecided reports whether both win percentages are so lopsided that the
// game is over either way.
func gameDecided(a, b float64) bool {
	bothWon := a > 100-inferenceDecidedPct && b > 100-inferenceDecidedPct
	bothLost := a < inferenceDecidedPct && b < inferenceDecidedPct
	return bothWon || bothLost
}

// samePlay compares two candidates by the move rather than its notation,
// since the two sims render the same play from the same board and a string
// compare would work - but the move is what actually identifies it.
func samePlay(a, b *montecarlo.CandidateStats) bool {
	if a == nil || b == nil {
		return false
	}
	if a.Move != nil && b.Move != nil {
		return a.Move.Equals(b.Move, false, true)
	}
	return a.Play == b.Play
}

// totalUpside adds up what a play's big follow-up chances are worth. Chances
// that didn't clear the bar contribute nothing: a long tail of slightly
// above-average plays isn't what anyone means by upside.
func totalUpside(cs []*FollowupCluster) float64 {
	total := 0.0
	for _, c := range cs {
		if c.IsBigChance {
			total += c.Upside
		}
	}
	return total
}

// deltas is best minus rival throughout, so a positive number always reads as
// the best play's advantage.
func (a *Analyzer) deltas(f *PositionFacts, rival *montecarlo.CandidateStats,
	rivalStats *stats.PlayStats) Deltas {

	best := f.Best
	d := Deltas{
		WinPct: best.WinPct - rival.WinPct,
		Equity: best.Equity - rival.Equity,
		Score:  best.Score - rival.Score,
		// Non-overlapping 99% intervals. When they do overlap the simulation
		// has not shown one play to be better, and the explanation should not
		// pretend otherwise.
		Established: best.WinPct-best.WinPctCI > rival.WinPct+rival.WinPctCI,
	}
	bestLeave, bestErr := a.EvaluateLeave(best.Leave)
	rivalLeave, rivalErr := a.EvaluateLeave(rival.Leave)
	if bestErr == nil && rivalErr == nil {
		d.LeaveValue = bestLeave - rivalLeave
	}
	d.OppMeanScore, d.OppStdev, d.OppBingoPct = plyDelta(best, rival, 1)
	d.OurMeanScore, _, d.OurBingoPct = plyDelta(best, rival, 2)
	return d
}

func plyDelta(best, rival *montecarlo.CandidateStats, ply int) (mean, stdev, bingo float64) {
	b, r := plyStats(best, ply), plyStats(rival, ply)
	if b == nil || r == nil {
		return 0, 0, 0
	}
	return b.MeanScore - r.MeanScore, b.Stdev - r.Stdev, b.BingoPct - r.BingoPct
}

func plyStats(c *montecarlo.CandidateStats, ply int) *montecarlo.CandidatePlyStats {
	for i := range c.Plies {
		if c.Plies[i].Ply == ply {
			return &c.Plies[i]
		}
	}
	return nil
}

// followupDiff lists the worthwhile follow-ups each play has that the other's
// sample doesn't show. This is the concrete "what you gave up" list, and it is
// about what the simulation sampled - a play missing from one side's table is
// not proof that it is impossible there.
func followupDiff(best, rival []*FollowupFact) (onlyBest, onlyRival []*FollowupFact) {
	key := func(f *FollowupFact) string { return foldKey(f.Play, true) }
	seen := func(fs []*FollowupFact) map[string]bool {
		m := map[string]bool{}
		for _, f := range fs {
			m[key(f)] = true
		}
		return m
	}
	inBest, inRival := seen(best), seen(rival)
	for _, f := range best {
		if f.Worthwhile() && !inRival[key(f)] {
			onlyBest = append(onlyBest, f)
		}
	}
	for _, f := range rival {
		if f.Worthwhile() && !inBest[key(f)] {
			onlyRival = append(onlyRival, f)
		}
	}
	return onlyBest, onlyRival
}

// addPositionFacts fills in the rack, the score, and what is left unseen -
// the same things the shell's `gamestate` prose covers, as data.
func (a *Analyzer) addPositionFacts(f *PositionFacts) {
	g := a.game
	ld := g.Bag().LetterDistribution()

	f.BagCount = g.Bag().TilesRemaining()
	f.OppRackSize = int(g.RackFor(g.NextPlayer()).NumTiles())
	f.UnseenCount = f.BagCount + f.OppRackSize
	f.Rack = g.RackFor(g.PlayerOnTurn()).TilesOn().UserVisible(g.Alphabet())
	f.Spread = g.CurrentSpread()

	switch {
	case f.BagCount <= preEndgameBagMax:
		f.Phase = PhasePreEndgame
	case f.BagCount <= lateGameBagMax:
		f.Phase = PhaseLateMidgame
	case g.Board().IsEmpty():
		f.Phase = PhaseOpening
	default:
		f.Phase = PhaseMidgame
	}

	// Unseen means the bag plus whatever the opponent is holding: from our
	// side of the table those are the same thing.
	unseen := append(g.Bag().Peek(), g.RackFor(g.NextPlayer()).TilesOn()...)
	for _, tile := range unseen {
		switch {
		case tile == 0:
			f.UnseenBlanks++
		case tile.IsVowel(ld):
			f.UnseenVowels++
		default:
			f.UnseenConsonants++
		}
		if ld.Score(tile) > 5 || tile == 0 || tile.UserVisible(ld.TileMapping(), false) == "S" {
			f.UnseenPowerTiles = append(f.UnseenPowerTiles, tile.UserVisible(ld.TileMapping(), false))
		}
	}
	sort.Strings(f.UnseenPowerTiles)
}

// analyzeFollowups works out, for each sampled follow-up play, whether it is
// available anyway or whether something has to happen first - most importantly
// whether it only exists because of the play we're recommending. That last
// case is the difference between a setup and a coincidence, and it can only be
// answered by putting the play on the board and looking.
func (a *Analyzer) analyzeFollowups(bestPlay string, typicalScore float64,
	ps *stats.PlayStats) ([]*FollowupFact, error) {

	// Play our best play once and reuse the resulting position for every
	// follow-up, rather than replaying it per play as the old tool did.
	normalizedBestPlay := stats.Normalize(strings.TrimSpace(bestPlay))
	ourBestPlay, err := a.game.ParseMove(a.game.PlayerOnTurn(), false, strings.Fields(normalizedBestPlay), false)
	if err != nil {
		return nil, fmt.Errorf("failed to parse our best play %s: %w", normalizedBestPlay, err)
	}
	after := a.game.Copy()
	if err := after.PlayMove(ourBestPlay, false, 0); err != nil {
		return nil, fmt.Errorf("failed to play our best play %s: %w", normalizedBestPlay, err)
	}
	// After our play it is the opponent's turn, so our rack is the other
	// player's.
	ourRack := after.RackLettersFor(1 - after.PlayerOnTurn())

	out := make([]*FollowupFact, 0, len(ps.OurFollowups))
	for _, fam := range ps.OurFollowups {
		fact := &FollowupFact{FollowupFamily: fam}
		for _, way := range fam.Ways {
			if !fam.TilePlay {
				// Nothing has to happen first for an exchange or a pass. The
				// follow-up table only holds tile plays today; this keeps the
				// coordinate parsing below from ever seeing one anyway.
				fact.WayRequirements = append(fact.WayRequirements, "none")
				continue
			}
			fact.WayRequirements = append(fact.WayRequirements,
				a.requirementFor(bestPlay, ourRack, after, way))
		}
		fact.IsSetup = fact.Requirement() == requiresBestPlay(bestPlay) &&
			fam.Pct >= setupMinPct && fam.MaxScore >= setupMinScore
		fact.Upside, fact.IsBigChance = bigChance(fact, typicalScore)
		out = append(out, fact)
	}
	return out, nil
}

func requiresBestPlay(bestPlay string) string {
	return "requires us to play " + strings.TrimSpace(bestPlay) + " first"
}

// bigChance decides whether a follow-up is one of the plays worth building a
// turn around, and by how much. typicalScore is what an ordinary next turn is
// worth after this play, so every chance is judged against the alternative
// the player actually faces rather than against a fixed number of points.
//
// The old rule here was a flat "at least 10% of the time and at least 40
// points", which quietly discarded the most interesting plays on the board: a
// 6% chance at 130 failed it, while an 11% chance at 23 passed. Weighing size
// against frequency is the whole point.
func bigChance(f *FollowupFact, typicalScore float64) (upside float64, big bool) {
	return bigChanceOf(f.Pct, f.AvgScore(), typicalScore)
}

// bigChanceOf is the same judgment on bare numbers, so that a cluster of plays
// taking one opportunity is weighed exactly as a single play is.
func bigChanceOf(pct, score, typicalScore float64) (upside float64, big bool) {
	if typicalScore <= 0 {
		// No baseline to judge against - the sim didn't look far enough
		// ahead - so claim nothing.
		return 0, false
	}
	upside = pct / 100 * (score - typicalScore)
	return upside, score >= typicalScore*bigChanceScoreRatio && upside >= bigChanceMinUpside
}

// typicalNextScore is what an ordinary turn is worth for us after a play: the
// mean of our own scores two plies out.
// reCoord is a board coordinate: a row number and a column letter, in either
// order. Anything else came from a play that isn't a placement, and a
// coordinate parser will quietly return the top-left square for it.
var reCoord = regexp.MustCompile(`^(?:[0-9]{1,2}[A-Z]|[A-Z][0-9]{1,2})$`)

// anchorOf identifies the opportunity a follow-up takes, and names it.
//
// The identity is where the *playthrough* sits - which tiles already on the
// board the play runs through, and where they are - never where the play
// starts. A front extension moves its own starting square: with ZOIC on the
// board, CENO(ZOIC) and SAPRO(ZOIC) begin four and five squares to its left
// and would look like different opportunities keyed on the coordinate, when
// they are one hook taken two ways. Keying on ZOIC itself, they group, and so
// do HI(THERMOS)T and NE(THERMOS)T, which extend in both directions at once.
//
// A play with no playthrough has nothing to hang on to, so its starting square
// is the opportunity. That groups plays making unrelated words on one open
// square, which is right: it is one place on the board, reached by one leave.
func anchorOf(play string, tilePlay bool) (key, label, through string) {
	fields := strings.Fields(strings.TrimSpace(play))
	// Exchanges and passes are not places on a board and never group. They
	// also split into fields like a play does - "(exch AEI)" looks like a
	// coordinate and a word - so this has to be settled before any parsing.
	if !tilePlay || len(fields) < 2 || !reCoord.MatchString(strings.ToUpper(fields[0])) {
		p := strings.TrimSpace(play)
		return p, p, ""
	}
	coord := strings.ToUpper(fields[0])
	row, col, vertical := move.FromBoardGameCoords(coord, false)

	// Walk the word a square at a time, noting the letters already on the
	// board and how far in the first of them lies.
	var runs strings.Builder
	offset, square, depth := -1, 0, 0
	for _, r := range fields[1] {
		switch r {
		case '(':
			depth++
		case ')':
			depth--
		default:
			if depth > 0 || r == '.' {
				if offset < 0 {
					offset = square
				}
				runs.WriteRune(r)
			}
			square++
		}
	}
	if runs.Len() == 0 {
		return coord, coord, ""
	}

	// Where the playthrough begins, in board coordinates rather than in the
	// play's own.
	index := row
	if vertical {
		row += offset
		index = col
	} else {
		col += offset
	}
	return fmt.Sprintf("%t/%d/%d/%s", vertical, row, col, runs.String()),
		fmt.Sprintf("(%s) in %s", runs.String(), stats.LaneLabel(vertical, index)),
		runs.String()
}

// SetFollowups records the sampled follow-ups and gathers them into the
// opportunities they take. The two are always set together - the flat list is
// the table, the clusters are what gets judged and reported - and going
// through here is what keeps them from disagreeing.
//
// Best and TypicalNextScore have to be set first: a chance is only big or
// small relative to what an ordinary turn after this play is worth.
func (f *PositionFacts) SetFollowups(fs []*FollowupFact) {
	f.Followups = fs
	f.Chances = clusterFollowups(fs, f.Best.Play, f.TypicalNextScore)
}

// clusterFollowups gathers the follow-ups that take the same opportunity and
// judges each opportunity whole. Sizing them one play at a time asks the wrong
// question: what a player wants to know is how often the hook pays, not how
// often it pays with a particular spelling.
//
// Ordering is by weight, and within a cluster the likeliest play leads, so the
// explanation names the play a reader is most likely to actually make.
func clusterFollowups(fs []*FollowupFact, bestPlay string, typical float64) []*FollowupCluster {
	byAnchor := map[string]*FollowupCluster{}
	order := []string{}
	for _, f := range fs {
		key, label, through := anchorOf(f.Play, f.TilePlay)
		c, ok := byAnchor[key]
		if !ok {
			c = &FollowupCluster{Anchor: label, Through: through, MinScore: f.MinScore}
			byAnchor[key] = c
			order = append(order, key)
		}
		c.Plays = append(c.Plays, f)
		c.Pct += f.Pct
		c.MinScore = min(c.MinScore, f.MinScore)
		c.MaxScore = max(c.MaxScore, f.MaxScore)
	}

	out := make([]*FollowupCluster, 0, len(order))
	for _, anchor := range order {
		c := byAnchor[anchor]
		sort.SliceStable(c.Plays, func(i, j int) bool { return c.Plays[i].Pct > c.Plays[j].Pct })
		c.AvgScore = clusterAvgScore(c)
		// A setup stays a setup however many spellings take it, and a hook that
		// no single spelling reaches often enough is still a setup once they
		// are counted together.
		c.IsSetup = c.Requirement() == requiresBestPlay(bestPlay) &&
			c.Pct >= setupMinPct && c.MaxScore >= setupMinScore
		c.Upside, c.IsBigChance = bigChanceOf(c.Pct, c.AvgScore, typical)
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Upside > out[j].Upside })
	return out
}

// clusterAvgScore is the score across every play in the cluster, weighted by
// how often each came up - the same reasoning as FollowupFact.AvgScore, one
// level up. The maximum would be the luckiest spelling of the luckiest draw.
func clusterAvgScore(c *FollowupCluster) float64 {
	total, weight := 0.0, 0.0
	for _, p := range c.Plays {
		total += p.Pct * p.AvgScore()
		weight += p.Pct
	}
	if weight == 0 {
		return float64(c.MaxScore)
	}
	return total / weight
}

func typicalNextScore(c *montecarlo.CandidateStats) float64 {
	if p := plyStats(c, 2); p != nil {
		return p.MeanScore
	}
	return 0
}

// requirementFor asks whether a follow-up play works on the board as it stands
// and on the board after our best play, imagining we drew whatever the
// simulation says it needs. A play that only comes out right after our play is
// one our play set up.
func (a *Analyzer) requirementFor(bestPlay, ourRack string, after *game.Game, way *stats.FollowupWay) string {
	npfields := strings.Fields(stats.Normalize(way.Play))
	if len(npfields) < 2 {
		// Exchanges and passes need nothing.
		return "none"
	}
	rackLetters := ourRack
	for _, char := range way.NeededDraw {
		if (char >= 'A' && char <= 'Z') || char == '?' {
			rackLetters += string(char)
		}
	}

	if !scoresAs(after.CreateAndScorePlacementMove(npfields[0], npfields[1], rackLetters, false))(way.Score) {
		// It doesn't even work after our play, so the opponent must add
		// something first.
		return "requires opponent play"
	}
	if !scoresAs(a.game.CreateAndScorePlacementMove(npfields[0], npfields[1], rackLetters, false))(way.Score) {
		return requiresBestPlay(bestPlay)
	}
	return "none"
}

// scoresAs curries the "did this move come out legal, and did it score what
// the simulation saw?" check, which we run against two different boards.
func scoresAs(m *move.Move, err error) func(int) bool {
	return func(want int) bool {
		return err == nil && m != nil && m.Score() == want
	}
}

// laneComparisons computes where replies land after the top few candidates,
// plus alsoPlay if it isn't already among them - that is how the play the
// player made gets into the board-dynamics section however low it ranked. The
// best play stays first; laneDifferences contrasts everything against it.
func laneComparisons(ss *stats.SimStats, candidates []montecarlo.CandidateStats, alsoPlay string) []*LaneComparison {
	plays := []string{}
	for i := range candidates {
		if i >= laneCandidates {
			break
		}
		plays = append(plays, candidates[i].Play)
	}
	if alsoPlay != "" && !slices.Contains(plays, alsoPlay) {
		plays = append(plays, alsoPlay)
	}

	out := []*LaneComparison{}
	for i, play := range plays {
		ls, err := ss.CalculateLaneStats(play, 0)
		if err != nil {
			// Lane data is a bonus; a position is still explainable without
			// it, and the board-dynamics card simply won't ship.
			log.Err(err).Str("play", play).Msg("could not compute lane stats")
			continue
		}
		out = append(out, &LaneComparison{Play: play, Best: i == 0, Stats: ls})
	}
	return out
}

// computeFlags turns the fact pack into the yes/no answers that select concept
// cards. A flag that is false means a whole paragraph of prompt never gets
// sent.
func computeFlags(f *PositionFacts) Flags {
	fl := Flags{}
	for _, name := range knownFlags {
		fl[name] = false
	}
	fl["always"] = true

	// The card explains where the pre-endgame line actually falls, so it ships
	// for the whole late stretch, not only past the line.
	fl["pre_endgame"] = f.BagCount <= lateGameBagMax
	fl["behind_early"] = f.Spread < 0 && f.BagCount > lateGameBagMax

	// Turnover is only worth raising in the second half, and only when there
	// is something left in the bag worth racing to.
	secondHalf := f.BagCount <= secondHalfBagMax
	fl["turnover_relevant"] = secondHalf &&
		(f.UnseenBlanks == 2 || len(f.UnseenPowerTiles) > 0)

	fl["leave_matters"] = f.Best.Leave != ""
	fl["uses_blank"] = f.Best.UsesBlank

	// Setups and big chances are judged per opportunity, not per play: four
	// spellings of one hook are one thing to talk about, and each is too rare
	// on its own to clear a bar the four of them clear together.
	for _, ch := range f.Chances {
		if ch.IsSetup {
			fl["has_setup"] = true
		}
		if ch.IsBigChance {
			fl["has_big_chance"] = true
		}
	}
	for _, fu := range f.Followups {
		if fu.IsSetup {
			fl["has_setup"] = true
		}
		if fu.Grouped() {
			fl["has_grouped_followup"] = true
		}
		for _, d := range fu.NeededDraws {
			if d != "" {
				fl["has_needed_draw"] = true
			}
		}
	}

	ourBingoHigh, oppBingoHigh := false, false
	if f.PlayStats != nil {
		ourBingoHigh = f.PlayStats.OurBingoPct >= ourBingoPctHigh
		oppBingoHigh = f.PlayStats.OppBingoPct >= oppBingoPctHigh
	}
	// The opponent card is about reading their side of the table, so it needs
	// its own trigger rather than folding into the bingo card.
	fl["opp_bingo_pct_high"] = oppBingoHigh
	fl["bingo_matters"] = f.Best.IsBingo || ourBingoHigh || oppBingoHigh

	// Volatility only matters if the candidates actually differ in it, and
	// only the next two plies are close enough to reason about. Each ply is
	// compared against itself across candidates, not against other plies.
	for ply := 1; ply <= 2; ply++ {
		var minStdev, maxStdev float64
		seen := false
		for _, c := range f.Candidates {
			for _, p := range c.Plies {
				if p.Ply != ply {
					continue
				}
				if !seen {
					minStdev, maxStdev, seen = p.Stdev, p.Stdev, true
					continue
				}
				minStdev = min(minStdev, p.Stdev)
				maxStdev = max(maxStdev, p.Stdev)
			}
		}
		if seen && maxStdev-minStdev >= volatilitySpreadMin {
			fl["volatility_matters"] = true
		}
	}

	fl["equity_sacrifice"] = f.BestByEquity != nil && f.BestByEquity.Play != f.Best.Play

	for _, lc := range f.Lanes {
		if lc.Stats != nil && len(lc.Stats.Lanes) > 0 {
			fl["has_lane_data"] = true
		}
	}

	if f.Comparison != nil && f.Comparison.Rival != nil {
		fl["has_comparison"] = true
		fl["comparison_was_best"] = f.Comparison.WasBest
	}

	// A read is only a subject when it says something and that something
	// changed the answer. Otherwise the model never learns there was one.
	if f.Inference != nil && f.Inference.Matters {
		fl["has_inference"] = true
		fl["inference_changed_play"] = f.Inference.ChangedTopPlay
	}
	return fl
}

// bonusLabel names the premium squares a lane's replies covered, e.g. "TWS,
// DLS", for prose that talks about what a play opens.
func bonusLabel(premiums []stats.PremiumUse) string {
	names := []string{}
	for _, p := range premiums {
		if p.Name != "" && p.Bonus != board.NoBonus {
			names = append(names, p.Name)
		}
		if len(names) >= 2 {
			break
		}
	}
	return strings.Join(names, ", ")
}
