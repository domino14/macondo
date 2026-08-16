package montecarlo

import (
	"strings"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/stats"
)

// EquityStats and ScoreDetails render the simulation to fixed-width tables for
// a human to read. Anything that wants to reason about the numbers - the AI
// explainer, in particular - should use CandidateStats instead, so that nobody
// has to parse those tables back into data.

// CandidatePlyStats is one ply's worth of results for a candidate play. Ply is
// 1-based, matching the "**Ply N**" headings in ScoreDetails.
type CandidatePlyStats struct {
	Ply        int     `json:"ply"`
	Ours       bool    `json:"ours"`
	MeanScore  float64 `json:"mean_score"`
	Stdev      float64 `json:"stdev"`
	BingoPct   float64 `json:"bingo_pct"`
	Iterations int     `json:"iterations"`
}

// CandidateStats is everything the simulation knows about one candidate play.
type CandidateStats struct {
	// Move is the play itself. Match against it rather than against Play when
	// it matters that two notations can mean the same move - on an empty
	// board a vertical opener is a transposition of its horizontal twin.
	Move *move.Move `json:"-"`
	// Play is in playthrough notation, e.g. "5D (S)PIC(A)".
	Play        string `json:"play"`
	Leave       string `json:"leave"`
	Score       int    `json:"score"`
	TilesPlayed int    `json:"tiles_played"`
	IsBingo     bool   `json:"is_bingo"`
	// UsesBlank is true if the play spends a blank.
	UsesBlank bool `json:"uses_blank"`
	// WinPct and Equity carry 99% confidence half-widths, the same intervals
	// EquityStats prints.
	WinPct   float64 `json:"win_pct"`
	WinPctCI float64 `json:"win_pct_ci"`
	Equity   float64 `json:"equity"`
	EquityCI float64 `json:"equity_ci"`
	// Ignored plays were cut off early by the stopping condition; they are
	// the ones EquityStats marks with an ❌.
	Ignored bool                `json:"ignored"`
	Plies   []CandidatePlyStats `json:"plies"`
}

// CandidateStats returns the simulated plays, best win% first - the same order
// and the same numbers as EquityStats and ScoreDetails.
func (s *Simmer) CandidateStats() []CandidateStats {
	if s.simmedPlays == nil {
		return nil
	}
	s.sortPlaysByWinRate(true)
	s.simmedPlays.RLock()
	defer s.simmedPlays.RUnlock()

	out := make([]CandidateStats, 0, len(s.simmedPlays.plays))
	for _, play := range s.simmedPlays.plays {
		m := play.play
		usesBlank := false
		for _, t := range m.Tiles() {
			if t.IsBlanked() {
				usesBlank = true
				break
			}
		}
		cs := CandidateStats{
			Move: m,
			// The description right-aligns the coordinate for table layout;
			// that padding is presentation, not part of the play.
			Play:        strings.TrimSpace(s.origGame.Board().MoveDescriptionWithPlaythrough(m)),
			Leave:       m.LeaveString(),
			Score:       m.Score(),
			TilesPlayed: m.TilesPlayed(),
			IsBingo:     m.BingoPlayed(),
			UsesBlank:   usesBlank,
			WinPct:      100.0 * play.winPctStats.Mean(),
			WinPctCI:    100.0 * stats.Z99 * play.winPctStats.StandardError(),
			Equity:      play.equityStats.Mean(),
			EquityCI:    stats.Z99 * play.equityStats.StandardError(),
			Ignored:     play.ignore.Load(),
		}
		for ply := 0; ply < s.maxPlies && ply < len(play.scoreStats); ply++ {
			cs.Plies = append(cs.Plies, CandidatePlyStats{
				Ply:        ply + 1,
				Ours:       ply%2 == 1,
				MeanScore:  play.scoreStats[ply].Mean(),
				Stdev:      play.scoreStats[ply].Stdev(),
				BingoPct:   100.0 * play.bingoStats[ply].Mean(),
				Iterations: play.scoreStats[ply].Iterations(),
			})
		}
		out = append(out, cs)
	}
	return out
}
