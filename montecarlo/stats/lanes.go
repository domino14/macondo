package stats

import (
	"sort"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/move"
	"github.com/domino14/word-golib/tilemapping"
)

// The heat map knows how often a square gets covered but not what the plays
// covering it were doing. A player doesn't think in squares, though - they
// think in lanes: "that opens the O column", "row 15 is live now". This file
// re-reads the same simulation log the heat map reads and keeps the geometry
// the heat map throws away, so the AI explainer can make positional claims
// that come from sampled data rather than from reading a picture of a board.

// BigReplyScore is what makes a reply a big one. What an open board costs you
// is the occasional huge turn rather than a point or two on the average one,
// so this is the figure a claim about openness rests on - and unlike a lane
// share, it is board-wide: the opponent replies somewhere every turn, so
// closing one lane moves their plays around without necessarily making any of
// them worse.
const BigReplyScore = 50

// Footprint is where a play puts its new tiles: the lane it sits in, and how
// far it reaches across the perpendicular ones. A lane can fall silent after a
// play for a reason that has nothing to do with defense - the tiles that made
// it a lane at all are the other candidate's - and this is what tells that
// case apart from a play that really did shut the lane down.
type Footprint struct {
	Vertical bool `json:"vertical"`
	// Index is the play's own lane: its row if horizontal, its column if
	// vertical.
	Index int `json:"index"`
	// Start and End bound the perpendicular lanes it drops a new tile in.
	// Playthrough tiles are left out: they were on the board already, so the
	// lanes crossing them were reachable before this play too.
	Start int `json:"start"`
	End   int `json:"end"`
}

// Touches reports whether the play puts a new tile in the given lane.
func (fp *Footprint) Touches(vertical bool, index int) bool {
	if fp == nil {
		return false
	}
	if vertical == fp.Vertical {
		return index == fp.Index
	}
	return index >= fp.Start && index <= fp.End
}

// LaneStat summarizes the sampled plays that landed in one row or column.
type LaneStat struct {
	// Label is how a player would name the lane: "row 12", "column K".
	Label    string `json:"label"`
	Vertical bool   `json:"vertical"`
	// Index is the 0-based row (horizontal) or column (vertical).
	Index int `json:"index"`
	// The premium squares a lane's replies covered used to be counted here
	// too, and named in the prompt as "covers TLS, DWS". The model read the
	// abbreviations back out as prose about "the dangerous triple lane", which
	// is a claim about the board nobody measured, so nothing counts them now.
	Count      int     `json:"count"`
	Pct        float64 `json:"pct"`
	MeanScore  float64 `json:"mean_score"`
	MaxScore   int     `json:"max_score"`
	BestPlay   string  `json:"best_play"`
	BingoCount int     `json:"bingo_count"`

	totalScore int
}

// LaneStats is the lane breakdown of every sampled continuation of one root
// play at one ply. Ply 0 is the opponent's reply - what the play opens up for
// them - and ply 1 is our own follow-up.
type LaneStats struct {
	Play string `json:"play"`
	Ply  int    `json:"ply"`
	// Total is every sampled continuation, whether or not it was attributed
	// to a lane. Pcts are shares of this.
	Total int `json:"total"`
	// Placements is the continuations that were attributed to a lane, i.e.
	// placements of two or more tiles.
	Placements int `json:"placements"`
	// SingleTile counts one-tile plays. They are deliberately left out of the
	// lanes: dropping a tile somewhere doesn't tell you a lane is open, and
	// a high share of them says the board is tight.
	SingleTile int `json:"single_tile"`
	// Scoreless counts passes and exchanges.
	Scoreless int `json:"scoreless"`
	// BigReplies counts sampled continuations worth BigReplyScore or more,
	// wherever they landed - one-tile plays included. This is the board-wide
	// measure of what the root play left open, as against the per-lane shares
	// below, which only say where the opponent went.
	BigReplies int     `json:"big_replies"`
	BigPct     float64 `json:"big_pct"`
	// Footprint is where the root play itself sits.
	Footprint *Footprint `json:"footprint,omitempty"`
	// Lanes is sorted by Count, most frequent first.
	Lanes []*LaneStat `json:"lanes"`
}

// Lane finds the stats for a lane, or nil. Vertical lanes are columns.
func (ls *LaneStats) Lane(vertical bool, index int) *LaneStat {
	for _, l := range ls.Lanes {
		if l.Vertical == vertical && l.Index == index {
			return l
		}
	}
	return nil
}

// LaneLabel names a lane the way a player would: rows are numbered from 1,
// columns are lettered from A.
func LaneLabel(vertical bool, index int) string {
	if vertical {
		return "column " + string(rune('A'+index))
	}
	return "row " + strconv.Itoa(index+1)
}

// CalculateLaneStats buckets every sampled continuation of the given root play
// by the row or column it was played in.
func (ss *SimStats) CalculateLaneStats(play string, ply int) (*LaneStats, error) {
	iters, err := ss.simmer.ReadHeatmap()
	if err != nil {
		return nil, err
	}
	normalizedPlay := Normalize(play)
	ls := &LaneStats{Play: play, Ply: ply}
	if p, err := ss.parsePlacement(normalizedPlay); err != nil {
		return nil, err
	} else if p != nil {
		ls.Footprint = &Footprint{
			Vertical: p.vertical, Index: p.index, Start: p.start, End: p.end,
		}
	}
	byLane := map[[2]int]*LaneStat{}

	for i := range iters {
		for j := range iters[i].Plays {
			if normalizedPlay != Normalize(iters[i].Plays[j].Play) {
				continue
			}
			if len(iters[i].Plays[j].Plies) <= ply {
				continue
			}
			logPlay := iters[i].Plays[j].Plies[ply]
			analyzedPlay := Normalize(logPlay.Play)
			ls.Total++
			if logPlay.Pts >= BigReplyScore {
				ls.BigReplies++
			}

			if strings.HasPrefix(analyzedPlay, "exchange ") ||
				analyzedPlay == "pass" || analyzedPlay == "UNHANDLED" {
				ls.Scoreless++
				continue
			}

			p, err := ss.parsePlacement(analyzedPlay)
			if err != nil {
				return nil, err
			}
			if p == nil {
				continue
			}
			if p.placed < 2 {
				ls.SingleTile++
				continue
			}
			ls.Placements++

			vertical, index := p.vertical, p.index
			key := [2]int{index, boolToInt(vertical)}
			l, ok := byLane[key]
			if !ok {
				l = &LaneStat{
					Label:    LaneLabel(vertical, index),
					Vertical: vertical,
					Index:    index,
				}
				byLane[key] = l
			}
			l.Count++
			l.totalScore += logPlay.Pts
			if logPlay.Pts > l.MaxScore || l.BestPlay == "" {
				l.MaxScore = logPlay.Pts
				l.BestPlay = strings.TrimSpace(logPlay.Play)
			}
			if logPlay.Bingo {
				l.BingoCount++
			}
		}
	}

	if ls.Total > 0 {
		ls.BigPct = float64(ls.BigReplies*100) / float64(ls.Total)
	}

	for _, l := range byLane {
		if l.Count > 0 {
			l.MeanScore = float64(l.totalScore) / float64(l.Count)
		}
		if ls.Total > 0 {
			l.Pct = float64(l.Count*100) / float64(ls.Total)
		}
		ls.Lanes = append(ls.Lanes, l)
	}
	sort.Slice(ls.Lanes, func(i, j int) bool {
		if ls.Lanes[i].Count != ls.Lanes[j].Count {
			return ls.Lanes[i].Count > ls.Lanes[j].Count
		}
		if ls.Lanes[i].MaxScore != ls.Lanes[j].MaxScore {
			return ls.Lanes[i].MaxScore > ls.Lanes[j].MaxScore
		}
		return ls.Lanes[i].Label < ls.Lanes[j].Label
	})

	return ls, nil
}

// placement is where one play put its new tiles, and what those squares are
// worth. The root play and every sampled reply are read the same way, so a
// footprint and a lane are measured on the same terms.
type placement struct {
	vertical bool
	// index is the play's own lane, start and end the perpendicular lanes it
	// places tiles in.
	index      int
	start, end int
	placed     int
}

// parsePlacement reads a normalized play - "8D W.RD", with dots for the tiles
// already on the board. It returns nil, nil for anything it cannot read as a
// placement, which is not an error: an unparseable line in the log costs us
// one sample, not the whole analysis.
func (ss *SimStats) parsePlacement(analyzedPlay string) (*placement, error) {
	playFields := strings.Fields(analyzedPlay)
	if len(playFields) != 2 {
		log.Debug().Str("play", analyzedPlay).Msg("skipping unparseable play")
		return nil, nil
	}
	row, col, vertical := move.FromBoardGameCoords(strings.ToUpper(playFields[0]), false)
	mw, err := tilemapping.ToMachineWord(playFields[1], ss.game.Alphabet())
	if err != nil {
		return nil, err
	}
	ri, ci := 1, 0
	if !vertical {
		ri, ci = 0, 1
	}

	p := &placement{vertical: vertical, index: row}
	if vertical {
		p.index = col
	}
	// Only the squares this play actually covers. Playthrough tiles were on
	// the board already, so they open no lane that wasn't open before.
	for idx := range mw {
		if mw[idx] == 0 {
			continue
		}
		r, c := row+(ri*idx), col+(ci*idx)
		cross := c
		if vertical {
			cross = r
		}
		if p.placed == 0 {
			p.start, p.end = cross, cross
		}
		p.start, p.end = min(p.start, cross), max(p.end, cross)
		p.placed++
	}
	if p.placed == 0 {
		return nil, nil
	}
	return p, nil
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
