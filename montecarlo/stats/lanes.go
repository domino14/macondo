package stats

import (
	"sort"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/word-golib/tilemapping"
)

// The heat map knows how often a square gets covered but not what the plays
// covering it were doing. A player doesn't think in squares, though - they
// think in lanes: "that opens the O column", "row 15 is live now". This file
// re-reads the same simulation log the heat map reads and keeps the geometry
// the heat map throws away, so the AI explainer can make positional claims
// that come from sampled data rather than from reading a picture of a board.

// PremiumUse counts how many sampled plays in a lane covered a premium square
// of a given kind.
type PremiumUse struct {
	Bonus board.BonusSquare `json:"-"`
	Name  string            `json:"name"`
	Count int               `json:"count"`
}

// LaneStat summarizes the sampled plays that landed in one row or column.
type LaneStat struct {
	// Label is how a player would name the lane: "row 12", "column K".
	Label    string `json:"label"`
	Vertical bool   `json:"vertical"`
	// Index is the 0-based row (horizontal) or column (vertical).
	Index      int          `json:"index"`
	Count      int          `json:"count"`
	Pct        float64      `json:"pct"`
	MeanScore  float64      `json:"mean_score"`
	MaxScore   int          `json:"max_score"`
	BestPlay   string       `json:"best_play"`
	BingoCount int          `json:"bingo_count"`
	Premiums   []PremiumUse `json:"premiums,omitempty"`

	totalScore int
	premiums   map[board.BonusSquare]int
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

			if strings.HasPrefix(analyzedPlay, "exchange ") ||
				analyzedPlay == "pass" || analyzedPlay == "UNHANDLED" {
				ls.Scoreless++
				continue
			}

			playFields := strings.Fields(analyzedPlay)
			if len(playFields) != 2 {
				log.Debug().Str("play", analyzedPlay).Msg("skipping unparseable ply play")
				continue
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

			// Collect the squares this play actually covers. Playthrough
			// tiles were already on the board, so they cover nothing.
			placed := 0
			bonuses := map[board.BonusSquare]bool{}
			for idx := range mw {
				if mw[idx] == 0 {
					continue
				}
				placed++
				b := ss.board.GetBonus(row+(ri*idx), col+(ci*idx))
				if b != board.NoBonus {
					bonuses[b] = true
				}
			}
			if placed < 2 {
				ls.SingleTile++
				continue
			}
			ls.Placements++

			index := row
			if vertical {
				index = col
			}
			key := [2]int{index, boolToInt(vertical)}
			l, ok := byLane[key]
			if !ok {
				l = &LaneStat{
					Label:    LaneLabel(vertical, index),
					Vertical: vertical,
					Index:    index,
					premiums: map[board.BonusSquare]int{},
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
			for b := range bonuses {
				l.premiums[b]++
			}
		}
	}

	for _, l := range byLane {
		if l.Count > 0 {
			l.MeanScore = float64(l.totalScore) / float64(l.Count)
		}
		if ls.Total > 0 {
			l.Pct = float64(l.Count*100) / float64(ls.Total)
		}
		for b, ct := range l.premiums {
			l.Premiums = append(l.Premiums, PremiumUse{Bonus: b, Name: b.Name(), Count: ct})
		}
		sort.Slice(l.Premiums, func(i, j int) bool {
			if l.Premiums[i].Count != l.Premiums[j].Count {
				return l.Premiums[i].Count > l.Premiums[j].Count
			}
			return l.Premiums[i].Name < l.Premiums[j].Name
		})
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

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
