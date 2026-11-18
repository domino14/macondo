package main

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
)

// -dumpreads path writes, for every complete read the bot builds, how it
// scores the opponent's true leave, to check the read's calibration: does a
// leave read at 10% turn out to be the truth one time in ten, and a tile read
// at 70% sit on the rack 70% of the time? The read itself is left alone.
var (
	readDumpMu sync.Mutex
	readDump   *json.Encoder
)

// calBins are the edges of the leave-probability bins, on a log scale.
var calBins = []float64{1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.01}

type calTile struct {
	Tile string  `json:"t"`
	C    int     `json:"c"` // holds at least C of it
	P    float64 `json:"p"`
	Hit  bool    `json:"hit"`
}

type calRecord struct {
	Bag      int     `json:"bag"`
	K        int     `json:"k"` // leave length
	OppType  string  `json:"oppType"`
	OppTiles int     `json:"oppTiles"`
	OppScore int32   `json:"oppScore"`
	N        int     `json:"n"`
	Truth    string  `json:"truth"`
	PTrue    float64 `json:"pTrue"`
	PriorTr  float64 `json:"priorTrue"`
	Rank     int     `json:"rank"` // 1 = the read's top leave; 0 = not in the read
	Entropy  float64 `json:"entropyBits"`
	Top      float64 `json:"top"`
	// Bins[i] = [summed probability, leaves, truth in bin (0/1)] for leaves
	// whose probability falls below calBins[i] and at or above calBins[i-1].
	Bins  [][3]float64 `json:"bins"`
	Tiles []calTile    `json:"tiles"`
}

func openReadDump(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	readDump = json.NewEncoder(f)
	return nil
}

func dumpRead(g *game.Game, racks []montecarlo.InferredRack) []montecarlo.InferredRack {
	truth, err := game.ExtractLastOppLeave(g)
	if err != nil || len(racks) == 0 {
		return racks
	}
	truth = sorted(truth)
	alph := g.Alphabet()
	pool := unseenPool(g)
	rec := calRecord{Bag: g.Bag().TilesRemaining(), K: len(truth), N: len(racks),
		Truth: tilemapping.MachineWord(truth).UserVisible(alph), Bins: make([][3]float64, len(calBins))}
	evts := g.History().Events[:g.Turn()]
	for i := len(evts) - 1; i >= 0; i-- {
		e := evts[i]
		if e.Type == pb.GameEvent_CHALLENGE_BONUS {
			continue
		}
		switch e.Type {
		case pb.GameEvent_EXCHANGE:
			rec.OppType, rec.OppTiles = "exch", len([]rune(e.Exchanged))
		case pb.GameEvent_TILE_PLACEMENT_MOVE:
			rec.OppType, rec.OppTiles = "play", len(strings.ReplaceAll(e.PlayedTiles, ".", ""))
		default:
			rec.OppType = strings.ToLower(e.Type.String())
		}
		rec.OppScore = e.Score
		break
	}
	total, priorTotal := 0.0, 0.0
	priorTruth := 0.0
	for _, r := range racks {
		total += r.Weight
		pr := leavePrior(r.Leave, pool)
		priorTotal += pr
		if slices.Equal(sorted(r.Leave), truth) {
			priorTruth = pr
		}
	}
	if total <= 0 {
		return racks
	}
	var atLeast [64][7]float64
	for _, r := range racks {
		p := r.Weight / total
		rec.Top = max(rec.Top, p)
		if p > 0 {
			rec.Entropy -= p * math.Log2(p)
		}
		b := 0
		for b < len(calBins)-1 && p >= calBins[b] {
			b++
		}
		rec.Bins[b][0] += p
		rec.Bins[b][1]++
		if slices.Equal(sorted(r.Leave), truth) {
			rec.PTrue = p
			rec.Bins[b][2] = 1
		}
		var counts [64]int
		for _, t := range r.Leave {
			if counts[t] < 7 {
				atLeast[t][counts[t]] += p
			}
			counts[t]++
		}
	}
	if rec.PTrue > 0 {
		rec.Rank = 1
		for _, r := range racks {
			if r.Weight/total > rec.PTrue {
				rec.Rank++
			}
		}
	}
	if priorTotal > 0 {
		rec.PriorTr = priorTruth / priorTotal
	}
	var truthCounts [64]int
	for _, t := range truth {
		truthCounts[t]++
	}
	for t := range atLeast {
		for c := range atLeast[t] {
			p, hit := atLeast[t][c], truthCounts[t] > c
			if p < 0.005 && !hit {
				continue
			}
			rec.Tiles = append(rec.Tiles, calTile{Tile: tilemapping.MachineLetter(t).UserVisible(alph, false),
				C: c + 1, P: p, Hit: hit})
		}
	}
	readDumpMu.Lock()
	readDump.Encode(rec)
	readDumpMu.Unlock()
	return racks
}
