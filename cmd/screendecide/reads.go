package main

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
)

// A netRead is an opponent read computed elsewhere (cmd/netread on the
// network-improvements branch): a posterior over the leaves the opponent may
// have kept, handed to the sim in place of the bot's own inference.
type netRead struct {
	racks []montecarlo.InferredRack
	k     int
}

// loadReads turns netread's raw net outputs into posteriors: each leave's
// prior weight times the softmax share, at temperature tau, of the play the
// opponent made among its alternatives. head "v" uses the value head in atanh
// space (undoing its saturation in decided games), "s" the spread head.
func loadReads(path, head string, tau float64, alph *tilemapping.TileMapping) (map[string]netRead, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		r = gz
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<28)
	reads := map[string]netRead{}
	for sc.Scan() {
		var rec struct {
			Key    string      `json:"key"`
			K      int         `json:"k"`
			Leaves []string    `json:"leaves"`
			W      []float64   `json:"w"`
			V      [][]float64 `json:"v"`
			S      [][]float64 `json:"s"`
		}
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return nil, err
		}
		vals := rec.V
		if head == "s" {
			vals = rec.S
		}
		if len(rec.Leaves) == 0 || len(vals) != len(rec.Leaves) {
			continue
		}
		nr := netRead{k: rec.K}
		for i, l := range rec.Leaves {
			mw, err := tilemapping.ToMachineWord(l, alph)
			if err != nil {
				return nil, fmt.Errorf("%s: leave %q: %w", rec.Key, l, err)
			}
			nr.racks = append(nr.racks, montecarlo.InferredRack{Leave: mw, Weight: rec.W[i] * share(vals[i], head, tau)})
		}
		reads[rec.Key] = nr
	}
	return reads, sc.Err()
}

// share is the softmax weight of v[0], the actual play, among all of v.
func share(v []float64, head string, tau float64) float64 {
	x := make([]float64, len(v))
	m := math.Inf(-1)
	for i, y := range v {
		if head == "v" {
			y = math.Atanh(max(-0.9999, min(0.9999, y)))
		}
		x[i] = y
		m = max(m, y)
	}
	z := 0.0
	for _, y := range x {
		z += math.Exp((y - m) / tau)
	}
	return math.Exp((x[0]-m)/tau) / z
}

// readFor hands each position's read to the bot (bot.ExternalRead) by its
// game, since many positions are decided at once.
var readFor sync.Map // *game.Game -> netRead

func externalRead(g *game.Game) ([]montecarlo.InferredRack, int, bool) {
	v, ok := readFor.Load(g)
	if !ok {
		return nil, 0, false
	}
	nr := v.(netRead)
	return nr.racks, nr.k, true
}
