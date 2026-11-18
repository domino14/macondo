package main

import (
	"cmp"
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"sync"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
)

// -oraclemode: what an -oracle bot is told in place of the opponent's true
// leave (bot.OracleLeave), to ask how much of the oracle's gain a leave that
// is like the truth keeps, and like it in which sense:
//
//	similar  the nearest leave by hand-built features (bingo chance, leave
//	         value, scoring tiles, balance)
//	value    a leave of about the same leave value, otherwise random
//	swap     the truth with one tile swapped for another the pool allows
//	random   a leave drawn from the tile-count prior (no information)
//
// The mix- modes hand the sim a posterior instead (bot.OracleRead): the truth
// at weight -oraclep, the rest spread over its mixNeighbours nearest leaves
// (mix-similar), its one-tile swaps (mix-swap), leaves of about its value
// (mix-value), or over leaves drawn from the prior, by frequency (mix-prior).
// They ask whether a read that misses the truth does better when its misses
// are like the truth.
//
// The part- modes tell the sim only some of the truth's tiles and draw the
// rest of the opponent's rack fresh: part1, part2 and part3 a random k of
// them (all of them when the leave is that short), part-power those that are
// blanks, S or J, Q, X, Z (none: a plain sim). They ask how the oracle's gain
// builds up tile by tile.
//
// Every substitute has the truth's length, differs from it, and is one the
// opponent could hold. Candidates for similar and value are drawn from the
// prior, so the nearest is the nearest among leaves the opponent might
// plausibly have.

const (
	substCandidates = 2000
	substDraws      = 100
	mixNeighbours   = 8
)

type substituter struct {
	mode string
	p    float64 // the truth's weight in the mix- modes
	ld   *tilemapping.LetterDistribution
	// bingos[b] holds the alphagrams of 7-b letters that, with b blanks,
	// make a seven-letter word.
	bingos [3]map[uint64]bool
	vowel  [64]bool
	power  [64]bool
}

// A substitution is what the bot was told at one position, for the record.
type substitution struct {
	True, Sub string
	Dist      float64
}

var substFor sync.Map // *game.Game -> *substitution

func newSubstituter(mode string, gd *kwg.KWG, ld *tilemapping.LetterDistribution) (*substituter, error) {
	switch mode {
	case "similar", "value", "swap", "random", "mix-similar", "mix-prior", "mix-swap", "mix-value",
		"part1", "part2", "part3", "part-power":
	default:
		return nil, fmt.Errorf("unknown -oraclemode %q", mode)
	}
	s := &substituter{mode: mode, ld: ld}
	for b := range s.bingos {
		s.bingos[b] = map[uint64]bool{}
	}
	walkWords(gd, 7, func(w []tilemapping.MachineLetter) {
		a := slices.Clone(w)
		slices.Sort(a)
		s.bingos[0][key(a)] = true
		for i := range a {
			if i > 0 && a[i] == a[i-1] {
				continue
			}
			one := slices.Delete(slices.Clone(a), i, i+1)
			s.bingos[1][key(one)] = true
			for j := range one {
				if j > 0 && one[j] == one[j-1] {
					continue
				}
				s.bingos[2][key(slices.Delete(slices.Clone(one), j, j+1))] = true
			}
		}
	})
	alph := ld.TileMapping()
	for _, l := range "AEIOU" {
		ml, _ := alph.Val(string(l))
		s.vowel[ml] = true
	}
	for _, l := range "JQXZ" {
		ml, _ := alph.Val(string(l))
		s.power[ml] = true
	}
	return s, nil
}

// walkWords calls f with every word of length n in the KWG's DAWG.
func walkWords(gd *kwg.KWG, n int, f func([]tilemapping.MachineLetter)) {
	word := make([]tilemapping.MachineLetter, 0, n)
	var walk func(node uint32)
	walk = func(node uint32) {
		if node == 0 {
			return
		}
		for i := node; ; i++ {
			word = append(word, tilemapping.MachineLetter(gd.Tile(i)))
			if len(word) == n {
				if gd.Accepts(i) {
					f(word)
				}
			} else {
				walk(gd.ArcIndex(i))
			}
			word = word[:len(word)-1]
			if gd.IsEnd(i) {
				break
			}
		}
	}
	walk(gd.ArcIndex(0))
}

func key(sorted []tilemapping.MachineLetter) uint64 {
	var k uint64
	for _, t := range sorted {
		k = k<<5 | uint64(t)
	}
	return k
}

// substitute is the bot.OracleLeave hook.
func (s *substituter) substitute(g *game.Game, truth []tilemapping.MachineLetter,
	value func(tilemapping.MachineWord) float64) []tilemapping.MachineLetter {

	pool, rng := s.setup(g)
	truth = sorted(truth)
	var sub []tilemapping.MachineLetter
	dist := math.NaN()
	switch s.mode {
	case "random":
		// A small pool may hold no leave but the truth; then keep it.
		for range substCandidates {
			if sub = drawLeave(pool, len(truth), rng); sub != nil && !slices.Equal(sub, truth) {
				break
			}
			sub = nil
		}
	case "swap":
		sub = s.swap(pool, truth, rng)
	default:
		sub, dist = s.nearest(pool, truth, value, rng)
	}
	if sub == nil {
		return truth
	}
	alph := g.Alphabet()
	substFor.Store(g, &substitution{True: tilemapping.MachineWord(truth).UserVisible(alph),
		Sub: tilemapping.MachineWord(sub).UserVisible(alph), Dist: dist})
	return sub
}

// setup returns the mover's unseen pool (the bag and the opponent's rack) and
// a generator seeded by the game and mode.
func (s *substituter) setup(g *game.Game) ([]uint8, *rand.Rand) {
	pool := g.Bag().PeekMap()
	for _, t := range g.RackFor(g.NextPlayer()).TilesOn() {
		pool[t]++
	}
	seed := g.Seed()
	h := sha256.Sum256(append(seed[:], []byte(s.mode)...))
	return pool, rand.New(rand.NewSource(int64(h[0]) | int64(h[1])<<8 | int64(h[2])<<16 | int64(h[3])<<24))
}

// read is the bot.OracleRead hook of the mix- modes.
func (s *substituter) read(g *game.Game, truth []tilemapping.MachineLetter,
	value func(tilemapping.MachineWord) float64) []montecarlo.InferredRack {

	pool, rng := s.setup(g)
	truth = sorted(truth)
	if strings.HasPrefix(s.mode, "part") {
		return s.partial(g, truth, rng)
	}
	racks := []montecarlo.InferredRack{{Leave: truth, Weight: s.p}}
	var rest [][]tilemapping.MachineLetter
	switch s.mode {
	case "mix-similar":
		rest, _ = s.nearestN(pool, truth, value, rng, mixNeighbours)
	case "mix-swap", "mix-value":
		seen := map[uint64]bool{key(truth): true}
		for range substCandidates {
			if len(rest) == mixNeighbours {
				break
			}
			var c []tilemapping.MachineLetter
			if s.mode == "mix-swap" {
				c = s.swap(pool, truth, rng)
			} else {
				c = drawLeave(pool, len(truth), rng)
				if c != nil && math.Abs(value(tilemapping.MachineWord(c))-value(tilemapping.MachineWord(truth))) > 1 {
					continue
				}
			}
			if c != nil && !seen[key(c)] {
				seen[key(c)] = true
				rest = append(rest, c)
			}
		}
	default:
		for range substCandidates / 10 {
			if c := drawLeave(pool, len(truth), rng); c != nil && !slices.Equal(c, truth) {
				rest = append(rest, c)
			}
		}
	}
	// Repeats of a prior draw stay separate entries: the sim samples by
	// weight, so each counts by its frequency.
	for _, c := range rest {
		racks = append(racks, montecarlo.InferredRack{Leave: c, Weight: (1 - s.p) / float64(len(rest))})
	}
	alph := g.Alphabet()
	sub := ""
	if len(rest) > 0 {
		sub = tilemapping.MachineWord(rest[0]).UserVisible(alph)
	}
	substFor.Store(g, &substitution{True: tilemapping.MachineWord(truth).UserVisible(alph), Sub: sub, Dist: float64(len(rest))})
	return racks
}

// partial tells the sim a subset of the truth's tiles as the opponent's whole
// read; the rest of the rack is drawn as in a plain sim.
func (s *substituter) partial(g *game.Game, truth []tilemapping.MachineLetter, rng *rand.Rand) []montecarlo.InferredRack {
	var part []tilemapping.MachineLetter
	if s.mode == "part-power" {
		for _, t := range truth {
			if l := s.ld.TileMapping().Letter(t); t == 0 || s.power[t] || l == "S" {
				part = append(part, t)
			}
		}
	} else {
		k := int(s.mode[len(s.mode)-1] - '0')
		perm := rng.Perm(len(truth))
		for _, i := range perm[:min(k, len(truth))] {
			part = append(part, truth[i])
		}
		part = sorted(part)
	}
	alph := g.Alphabet()
	substFor.Store(g, &substitution{True: tilemapping.MachineWord(truth).UserVisible(alph),
		Sub: tilemapping.MachineWord(part).UserVisible(alph), Dist: float64(len(part))})
	if len(part) == 0 {
		return nil
	}
	return []montecarlo.InferredRack{{Leave: part, Weight: 1}}
}

func sorted(l []tilemapping.MachineLetter) []tilemapping.MachineLetter {
	c := slices.Clone(l)
	slices.Sort(c)
	return c
}

// drawLeave draws n tiles from pool without replacement.
func drawLeave(pool []uint8, n int, rng *rand.Rand) []tilemapping.MachineLetter {
	tiles := make([]tilemapping.MachineLetter, 0, 100)
	for t, c := range pool {
		for range c {
			tiles = append(tiles, tilemapping.MachineLetter(t))
		}
	}
	if len(tiles) < n {
		return nil
	}
	for i := 0; i < n; i++ {
		j := i + rng.Intn(len(tiles)-i)
		tiles[i], tiles[j] = tiles[j], tiles[i]
	}
	return sorted(tiles[:n])
}

func (s *substituter) swap(pool []uint8, truth []tilemapping.MachineLetter, rng *rand.Rand) []tilemapping.MachineLetter {
	// The pool the swapped-in tile comes from excludes the truth's own tiles.
	rest := slices.Clone(pool)
	for _, t := range truth {
		rest[t]--
	}
	type option struct{ i, t int }
	var opts []option
	for i := range truth {
		if i > 0 && truth[i] == truth[i-1] {
			continue
		}
		for t, c := range rest {
			if c > 0 && tilemapping.MachineLetter(t) != truth[i] {
				opts = append(opts, option{i, t})
			}
		}
	}
	if len(opts) == 0 {
		return nil
	}
	o := opts[rng.Intn(len(opts))]
	sub := slices.Clone(truth)
	sub[o.i] = tilemapping.MachineLetter(o.t)
	return sorted(sub)
}

// features describes a leave in the pool it was kept from.
func (s *substituter) features(leave []tilemapping.MachineLetter, pool []uint8,
	value func(tilemapping.MachineWord) float64, rng *rand.Rand) []float64 {

	k := len(leave)
	var face, power, vowels, blanks, ss, q, u, dups float64
	for i, t := range leave {
		face += float64(s.ld.Score(t))
		if s.power[t] {
			power++
		}
		if s.vowel[t] {
			vowels++
		}
		switch s.ld.TileMapping().Letter(t) {
		case "?":
			blanks++
		case "S":
			ss++
		case "Q":
			q++
		case "U":
			u++
		}
		if i > 0 && leave[i] == leave[i-1] {
			dups++
		}
	}
	qNoU := 0.0
	if q > 0 && u == 0 {
		qNoU = 1
	}
	return []float64{
		s.bingoChance(leave, pool, rng),
		value(tilemapping.MachineWord(leave)),
		face,
		power,
		blanks,
		ss,
		math.Abs(vowels - 0.4*float64(k)),
		dups,
		qNoU,
	}
}

// Feature weights in the standardised distance: bingo chance and leave value
// count double.
var featureWeights = []float64{2, 2, 1, 1, 1, 1, 1, 1, 1}

// bingoChance is the chance that the leave plus a random draw from the rest
// of the pool anagrams to a seven-letter word (blanks as anything).
func (s *substituter) bingoChance(leave []tilemapping.MachineLetter, pool []uint8, rng *rand.Rand) float64 {
	rest := slices.Clone(pool)
	for _, t := range leave {
		rest[t]--
	}
	need := 7 - len(leave)
	if need <= 0 {
		return 0
	}
	hits := 0
	rack := make([]tilemapping.MachineLetter, 0, 7)
	for range substDraws {
		d := drawLeave(rest, need, rng)
		if d == nil {
			return 0
		}
		rack = append(append(rack[:0], leave...), d...)
		slices.Sort(rack)
		b := 0
		for b < len(rack) && rack[b] == 0 {
			b++
		}
		if b > 2 || s.bingos[b][key(rack[b:])] {
			hits++
		}
	}
	return float64(hits) / substDraws
}

func (s *substituter) nearest(pool []uint8, truth []tilemapping.MachineLetter,
	value func(tilemapping.MachineWord) float64, rng *rand.Rand) ([]tilemapping.MachineLetter, float64) {

	if s.mode != "value" {
		n, d := s.nearestN(pool, truth, value, rng, 1)
		if len(n) == 0 {
			return nil, math.NaN()
		}
		return n[0], d
	}
	seen := map[uint64]bool{key(truth): true}
	var cands [][]tilemapping.MachineLetter
	for range substCandidates {
		c := drawLeave(pool, len(truth), rng)
		if c == nil {
			return nil, math.NaN()
		}
		if k := key(c); !seen[k] {
			seen[k] = true
			cands = append(cands, c)
		}
	}
	if len(cands) == 0 {
		return nil, math.NaN()
	}
	tv := value(tilemapping.MachineWord(truth))
	var near [][]tilemapping.MachineLetter
	best, bestD := cands[0], math.Inf(1)
	for _, c := range cands {
		d := math.Abs(value(tilemapping.MachineWord(c)) - tv)
		if d <= 1 {
			near = append(near, c)
		}
		if d < bestD {
			best, bestD = c, d
		}
	}
	if len(near) > 0 {
		c := near[rng.Intn(len(near))]
		return c, math.Abs(value(tilemapping.MachineWord(c)) - tv)
	}
	return best, bestD
}

// nearestN returns up to n candidates drawn from the prior nearest the truth by
// the weighted, standardised feature distance, nearest first, and the nearest
// one's distance.
func (s *substituter) nearestN(pool []uint8, truth []tilemapping.MachineLetter,
	value func(tilemapping.MachineWord) float64, rng *rand.Rand, n int) ([][]tilemapping.MachineLetter, float64) {

	seen := map[uint64]bool{key(truth): true}
	var cands [][]tilemapping.MachineLetter
	for range substCandidates {
		c := drawLeave(pool, len(truth), rng)
		if c == nil {
			return nil, math.NaN()
		}
		if k := key(c); !seen[k] {
			seen[k] = true
			cands = append(cands, c)
		}
	}
	if len(cands) == 0 {
		return nil, math.NaN()
	}
	tf := s.features(truth, pool, value, rng)
	fs := make([][]float64, len(cands))
	for i, c := range cands {
		fs[i] = s.features(c, pool, value, rng)
	}
	// Standardise each feature over the candidates.
	sd := make([]float64, len(tf))
	for j := range tf {
		var m, v float64
		for _, f := range fs {
			m += f[j]
		}
		m /= float64(len(fs))
		for _, f := range fs {
			v += (f[j] - m) * (f[j] - m)
		}
		sd[j] = math.Sqrt(v / float64(len(fs)))
	}
	ds := make([]float64, len(fs))
	for i, f := range fs {
		for j := range tf {
			if sd[j] == 0 {
				continue
			}
			z := (f[j] - tf[j]) / sd[j]
			ds[i] += featureWeights[j] * z * z
		}
	}
	idx := make([]int, len(cands))
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int { return cmp.Compare(ds[a], ds[b]) })
	out := make([][]tilemapping.MachineLetter, 0, n)
	for _, i := range idx[:min(n, len(idx))] {
		out = append(out, cands[i])
	}
	return out, math.Sqrt(ds[idx[0]])
}
