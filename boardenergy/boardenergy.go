// Package boardenergy scores board shapes with the pairwise maximum-entropy
// (Ising) model of tile occupancy fitted by pytorch/ising_fit.py: each square
// is a spin, +1 holding a tile and -1 empty, and
//
//	E(s) = -sum_i h_i s_i - 1/2 sum_{i != j} W_ij s_i s_j
//
// with one (h, W) per band of tiles on the board. High energy means a shape
// the fitted games rarely make; low energy, a very typical one.
package boardenergy

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

// N is the number of squares on the standard board.
const N = 225

// Band holds the parameters fitted on boards with Lo..Hi tiles.
type Band struct {
	Lo, Hi int
	H      [N]float64
	W      []float64 // N*N, symmetric, zero diagonal
}

// Model is a set of bands covering the game.
type Model struct {
	Bands []Band
}

// Load reads the binary written by pytorch/ising_export.py.
func Load(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 12 || string(b[:4]) != "ISNG" {
		return nil, errors.New("boardenergy: not an ISNG parameter file")
	}
	if v := binary.LittleEndian.Uint32(b[4:]); v != 1 {
		return nil, fmt.Errorf("boardenergy: version %d, want 1", v)
	}
	nb := int(binary.LittleEndian.Uint32(b[8:]))
	off := 12
	per := 8 + 4*N + 4*N*N
	if len(b) != off+nb*per {
		return nil, fmt.Errorf("boardenergy: %d bytes for %d bands, want %d", len(b), nb, off+nb*per)
	}
	m := &Model{}
	f32 := func(o int) float64 { return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[o:]))) }
	for k := 0; k < nb; k++ {
		bd := Band{Lo: int(binary.LittleEndian.Uint32(b[off:])), Hi: int(binary.LittleEndian.Uint32(b[off+4:])), W: make([]float64, N*N)}
		off += 8
		for i := 0; i < N; i++ {
			bd.H[i] = f32(off + 4*i)
		}
		off += 4 * N
		for i := 0; i < N*N; i++ {
			bd.W[i] = f32(off + 4*i)
		}
		off += 4 * N * N
		m.Bands = append(m.Bands, bd)
	}
	return m, nil
}

// BandFor returns the band whose range holds tiles, or the nearest one.
func (m *Model) BandFor(tiles int) *Band {
	best, dist := 0, math.MaxInt
	for i := range m.Bands {
		b := &m.Bands[i]
		d := 0
		if tiles < b.Lo {
			d = b.Lo - tiles
		} else if tiles > b.Hi {
			d = tiles - b.Hi
		}
		if d < dist {
			best, dist = i, d
		}
	}
	return &m.Bands[best]
}

func spins(occ *[N]bool) (s [N]float64, tiles int) {
	for i, o := range occ {
		if o {
			s[i] = 1
			tiles++
		} else {
			s[i] = -1
		}
	}
	return
}

// Energy of an occupancy pattern under the band of its own tile count.
func (m *Model) Energy(occ *[N]bool) float64 {
	s, tiles := spins(occ)
	return m.BandFor(tiles).energy(&s)
}

func (b *Band) energy(s *[N]float64) float64 {
	e := 0.0
	for i := 0; i < N; i++ {
		row := b.W[i*N : (i+1)*N]
		ws := 0.0
		for j := 0; j < N; j++ {
			ws += row[j] * s[j]
		}
		e -= b.H[i]*s[i] + 0.5*s[i]*ws
	}
	return e
}

// Position precomputes what is needed to score the energy change of many
// candidate plays on one board, all under the same band so that the changes
// are comparable.
type Position struct {
	b  *Band
	ws [N]float64 // (W s)_i
}

// NewPosition prepares occ for ΔE queries using the band for tiles on the
// board after a typical play (current tiles + 4).
func (m *Model) NewPosition(occ *[N]bool) *Position {
	s, tiles := spins(occ)
	p := &Position{b: m.BandFor(tiles + 4)}
	for i := 0; i < N; i++ {
		row := p.b.W[i*N : (i+1)*N]
		v := 0.0
		for j := 0; j < N; j++ {
			v += row[j] * s[j]
		}
		p.ws[i] = v
	}
	return p
}

// DeltaE is the energy change from placing tiles on the empty squares
// given (each flips -1 -> +1): with δ = 2 on those squares,
// ΔE = -h·δ - δ·(W s) - 1/2 δ'Wδ.
func (p *Position) DeltaE(squares []int) float64 {
	d := 0.0
	for _, i := range squares {
		d -= 2*p.b.H[i] + 2*p.ws[i]
		for _, j := range squares {
			d -= 2 * p.b.W[i*N+j]
		}
	}
	return d
}

// EnergyWith is the energy, under this position's band, of the board with
// the given squares filled; for tests.
func (p *Position) EnergyWith(occ *[N]bool, squares []int) float64 {
	o := *occ
	for _, i := range squares {
		o[i] = true
	}
	s, _ := spins(&o)
	return p.b.energy(&s)
}

// bandSums caches per-band constants for EnergyOcc.
type bandSums struct {
	r       [N]float64 // row sums of W
	hSum, c float64    // sum h, sum W
}

var sumsCache = map[*Band]*bandSums{}

func (b *Band) sums() *bandSums {
	if s, ok := sumsCache[b]; ok {
		return s
	}
	s := &bandSums{}
	for i := 0; i < N; i++ {
		s.hSum += b.H[i]
		for j := 0; j < N; j++ {
			s.r[i] += b.W[i*N+j]
		}
		s.c += s.r[i]
	}
	sumsCache[b] = s
	return s
}

// PrepareSums precomputes the constants EnergyOcc needs for every band; call
// it once before using EnergyOcc from several goroutines.
func (m *Model) PrepareSums() {
	for i := range m.Bands {
		m.Bands[i].sums()
	}
}

// EnergyOcc is Energy for a board given as the list of occupied squares, in
// O(tiles^2) instead of O(N^2): with s = 2o - 1,
// E = -2 h.o + sum h - 2 o'Wo + 2 r.o - (sum W)/2, r = W 1.
func (m *Model) EnergyOcc(occupied []int) float64 {
	b := m.BandFor(len(occupied))
	s := b.sums()
	e := s.hSum - s.c/2
	for _, i := range occupied {
		e += -2*b.H[i] + 2*s.r[i]
		row := b.W[i*N : (i+1)*N]
		for _, j := range occupied {
			e -= 2 * row[j]
		}
	}
	return e
}
