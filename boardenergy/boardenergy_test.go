package boardenergy

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

const paramsPath = "../data/strategy/default/ising/open-v1.bin"

func load(t *testing.T) *Model {
	t.Helper()
	if _, err := os.Stat(paramsPath); err != nil {
		t.Skip("parameter file not present (run pytorch/ising_export.py)")
	}
	m, err := Load(paramsPath)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestEnergyMatchesPython(t *testing.T) {
	m := load(t)
	// fixture.bin is git-ignored like the parameters; both come from
	// pytorch/ising_export.py --fixture.
	b, err := os.ReadFile(filepath.Join("testdata", "fixture.bin"))
	if err != nil {
		t.Skip("parity fixture not present (run pytorch/ising_export.py --fixture)")
	}
	n := int(binary.LittleEndian.Uint32(b))
	off := 4
	for k := 0; k < n; k++ {
		var occ [N]bool
		for i := 0; i < N; i++ {
			occ[i] = b[off+i] == 1
		}
		want := math.Float64frombits(binary.LittleEndian.Uint64(b[off+N:]))
		off += N + 8
		if got := m.Energy(&occ); math.Abs(got-want) > 1e-3*math.Max(1, math.Abs(want)) {
			t.Errorf("board %d: energy %.6f, python %.6f", k, got, want)
		}
	}
}

func TestDeltaEMatchesDirect(t *testing.T) {
	m := load(t)
	r := rand.New(rand.NewSource(1))
	for trial := 0; trial < 50; trial++ {
		var occ [N]bool
		for i := range occ {
			occ[i] = r.Float64() < 0.2
		}
		var empty []int
		for i, o := range occ {
			if !o {
				empty = append(empty, i)
			}
		}
		r.Shuffle(len(empty), func(i, j int) { empty[i], empty[j] = empty[j], empty[i] })
		sq := empty[:1+r.Intn(7)]
		p := m.NewPosition(&occ)
		before := p.EnergyWith(&occ, nil)
		after := p.EnergyWith(&occ, sq)
		if d := p.DeltaE(sq); math.Abs(d-(after-before)) > 1e-6*math.Max(1, math.Abs(d)) {
			t.Fatalf("trial %d: ΔE %.9f, direct %.9f", trial, d, after-before)
		}
	}
}

func TestEnergyOccMatchesEnergy(t *testing.T) {
	m := load(t)
	r := rand.New(rand.NewSource(2))
	for trial := 0; trial < 30; trial++ {
		var occ [N]bool
		var list []int
		p := r.Float64() * 0.5
		for i := range occ {
			if r.Float64() < p {
				occ[i] = true
				list = append(list, i)
			}
		}
		if a, b := m.Energy(&occ), m.EnergyOcc(list); math.Abs(a-b) > 1e-6*math.Max(1, math.Abs(a)) {
			t.Fatalf("trial %d (%d tiles): Energy %.9f, EnergyOcc %.9f", trial, len(list), a, b)
		}
	}
}

func TestLeafWinSanity(t *testing.T) {
	m := load(t)
	const path = "../data/strategy/default/ising/leafwin-v1.json"
	if _, err := os.Stat(path); err != nil {
		t.Skip("leaf win file not present")
	}
	for _, withE := range []bool{false, true} {
		lw, err := LoadLeafWin(path, withE, m)
		if err != nil {
			t.Fatal(err)
		}
		occ := []int{112, 113, 114, 115, 116}
		lo, mid, hi := lw.Prob(-100, 50, occ), lw.Prob(0, 50, occ), lw.Prob(100, 50, occ)
		if !(lo < mid && mid < hi) || lo < 0 || hi > 1 {
			t.Fatalf("energy=%v: P(-100)=%.3f P(0)=%.3f P(100)=%.3f not increasing in (0,1)", withE, lo, mid, hi)
		}
		// Later in the game a lead is worth more.
		if lw.Prob(50, 15, occ) <= lw.Prob(50, 80, occ) {
			t.Fatalf("energy=%v: +50 with 15 unseen (%.3f) should beat +50 with 80 unseen (%.3f)", withE, lw.Prob(50, 15, occ), lw.Prob(50, 80, occ))
		}
	}
}
