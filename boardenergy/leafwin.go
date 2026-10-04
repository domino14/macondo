package boardenergy

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
)

// LeafWin is the sim's end-of-line win model fitted by
// pytorch/ising_leafwin.py: P(the player on turn wins) from their spread,
// the unseen tiles and, optionally, the board's energy z-score among boards
// with the same tile count. It was fitted on 8..93 unseen tiles; callers
// fall back to their table outside MinUnseen.
type LeafWin struct {
	coef      []float64
	useEnergy bool
	mean, sd  map[int]float64
	model     *Model
}

const MinUnseen = 8

type leafWinFile struct {
	CoefBase    []float64             `json:"coef_base"`
	CoefEnergy  []float64             `json:"coef_energy"`
	EnergyStats map[string][2]float64 `json:"energy_stats"`
}

// LoadLeafWin reads the JSON; withEnergy selects the model with the energy
// terms (then model must be the Ising parameters the stats were made with).
func LoadLeafWin(path string, withEnergy bool, model *Model) (*LeafWin, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f leafWinFile
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, err
	}
	lw := &LeafWin{coef: f.CoefBase, useEnergy: withEnergy, mean: map[int]float64{}, sd: map[int]float64{}, model: model}
	if withEnergy {
		lw.coef = f.CoefEnergy
		for k, v := range f.EnergyStats {
			t, err := strconv.Atoi(k)
			if err != nil {
				return nil, err
			}
			lw.mean[t], lw.sd[t] = v[0], v[1]
		}
		model.PrepareSums()
	}
	return lw, nil
}

// UsesEnergy reports whether the board is needed.
func (lw *LeafWin) UsesEnergy() bool { return lw.useEnergy }

// Prob is P(the player on turn wins) given their spread, the unseen tiles
// and the occupied squares (ignored without energy).
func (lw *LeafWin) Prob(spreadOnTurn, unseen int, occupied []int) float64 {
	u := float64(unseen) / 100
	s := float64(spreadOnTurn) / 100
	c := lw.coef
	x := c[0] + c[1]*u + c[2]*u*u + c[3]*s + c[4]*s*u + c[5]*s*u*u + c[6]*s*s*s
	if lw.useEnergy {
		z := 0.0
		if sd, ok := lw.sd[len(occupied)]; ok && sd > 0 {
			z = (lw.model.EnergyOcc(occupied) - lw.mean[len(occupied)]) / sd
			z = math.Max(-3, math.Min(3, z))
		}
		x += c[7]*z + c[8]*s*z + c[9]*s*z*u + c[10]*z*z + c[11]*s*z*z
	}
	return 1 / (1 + math.Exp(-x))
}
