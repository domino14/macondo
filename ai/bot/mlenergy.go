package bot

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/boardenergy"
	"github.com/domino14/macondo/move"
)

// The fast ML bot ranks HastyBot's top plays by equity. With
// MACONDO_ML_ENERGY_EXTRA=k it also sends the net the k plays outside that
// list that lower the board's Ising energy most and the k that raise it
// most (pytorch/ising_fit.py; boardenergy): shape-changing plays the
// equity cut would hide. Parameters from MACONDO_ML_ENERGY_PARAMS, else
// <data>/strategy/default/ising/open-v1.bin.

func mlEnergyExtra() int {
	if v := os.Getenv("MACONDO_ML_ENERGY_EXTRA"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

var (
	energyOnce  sync.Once
	energyModel *boardenergy.Model
)

func loadEnergyModel(dataPath string) *boardenergy.Model {
	energyOnce.Do(func() {
		path := os.Getenv("MACONDO_ML_ENERGY_PARAMS")
		if path == "" {
			path = filepath.Join(dataPath, "strategy", "default", "ising", "open-v1.bin")
		}
		m, err := boardenergy.Load(path)
		if err != nil {
			log.Error().Err(err).Str("path", path).Msg("could not load the board-energy model; no energy candidates")
			return
		}
		energyModel = m
		log.Info().Str("path", path).Int("bands", len(m.Bands)).Msg("board-energy model loaded")
	})
	return energyModel
}

// placedSquares lists the board squares a play puts new tiles on.
func placedSquares(m *move.Move) []int {
	if m.Action() != move.MoveTypePlay {
		return nil
	}
	r, c, vertical := m.CoordsAndVertical()
	var sq []int
	for i, t := range m.Tiles() {
		if t == 0 {
			continue // played through
		}
		rr, cc := r, c+i
		if vertical {
			rr, cc = r+i, c
		}
		sq = append(sq, rr*15+cc)
	}
	return sq
}

// extremeIndices returns the indices of the k smallest and k largest values,
// smallest first, without repeats.
func extremeIndices(vals []float64, k int) []int {
	idx := make([]int, len(vals))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return vals[idx[a]] < vals[idx[b]] })
	if 2*k >= len(idx) {
		return idx
	}
	return append(append([]int{}, idx[:k]...), idx[len(idx)-k:]...)
}

// withEnergyExtras returns the first top plays of all (sorted by equity)
// plus the k lowest- and k highest-ΔE tile plays among the rest.
func withEnergyExtras(all []*move.Move, top, k int, b *board.GameBoard, model *boardenergy.Model) []*move.Move {
	if top > len(all) {
		top = len(all)
	}
	out := append([]*move.Move{}, all[:top]...)
	if model == nil || k <= 0 || b.Dim() != 15 {
		return out
	}
	var occ [boardenergy.N]bool
	for r := 0; r < 15; r++ {
		for c := 0; c < 15; c++ {
			occ[r*15+c] = b.GetLetter(r, c) != 0
		}
	}
	pos := model.NewPosition(&occ)
	var rest []*move.Move
	var deltas []float64
	for _, m := range all[top:] {
		if sq := placedSquares(m); len(sq) > 0 {
			rest = append(rest, m)
			deltas = append(deltas, pos.DeltaE(sq))
		}
	}
	for _, i := range extremeIndices(deltas, k) {
		out = append(out, rest[i])
	}
	return out
}

// MACONDO_SIM_LEAFWIN picks how a simming bot scores the end of each
// simulated line: unset or "table" for the win-percentage table, "base" for
// the logistic model fitted on our games (pytorch/ising_leafwin.py), "energy"
// for that model with the board-energy terms. The file is
// MACONDO_SIM_LEAFWIN_FILE, else <data>/strategy/default/ising/leafwin-v1.json.
var (
	leafWinMu    sync.Mutex
	leafWinCache = map[string]*boardenergy.LeafWin{}
)

// simLeafWin returns the leaf model for mode ("" means MACONDO_SIM_LEAFWIN),
// or nil for the table. Loaded once per mode.
func simLeafWin(mode, dataPath string) *boardenergy.LeafWin {
	if mode == "" {
		mode = os.Getenv("MACONDO_SIM_LEAFWIN")
	}
	if mode == "" || mode == "table" {
		return nil
	}
	leafWinMu.Lock()
	defer leafWinMu.Unlock()
	if lw, ok := leafWinCache[mode]; ok {
		return lw
	}
	path := os.Getenv("MACONDO_SIM_LEAFWIN_FILE")
	if path == "" {
		path = filepath.Join(dataPath, "strategy", "default", "ising", "leafwin-v1.json")
	}
	var model *boardenergy.Model
	if mode == "energy" {
		if model = loadEnergyModel(dataPath); model == nil {
			leafWinCache[mode] = nil
			return nil
		}
	}
	lw, err := boardenergy.LoadLeafWin(path, mode == "energy", model)
	if err != nil {
		log.Error().Err(err).Str("path", path).Msg("could not load the leaf win model; using the table")
		lw = nil
	} else {
		log.Info().Str("mode", mode).Str("path", path).Msg("sim leaf win model loaded")
	}
	leafWinCache[mode] = lw
	return lw
}
