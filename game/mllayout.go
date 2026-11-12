package game

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/domino14/word-golib/tilemapping"
)

// The net's input depends on the alphabet. With L letters (not counting the
// blank) a position is NumPlanes() board planes of NN_H x NN_W:
//
//	0 .. L-1        a letter on the square, one plane per letter
//	L               the tile there is a blank
//	L+1 .. 2L       horizontal cross-checks, one plane per letter
//	2L+1 .. 3L      vertical cross-checks
//	3L+1 .. 3L+4    empty 2L, 3L, 2W, 3W squares
//	3L+5, 3L+6      history: the candidate's squares, the opponent's last move
//
// and NumScalars() scalars:
//
//	0 .. L          the mover's rack, tiles of each kind (blank first) / 7
//	L+1 .. 2L+1     unseen tiles of each kind / unseen tiles
//	2L+2 ..         the opponent's last move (score, tiles played, exchanged),
//	                6 power tiles unseen, vowel/consonant ratio of the unseen
//	                tiles and of the rack, turns since the opponent's last
//	                bingo, the candidate's score and leave, tiles in the bag,
//	                the mover's spread
//
// English (26 letters) is 85 planes and 72 scalars, the layout every
// shipped model was trained on; TestEnglishMLLayout pins it.
type MLLayout struct {
	Letters    int // L
	NumPlanes  int // board planes, 3L+7
	NumScalars int // 2L+20
	PlaneLen   int // NumPlanes * NN_H * NN_W values
	RowLen     int // PlaneLen + NumScalars

	blankPlane, horCCPlanes, verCCPlanes, bonusPlanes, historyPlanes int // offsets, in values

	historyScal, powerScal, vcBagScal, vcRackScal, addlScal int // offsets into the scalars

	vowels []tilemapping.MachineLetter
	// The power tiles: the blank, then five letters, each counted unseen
	// over how many the bag starts with.
	power      [mlNumPowerTiles]tilemapping.MachineLetter
	powerCount [mlNumPowerTiles]float32
}

const (
	mlPlaneSize      = NN_H * NN_W
	mlNumBonusPlanes = 4
	mlNumPowerTiles  = 6
)

// mlEnglishPowerLetters are English's power tiles. Other alphabets take
// their five highest-scoring letters.
var mlEnglishPowerLetters = []string{"J", "Q", "S", "X", "Z"}

var mlLayouts sync.Map // *tilemapping.LetterDistribution -> *MLLayout

// MLLayoutFor is the net's input layout for games with this letter
// distribution.
func MLLayoutFor(ld *tilemapping.LetterDistribution) (*MLLayout, error) {
	if l, ok := mlLayouts.Load(ld); ok {
		return l.(*MLLayout), nil
	}
	l, err := newMLLayout(ld)
	if err != nil {
		return nil, err
	}
	mlLayouts.Store(ld, l)
	return l, nil
}

func newMLLayout(ld *tilemapping.LetterDistribution) (*MLLayout, error) {
	dist := ld.Distribution()
	L := len(dist) - 1 // dist[0] is the blank
	if L < mlNumPowerTiles-1 || L > 62 {
		return nil, fmt.Errorf("letter distribution %s: %d letters, the net needs 5 to 62", ld.Name, L)
	}
	planes := 3*L + 7
	l := &MLLayout{
		Letters:       L,
		NumPlanes:     planes,
		NumScalars:    2*L + 20,
		PlaneLen:      planes * mlPlaneSize,
		blankPlane:    L * mlPlaneSize,
		horCCPlanes:   (L + 1) * mlPlaneSize,
		verCCPlanes:   (2*L + 1) * mlPlaneSize,
		bonusPlanes:   (3*L + 1) * mlPlaneSize,
		historyPlanes: (3*L + 5) * mlPlaneSize,
		historyScal:   2 * (L + 1),
		vowels:        ld.Vowels,
	}
	l.RowLen = l.PlaneLen + l.NumScalars
	l.powerScal = l.historyScal + 3
	l.vcBagScal = l.powerScal + mlNumPowerTiles
	l.vcRackScal = l.vcBagScal + 2
	l.addlScal = l.vcRackScal + 2

	var power []tilemapping.MachineLetter
	if strings.EqualFold(ld.Name, "english") {
		for _, s := range mlEnglishPowerLetters {
			ml, err := ld.TileMapping().Val(s)
			if err != nil {
				return nil, err
			}
			power = append(power, ml)
		}
	} else {
		for ml := 1; ml <= L; ml++ {
			power = append(power, tilemapping.MachineLetter(ml))
		}
		// Highest score first, then alphabet order; keep five, in
		// alphabet order.
		slices.SortStableFunc(power, func(a, b tilemapping.MachineLetter) int {
			return cmp.Compare(ld.Score(b), ld.Score(a))
		})
		power = power[:mlNumPowerTiles-1]
		slices.Sort(power)
	}
	l.power[0] = 0
	copy(l.power[1:], power)
	for i, ml := range l.power {
		if dist[ml] == 0 {
			return nil, fmt.Errorf("letter distribution %s: power tile %d has no tiles", ld.Name, ml)
		}
		l.powerCount[i] = float32(dist[ml])
	}
	return l, nil
}

// mlLayout is the layout of this game's letter distribution.
func (g *Game) mlLayout() (*MLLayout, error) {
	if g.mlLayoutCache == nil {
		l, err := MLLayoutFor(g.letterDistribution)
		if err != nil {
			return nil, err
		}
		g.mlLayoutCache = l
	}
	return g.mlLayoutCache, nil
}

// MLLayout is the net's input layout for this game.
func (g *Game) MLLayout() (*MLLayout, error) {
	return g.mlLayout()
}
