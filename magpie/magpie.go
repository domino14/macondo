// Package magpie runs MAGPIE's move generator and board (cross-sets, anchors,
// play/unplay) from Go. MAGPIE is a git submodule at third_party/magpie; run
// `make magpie` once to build the static libraries this package links.
//
// MAGPIE fixes the board size at compile time, so there are two builds linked
// side by side, for 15x15 and 21x21 boards, and every call dispatches on the
// board size of the Data it was made from.
package magpie

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/magpie/src -I${SRCDIR}/../build/magpie -DRACK_SIZE=7 -DNDEBUG -O3 -Wno-trigraphs
#cgo LDFLAGS: ${SRCDIR}/../build/magpie/libmagpie_b15.a ${SRCDIR}/../build/magpie/libmagpie_b21.a -lm -lpthread
#cgo linux LDFLAGS: -ldl
#include <stdlib.h>
#include "shim.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/domino14/word-golib/tilemapping"
)

// Variants, as MAGPIE numbers them.
const (
	VariantClassic  = C.MG_VARIANT_CLASSIC
	VariantWordSmog = C.MG_VARIANT_WORDSMOG
)

// Record types: which moves a generation keeps.
type Record int

const (
	// RecordAll keeps every move, up to the position's move capacity: with a
	// capacity of n, the n best.
	RecordAll Record = C.MG_RECORD_ALL
	// RecordBest keeps only the best move.
	RecordBest Record = C.MG_RECORD_BEST
	// RecordWithin keeps every move within a margin of the best.
	RecordWithin Record = C.MG_RECORD_WITHIN_X
)

// Sort is what a generation ranks moves by.
type Sort int

const (
	SortEquity Sort = C.MG_SORT_EQUITY
	SortScore  Sort = C.MG_SORT_SCORE
)

// Move types in Move.Type.
const (
	MovePlacement = C.MG_MOVE_PLACEMENT
	MovePass      = C.MG_MOVE_PASS
	MoveExchange  = C.MG_MOVE_EXCHANGE
)

// EquityResolution is how many MAGPIE Equity units make one point.
const EquityResolution = 1000

// MaxDim is the largest board supported.
const MaxDim = C.MG_MAX_DIM

// Move is MAGPIE's move, laid out exactly like the C mg_move so a slice of
// them is filled in place. Score and Equity are in millipoints.
type Move struct {
	Score       int32
	Equity      int32
	Type        uint8
	Row         uint8
	Col         uint8
	Vertical    uint8
	TilesPlayed uint8
	TilesLength uint8
	Tiles       [MaxDim]tilemapping.MachineLetter
}

func init() {
	if unsafe.Sizeof(Move{}) != unsafe.Sizeof(C.mg_move{}) {
		panic(fmt.Sprintf("magpie.Move is %d bytes but mg_move is %d",
			unsafe.Sizeof(Move{}), unsafe.Sizeof(C.mg_move{})))
	}
}

// Word returns the move's tiles: for a placement, 0 marks a played-through
// square.
func (m *Move) Word() tilemapping.MachineWord {
	return tilemapping.MachineWord(m.Tiles[:m.TilesLength])
}

// DataOptions describes the lexicon data to load.
type DataOptions struct {
	KWGPath string
	// KLVPath is a .klv2 leave file; empty for no leaves.
	KLVPath string
	// WMPPath is a .wmp word map; empty to build one in memory when BuildWMP
	// is set, or run without one.
	WMPPath  string
	BuildWMP bool
	// Dist and Layout are Macondo's: the layout's rows use the bonus
	// characters MAGPIE does.
	Dist       *tilemapping.LetterDistribution
	Layout     []string
	Variant    int
	BingoBonus int
}

// Data is lexicon data (KWG, leaves, word map, distribution, layout) shared
// read-only by every Position made from it. It must outlive them.
type Data struct {
	ptr  *C.mg_data
	dim  int
	dist *tilemapping.LetterDistribution
}

// distCSV writes a distribution in MAGPIE's letter,lower,count,score,vowel
// format, in machine-letter order.
func distCSV(ld *tilemapping.LetterDistribution) string {
	tm := ld.TileMapping()
	vowel := map[tilemapping.MachineLetter]bool{}
	for _, v := range ld.Vowels {
		vowel[v] = true
	}
	var sb strings.Builder
	dist := ld.Distribution()
	for i := range dist {
		ml := tilemapping.MachineLetter(i)
		letter := tm.Letter(ml)
		lower := strings.ToLower(letter)
		if i == 0 {
			letter, lower = "?", "?"
		}
		v := 0
		if vowel[ml] {
			v = 1
		}
		fmt.Fprintf(&sb, "%s,%s,%d,%d,%d\n", letter, lower, dist[i], ld.Score(ml), v)
	}
	return sb.String()
}

// NewData loads lexicon data.
func NewData(opts DataOptions) (*Data, error) {
	dim := len(opts.Layout)
	if dim != 15 && dim != 21 {
		return nil, fmt.Errorf("magpie: unsupported board size %d", dim)
	}
	var layout strings.Builder
	fmt.Fprintf(&layout, "%d, %d\n", dim/2, dim/2)
	for _, row := range opts.Layout {
		if len(row) != dim {
			return nil, fmt.Errorf("magpie: layout row %q is not %d wide", row, dim)
		}
		layout.WriteString(row)
		layout.WriteByte('\n')
	}
	cstr := func(s string) *C.char {
		if s == "" {
			return nil
		}
		return C.CString(s)
	}
	kwg, klv, wmp := cstr(opts.KWGPath), cstr(opts.KLVPath), cstr(opts.WMPPath)
	ld, lay := C.CString(distCSV(opts.Dist)), C.CString(layout.String())
	defer func() {
		for _, p := range []*C.char{kwg, klv, wmp, ld, lay} {
			C.free(unsafe.Pointer(p))
		}
	}()
	buildWMP := C.int(0)
	if opts.BuildWMP {
		buildWMP = 1
	}
	var errbuf [512]C.char
	var ptr *C.mg_data
	switch dim {
	case 15:
		ptr = C.mgs15_data_create(kwg, klv, wmp, buildWMP, ld, lay,
			C.int(opts.Variant), C.int(opts.BingoBonus), &errbuf[0], C.int(len(errbuf)))
	case 21:
		ptr = C.mgs21_data_create(kwg, klv, wmp, buildWMP, ld, lay,
			C.int(opts.Variant), C.int(opts.BingoBonus), &errbuf[0], C.int(len(errbuf)))
	}
	if ptr == nil {
		return nil, errors.New("magpie: " + C.GoString(&errbuf[0]))
	}
	return &Data{ptr: ptr, dim: dim, dist: opts.Dist}, nil
}

// Close frees the data. No Position made from it may be used afterwards.
func (d *Data) Close() {
	if d.ptr == nil {
		return
	}
	if d.dim == 15 {
		C.mgs15_data_destroy(d.ptr)
	} else {
		C.mgs21_data_destroy(d.ptr)
	}
	d.ptr = nil
}

// Dim is the board size.
func (d *Data) Dim() int { return d.dim }

// HasWMP reports whether a word map is loaded.
func (d *Data) HasWMP() bool {
	if d.dim == 15 {
		return C.mgs15_data_has_wmp(d.ptr) != 0
	}
	return C.mgs21_data_has_wmp(d.ptr) != 0
}

// LeaveValue is the leave's value in points.
func (d *Data) LeaveValue(leave tilemapping.MachineWord) float64 {
	if len(leave) == 0 {
		return 0
	}
	p := (*C.uint8_t)(unsafe.Pointer(&leave[0]))
	var v C.int32_t
	if d.dim == 15 {
		v = C.mgs15_data_leave_value(d.ptr, p, C.int(len(leave)))
	} else {
		v = C.mgs21_data_leave_value(d.ptr, p, C.int(len(leave)))
	}
	return float64(v) / EquityResolution
}

// Position is a board with its cross-sets and anchors, both racks and the
// bag. It is not safe for concurrent use: give each worker its own.
type Position struct {
	ptr   *C.mg_pos
	data  *Data
	moves []Move
}

// NewPosition makes an empty position. moveCapacity bounds RecordAll.
func NewPosition(d *Data, moveCapacity int, seed uint64) *Position {
	if moveCapacity < 1 {
		moveCapacity = 1
	}
	p := &Position{data: d, moves: make([]Move, moveCapacity)}
	if d.dim == 15 {
		p.ptr = C.mgs15_pos_create(d.ptr, C.int(moveCapacity), C.uint64_t(seed))
	} else {
		p.ptr = C.mgs21_pos_create(d.ptr, C.int(moveCapacity), C.uint64_t(seed))
	}
	return p
}

// Close frees the position.
func (p *Position) Close() {
	if p.ptr == nil {
		return
	}
	if p.data.dim == 15 {
		C.mgs15_pos_destroy(p.ptr)
	} else {
		C.mgs21_pos_destroy(p.ptr)
	}
	p.ptr = nil
}

// CopyFrom makes p the same position as src (backups are not copied, and are
// turned off).
func (p *Position) CopyFrom(src *Position) {
	if p.data.dim == 15 {
		C.mgs15_pos_copy(p.ptr, src.ptr)
	} else {
		C.mgs21_pos_copy(p.ptr, src.ptr)
	}
}

func bytesPtr(b []tilemapping.MachineLetter) *C.uint8_t {
	if len(b) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&b[0]))
}

// State is a whole position, for Position.Set.
type State struct {
	// Squares is dim*dim, row-major, 0 for empty.
	Squares []tilemapping.MachineLetter
	Racks   [2]tilemapping.MachineWord
	Bag     tilemapping.MachineWord
	OnTurn  int
	// Scores in points.
	Scores         [2]int
	ScorelessTurns int
}

// Set replaces the whole position and recomputes cross-sets and anchors.
func (p *Position) Set(s State) error {
	dim := p.data.dim
	if len(s.Squares) != dim*dim {
		return fmt.Errorf("magpie: %d squares for a %dx%d board", len(s.Squares), dim, dim)
	}
	var errbuf [256]C.char
	sq, bag := bytesPtr(s.Squares), bytesPtr(s.Bag)
	r0, r1 := bytesPtr(s.Racks[0]), bytesPtr(s.Racks[1])
	n0, n1, nbag := C.int(len(s.Racks[0])), C.int(len(s.Racks[1])), C.int(len(s.Bag))
	s0 := C.int32_t(s.Scores[0] * EquityResolution)
	s1 := C.int32_t(s.Scores[1] * EquityResolution)
	var rc C.int
	if dim == 15 {
		rc = C.mgs15_pos_set(p.ptr, sq, r0, n0, r1, n1, bag, nbag, C.int(s.OnTurn),
			s0, s1, C.int(s.ScorelessTurns), &errbuf[0], C.int(len(errbuf)))
	} else {
		rc = C.mgs21_pos_set(p.ptr, sq, r0, n0, r1, n1, bag, nbag, C.int(s.OnTurn),
			s0, s1, C.int(s.ScorelessTurns), &errbuf[0], C.int(len(errbuf)))
	}
	if rc != 0 {
		return errors.New("magpie: " + C.GoString(&errbuf[0]))
	}
	return nil
}

// SetRacksAndBag replaces both racks and the bag.
func (p *Position) SetRacksAndBag(rack0, rack1, bag tilemapping.MachineWord) {
	if p.data.dim == 15 {
		C.mgs15_pos_set_racks_and_bag(p.ptr, bytesPtr(rack0), C.int(len(rack0)),
			bytesPtr(rack1), C.int(len(rack1)), bytesPtr(bag), C.int(len(bag)))
	} else {
		C.mgs21_pos_set_racks_and_bag(p.ptr, bytesPtr(rack0), C.int(len(rack0)),
			bytesPtr(rack1), C.int(len(rack1)), bytesPtr(bag), C.int(len(bag)))
	}
}

// SetOnTurn sets the player on turn.
func (p *Position) SetOnTurn(player int) {
	if p.data.dim == 15 {
		C.mgs15_pos_set_on_turn(p.ptr, C.int(player))
	} else {
		C.mgs21_pos_set_on_turn(p.ptr, C.int(player))
	}
}

// SetBackups turns the backup stack, which Unplay pops, on or off.
func (p *Position) SetBackups(on bool) {
	v := C.int(0)
	if on {
		v = 1
	}
	if p.data.dim == 15 {
		C.mgs15_pos_set_backups(p.ptr, v)
	} else {
		C.mgs21_pos_set_backups(p.ptr, v)
	}
}

// Generate generates moves for the player on turn, best first. margin (in
// points) is the window for RecordWithin. The returned slice is reused by
// the next call.
func (p *Position) Generate(record Record, sort Sort, margin float64) []Move {
	out := (*C.mg_move)(unsafe.Pointer(&p.moves[0]))
	m := C.int32_t(margin * EquityResolution)
	var n C.int
	if p.data.dim == 15 {
		n = C.mgs15_gen(p.ptr, C.int(record), C.int(sort), m, out, C.int(len(p.moves)))
	} else {
		n = C.mgs21_gen(p.ptr, C.int(record), C.int(sort), m, out, C.int(len(p.moves)))
	}
	return p.moves[:n]
}

// Play plays m for the player on turn, who then draws the tiles in drawn
// (taken out of the bag). It backs up first if backups are on.
func (p *Position) Play(m *Move, drawn tilemapping.MachineWord) error {
	cm := (*C.mg_move)(unsafe.Pointer(m))
	var rc C.int
	if p.data.dim == 15 {
		rc = C.mgs15_play(p.ptr, cm, bytesPtr(drawn), C.int(len(drawn)))
	} else {
		rc = C.mgs21_play(p.ptr, cm, bytesPtr(drawn), C.int(len(drawn)))
	}
	if rc != 0 {
		return errors.New("magpie: drawn tile not in the bag")
	}
	return nil
}

// Unplay undoes the last Play (backups must be on).
func (p *Position) Unplay() {
	if p.data.dim == 15 {
		C.mgs15_unplay(p.ptr)
	} else {
		C.mgs21_unplay(p.ptr)
	}
}

// Board is a copy of a position's board.
type Board struct {
	Dim     int
	Letters []tilemapping.MachineLetter
	// Indexed [dir*dim*dim + row*dim + col]; dir 0 is across plays, 1 down.
	CrossSets   []uint64
	CrossScores []int32
	Anchors     []uint8
}

// NewBoard allocates a Board for the data's board size.
func (d *Data) NewBoard() *Board {
	n := d.dim * d.dim
	return &Board{Dim: d.dim, Letters: make([]tilemapping.MachineLetter, n),
		CrossSets: make([]uint64, 2*n), CrossScores: make([]int32, 2*n),
		Anchors: make([]uint8, 2*n)}
}

// ReadBoard copies the position's board into b.
func (p *Position) ReadBoard(b *Board) {
	l := bytesPtr(b.Letters)
	cs := (*C.uint64_t)(unsafe.Pointer(&b.CrossSets[0]))
	sc := (*C.int32_t)(unsafe.Pointer(&b.CrossScores[0]))
	an := (*C.uint8_t)(unsafe.Pointer(&b.Anchors[0]))
	if p.data.dim == 15 {
		C.mgs15_pos_read_board(p.ptr, l, cs, sc, an)
	} else {
		C.mgs21_pos_read_board(p.ptr, l, cs, sc, an)
	}
}

// RackCounts returns both racks as per-letter counts, and the bag size.
func (p *Position) RackCounts() (r0, r1 []uint8, bag int) {
	n := len(p.data.dist.Distribution())
	r0, r1 = make([]uint8, n), make([]uint8, n)
	var cbag C.int
	if p.data.dim == 15 {
		C.mgs15_pos_read_racks(p.ptr, (*C.uint8_t)(unsafe.Pointer(&r0[0])),
			(*C.uint8_t)(unsafe.Pointer(&r1[0])), &cbag)
	} else {
		C.mgs21_pos_read_racks(p.ptr, (*C.uint8_t)(unsafe.Pointer(&r0[0])),
			(*C.uint8_t)(unsafe.Pointer(&r1[0])), &cbag)
	}
	return r0, r1, int(cbag)
}
