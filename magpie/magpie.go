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
	"os"
	"runtime"
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
	// KWG is a .kwg file's contents (a .kad for WordSmog), or KWGPath names
	// the file.
	KWG     []byte
	KWGPath string
	// KLV is a .klv2 leave file's contents, or KLVPath names the file.
	// MAGPIE's generator always reads leaves, so one is required.
	KLV     []byte
	KLVPath string
	// WMPPath is a .wmp word map; empty to build one in memory when BuildWMP
	// is set, or run without one.
	WMPPath  string
	BuildWMP bool
	// RITPath is a rack info table (see WriteRIT), memory-mapped. It needs a
	// word map. Empty for none; the generator then computes each rack's
	// leaves on first sight.
	RITPath string
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
	kwgBytes, klvBytes := opts.KWG, opts.KLV
	var err error
	if kwgBytes == nil {
		if kwgBytes, err = os.ReadFile(opts.KWGPath); err != nil {
			return nil, fmt.Errorf("magpie: %w", err)
		}
	}
	if klvBytes == nil {
		if opts.KLVPath == "" {
			return nil, errors.New("magpie: leaves (KLV or KLVPath) are required")
		}
		if klvBytes, err = os.ReadFile(opts.KLVPath); err != nil {
			return nil, fmt.Errorf("magpie: %w", err)
		}
	}
	if len(kwgBytes) < 4 || len(klvBytes) < 4 {
		return nil, errors.New("magpie: empty word graph or leave file")
	}
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
	wmp, rit := cstr(opts.WMPPath), cstr(opts.RITPath)
	ld, lay := C.CString(distCSV(opts.Dist)), C.CString(layout.String())
	// The shim copies the graphs out of these buffers before returning.
	kwg := (*C.uint8_t)(unsafe.Pointer(&kwgBytes[0]))
	klv := (*C.uint8_t)(unsafe.Pointer(&klvBytes[0]))
	kwgLen, klvLen := C.int64_t(len(kwgBytes)), C.int64_t(len(klvBytes))
	defer func() {
		for _, p := range []*C.char{wmp, rit, ld, lay} {
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
		ptr = C.mgs15_data_create(kwg, kwgLen, klv, klvLen, wmp, buildWMP, rit, ld, lay,
			C.int(opts.Variant), C.int(opts.BingoBonus), &errbuf[0], C.int(len(errbuf)))
	case 21:
		ptr = C.mgs21_data_create(kwg, kwgLen, klv, klvLen, wmp, buildWMP, rit, ld, lay,
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

// WriteWMP writes the data's word map to path.
func (d *Data) WriteWMP(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var errbuf [512]C.char
	var rc C.int
	if d.dim == 15 {
		rc = C.mgs15_data_write_wmp(d.ptr, cpath, &errbuf[0], C.int(len(errbuf)))
	} else {
		rc = C.mgs21_data_write_wmp(d.ptr, cpath, &errbuf[0], C.int(len(errbuf)))
	}
	if rc != 0 {
		return errors.New("magpie: " + C.GoString(&errbuf[0]))
	}
	return nil
}

// WriteRIT builds the rack info table for this data's leaves and word map
// and writes it to path. It takes minutes and gigabytes for English.
func (d *Data) WriteRIT(path string, threads int) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var errbuf [512]C.char
	var rc C.int
	if d.dim == 15 {
		rc = C.mgs15_data_write_rit(d.ptr, cpath, C.int(threads), &errbuf[0], C.int(len(errbuf)))
	} else {
		rc = C.mgs21_data_write_rit(d.ptr, cpath, C.int(threads), &errbuf[0], C.int(len(errbuf)))
	}
	if rc != 0 {
		return errors.New("magpie: " + C.GoString(&errbuf[0]))
	}
	return nil
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
// bag. It is not safe for concurrent use: give each worker its own. Its C
// memory is freed by Close, or when the garbage collector finds it
// unreachable.
type Position struct {
	ptr   *C.mg_pos
	h     *posHandle
	data  *Data
	moves []Move
}

// posHandle owns a position's C memory; the cleanup the garbage collector
// runs holds it rather than the Position.
type posHandle struct {
	ptr *C.mg_pos
	dim int
}

func (h *posHandle) free() {
	if h.ptr == nil {
		return
	}
	if h.dim == 15 {
		C.mgs15_pos_destroy(h.ptr)
	} else {
		C.mgs21_pos_destroy(h.ptr)
	}
	h.ptr = nil
}

// DefaultMoveCapacity is how many moves RecordAll keeps unless Generate is
// asked for more.
const DefaultMoveCapacity = 1000

// NewPosition makes an empty position. moveCapacity is how many moves
// RecordAll keeps by default (the best moveCapacity).
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
	p.h = &posHandle{ptr: p.ptr, dim: d.dim}
	runtime.AddCleanup(p, func(h *posHandle) { h.free() }, p.h)
	return p
}

// Close frees the position now.
func (p *Position) Close() {
	p.h.free()
	p.ptr = nil
}

// CopyFrom makes p the same position as src. Saved plays are not copied.
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
	// MAGPIE aborts on a bag bigger than the distribution, so check here.
	// (Racks and the board may hold more of a tile than exist: some tests
	// set up such positions, and MAGPIE does not mind.)
	dist := p.data.dist.Distribution()
	total := 0
	for _, n := range dist {
		total += int(n)
	}
	if len(s.Bag) > total {
		return fmt.Errorf("magpie: a bag of %d tiles, more than the distribution's %d", len(s.Bag), total)
	}
	for _, w := range [][]tilemapping.MachineLetter{s.Squares, s.Racks[0], s.Racks[1], s.Bag} {
		for _, t := range w {
			if int(t.Unblank()) >= len(dist) {
				return fmt.Errorf("magpie: tile %d is outside the distribution", t)
			}
		}
	}
	if len(s.Racks[0]) > 7 || len(s.Racks[1]) > 7 {
		return errors.New("magpie: a rack holds more than 7 tiles")
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

// SetRacksAndBag replaces both racks, the bag and the player on turn. The
// tiles must be ones the distribution holds (it is not checked here, for
// speed; see Set).
func (p *Position) SetRacksAndBag(rack0, rack1, bag tilemapping.MachineWord, onTurn int) {
	if p.data.dim == 15 {
		C.mgs15_pos_set_racks_and_bag(p.ptr, bytesPtr(rack0), C.int(len(rack0)),
			bytesPtr(rack1), C.int(len(rack1)), bytesPtr(bag), C.int(len(bag)), C.int(onTurn))
	} else {
		C.mgs21_pos_set_racks_and_bag(p.ptr, bytesPtr(rack0), C.int(len(rack0)),
			bytesPtr(rack1), C.int(len(rack1)), bytesPtr(bag), C.int(len(bag)), C.int(onTurn))
	}
}

// Data is the data the position generates with.
func (p *Position) Data() *Data { return p.data }

// SetData switches the data (leaves, word map) the position generates with,
// keeping the position. d must have the same word graph, distribution and
// board.
func (p *Position) SetData(d *Data) {
	if d == p.data {
		return
	}
	if d.dim != p.data.dim {
		panic("magpie: SetData across board sizes")
	}
	if d.dim == 15 {
		C.mgs15_pos_set_data(p.ptr, d.ptr)
	} else {
		C.mgs21_pos_set_data(p.ptr, d.ptr)
	}
	p.data = d
}

// SetOnTurn sets the player on turn.
func (p *Position) SetOnTurn(player int) {
	if p.data.dim == 15 {
		C.mgs15_pos_set_on_turn(p.ptr, C.int(player))
	} else {
		C.mgs21_pos_set_on_turn(p.ptr, C.int(player))
	}
}

// BackupMode is what Play saves for Unplay.
type BackupMode int

const (
	// BackupOff saves nothing.
	BackupOff BackupMode = C.MG_BACKUP_OFF
	// BackupStack saves every play, to any depth.
	BackupStack BackupMode = C.MG_BACKUP_STACK
	// BackupSingle saves only the last play.
	BackupSingle BackupMode = C.MG_BACKUP_SINGLE
)

// SetBackupMode sets what Play saves for Unplay. Plays already saved stay.
func (p *Position) SetBackupMode(m BackupMode) {
	if p.data.dim == 15 {
		C.mgs15_pos_set_backup_mode(p.ptr, C.int(m))
	} else {
		C.mgs21_pos_set_backup_mode(p.ptr, C.int(m))
	}
}

// BackupDepth is the number of plays Unplay can undo.
func (p *Position) BackupDepth() int {
	if p.data.dim == 15 {
		return int(C.mgs15_pos_backup_depth(p.ptr))
	}
	return int(C.mgs21_pos_backup_depth(p.ptr))
}

// Generate generates moves for the player on turn, best first. margin (in
// points) is the window for RecordWithin. RecordAll keeps the position's
// move capacity's worth; see GenerateN. The returned slice is reused by the
// next call.
func (p *Position) Generate(record Record, sort Sort, margin float64) []Move {
	return p.GenerateN(record, sort, margin, len(p.moves))
}

// GenerateN is Generate keeping at most n moves (the best n).
func (p *Position) GenerateN(record Record, sort Sort, margin float64, n int) []Move {
	if n < 1 {
		n = 1
	}
	if n > len(p.moves) {
		p.moves = make([]Move, n)
	}
	out := (*C.mg_move)(unsafe.Pointer(&p.moves[0]))
	capacity := C.int(n)
	m := C.int32_t(margin * EquityResolution)
	var got C.int
	if p.data.dim == 15 {
		got = C.mgs15_gen(p.ptr, C.int(record), C.int(sort), m, out, capacity)
	} else {
		got = C.mgs21_gen(p.ptr, C.int(record), C.int(sort), m, out, capacity)
	}
	return p.moves[:got]
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

// ErrNoBackup is Unplay's error when no play is saved.
var ErrNoBackup = errors.New("magpie: no play to undo")

// Unplay undoes the last saved Play.
func (p *Position) Unplay() error {
	var rc C.int
	if p.data.dim == 15 {
		rc = C.mgs15_unplay(p.ptr)
	} else {
		rc = C.mgs21_unplay(p.ptr)
	}
	if rc != 0 {
		return ErrNoBackup
	}
	return nil
}

// ResetToFirst undoes every saved play, back to the oldest, along with any
// played since without saving.
func (p *Position) ResetToFirst() error {
	var rc C.int
	if p.data.dim == 15 {
		rc = C.mgs15_reset_to_first(p.ptr)
	} else {
		rc = C.mgs21_reset_to_first(p.ptr)
	}
	if rc != 0 {
		return ErrNoBackup
	}
	return nil
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

// SmallMove is an endgame move: TM is Macondo's tinymove.TinyMove encoding
// (0 for a pass). Laid out like the C mg_small_move.
type SmallMove struct {
	TM          uint64
	Score       int16
	TilesPlayed uint8
	PlayLength  uint8
}

func init() {
	if unsafe.Sizeof(SmallMove{}) != unsafe.Sizeof(C.mg_small_move{}) {
		panic("magpie.SmallMove does not match mg_small_move")
	}
}

// GenerateSmall generates every play for the player on turn as small moves
// (a pass too if pass is set), into buf, using the override word graph if
// one is set. It returns the moves, which alias buf; buf's length is the
// most it can hold.
func (p *Position) GenerateSmall(pass bool, buf []SmallMove) []SmallMove {
	return p.GenerateSmallFor(-1, pass, buf)
}

// GenerateSmallFor is GenerateSmall for the given player, on turn or not
// (-1 for the player on turn).
func (p *Position) GenerateSmallFor(player int, pass bool, buf []SmallMove) []SmallMove {
	if len(buf) == 0 {
		return buf
	}
	gp := C.int(0)
	if pass {
		gp = 1
	}
	out := (*C.mg_small_move)(unsafe.Pointer(&buf[0]))
	var n C.int
	if p.data.dim == 15 {
		n = C.mgs15_gen_small(p.ptr, C.int(player), gp, out, C.int(len(buf)))
	} else {
		n = C.mgs21_gen_small(p.ptr, C.int(player), gp, out, C.int(len(buf)))
	}
	return buf[:n]
}

// TilesPlayed is the set of tiles player can play somewhere (a bit per
// machine letter), ORed into known. It stops looking once every tile on the
// player's rack is known playable.
func (p *Position) TilesPlayed(player int, known uint64) uint64 {
	if p.data.dim == 15 {
		return uint64(C.mgs15_tiles_played(p.ptr, C.int(player), C.uint64_t(known)))
	}
	return uint64(C.mgs21_tiles_played(p.ptr, C.int(player), C.uint64_t(known)))
}

// PlaySmall plays a small move worth score points with an incremental undo
// (the bag must be empty). The move's tiles go into strip, 0 for a
// played-through square; it returns how many.
func (p *Position) PlaySmall(tm uint64, score int, strip []tilemapping.MachineLetter) (int, error) {
	sp := (*C.uint8_t)(unsafe.Pointer(&strip[0]))
	var n C.int
	if p.data.dim == 15 {
		n = C.mgs15_play_small(p.ptr, C.uint64_t(tm), C.int(score), sp)
	} else {
		n = C.mgs21_play_small(p.ptr, C.uint64_t(tm), C.int(score), sp)
	}
	if n < 0 {
		return 0, errors.New("magpie: small move does not fit the board")
	}
	return int(n), nil
}

// UnplaySmall undoes the last PlaySmall.
func (p *Position) UnplaySmall() error {
	var rc C.int
	if p.data.dim == 15 {
		rc = C.mgs15_unplay_small(p.ptr)
	} else {
		rc = C.mgs21_unplay_small(p.ptr)
	}
	if rc != 0 {
		return ErrNoBackup
	}
	return nil
}

// KWG is a word graph MAGPIE built, such as an endgame's pruned one.
type KWG struct {
	ptr *C.mg_kwg
	dim int
}

// PrunedKWG builds a word graph of every word playable from this position,
// for an endgame. Close it when done.
func (p *Position) PrunedKWG() *KWG {
	k := &KWG{dim: p.data.dim}
	if k.dim == 15 {
		k.ptr = C.mgs15_pos_pruned_kwg(p.ptr)
	} else {
		k.ptr = C.mgs21_pos_pruned_kwg(p.ptr)
	}
	return k
}

// Close frees the word graph.
func (k *KWG) Close() {
	if k == nil || k.ptr == nil {
		return
	}
	if k.dim == 15 {
		C.mgs15_kwg_destroy(k.ptr)
	} else {
		C.mgs21_kwg_destroy(k.ptr)
	}
	k.ptr = nil
}

// SetOverrideKWG makes the position generate (and build cross-sets) from k
// rather than the full word graph; nil goes back to the full one.
func (p *Position) SetOverrideKWG(k *KWG) {
	var kp *C.mg_kwg
	if k != nil {
		kp = k.ptr
	}
	if p.data.dim == 15 {
		C.mgs15_pos_set_override_kwg(p.ptr, kp)
	} else {
		C.mgs21_pos_set_override_kwg(p.ptr, kp)
	}
}

// PlaySmallDrawing plays a small move worth score points saved as the backup
// mode says, the player drawing drawn (a pre-endgame's bag is not empty).
// The move's letters go into strip, as for PlaySmall.
func (p *Position) PlaySmallDrawing(tm uint64, score int, drawn, strip []tilemapping.MachineLetter) (int, error) {
	sp := (*C.uint8_t)(unsafe.Pointer(&strip[0]))
	var n C.int
	if p.data.dim == 15 {
		n = C.mgs15_play_small_drawing(p.ptr, C.uint64_t(tm), C.int(score), bytesPtr(drawn), C.int(len(drawn)), sp)
	} else {
		n = C.mgs21_play_small_drawing(p.ptr, C.uint64_t(tm), C.int(score), bytesPtr(drawn), C.int(len(drawn)), sp)
	}
	if n < 0 {
		return 0, errors.New("magpie: small move does not fit the board, or a drawn tile is not in the bag")
	}
	return int(n), nil
}
