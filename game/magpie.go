package game

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/dataloaders"
	"github.com/domino14/macondo/lexicon"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tinymove"
	"github.com/domino14/macondo/wmp"
)

// magpieDataKey identifies one set of lexicon data for MAGPIE.
type magpieDataKey struct {
	dataPath, lexicon, dist, leaves string
	dim                             int
	wordSmog                        bool
}

type magpieDataEntry struct {
	once sync.Once
	data *magpie.Data
	err  error
}

// noLeavesKLV is a .klv2 with an empty word graph and a single leave worth
// 0, which every leave lookup lands on.
var noLeavesKLV = []byte{
	2, 0, 0, 0, // two word graph nodes, both empty
	0, 0, 0, 0,
	0, 0, 0, 0,
	1, 0, 0, 0, // one leave value: 0.0
	0, 0, 0, 0,
}

// magpieDataCache holds the MAGPIE data loaded in this process, which is
// shared read-only by every game and never freed.
var magpieDataCache sync.Map

// readDataFile reads a whole file through word-golib's cache layer, which
// knows the data path's storage.
func readDataFile(path string) ([]byte, error) {
	f, _, err := cache.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

// MagpieData returns MAGPIE's data for a lexicon, distribution and board,
// loading it the first time it is asked for. leavefile picks the leaves as
// dataloaders.LeavesFileForLexicon does ("" for the lexicon's standard ones).
func MagpieData(cfg *config.Config, lexiconName string, dist *tilemapping.LetterDistribution,
	distName string, layout []string, wordSmog bool, leavefile string) (*magpie.Data, error) {

	wcfg := cfg.WGLConfig()
	if leavefile == "leaves.klv2" {
		leavefile = "" // the lexicon's standard leaves
	}
	if leavefile == "" && len(layout) == 21 {
		leavefile = "super-leaves.klv2"
	}
	key := magpieDataKey{dataPath: wcfg.DataPath, lexicon: lexiconName, dist: distName,
		leaves: leavefile, dim: len(layout), wordSmog: wordSmog}
	v, _ := magpieDataCache.LoadOrStore(key, &magpieDataEntry{})
	e := v.(*magpieDataEntry)
	e.once.Do(func() {
		e.data, e.err = loadMagpieData(cfg, lexiconName, dist, layout, wordSmog, leavefile)
	})
	return e.data, e.err
}

func loadMagpieData(cfg *config.Config, lexiconName string, dist *tilemapping.LetterDistribution,
	layout []string, wordSmog bool, leavefile string) (*magpie.Data, error) {

	wcfg := cfg.WGLConfig()
	lexDir := dataloaders.LexiconDataPath(wcfg)
	ext := ".kwg"
	variant := magpie.VariantClassic
	if wordSmog {
		// WordSmog plays out of an alpha KWG, the .kad.
		ext, variant = ".kad", magpie.VariantWordSmog
	}
	if err := lexicon.EnsureLexiconFileOnce(lexiconName, ext, wcfg); err != nil {
		log.Info().Err(err).Str("lexicon", lexiconName).Msg("could not fetch word graph; trying disk")
	}
	kwgBytes, err := readDataFile(filepath.Join(lexDir, lexiconName+ext))
	if err != nil {
		return nil, fmt.Errorf("word graph for %s: %w", lexiconName, err)
	}
	var klvBytes []byte
	lf, err := dataloaders.LeavesFileForLexicon(wcfg, leavefile, lexiconName)
	if err == nil {
		klvBytes, err = io.ReadAll(lf)
		lf.Close()
	}
	if err != nil {
		// Some lexica (test ones, experiments) have no leaves. MAGPIE always
		// wants some, so give it leaves that are all worth nothing, as
		// Macondo's equity calculators do in this case.
		log.Warn().Err(err).Str("lexicon", lexiconName).Msg("no leaves; generating with none")
		klvBytes = noLeavesKLV
	}

	opts := magpie.DataOptions{KWG: kwgBytes, KLV: klvBytes, Dist: dist, Layout: layout,
		Variant: variant, BingoBonus: 50}
	// A word map, as Macondo's policy says: read from disk if there, else
	// built in memory. They are only made for 15x15 classic boards.
	pol := wmp.CurrentPolicy()
	if pol.Enabled && !wordSmog && len(layout) == 15 && bitRackCompatible(dist, len(layout)) {
		wmpPath := filepath.Join(wcfg.DataPath, "lexica", lexiconName+".wmp")
		if fi, err := os.Stat(wmpPath); err == nil && (pol.MaxBytes == 0 || fi.Size() <= pol.MaxBytes) {
			opts.WMPPath = wmpPath
		} else if pol.BuildIfMissing {
			opts.BuildWMP = true
		}
		// A rack info table beside it is used if present (see magpie.WriteRIT).
		ritPath := filepath.Join(wcfg.DataPath, "lexica", lexiconName+".rit")
		if _, err := os.Stat(ritPath); err == nil && (opts.WMPPath != "" || opts.BuildWMP) {
			opts.RITPath = ritPath
		}
	}
	d, err := magpie.NewData(opts)
	if err != nil {
		return nil, err
	}
	if opts.BuildWMP && d.HasWMP() {
		// Building one can take tens of seconds for a big lexicon; keep it
		// for next time, as Macondo's own word map policy does.
		saveWMP(d, filepath.Join(wcfg.DataPath, "lexica", lexiconName+".wmp"))
	}
	log.Info().Str("lexicon", lexiconName).Int("dim", len(layout)).Bool("wordsmog", wordSmog).
		Bool("wmp", d.HasWMP()).Bool("rit", opts.RITPath != "").Msg("loaded-magpie-data")
	return d, nil
}

// The board of a game lives in MAGPIE: its position (board, cross-sets,
// anchors, racks and bag) is where moves are played and unplayed and where
// moves are generated. The Go board is a copy of it, refreshed when Board()
// is asked for after a move (viewStale), which code that only reads the board
// -- validation, scoring by hand, display, NN features -- uses. Code that
// changes the Go board directly (setting up a position, replaying a history
// from scratch) marks MAGPIE's copy stale (posStale), and it is rebuilt from
// the Go board on next use. Racks and the bag are Go's and are copied to
// MAGPIE whenever it is about to use them.

// ReleaseMagpie frees this game's MAGPIE position now rather than when the
// garbage collector gets to it: the collector only sees the small Go side,
// so a program that churns through games (mlproducer) can hold gigabytes of
// MAGPIE positions between collections. The Go board is brought up to date
// first, so the game stays usable: it builds a new position from the board if
// it is used again. Only call it on a game that is not in the middle of
// playing and unplaying moves (MAGPIE's backup stack goes with the position).
func (g *Game) ReleaseMagpie() {
	if g.pos != nil {
		g.refreshView()
		g.pos.Close()
		g.pos = nil
	}
}

// MagpiePosition returns this game's MAGPIE position, for move generation,
// with the racks, bag and player on turn brought up to date. Callers should
// ask for it again rather than hold it across changes they make to the game.
func (g *Game) MagpiePosition() (*magpie.Position, error) {
	if g.goBoard {
		return nil, errNoMagpie
	}
	if g.pos == nil {
		d, err := g.rules.MagpieData()
		if err != nil {
			return nil, err
		}
		g.pos = magpie.NewPosition(d, magpie.DefaultMoveCapacity, 0)
		g.pos.SetBackupMode(posBackupMode(g.backupMode))
		g.posStale = true
	}
	if err := g.syncPos(); err != nil {
		return nil, err
	}
	return g.pos, nil
}

// errNoMagpie is MagpiePosition's error for a game without a lexicon, whose
// board is kept on the Go side (see Game.goBoard).
var errNoMagpie = errors.New("a game without a lexicon has no MAGPIE position")

func posBackupMode(m BackupMode) magpie.BackupMode {
	switch m {
	case SimulationMode:
		return magpie.BackupStack
	case InteractiveGameplayMode:
		return magpie.BackupSingle
	}
	return magpie.BackupOff
}

// markPosStale says the Go board was changed directly, so MAGPIE's position
// is rebuilt from it on next use. Callers change the board they got from
// Board(), which is up to date.
func (g *Game) markPosStale() {
	if g.pos != nil {
		g.posStale = true
	}
	g.viewStale = false
}

// pushRacks copies both racks, the bag and the player on turn to MAGPIE.
// They are cheap to send, and code outside the game changes the bag
// directly, so they are sent rather than tracked.
func (g *Game) pushRacks() {
	r0 := g.posScratch[0][:g.players[0].rack.NoAllocTilesOn(g.posScratch[0][:])]
	r1 := g.posScratch[1][:g.players[1].rack.NoAllocTilesOn(g.posScratch[1][:])]
	g.pos.SetRacksAndBag(r0, r1, g.bag.Tiles(), g.onturn)
}

func (g *Game) syncPos() error {
	if !g.posStale {
		g.pushRacks()
		return nil
	}
	dim := g.board.Dim()
	squares := make([]tilemapping.MachineLetter, dim*dim)
	for r := 0; r < dim; r++ {
		for c := 0; c < dim; c++ {
			squares[r*dim+c] = g.board.GetLetter(r, c)
		}
	}
	err := g.pos.Set(magpie.State{
		Squares: squares,
		Racks: [2]tilemapping.MachineWord{
			g.players[0].rack.TilesOn(), g.players[1].rack.TilesOn()},
		Bag:            g.bag.Tiles(),
		OnTurn:         g.onturn,
		Scores:         [2]int{g.players[0].points, g.players[1].points},
		ScorelessTurns: g.scorelessTurns,
	})
	if err != nil {
		return err
	}
	g.posStale = false
	// The tiles agree; the cross-sets come from MAGPIE.
	g.viewStale = true
	// Set drops MAGPIE's saved plays, so no backup taken so far has one.
	for _, st := range g.stateStack {
		st.posPlayed, st.posSmall = false, false
	}
	return nil
}

// refreshView brings the Go board up to date with MAGPIE's.
func (g *Game) refreshView() {
	if !g.viewStale || g.pos == nil {
		return
	}
	if g.mboard == nil {
		g.mboard = g.pos.Data().NewBoard()
	}
	g.pos.ReadBoard(g.mboard)
	g.board.SetFromMagpie(g.mboard.Letters, g.mboard.CrossSets, g.mboard.CrossScores)
	g.viewStale = false
}

// toMagpieMove converts a move for MAGPIE.
func toMagpieMove(m *move.Move) magpie.Move {
	var mm magpie.Move
	switch m.Action() {
	case move.MoveTypePlay:
		mm.Type = magpie.MovePlacement
		r, c, v := m.CoordsAndVertical()
		mm.Row, mm.Col = uint8(r), uint8(c)
		if v {
			mm.Vertical = 1
		}
		mm.Score = int32(m.Score() * magpie.EquityResolution)
		mm.TilesPlayed = uint8(m.TilesPlayed())
		mm.TilesLength = uint8(copy(mm.Tiles[:], m.Tiles()))
	case move.MoveTypeExchange:
		mm.Type = magpie.MoveExchange
		mm.TilesPlayed = uint8(len(m.Tiles()))
		mm.TilesLength = uint8(copy(mm.Tiles[:], m.Tiles()))
	default:
		mm.Type = magpie.MovePass
	}
	return mm
}

// posPlay plays m on the board in MAGPIE's position, saved for unplay as the
// backup mode says, or on the Go board in a game without MAGPIE. The racks
// and bag are Go's; MAGPIE gets them the next time it is asked for.
func (g *Game) posPlay(m *move.Move) error {
	if g.goBoard {
		if m.Action() == move.MoveTypePlay {
			g.board.PlayMove(m)
			g.crossScores.UpdateForMove(g.board, m)
		}
		return nil
	}
	mm := toMagpieMove(m)
	g.pos.PlayBoard(&mm)
	g.viewStale = true
	if g.backupMode != NoBackup {
		st := g.stateStack[max(g.stackPtr-1, 0)]
		st.posPlayed, st.posSmall = true, false
	}
	return nil
}

// posPlaySmall plays a small move in MAGPIE's position, filling
// g.stripBackup. With drawn non-nil (a pre-endgame) it is a backed-up play
// that draws them; otherwise an incremental one (the endgame).
func (g *Game) posPlaySmall(m *tinymove.SmallMove, drawn []tilemapping.MachineLetter) {
	var err error
	if drawn != nil {
		_, err = g.pos.PlaySmallDrawing(uint64(m.TinyMove()), m.Score(), drawn, g.stripBackup[:])
	} else {
		_, err = g.pos.PlaySmall(uint64(m.TinyMove()), m.Score(), g.stripBackup[:])
	}
	if err != nil {
		panic(fmt.Sprintf("playing %v: %v", m.ShortDescription(), err))
	}
	g.viewStale = true
	if g.backupMode != NoBackup {
		st := g.stateStack[max(g.stackPtr-1, 0)]
		st.posPlayed = drawn != nil
		st.posSmall = drawn == nil
	}
}

// posUnplay undoes in MAGPIE's position what the backup st was taken for.
// A backup whose move did not touch the board (a pass replayed from a
// history, say) has nothing there to undo.
func (g *Game) posUnplay(st *stateBackup) {
	var err error
	switch {
	case g.pos == nil || g.posStale:
	case st.posSmall:
		err = g.pos.UnplaySmall()
		g.viewStale = true
	case st.posPlayed:
		err = g.pos.Unplay()
		g.viewStale = true
	}
	st.posPlayed, st.posSmall = false, false
	if err != nil {
		panic(fmt.Sprintf("MAGPIE's position has no backup to unplay: %v", err))
	}
}

// MoveFromMagpie converts a move MAGPIE generated for the player on turn
// into a Macondo move, with its leave and equity (MAGPIE's, in points).
func (g *Game) MoveFromMagpie(m *magpie.Move) *move.Move {
	rack := g.players[g.onturn].rack
	alph := g.alph
	equity := float64(m.Equity) / magpie.EquityResolution
	var mv *move.Move
	switch m.Type {
	case magpie.MovePlacement:
		tiles := append(tilemapping.MachineWord(nil), m.Word()...)
		leave, err := tilemapping.Leave(rack.TilesOn(), tiles, true)
		if err != nil {
			panic(fmt.Sprintf("magpie move %v is not on rack %v: %v", tiles, rack.String(), err))
		}
		mv = move.NewScoringMove(int(m.Score/magpie.EquityResolution), tiles, leave,
			m.Vertical == 1, int(m.TilesPlayed), alph, int(m.Row), int(m.Col))
	case magpie.MoveExchange:
		tiles := append(tilemapping.MachineWord(nil), m.Word()...)
		leave, err := tilemapping.Leave(rack.TilesOn(), tiles, false)
		if err != nil {
			panic(fmt.Sprintf("magpie exchange %v is not on rack %v: %v", tiles, rack.String(), err))
		}
		mv = move.NewExchangeMove(tiles, leave, alph)
	default:
		mv = move.NewPassMove(rack.TilesOn(), alph)
	}
	mv.SetEquity(equity)
	return mv
}

// takeSmallMoveTiles takes a small move's placed tiles off the player's rack.
func takeSmallMoveTiles(m *tinymove.SmallMove, rack *tilemapping.Rack) {
	tm := m.TinyMove()
	blanks := uint64(tm) >> 12 & 127
	for i := 0; i < m.TilesPlayed(); i++ {
		t := tilemapping.MachineLetter(uint64(tm) >> (20 + 6*i) & 63)
		if blanks&(1<<i) != 0 {
			t = 0
		}
		rack.Take(t)
	}
}

// saveWMP writes a word map MAGPIE built to path, if the directory is
// writable. A failure only costs the next process another build.
func saveWMP(d *magpie.Data, path string) {
	tmp := fmt.Sprintf("%s.%d.tmp", path, os.Getpid())
	if err := d.WriteWMP(tmp); err != nil {
		os.Remove(tmp)
		log.Info().Err(err).Str("path", path).Msg("could not save word map")
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		log.Info().Err(err).Str("path", path).Msg("could not save word map")
		return
	}
	log.Info().Str("path", path).Msg("saved word map")
}

// bitRackCompatible reports whether MAGPIE can key word maps and rack info
// tables on this distribution's racks: at most 32 tiles (blank included),
// and no letter that could appear more than 15 times counting blanks (as
// MAGPIE's bit_rack_is_compatible_with_ld decides). Polish, with 33, can't.
func bitRackCompatible(dist *tilemapping.LetterDistribution, dim int) bool {
	counts := dist.Distribution()
	if len(counts) > 32 {
		return false
	}
	for ml, n := range counts {
		c := int(n)
		if ml != 0 {
			c += int(counts[0])
		}
		if min(c, dim) > 15 {
			return false
		}
	}
	return true
}

// MagpieMoves generates up to n moves for the player on turn, best first by
// sort (MAGPIE's static equity, or score), as Macondo moves. It is for tests
// and tools; players generate through ai/turnplayer.
func (g *Game) MagpieMoves(n int, sort magpie.Sort) ([]*move.Move, error) {
	pos, err := g.MagpiePosition()
	if err != nil {
		return nil, err
	}
	var gen []magpie.Move
	if n == 1 {
		gen = pos.Generate(magpie.RecordBest, sort, 0)
	} else {
		gen = pos.GenerateN(magpie.RecordAll, sort, 0, n)
	}
	moves := make([]*move.Move, len(gen))
	for i := range gen {
		moves[i] = g.MoveFromMagpie(&gen[i])
	}
	return moves, nil
}
