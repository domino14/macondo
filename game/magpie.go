package game

import (
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
	lf, err := dataloaders.LeavesFileForLexicon(wcfg, leavefile, lexiconName)
	if err != nil {
		return nil, fmt.Errorf("leaves for %s: %w", lexiconName, err)
	}
	klvBytes, err := io.ReadAll(lf)
	lf.Close()
	if err != nil {
		return nil, fmt.Errorf("leaves for %s: %w", lexiconName, err)
	}

	opts := magpie.DataOptions{KWG: kwgBytes, KLV: klvBytes, Dist: dist, Layout: layout,
		Variant: variant, BingoBonus: 50}
	// A word map, as Macondo's policy says: read from disk if there, else
	// built in memory. They are only made for 15x15 classic boards.
	pol := wmp.CurrentPolicy()
	if pol.Enabled && !wordSmog && len(layout) == 15 {
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
	log.Info().Str("lexicon", lexiconName).Int("dim", len(layout)).Bool("wordsmog", wordSmog).
		Bool("wmp", d.HasWMP()).Bool("rit", opts.RITPath != "").Msg("loaded-magpie-data")
	return d, nil
}

// MagpiePosition returns MAGPIE's copy of this game's position, for move
// generation: board, cross-sets, both racks and the bag, with the player on
// turn. It is made the first time it is asked for, and from then on kept in
// step with the game as moves are played and unplayed. Each call brings the
// racks and bag up to date, so callers should ask for it again rather than
// hold it across changes they make to the game.
func (g *Game) MagpiePosition() (*magpie.Position, error) {
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

func posBackupMode(m BackupMode) magpie.BackupMode {
	switch m {
	case SimulationMode:
		return magpie.BackupStack
	case InteractiveGameplayMode:
		return magpie.BackupSingle
	}
	return magpie.BackupOff
}

// markPosStale says the Go board changed in a way MAGPIE's copy did not
// follow, so the next MagpiePosition call rebuilds it.
func (g *Game) markPosStale() {
	if g.pos != nil {
		g.posStale = true
	}
}

// posLive reports whether MAGPIE's copy is following moves.
func (g *Game) posLive() bool {
	return g.pos != nil && !g.posStale
}

// pushRacks copies both racks, the bag and the player on turn to MAGPIE.
// They are cheap to send, and code outside the game changes the bag
// directly, so they are sent rather than tracked.
func (g *Game) pushRacks() {
	r0 := g.posScratch[0][:g.players[0].rack.NoAllocTilesOn(g.posScratch[0][:])]
	r1 := g.posScratch[1][:g.players[1].rack.NoAllocTilesOn(g.posScratch[1][:])]
	g.pos.SetRacksAndBag(r0, r1, g.bag.Tiles())
	g.pos.SetOnTurn(g.onturn)
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
	// Set drops MAGPIE's saved plays, so no backup taken so far has one.
	for _, st := range g.stateStack {
		st.posPlayed = false
	}
	return nil
}

// posPlay plays m in MAGPIE's copy, if it is following moves, with the tiles
// the player drew. Called after the Go side has played it.
func (g *Game) posPlay(m *move.Move, drawn []tilemapping.MachineLetter) {
	if !g.posLive() {
		return
	}
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
	if err := g.pos.Play(&mm, drawn); err != nil {
		// The copies disagree; rebuild MAGPIE's from the Go side.
		log.Warn().Err(err).Msg("magpie-position-out-of-step")
		g.posStale = true
		return
	}
	if g.backupMode != NoBackup {
		g.stateStack[max(g.stackPtr-1, 0)].posPlayed = true
	}
}

// posUnplay undoes MAGPIE's copy of the play the backup st undid.
func (g *Game) posUnplay(st *stateBackup) {
	if g.pos == nil {
		return
	}
	if !st.posPlayed || g.posStale || g.pos.Unplay() != nil {
		g.posStale = true
	}
	st.posPlayed = false
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
