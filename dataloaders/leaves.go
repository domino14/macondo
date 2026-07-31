package dataloaders

import (
	"io"
	"path/filepath"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/cache"
	wglconfig "github.com/domino14/word-golib/config"

	"github.com/domino14/macondo/lexicon"
)

const (
	// LeavesExtension is the extension of a leave-values file.
	LeavesExtension = ".klv2"
	// legacyLeavesName and legacySuperLeavesName are the names leave files used
	// to have inside their per-lexicon strategy folder. They are still accepted
	// as the way a caller asks for "the normal leaves" or "the super leaves",
	// but the file itself now lives next to the word graphs, named after its
	// lexicon.
	legacyLeavesName      = "leaves.klv2"
	legacySuperLeavesName = "super-leaves.klv2"
)

// klvDownloadTried remembers which leave files this process has already tried
// to download, so a missing one is fetched at most once.
var klvDownloadTried sync.Map

// LexiconDataPath is the directory holding per-lexicon binary data: word
// graphs (.kwg, .kad) and leave values (.klv2).
func LexiconDataPath(cfg *wglconfig.Config) string {
	return filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix)
}

// leavesNameFor returns the name a lexicon's leave file has in the lexicon
// data directory. The second return value is false when leavefile names a
// specific file rather than asking for a lexicon's standard leaves -- those
// are one-off experiments and only ever live in the strategy folders.
func leavesNameFor(leavefile, lexiconName string) (string, bool) {
	switch leavefile {
	case "", legacyLeavesName:
		return lexiconName + LeavesExtension, true
	case legacySuperLeavesName:
		return "super-" + lexiconName + LeavesExtension, true
	}
	return "", false
}

// LeavesFileForLexicon opens the leave-values file for a lexicon. It looks, in
// order:
//
//  1. <data>/lexica/gaddag/<prefix>/<LEXICON>.klv2, the current layout, which
//     keeps a lexicon's leaves beside its word graphs
//  2. the same for the lexicon this one borrows from (CSW19 uses CSW24's)
//  3. the legacy per-lexicon strategy folders
//  4. failing all of that, a download, since these files are published
//     alongside the word graphs
//
// leavefile selects which leaves are wanted: "" or "leaves.klv2" for the
// normal ones, "super-leaves.klv2" for a SuperCrosswordGame's, or an explicit
// filename to pick a specific experiment out of a strategy folder.
func LeavesFileForLexicon(cfg *wglconfig.Config, leavefile, lexiconName string) (io.ReadCloser, error) {
	name, standard := leavesNameFor(leavefile, lexiconName)
	dir := LexiconDataPath(cfg)

	if standard {
		if file, _, err := cache.Open(filepath.Join(dir, name)); err == nil {
			return file, nil
		}
		// A lexicon with no leaves of its own borrows another's, the same way
		// the strategy folders do.
		if def := defaultForLexicon(lexiconName); def != "" && def != lexiconName {
			defName, _ := leavesNameFor(leavefile, def)
			if file, _, err := cache.Open(filepath.Join(dir, defName)); err == nil {
				log.Debug().Str("lexicon", lexiconName).Str("leaves", defName).
					Msg("no lexicon-specific leaves; borrowing")
				return file, nil
			}
		}
	}

	strategyName := leavefile
	if strategyName == "" {
		strategyName = legacyLeavesName
	}
	file, err := StratFileForLexicon(StrategyParamsPath(cfg), strategyName, lexiconName)
	if err == nil {
		return file, nil
	}
	if !standard {
		return nil, err
	}

	// Nothing on disk anywhere. These files are published next to the word
	// graphs, so try to fetch this lexicon's -- once per process. The object
	// cache doesn't remember failures, so without this a lexicon that has no
	// published leaves (an experimental one, say) would retry the download
	// every time an equity calculator is built.
	if _, tried := klvDownloadTried.LoadOrStore(name, struct{}{}); tried {
		return nil, err
	}
	if derr := EnsureKLV(name, cfg); derr != nil {
		log.Info().Err(derr).Str("leaves", name).Msg("could not download leave values")
		return nil, err
	}
	file, _, oerr := cache.Open(filepath.Join(dir, name))
	if oerr != nil {
		return nil, err
	}
	return file, nil
}

// EnsureKLV makes sure a leave-values file is on disk, downloading it if not.
// name is the bare file name without its extension, e.g. "NWL23" or
// "super-NWL23".
func EnsureKLV(name string, cfg *wglconfig.Config) error {
	if filepath.Ext(name) == LeavesExtension {
		name = name[:len(name)-len(LeavesExtension)]
	}
	return lexicon.EnsureLexiconFile(name, LeavesExtension, cfg)
}
