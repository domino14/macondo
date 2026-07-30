package wmp

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/cache"
	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// ensureMu serializes the check-build-write sequence in EnsureWMP so that
// concurrent callers (e.g. autoplay goroutines) don't all race to build the
// same lexicon. It also guards builtWMPs.
var ensureMu sync.Mutex

// builtWMPs memoizes WMPs built by EnsureWMP, keyed by .wmp path. Built
// WMPs can't go into word-golib's global object cache: that cache only
// populates itself through a load function which it calls with its own
// non-reentrant mutex held, and building needs kwg.GetKWG and
// tilemapping.ProbableLetterDistribution, both of which take that same
// mutex. Building inside a load function deadlocks. So the build happens
// out here and the result is memoized here.
var builtWMPs = map[string]*WMP{}

const CacheKeyPrefixWMP = "wmp:"

func wmpPathFor(cfg *wglconfig.Config, name string) string {
	return filepath.Join(cfg.DataPath, "lexica", name+".wmp")
}

// CacheLoadFuncWMP loads a WMP for the given lexicon key from disk.
// It does NOT build the WMP if the file is absent — callers that want
// build-on-miss should use EnsureWMP instead. Errors from this function
// are not cached by word-golib/cache, so a subsequent EnsureWMP call
// that writes the file to disk will succeed on the next GetWMP call.
func CacheLoadFuncWMP(cfg *wglconfig.Config, key string) (interface{}, error) {
	name := strings.TrimPrefix(key, CacheKeyPrefixWMP)
	wmpPath := wmpPathFor(cfg, name)
	if _, err := os.Stat(wmpPath); err != nil {
		return nil, fmt.Errorf("WMP file not found for %s at %s", name, wmpPath)
	}
	return LoadFromFile(name, wmpPath)
}

// GetWMP returns the WMP for the named lexicon from the global object cache,
// loading it from disk on first access. It does NOT build the WMP if absent;
// use EnsureWMP for that. The WMP is read-only after load and safe to share
// across goroutines.
func GetWMP(cfg *wglconfig.Config, name string) (*WMP, error) {
	obj, err := cache.Load(cfg, CacheKeyPrefixWMP+name, CacheLoadFuncWMP)
	if err != nil {
		return nil, err
	}
	w, ok := obj.(*WMP)
	if !ok {
		return nil, errors.New("could not convert cached object to WMP")
	}
	return w, nil
}

// EnsureWMP returns the WMP for the named lexicon, building it from the KWG
// when the .wmp file is not on disk. A built WMP is saved for next time if
// the lexica directory is writable; if it isn't (a read-only data mount, for
// instance), the WMP is still returned and memoized, so the build cost is
// paid once per process instead of once per call.
//
// Building takes a second or two per lexicon and a few hundred MB of
// transient memory, so callers on a latency budget should prefer a
// pre-built .wmp file on disk.
func EnsureWMP(cfg *wglconfig.Config, name string) (*WMP, error) {
	wmpPath := wmpPathFor(cfg, name)

	ensureMu.Lock()
	defer ensureMu.Unlock()

	if w, ok := builtWMPs[wmpPath]; ok {
		return w, nil
	}
	if _, err := os.Stat(wmpPath); err == nil {
		// Already on disk; the global object cache dedupes the load.
		w, err := GetWMP(cfg, name)
		if err == nil {
			return w, nil
		}
		// The file is there but unreadable — truncated by an interrupted
		// write, say. Fall through and rebuild rather than failing on it
		// for the life of the process (and every process after).
		log.Warn().Err(err).Str("path", wmpPath).Msg("could not load WMP; rebuilding it")
	}

	log.Info().Str("lexicon", name).Msg("WMP not found; building from KWG (this takes a second or two)...")
	gd, err := kwg.GetKWG(cfg, name)
	if err != nil {
		return nil, fmt.Errorf("cannot build WMP for %s: KWG not available: %w", name, err)
	}
	ld, err := tilemapping.ProbableLetterDistribution(cfg, name)
	if err != nil {
		return nil, fmt.Errorf("cannot build WMP for %s: letter distribution unavailable: %w", name, err)
	}
	// Use boardDim=15 for standard crossword boards. WMP only supports 15×15 boards.
	w, err := MakeFromKWG(gd, ld, 15, runtime.NumCPU())
	if err != nil {
		return nil, fmt.Errorf("WMP build failed for %s: %w", name, err)
	}
	w.Name = name

	if wErr := w.WriteToFile(wmpPath); wErr != nil {
		log.Warn().Err(wErr).Str("path", wmpPath).
			Msg("WMP built but could not be saved to disk; using it from memory and rebuilding next session")
	} else {
		log.Info().Str("lexicon", name).Str("path", wmpPath).Msg("WMP built and saved")
	}
	builtWMPs[wmpPath] = w
	return w, nil
}
