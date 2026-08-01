package wmp

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog"
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
	p := CurrentPolicy()

	if !p.useWordMap() {
		logWMP(name, "wmp-disabled").Msg("word maps are turned off; using the KWG move generator")
		return nil, ErrDisabled
	}

	ensureMu.Lock()
	defer ensureMu.Unlock()

	if w, ok := builtWMPs[wmpPath]; ok {
		// Already in this process's memory. On a platform that reuses an
		// execution environment across invocations -- Lambda does -- this is
		// the common case after the first one, and the whole reason the build
		// cost is worth paying at all.
		logWMP(name, "wmp-from-memory").Msg("word map already in memory")
		return w, nil
	}

	rebuild := false
	if fi, err := os.Stat(wmpPath); err == nil {
		size := fi.Size()
		switch {
		case !p.allowsSize(size):
			logWMP(name, "wmp-too-large").Int64("bytes", size).
				Int64("max_bytes", p.MaxBytes).
				Msg("word map is larger than the limit; using the KWG move generator")
			return nil, fmt.Errorf("%s is %d MB, over the %d MB limit: %w",
				name, size>>20, p.MaxBytes>>20, ErrDisabled)
		case p.prefersRebuild(size):
			logWMP(name, "wmp-rebuild-preferred").Int64("bytes", size).
				Msg("word map is larger than the read limit; building it instead of reading it")
			rebuild = true
		}

		if !rebuild {
			start := time.Now()
			// Already on disk; the global object cache dedupes the load.
			w, err := GetWMP(cfg, name)
			if err == nil {
				logWMP(name, "wmp-loaded-from-disk").Int64("bytes", size).
					Dur("took", time.Since(start)).Msg("word map read from disk")
				return w, nil
			}
			// The file is there but unreadable — truncated by an interrupted
			// write, say. Fall through and rebuild rather than failing on it
			// for the life of the process (and every process after).
			log.Warn().Err(err).Str("path", wmpPath).Msg("could not load WMP; rebuilding it")
		}
	}

	buildStart := time.Now()
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
	// The size of a word map can't be known before building it -- the ratio to
	// the KWG it comes from runs from 27x to 94x across lexica -- so a build
	// that turns out to be over the limit is used anyway rather than thrown
	// away. It is logged, because a lexicon that lands here every cold start
	// is one to seed on disk or to put over the limit deliberately.
	var built int64
	if fi, err := os.Stat(wmpPath); err == nil {
		built = fi.Size()
	}
	logWMP(name, "wmp-built").Int64("bytes", built).
		Dur("took", time.Since(buildStart)).Msg("word map built from the KWG")
	builtWMPs[wmpPath] = w
	return w, nil
}

// processStart and wmpEvents let a reader of the logs tell how often a word
// map is actually being loaded or built versus found in memory, and over what
// stretch of a process's life. On Lambda that answers how much an execution
// environment is being reused.
var (
	processStart = time.Now()
	wmpEvents    atomic.Int64
)

func logWMP(lexicon, event string) *zerolog.Event {
	return log.Info().
		Str("event", event).
		Str("lexicon", lexicon).
		Int64("wmp_events_this_process", wmpEvents.Add(1)).
		Dur("process_uptime", time.Since(processStart))
}
