package wmp

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/cache"
	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

const CacheKeyPrefixWMP = "wmp:"

// CacheLoadFuncWMP loads a WMP for the given lexicon key from disk.
// It does NOT build the WMP if the file is absent — callers that want
// build-on-miss should use EnsureWMP instead. Errors from this function
// are not cached by word-golib/cache, so a subsequent EnsureWMP call
// that writes the file to disk will succeed on the next GetWMP call.
func CacheLoadFuncWMP(cfg *wglconfig.Config, key string) (interface{}, error) {
	name := strings.TrimPrefix(key, CacheKeyPrefixWMP)
	wmpPath := filepath.Join(cfg.DataPath, "lexica", name+".wmp")
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

// EnsureWMP ensures the WMP for the named lexicon is on disk and in the
// global cache. If the .wmp file is already present, EnsureWMP is equivalent
// to GetWMP. If not, it builds the WMP from the KWG (which may take several
// minutes for large lexicons), saves it to disk, and caches it.
//
// This is an interactive / setup operation and should only be called from
// paths where the user is willing to wait (e.g. shell "set lexicon" or
// "load" commands). Automated / test paths should use GetWMP (or
// TryLoadWMP) so they never trigger an unexpectedly long build.
func EnsureWMP(cfg *wglconfig.Config, name string) (*WMP, error) {
	wmpPath := filepath.Join(cfg.DataPath, "lexica", name+".wmp")

	if _, err := os.Stat(wmpPath); err != nil {
		// File not on disk — build from KWG.
		log.Info().Str("lexicon", name).Msg("WMP not found; building from KWG (this may take a few seconds)...")
		gd, err := kwg.GetKWG(cfg, name)
		if err != nil {
			return nil, fmt.Errorf("cannot build WMP for %s: KWG not available: %w", name, err)
		}
		ld, err := tilemapping.ProbableLetterDistribution(cfg, name)
		if err != nil {
			return nil, fmt.Errorf("cannot build WMP for %s: letter distribution unavailable: %w", name, err)
		}
		// Use boardDim=21 (the largest supported crossword board) so the WMP
		// covers words of all lengths that appear in any standard lexicon.
		// CheckCompatible still passes for English (max effective count 14 ≤ 15).
		w, buildErr := MakeFromKWG(gd, ld, 21, runtime.NumCPU())
		if buildErr != nil {
			return nil, fmt.Errorf("WMP build failed for %s: %w", name, buildErr)
		}
		if wErr := w.WriteToFile(wmpPath); wErr != nil {
			log.Warn().Err(wErr).Str("path", wmpPath).
				Msg("WMP built but could not be saved to disk; it will be rebuilt next session")
		} else {
			log.Info().Str("lexicon", name).Str("path", wmpPath).Msg("WMP built and saved")
		}
	}

	// File is now on disk (either pre-existing or just written). Load via cache.
	return GetWMP(cfg, name)
}
