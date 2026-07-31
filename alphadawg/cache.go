package alphadawg

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/domino14/word-golib/cache"
	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/kwg"

	"github.com/domino14/macondo/lexicon"
)

const (
	// CacheKeyPrefixKAD namespaces alpha dawgs in the global object cache so
	// they never collide with the gaddag of the same lexicon.
	CacheKeyPrefixKAD = "kad:"
	// Extension is the file extension for an alpha dawg.
	Extension = ".kad"
)

func kadPath(cfg *wglconfig.Config, lexiconName string) string {
	return filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix,
		lexiconName+Extension)
}

// EnsureKAD makes sure the alpha dawg for lexname is on disk, downloading it
// if necessary. Call it from interactive paths before Get.
func EnsureKAD(lexname string, cfg *wglconfig.Config) error {
	return lexicon.EnsureWordGraphFile(lexname, Extension, cfg)
}

// Get returns the alpha dawg for the named lexicon from the global object
// cache, loading it from disk on first access. It does not download; call
// EnsureKAD first if that is wanted. The graph is read-only after load and
// safe to share across goroutines.
//
// distName may be empty, in which case the letter distribution (and thus the
// alphabet) is guessed from the lexicon name.
func Get(cfg *wglconfig.Config, name, distName string) (*kwg.KWG, error) {
	key := CacheKeyPrefixKAD + name
	if distName != "" {
		key += ":" + strings.ToLower(distName)
	}
	obj, err := cache.Load(cfg, key, func(cfg *wglconfig.Config, _ string) (any, error) {
		path := kadPath(cfg, name)
		if _, err := os.Stat(path); err != nil {
			return nil, fmt.Errorf("alpha dawg (WordSmog dictionary) not found for %s at %s", name, path)
		}
		var opts []kwg.LoadOption
		if distName != "" {
			opts = append(opts, kwg.WithDistribution(distName))
		}
		// A .kad is byte-wise a DawgOnly KWG, so the standard loader reads it
		// as-is. Only the root node index differs, which alphadawg handles.
		return kwg.LoadKWG(cfg, path, opts...)
	})
	if err != nil {
		return nil, err
	}
	k, ok := obj.(*kwg.KWG)
	if !ok {
		return nil, errors.New("could not convert cached object to alpha dawg")
	}
	return k, nil
}
