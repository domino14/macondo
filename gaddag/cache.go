package gaddag

import (
	"path/filepath"
	"strings"

	"github.com/domino14/macondo/config"
)

// CacheLoadFunc is the function that loads a gaddag object into the global cache.
func CacheLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	lexiconName := strings.TrimPrefix(key, "gaddag:")
	return LoadGaddag(filepath.Join(cfg.LexiconPath, "gaddag", lexiconName+".gaddag"))
}
