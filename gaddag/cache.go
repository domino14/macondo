package gaddag

import (
	"path/filepath"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
)

// Gaddags should be stored in a cache and not loaded multiple times, to save memory and time,
// particularly if we have loaded many games in memory.

// Cache is a, well, gaddag cache.
type cache struct {
	sync.Mutex
	gaddags map[string]GenericDawg
}

// GenericDawgCache is our global cache for gaddags/dawgs
var GenericDawgCache *cache

// CreateGaddagCache creates the global gaddag cache
func CreateGaddagCache() {
	GenericDawgCache = &cache{gaddags: make(map[string]GenericDawg)}
}

// Load loads the lexicon into the cache.
func (gc *cache) Load(cfg *config.Config, lexiconName string) error {
	log.Debug().Str("lexicon", lexiconName).Msg("loading GenericDawg into cache")
	gd, err := LoadGaddag(filepath.Join(cfg.LexiconPath, "gaddag", lexiconName+".gaddag"))
	if err != nil {
		return err
	}
	gc.Lock()
	defer gc.Unlock()
	gc.gaddags[lexiconName] = gd
	return nil
}

// Get gets the gaddag from the cache. If it's not in the cache it will try
// to load it in before getting it.
func (gc *cache) Get(cfg *config.Config, lexiconName string) (GenericDawg, error) {
	var ok bool
	var gd GenericDawg
	if gd, ok = gc.gaddags[lexiconName]; !ok {
		err := gc.Load(cfg, lexiconName)
		if err != nil {
			return nil, err
		}
		return gc.gaddags[lexiconName], nil
	}
	log.Debug().Str("lexicon", lexiconName).Msg("getting GenericDawg from cache")
	return gd, nil
}

func LoadFromCache(cfg *config.Config, name string) (GenericDawg, error) {
	if GenericDawgCache == nil {
		CreateGaddagCache()
	}
	return GenericDawgCache.Get(cfg, name)
}
