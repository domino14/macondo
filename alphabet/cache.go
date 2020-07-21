package alphabet

import (
	"sync"

	"github.com/domino14/macondo/config"
	"github.com/rs/zerolog/log"
)

type cache struct {
	sync.Mutex
	letterDistributions map[string]*LetterDistribution
}

// LetterDistributionCache is a global letter distribution cache.
var LetterDistributionCache *cache

// CreateLetterDistributionCache creates the global letter distribution cache.
func CreateLetterDistributionCache() {
	LetterDistributionCache = &cache{letterDistributions: make(map[string]*LetterDistribution)}
}

// Load loads the letter distribution into the cache.
func (ldc *cache) Load(cfg *config.Config, name string) error {
	log.Debug().Str("ldname", name).Msg("loading LetterDistribution into cache")

	ld, err := NamedLetterDistribution(cfg, name)
	if err != nil {
		return err
	}
	ldc.Lock()
	defer ldc.Unlock()
	ldc.letterDistributions[name] = ld
	return nil
}

// Get gets the letter distribution from the cache, loading it in if missing.
func (ldc *cache) Get(cfg *config.Config, name string) (*LetterDistribution, error) {
	var ok bool
	var ld *LetterDistribution
	if ld, ok = ldc.letterDistributions[name]; !ok {
		err := ldc.Load(cfg, name)
		if err != nil {
			return nil, err
		}
		return ldc.letterDistributions[name], nil
	}
	log.Debug().Str("ldname", name).Msg("getting LetterDistribution from cache")
	return ld, nil
}
