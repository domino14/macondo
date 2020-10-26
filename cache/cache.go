package cache

import (
	"sync"

	"github.com/domino14/macondo/config"
	"github.com/rs/zerolog/log"
)

// The cache is a package used for generic large objects that we want to cache,
// especially if we are trying to use Macondo as part of the backend for a
// server. For example, we want to cache gaddags, strategy files, alphabets, etc.

type cache struct {
	sync.Mutex
	objects map[string]interface{}
}

type loadFunc func(cfg *config.Config, key string) (interface{}, error)

// GlobalObjectCache is our global object cache, of course.
var GlobalObjectCache *cache

func (c *cache) load(cfg *config.Config, key string, loadFunc loadFunc) error {
	log.Debug().Str("key", key).Msg("loading into cache")

	obj, err := loadFunc(cfg, key)
	if err != nil {
		return err
	}
	c.objects[key] = obj

	return nil
}

func (c *cache) get(cfg *config.Config, key string, loadFunc loadFunc) (interface{}, error) {

	var ok bool
	var obj interface{}
	c.Lock()
	defer c.Unlock()
	if obj, ok = c.objects[key]; !ok {
		err := c.load(cfg, key, loadFunc)
		if err != nil {
			return nil, err
		}
		return c.objects[key], nil
	}
	log.Debug().Str("key", key).Msg("getting obj from cache")

	return obj, nil
}

func CreateGlobalObjectCache() {
	GlobalObjectCache = &cache{objects: make(map[string]interface{})}
}

func Load(cfg *config.Config, name string, loadFunc loadFunc) (interface{}, error) {
	if GlobalObjectCache == nil {
		CreateGlobalObjectCache()
	}
	return GlobalObjectCache.get(cfg, name, loadFunc)
}
