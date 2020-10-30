package gaddag

import (
	"bytes"
	"errors"
	"path/filepath"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/cache"
	"github.com/domino14/macondo/config"
)

var CacheKeyPrefix = "gaddag:"

// CacheLoadFunc is the function that loads an object into the global cache.
func CacheLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	lexiconName := strings.TrimPrefix(key, CacheKeyPrefix)
	return LoadGaddag(filepath.Join(cfg.LexiconPath, "gaddag", lexiconName+".gaddag"))
}

// LoadGaddag loads a gaddag from a file and returns a *SimpleGaddag structure.
func LoadGaddag(filename string) (*SimpleGaddag, error) {
	log.Debug().Msgf("Loading %v ...", filename)
	file, err := cache.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return ScanGaddag(file)
}

// CacheReadFunc converts raw data when populating the global cache
func CacheReadFunc(data []byte) (interface{}, error) {
	stream := bytes.NewReader(data)
	return ScanGaddag(stream)
}

// Set loads a gaddag from bytes and populates the cache
func Set(name string, data []byte) error {
	key := CacheKeyPrefix + name
	return cache.Populate(key, data, CacheReadFunc)
}

// Get loads a named gaddag from the cache or from a file
func Get(cfg *config.Config, name string) (*SimpleGaddag, error) {
	key := CacheKeyPrefix + name
	obj, err := cache.Load(cfg, key, CacheLoadFunc)
	if err != nil {
		return nil, err
	}
	ret, ok := obj.(*SimpleGaddag)
	if !ok {
		return nil, errors.New("Could not read gaddag from file")
	}
	return ret, nil
}
