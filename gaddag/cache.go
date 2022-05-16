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

const (
	gaddagCacheKeyPrefix = "gaddag:"
	dawgCacheKeyPrefix   = "dawg:"
)

// CacheLoadFunc is the function that loads an object into the global cache.
func CacheLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	var prefix string
	var t GenericDawgType
	if strings.HasPrefix(key, gaddagCacheKeyPrefix) {
		prefix = gaddagCacheKeyPrefix
		t = TypeGaddag
	} else if strings.HasPrefix(key, dawgCacheKeyPrefix) {
		prefix = dawgCacheKeyPrefix
		t = TypeDawg
	}
	lexiconName := strings.TrimPrefix(key, prefix)
	if t == TypeGaddag {
		return LoadGaddag(filepath.Join(cfg.LexiconPath, "gaddag", lexiconName+".gaddag"))
	} else if t == TypeDawg {
		return LoadDawg(filepath.Join(cfg.LexiconPath, "dawg", lexiconName+".dawg"))
	}
	return nil, errors.New("bad genericdawg type")
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
func gaddagCacheReadFunc(data []byte) (interface{}, error) {
	stream := bytes.NewReader(data)
	return ScanGaddag(stream)
}

func dawgCacheReadFunc(data []byte) (interface{}, error) {
	stream := bytes.NewReader(data)
	return ReadDawg(stream)
}

// Set loads a gaddag from bytes and populates the cache
func Set(name string, data []byte, t GenericDawgType) error {
	prefix := gaddagCacheKeyPrefix
	readFunc := gaddagCacheReadFunc
	if t == TypeDawg {
		prefix = dawgCacheKeyPrefix
		readFunc = dawgCacheReadFunc
	}
	key := prefix + name
	return cache.Populate(key, data, readFunc)
}

// Get loads a named gaddag from the cache or from a file
func Get(cfg *config.Config, name string) (*SimpleGaddag, error) {

	key := gaddagCacheKeyPrefix + name
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

// GetDawg loads a named dawg from the cache or from a file
func GetDawg(cfg *config.Config, name string) (*SimpleDawg, error) {
	key := dawgCacheKeyPrefix + name
	obj, err := cache.Load(cfg, key, CacheLoadFunc)
	if err != nil {
		return nil, err
	}
	ret, ok := obj.(*SimpleDawg)
	if !ok {
		return nil, errors.New("Could not read dawg from file")
	}
	return ret, nil
}
