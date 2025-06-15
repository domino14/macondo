package equity

import (
	"errors"
	"strings"

	"github.com/domino14/macondo/dataloaders"
	wglconfig "github.com/domino14/word-golib/config"
)

func LeaveCacheLoadFunc(cfg *wglconfig.Config, key string) (interface{}, error) {
	// Key looks like leavefile:lexicon:filename
	fields := strings.Split(key, ":")
	if fields[0] != "leavefile" {
		return nil, errors.New("leavecacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadKLV(dataloaders.StrategyParamsPath(cfg), fields[2], fields[1])
}

func PEGCacheLoadFunc(cfg *wglconfig.Config, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "pegfile" {
		return nil, errors.New("pegcacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadPEGParams(dataloaders.StrategyParamsPath(cfg), fields[2], fields[1])
}

func WinPCTLoadFunc(cfg *wglconfig.Config, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "winpctfile" {
		return nil, errors.New("winpctcacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadWinPCTParams(dataloaders.StrategyParamsPath(cfg), fields[2], fields[1])
}
