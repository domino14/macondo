package equity

import (
	"errors"
	"path/filepath"
	"strings"

	"github.com/domino14/macondo/config"
)

func strategyParamsPath(cfg map[string]any) string {
	return filepath.Join(cfg[config.ConfigDataPath].(string), "strategy")
}

func LeaveCacheLoadFunc(cfg map[string]any, key string) (interface{}, error) {
	// Key looks like leavefile:lexicon:filename
	fields := strings.Split(key, ":")
	if fields[0] != "leavefile" {
		return nil, errors.New("leavecacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadKLV(strategyParamsPath(cfg), fields[2], fields[1])
}

func PEGCacheLoadFunc(cfg map[string]any, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "pegfile" {
		return nil, errors.New("pegcacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadPEGParams(strategyParamsPath(cfg), fields[2], fields[1])
}

func WinPCTLoadFunc(cfg map[string]any, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "winpctfile" {
		return nil, errors.New("winpctcacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadWinPCTParams(strategyParamsPath(cfg), fields[2], fields[1])
}
