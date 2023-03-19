package equity

import (
	"errors"
	"strings"

	"github.com/domino14/macondo/config"
)

func LeaveCacheLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	// Key looks like leavefile:lexicon:filename
	fields := strings.Split(key, ":")
	if fields[0] != "leavefile" {
		return nil, errors.New("leavecacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadKLV(cfg.StrategyParamsPath, fields[2], fields[1])
}

func PEGCacheLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "pegfile" {
		return nil, errors.New("pegcacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadPEGParams(cfg.StrategyParamsPath, fields[2], fields[1])
}

func WinPCTLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	fields := strings.Split(key, ":")
	if fields[0] != "winpctfile" {
		return nil, errors.New("winpctcacheloadfunc - bad cache key: " + key)
	}
	if len(fields) != 3 {
		return nil, errors.New("cache key missing fields")
	}
	return loadWinPCTParams(cfg.StrategyParamsPath, fields[2], fields[1])
}
