package alphabet

import (
	"strings"

	"github.com/domino14/macondo/config"
)

func CacheLoadFunc(cfg *config.Config, key string) (interface{}, error) {
	dist := strings.TrimPrefix(key, "letterdist:")
	return NamedLetterDistribution(cfg, dist)
}
