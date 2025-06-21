package dataloaders

import (
	"io"
	"path/filepath"
	"strings"

	wglconfig "github.com/domino14/word-golib/config"

	"github.com/domino14/word-golib/cache"
	"github.com/rs/zerolog/log"
)

func StrategyParamsPath(cfg *wglconfig.Config) string {
	return filepath.Join(cfg.DataPath, "strategy")
}

func defaultForLexicon(lexiconName string) string {
	// If there doesn't exist a specific folder with the name of the
	// lexicon, we'll call this function.
	if strings.HasPrefix(lexiconName, "CSW") {
		return "CSW24"
	} else if strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") ||
		strings.HasPrefix(lexiconName, "NSWL") {

		return "NWL23"
	} else if strings.HasPrefix(lexiconName, "ECWL") || // obsolete name for CEL
		strings.HasPrefix(lexiconName, "CEL") { // common english words
		return "ECWL"
	} else if strings.HasPrefix(lexiconName, "RD") {
		return "RD29"
	} else if strings.HasPrefix(lexiconName, "NSF") {
		return "NSF23"
	} else if strings.HasPrefix(lexiconName, "FRA") {
		return "FRA24"
	} else if strings.HasPrefix(lexiconName, "DISC") {
		return "DISC2"
	} else if strings.HasPrefix(lexiconName, "OSPS") {
		return "OSPS50"
	} else if strings.HasPrefix(lexiconName, "FILE") {
		return "FILE2017"
	}
	return ""
}

func StratFileForLexicon(strategyDir string, filename string, lexiconName string) (io.ReadCloser, error) {
	file, _, err := cache.Open(filepath.Join(strategyDir, lexiconName, filename))
	if err != nil {
		defdir := defaultForLexicon(lexiconName)
		file, _, err = cache.Open(filepath.Join(strategyDir, defdir, filename))
		if err != nil {
			// last-ditch effort. Try "default" path.
			defdir = "default"
			file, _, err = cache.Open(filepath.Join(strategyDir, defdir, filename))
			if err != nil {
				return nil, err
			}
		}
		log.Debug().Str("strat-file", filename).Str("dir", defdir).Msgf(
			"no lexicon-specific strategy")
	}
	return file, nil
}
