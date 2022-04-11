package strategy

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"path/filepath"

	"github.com/domino14/macondo/cache"
	"github.com/rs/zerolog/log"
)

const (
	LeaveFilename         = "leaves.olv"
	PEGAdjustmentFilename = "preendgame.json"
)

func stratFileForLexicon(strategyDir string, filename string, lexiconName string) (io.ReadCloser, error) {
	file, err := cache.Open(filepath.Join(strategyDir, lexiconName, filename))
	if err != nil {
		defdir := defaultForLexicon(lexiconName)
		file, err = cache.Open(filepath.Join(strategyDir, defdir, filename))
		if err != nil {
			return nil, err
		}
		log.Info().Str("strat-file", filename).Str("dir", defdir).Msgf(
			"no lexicon-specific strategy")
	}
	return file, nil
}

// Load the exhaustive-leave minimal perfect hash.
func loadExhaustiveMPH(strategyPath, leavefile, lexiconName string) (*OldLeaves, error) {
	// XXX: This function doesn't take into account the different letter distributions
	// For now it doesn't matter but it will in the future when we have variants.

	file, err := stratFileForLexicon(strategyPath, leavefile, lexiconName)
	if err != nil {
		return nil, err
	}
	var leaves *OldLeaves

	defer file.Close()
	leaves, err = ReadOldLeaves(file)
	if err != nil {
		return nil, err
	}
	log.Debug().Str("lexiconName", lexiconName).
		Int("mph-size", leaves.Len()).
		Msg("loaded-mph")
	return leaves, nil
}

func loadPEGParams(strategyPath, filepath, lexiconName string) ([]float64, error) {
	pegfile, err := stratFileForLexicon(strategyPath, filepath, lexiconName)
	if err != nil {
		return nil, err
	}
	defer pegfile.Close()

	bts, err := ioutil.ReadAll(pegfile)
	if err != nil {
		return nil, err
	}

	var adjustmentVals []float64

	err = json.Unmarshal(bts, &adjustmentVals)
	if err != nil {
		return nil, err
	}
	log.Debug().Msgf("Size of pre-endgame adjustment array: %v", len(adjustmentVals))
	return adjustmentVals, nil
}
