package equity

import (
	"encoding/csv"
	"encoding/json"
	"io"
	"path/filepath"
	"strconv"

	"github.com/domino14/word-golib/cache"
	"github.com/rs/zerolog/log"
)

const (
	PEGAdjustmentFilename = "preendgame.json"
	LeavesFilename        = "leaves.klv2"
)

func stratFileForLexicon(strategyDir string, filename string, lexiconName string) (io.ReadCloser, error) {
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

func loadKLV(strategyPath, leavefile, lexiconName string) (*KLV, error) {
	file, err := stratFileForLexicon(strategyPath, leavefile, lexiconName)
	if err != nil {
		return nil, err
	}
	var leaves *KLV
	defer file.Close()
	leaves, err = ReadKLV(file)
	if err != nil {
		return nil, err
	}
	log.Debug().Str("lexiconName", lexiconName).
		Int("leaves-size", len(leaves.leaveValues)).
		Msg("loaded-klv")
	return leaves, nil
}

func loadPEGParams(strategyPath, filepath, lexiconName string) ([]float64, error) {
	pegfile, err := stratFileForLexicon(strategyPath, filepath, lexiconName)
	if err != nil {
		return nil, err
	}
	defer pegfile.Close()

	bts, err := io.ReadAll(pegfile)
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

const MaxRepresentedWinSpread = 300

func loadWinPCTParams(strategyPath, filepath, lexiconName string) ([][]float32, error) {
	winpctfile, err := stratFileForLexicon(strategyPath, filepath, lexiconName)
	if err != nil {
		return nil, err
	}

	defer winpctfile.Close()
	r := csv.NewReader(winpctfile)
	idx := -1

	// from 300 to -300 in spread, including 0
	wpct := make([][]float32, MaxRepresentedWinSpread*2+1)
	for i := range wpct {
		wpct[i] = make([]float32, 94)
	}

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		idx++
		// The first row is the header.
		if idx == 0 {
			continue
		}
		for i := range record {
			// The very first column is the spread
			if i == 0 {
				continue
			}
			f, err := strconv.ParseFloat(record[i], 32)
			if err != nil {
				return nil, err
			}
			wpct[idx-1][i-1] = float32(f)
		}

	}
	return wpct, nil
}
