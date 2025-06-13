package equity

import (
	"encoding/csv"
	"encoding/json"
	"io"
	"strconv"

	"github.com/domino14/macondo/dataloaders"
	"github.com/rs/zerolog/log"
)

const (
	PEGAdjustmentFilename = "preendgame.json"
	LeavesFilename        = "leaves.klv2"
)

func loadKLV(strategyPath, leavefile, lexiconName string) (*KLV, error) {
	file, err := dataloaders.StratFileForLexicon(strategyPath, leavefile, lexiconName)
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
	pegfile, err := dataloaders.StratFileForLexicon(strategyPath, filepath, lexiconName)
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
	winpctfile, err := dataloaders.StratFileForLexicon(strategyPath, filepath, lexiconName)
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
