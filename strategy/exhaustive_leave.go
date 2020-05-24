package strategy

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/alecthomas/mph"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

const (
	LeaveFilename         = "leaves.idx.gz"
	PEGAdjustmentFilename = "preendgame.json"
)

// ExhaustiveLeaveStrategy should apply an equity calculation for all leaves
// exhaustively.
type ExhaustiveLeaveStrategy struct {
	leaveValues                *mph.CHD
	preEndgameAdjustmentValues []float64
}

func float32FromBytes(bytes []byte) float32 {
	bits := binary.BigEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

func defaultForLexicon(lexiconName string) string {
	if strings.HasPrefix(lexiconName, "CSW") ||
		strings.HasPrefix(lexiconName, "TWL") ||
		strings.HasPrefix(lexiconName, "NWL") {

		return "default_english"
	}
	return ""
}

func stratFileForLexicon(strategyDir string, filename string, lexiconName string) (*os.File, error) {
	file, err := os.Open(filepath.Join(strategyDir, lexiconName, filename))
	if err != nil {
		defdir := defaultForLexicon(lexiconName)
		file, err = os.Open(filepath.Join(strategyDir, defdir, filename))
		if err != nil {
			return nil, err
		}
		log.Debug().Str("strat-file", filename).Str("dir", defdir).Msgf(
			"no lexicon-specific strategy")
	}
	return file, nil
}

func (els *ExhaustiveLeaveStrategy) Init(lexiconName string, alph *alphabet.Alphabet,
	strategyDir, leavefile string) error {

	if leavefile == "" {
		leavefile = LeaveFilename
	}

	file, err := stratFileForLexicon(strategyDir, leavefile, lexiconName)
	if err != nil {
		return err
	}

	defer file.Close()
	var gz *gzip.Reader
	if strings.HasSuffix(leavefile, ".gz") {
		gz, err = gzip.NewReader(file)
		defer gz.Close()
	}
	if gz != nil {
		log.Debug().Msg("reading from compressed file")
		els.leaveValues, err = mph.Read(gz)
	} else {
		els.leaveValues, err = mph.Read(file)
	}
	if err != nil {
		return err
	}

	log.Debug().Msgf("Size of MPH: %v", els.leaveValues.Len())

	err = els.SetPreendgameStrategy(strategyDir, PEGAdjustmentFilename, lexiconName)
	if err != nil {
		// Don't return this error. Just log it.
		log.Err(err).Msg("No pre-endgame equity adjustments will be used.")
	}
	return nil
}

func (els *ExhaustiveLeaveStrategy) SetPreendgameStrategy(strategyDir, filepath, lexiconName string) error {
	if filepath == "" {
		filepath = PEGAdjustmentFilename
	}
	pegfile, err := stratFileForLexicon(strategyDir, filepath, lexiconName)
	if err != nil {
		return err
	}
	defer pegfile.Close()

	bts, err := ioutil.ReadAll(pegfile)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bts, &els.preEndgameAdjustmentValues)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Size of pre-endgame adjustment array: %v", len(els.preEndgameAdjustmentValues))

	return nil
}

func NewExhaustiveLeaveStrategy(lexiconName string,
	alph *alphabet.Alphabet, strategyDir, leavefile string) (*ExhaustiveLeaveStrategy, error) {

	strategy := &ExhaustiveLeaveStrategy{}

	err := strategy.Init(lexiconName, alph, strategyDir, leavefile)
	if err != nil {
		return nil, err
	}
	return strategy, nil
}

func (els ExhaustiveLeaveStrategy) Equity(play *move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) float64 {

	leave := play.Leave()
	score := play.Score()

	leaveAdjustment := 0.0
	otherAdjustments := 0.0

	// Use global placement and endgame adjustments; this is only when
	// not overriding this with an endgame player.
	if board.IsEmpty() {
		otherAdjustments += placementAdjustment(play)
	}

	if bag.TilesRemaining() > 0 {
		leaveAdjustment = els.LeaveValue(leave)
		bagPlusSeven := bag.TilesRemaining() - play.TilesPlayed() + 7
		if bagPlusSeven < len(els.preEndgameAdjustmentValues) {
			preEndgameAdjustment := els.preEndgameAdjustmentValues[bagPlusSeven]
			// log.Debug().Float64("peg-adjust", preEndgameAdjustment).Int("bagPlusSeven", bagPlusSeven).Msg("equity calc")
			otherAdjustments += preEndgameAdjustment
		}
	} else {
		// The bag is empty.
		otherAdjustments += endgameAdjustment(play, oppRack, bag.LetterDistribution())
	}

	return float64(score) + leaveAdjustment + otherAdjustments
}

func (els ExhaustiveLeaveStrategy) LeaveValue(leave alphabet.MachineWord) float64 {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Recovered from panic; leave was %v\n", leave.UserVisible(alphabet.EnglishAlphabet()))
			// Panic anyway; the recover was just to figure out which leave did it.
			panic("panicking anyway")
		}
	}()
	if len(leave) == 0 {
		return 0
	}
	if len(leave) > 1 {
		sort.Slice(leave, func(i, j int) bool {
			return leave[i] < leave[j]
		})
	}
	if len(leave) <= 6 {
		// log.Debug().Msgf("Need to look up leave for %v", leave.UserVisible(alphabet.EnglishAlphabet()))
		val := els.leaveValues.Get(leave.Bytes())
		// log.Debug().Msgf("Value was %v", val)
		return float64(float32FromBytes(val))
	}
	// Only will happen if we have a pass. Passes are very rare and
	// we should ignore this a bit since there will be a negative
	// adjustment already from the fact that we're scoring 0.
	return float64(0)
}
