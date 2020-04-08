package strategy

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
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
	LeaveFilename = "leaves.idx.gz"
)

// ExhaustiveLeaveStrategy should apply an equity calculation for all leaves
// exhaustively.
type ExhaustiveLeaveStrategy struct {
	leaveValues *mph.CHD
	bag         *alphabet.Bag
}

func float32FromBytes(bytes []byte) float32 {
	bits := binary.BigEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

func (els *ExhaustiveLeaveStrategy) Init(lexiconName string, alph *alphabet.Alphabet,
	strategyDir string) error {

	file, err := os.Open(filepath.Join(strategyDir, lexiconName, LeaveFilename))
	if err != nil {
		return err
	}
	defer file.Close()
	var gz *gzip.Reader
	if strings.HasSuffix(LeaveFilename, ".gz") {
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
	return nil
}

func NewExhaustiveLeaveStrategy(bag *alphabet.Bag, lexiconName string,
	alph *alphabet.Alphabet, strategyDir string) *ExhaustiveLeaveStrategy {

	strategy := &ExhaustiveLeaveStrategy{bag: bag}

	err := strategy.Init(lexiconName, alph, strategyDir)
	if err != nil {
		log.Error().Err(err).Msg("initializing strategy")
		return nil
	}
	return strategy
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
	if bag.TilesRemaining() == 0 {
		otherAdjustments += endgameAdjustment(play, oppRack, els.bag.LetterDistribution())
	} else {
		// the leave doesn't matter if the bag is empty
		leaveAdjustment = els.LeaveValue(leave)
	}

	// also need a pre-endgame adjustment that biases towards leaving
	// one in the bag, etc.
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
