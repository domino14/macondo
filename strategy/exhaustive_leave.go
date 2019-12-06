package strategy

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/alecthomas/mph"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
)

func float64FromBytes(bytes []byte) float64 {
	bits := binary.BigEndian.Uint64(bytes)
	float := math.Float64frombits(bits)
	return float
}

func float64ToByte(f float64) []byte {
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.BigEndian, f)
	if err != nil {
		fmt.Println("binary.Write failed:", err)
	}
	return buf.Bytes()
}

func moveBlanksToEnd(letters string) string {
	blankCt := strings.Count(letters, "?")
	if strings.Contains(letters, "?") {
		letters = strings.ReplaceAll(letters, "?", "")
		letters += strings.Repeat("?", blankCt)
	}

	return letters
}

func (els *ExhaustiveLeaveStrategy) Init(lexiconName string, alph *alphabet.Alphabet,
	leavefile string) error {

	hb := mph.Builder()

	file, err := os.Open(filepath.Join(DataDir, lexiconName, leavefile))
	if err != nil {
		return err
	}

	defer file.Close()

	r := csv.NewReader(file)
	records, err := r.ReadAll()
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	for _, record := range records {
		letters := moveBlanksToEnd(record[0])

		// These bytes can be put in hash table right away.
		mw, err := alphabet.ToMachineWord(letters, alph)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		leaveVal, err := strconv.ParseFloat(record[1], 32)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		hb.Add(mw.Bytes(), float64ToByte(leaveVal))
	}

	els.leaveValues, err = hb.Build()
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}
	log.Debug().Msgf("Finished building MPH for leave file %v", leavefile)
	log.Debug().Msgf("Size of MPH: %v", els.leaveValues.Len())
	return nil
}

func NewExhaustiveLeaveStrategy(bag *alphabet.Bag, lexiconName string,
	alph *alphabet.Alphabet, leaveFilename string) *ExhaustiveLeaveStrategy {

	strategy := &ExhaustiveLeaveStrategy{bag: bag}

	err := strategy.Init(lexiconName, alph, leaveFilename)
	if err != nil {
		log.Printf("[ERROR] Initializing strategy: %v", err)
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
		otherAdjustments += endgameAdjustment(play, oppRack, els.bag)
	} else {
		// the leave doesn't matter if the bag is empty
		leaveAdjustment = els.lookup(leave)
	}

	// also need a pre-endgame adjustment that biases towards leaving
	// one in the bag, etc.
	return float64(score) + leaveAdjustment + otherAdjustments
}

func (els ExhaustiveLeaveStrategy) lookup(leave alphabet.MachineWord) float64 {
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
		return float64FromBytes(val)
	}
	// Only will happen if we have a pass. Passes are very rare and
	// we should ignore this a bit since there will be a negative
	// adjustment already from the fact that we're scoring 0.
	return float64(0)
}
