// This file implements the "dynamic programming" solver, the somewhat
// inaccurate V1 of the endgame solver.
package endgame

import (
	"log"
	"sort"
	"strconv"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

const VeryLowEvaluation = -100000
const (
	OurPlayTable = iota
	TheirPlayTable
)

type playTableEntry struct {
	move  *move.Move
	value int
}

type playTable map[string]map[int]*playTableEntry

func (pt *playTable) addEntry(rackRepr string, outIn int, value int,
	move *move.Move) *playTableEntry {

	t := *pt

	outIns, ok := t[rackRepr]
	if !ok {
		t[rackRepr] = make(map[int]*playTableEntry)
		outIns = t[rackRepr]
	}
	outIns[outIn] = &playTableEntry{
		move:  move,
		value: value,
	}
	*pt = t
	return outIns[outIn]
}

func (pt playTable) get(rackRepr string, outIn int) (*playTableEntry, bool) {
	outIns, ok := pt[rackRepr]
	if !ok {
		return nil, false
	}
	entry, ok := outIns[outIn]
	if !ok {
		return nil, false
	}
	return entry, true
}

func (pt playTable) Printable(alph *alphabet.Alphabet) string {
	log.Printf("table has %v items", len(pt))
	str := "rack\tN\tvalue\tmove\n"
	for k, v := range pt {
		rack := alphabet.HashableToUserVisible(k, alph)
		for outIn, entry := range v {
			moveStr := "<nil>"
			if entry.move != nil {
				moveStr = entry.move.String()
			}

			str += rack + "\t" + strconv.Itoa(outIn) + "\t" + strconv.Itoa(
				entry.value) + "\t" + moveStr + "\n"
		}
	}
	return str
}

// DynamicProgSolver implements an endgame solver that uses dynamic programming
// to build up tables of endgame move valuations, similar to approach #1
// in the Sheppard paper.
type DynamicProgSolver struct {
	movegen *movegen.GordonGenerator
	board   *board.GameBoard
	gd      *gaddag.SimpleGaddag
	bag     *alphabet.Bag

	// The play tables are maps of maps.
	// The first level is the rack, the second level is number of plays we
	// are out in.
	playTableOurs playTable
	playTableOpp  playTable
}

func (s *DynamicProgSolver) Init(board *board.GameBoard, movegen *movegen.GordonGenerator,
	bag *alphabet.Bag, gd *gaddag.SimpleGaddag) {

	s.board = board
	s.movegen = movegen
	s.bag = bag
	s.gd = gd
}

// Solve generates the tables and returns the best calculated move for playerOnTurn.
func (s *DynamicProgSolver) Solve(playerOnTurn, opponent *alphabet.Rack) *move.Move {
	s.movegen.SetSortingParameter(movegen.SortByScore)
	defer s.movegen.SetSortingParameter(movegen.SortByEquity)

	s.movegen.GenAll(playerOnTurn)
	s.generatePlayTables(playerOnTurn, s.movegen.Plays(), OurPlayTable)

	s.movegen.GenAll(opponent)
	s.generatePlayTables(opponent, s.movegen.Plays(), TheirPlayTable)

	// Go down the list of play tables now to determine the best play.

	return nil
}

func (s *DynamicProgSolver) generatePlayTables(rack *alphabet.Rack,
	plays []*move.Move, whose int) {

	var pt *playTable

	if whose == OurPlayTable {
		s.playTableOurs = make(playTable)
		pt = &s.playTableOurs
	} else if whose == TheirPlayTable {
		s.playTableOpp = make(playTable)
		pt = &s.playTableOpp
	}

	// Add base case
	s.addBaseCase(*pt, rack)

	for n := 1; n <= 3; n++ {
		for _, play := range plays {
			s.addPlayEvaluation(rack, plays, play, *pt, n)
		}
	}

}

func (s *DynamicProgSolver) addPlayEvaluation(rack *alphabet.Rack, plays []*move.Move,
	play *move.Move, pt playTable, numTurns int) {
	// Add the evaluation for a single play.
	// "For every move that can be played out of each rack we compute
	// the score of that move plus the evaluation of the remaining tiles
	// assuming that N − 1 turns remain. The value of a rack in N turns
	// is the largest value of that function over all possible moves."

	// log.Println(numTurns, "Call addPlayEvaluation for", play, rack.TilesOn().UserVisible(
	// 	s.gd.GetAlphabet()))
	eval := s.rackEvaluation(play.Leave(), numTurns-1, plays, pt)
	// log.Println("The evaluation of the leave", play.Leave().UserVisible(
	// 	s.gd.GetAlphabet()), "with N", numTurns-1, "was", eval)
	if eval == nil {
		// Not a possible evaluation, I guess
		return
	}
	value := play.Score() + eval.value

	rackToAdd := rack.Hashable()

	entry, exists := pt.get(rackToAdd, numTurns)
	// log.Println("The value is then", value, "and should add the rack",
	// 	alphabet.HashableToUserVisible(rackToAdd, s.gd.GetAlphabet()), numTurns)
	if !exists || entry.value < value {
		pt.addEntry(rackToAdd, numTurns, value, play)
	}
}

func (s *DynamicProgSolver) rackEvaluation(rack alphabet.MachineWord, numTurns int,
	plays []*move.Move, pt playTable) *playTableEntry {

	// 0. See if the play table already contains an evaluation for this rack,
	//		return it if so.
	// 1. find which moves in `plays` can be played out of the given rack
	// 2. for each such move, figure out the score of the move + the evaluation
	//		of the leave with numTurns -1. This is an evaluation.
	// 3. take the maximum of all the evaluations. This is the rack evaluation;
	// 		store and return this.
	// log.Println("Trying to call rackEvaluation with",
	// 	rack.UserVisible(s.gd.GetAlphabet()), numTurns)

	if entry, ok := pt.get(rack.String(), numTurns); ok {
		return entry
	}
	// XXX why does uncommenting this give the wrong results?
	// if len(rack) == 0 && numTurns == 0 {
	// 	return nil
	// }

	if numTurns == 0 {
		value := -2 * rack.Score(s.bag)
		return pt.addEntry(rack.String(), 0, value, nil)
	}

	var maxPlay *move.Move
	maxValue := VeryLowEvaluation

	for _, play := range plays {
		// Which of these moves can be played out of the given rack?
		playable, leave := canBePlayedWith(play.Tiles(), rack)
		if !playable {
			continue
		}
		leaveValue := 0
		leaveEvaluation := s.rackEvaluation(leave, numTurns-1, plays, pt)
		if leaveEvaluation != nil {
			leaveValue = leaveEvaluation.value
		}
		value := play.Score() + leaveValue
		if value > maxValue {
			maxValue = value
			maxPlay = play
		}
	}
	if maxValue == VeryLowEvaluation {
		// Found no matching plays.
		return nil
	}
	return pt.addEntry(rack.String(), numTurns, maxValue, maxPlay)
}

func canBePlayedWith(playWord alphabet.MachineWord, rack alphabet.MachineWord) (
	bool, alphabet.MachineWord) {
	// Return true if the playWord can be played with the given rack. The rack
	// must be a superset of the played tiles. The second value that is returned
	// is the leave, if any.

	m := make(map[alphabet.MachineLetter]int)

	for _, letter := range rack {
		m[letter]++
	}

	for _, letter := range playWord {
		if letter == alphabet.PlayedThroughMarker {
			continue
		}

		// Note: this loop assumes that the word had to have been generated
		// with some superset of rack. So it won't try to use a blank if
		// `letter` isn't a blank.
		if m[letter] == 0 {
			if letter.IsBlanked() {
				if m[alphabet.BlankMachineLetter] == 0 {
					return false, nil
				}
				// Otherwise, the letter is a blank and we have at least one
				// blank to use.
				m[alphabet.BlankMachineLetter]--
			} else {
				// Otherwise, the letter was not found at all.
				return false, nil
			}
		}
		m[letter]--
	}
	// If we've gotten here, playWord is a subset of rack. Note the leave;
	// i.e. whatever is in the map.
	leave := make(alphabet.MachineWord, 0)
	for letter, c := range m {
		if c == 0 {
			continue
		}
		for i := 0; i < c; i++ {
			leave = append(leave, letter)
		}
	}
	sort.Slice(leave, func(a, b int) bool {
		return leave[a] < leave[b]
	})
	return true, leave
}

// addBaseCase adds the "out in 0" case, where the value is just negative
// twice the score on our rack.
func (s *DynamicProgSolver) addBaseCase(pt playTable, rack *alphabet.Rack) {
	score := rack.ScoreOn(s.bag)
	rackRepr := rack.Hashable()
	pt.addEntry(rackRepr, 0, -score*2, nil)
}
