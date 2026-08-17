package rangefinder

import (
	"fmt"
	"strings"

	"github.com/domino14/word-golib/tilemapping"
)

// A read is not always about a tile. On a position where the opponent's play
// gave away that they kept consonants, no single letter moves much - the
// largest shift might be eight points of probability, a tenth of a tile - and
// yet "expect a consonant-heavy rack" is the most useful thing anyone could be
// told about the position. That finding lives in the sum over letters, not in
// any of them, so it has to be computed as its own thing.
//
// The other kind of read is the opposite: one tile, enormous, and the shape
// barely moved. Both are reported when present, and neither is invented when
// it isn't.

// CountPair is an expected number of tiles under the read, next to what a rack
// drawn at random from the same pool would hold.
type CountPair struct {
	Read   float64 `json:"read"`
	Chance float64 `json:"chance"`
}

// Diff is how far the read moved it, in tiles.
func (c CountPair) Diff() float64 { return c.Read - c.Chance }

// RackShape is what the read says about the makeup of the rack rather than
// about any particular tile.
type RackShape struct {
	RackLength int `json:"rack_length"`
	// Vowels, Consonants and Blanks are expected counts. Under either
	// hypothesis the three sum to RackLength, which is worth knowing when
	// reading them: they are three views of one distribution, not estimates
	// that might disagree.
	Vowels     CountPair `json:"vowels"`
	Consonants CountPair `json:"consonants"`
	Blanks     CountPair `json:"blanks"`
	// VowelCount[k] is the chance of holding exactly k vowels. A mean of 0.9
	// vowels is consistent with "usually one, sometimes none" and with "half
	// the time two, half the time none", and those are different reads.
	VowelCount []CountPair `json:"vowel_count"`
}

// ShapeNotable is how far the expected number of vowels has to move, in tiles,
// before the shape is worth reporting as a finding. Unlike a single tile, the
// shape sums 26 small shifts, so a much smaller number is meaningful here:
// a third of a tile of vowels is the difference between a rack that can play
// and one that is stuck.
const ShapeNotable = 0.25

// Notable reports whether the read moved the makeup of the rack enough to say
// so. It looks at vowels alone because consonants are their mirror: with the
// blank barely moving, one going up is the other going down.
func (s *RackShape) Notable() bool {
	return s != nil && abs(s.Vowels.Diff()) >= ShapeNotable
}

// rackShape computes the makeup of the rack under the read and under chance.
// Both come from the same two sources the per-tile figures do - the posterior
// for the read, the unseen pool for chance - so the two columns are always
// comparable.
func (r *RangeFinder) rackShape() *RackShape {
	if r.inference == nil || len(r.inference.InferredRacks) == 0 {
		return nil
	}
	rackLen := r.inference.RackLength
	if rackLen <= 0 {
		return nil
	}
	ld := r.origGame.Bag().LetterDistribution()

	// The read: walk the posterior, counting vowels in each rack against its
	// weight. Counting racks rather than tile shares is what makes the
	// distribution available - a share of slots can't say how many racks held
	// two.
	sumW := 0.0
	countW := make([]float64, rackLen+1)
	vowelW, blankW := 0.0, 0.0
	for _, ir := range r.inference.InferredRacks {
		sumW += ir.Weight
		vowels := 0
		for _, ml := range ir.Leave {
			switch {
			case ml == 0:
				blankW += ir.Weight
			case ml.IsVowel(ld):
				vowels++
			}
		}
		vowelW += float64(vowels) * ir.Weight
		if vowels <= rackLen {
			countW[vowels] += ir.Weight
		}
	}
	if sumW == 0 {
		return nil
	}

	// Chance: how many of each kind are left, and what a random draw gives.
	unseen, unseenVowels, unseenBlanks := 0, 0, 0
	for i := range r.inferenceBagMap {
		n := int(r.inferenceBagMap[i])
		unseen += n
		switch {
		case i == 0:
			unseenBlanks += n
		case tilemapping.MachineLetter(i).IsVowel(ld):
			unseenVowels += n
		}
	}
	if unseen == 0 {
		return nil
	}

	s := &RackShape{RackLength: rackLen}
	s.Vowels = CountPair{
		Read:   vowelW / sumW,
		Chance: float64(rackLen) * float64(unseenVowels) / float64(unseen),
	}
	s.Blanks = CountPair{
		Read:   blankW / sumW,
		Chance: float64(rackLen) * float64(unseenBlanks) / float64(unseen),
	}
	// Everything that is neither, by subtraction, so the three always add up
	// to a whole rack.
	s.Consonants = CountPair{
		Read:   float64(rackLen) - s.Vowels.Read - s.Blanks.Read,
		Chance: float64(rackLen) - s.Vowels.Chance - s.Blanks.Chance,
	}

	s.VowelCount = make([]CountPair, rackLen+1)
	for k := range s.VowelCount {
		s.VowelCount[k] = CountPair{
			Read:   100 * countW[k] / sumW,
			Chance: 100 * hypergeometric(unseen, unseenVowels, rackLen, k),
		}
	}
	return s
}

// hypergeometric is the chance of drawing exactly k of the wanted kind when
// taking draws tiles from a pool of size unseen holding wanted of them. Written
// as a running product so a 100-tile pool needs no factorials.
func hypergeometric(unseen, wanted, draws, k int) float64 {
	other := unseen - wanted
	if k < 0 || k > draws || k > wanted || draws-k > other {
		return 0
	}
	// C(wanted,k) * C(other,draws-k) / C(unseen,draws), one factor at a time.
	p := 1.0
	for i := range k {
		p *= float64(wanted-i) / float64(i+1)
	}
	for i := range draws - k {
		p *= float64(other-i) / float64(i+1)
	}
	for i := range draws {
		p *= float64(i+1) / float64(unseen-i)
	}
	return p
}

// ShapeSummary says what the read concluded about the makeup of the rack, or
// nothing at all when it concluded nothing worth saying.
func ShapeSummary(s *RackShape) string {
	if s == nil {
		return ""
	}
	var ss strings.Builder
	ss.WriteString("The read says they are holding:\n\n")
	for _, row := range []struct {
		name string
		pair CountPair
	}{
		{"consonants", s.Consonants},
		{"vowels", s.Vowels},
		{"blanks", s.Blanks},
	} {
		flag := ""
		if abs(row.pair.Diff()) >= ShapeNotable {
			flag = "   <-- the read"
		}
		fmt.Fprintf(&ss, "  %.2f %-11s where a random rack holds %.2f   %+.2f%s\n",
			row.pair.Read, row.name, row.pair.Chance, row.pair.Diff(), flag)
	}

	// The mean says which way; the distribution says how reliably. A read
	// worth acting on is one that moves the whole shape of it.
	if s.Notable() && len(s.VowelCount) > 0 {
		ss.WriteString("\n  vowels held    read   chance\n")
		for k, pair := range s.VowelCount {
			if pair.Read < 0.5 && pair.Chance < 0.5 {
				continue
			}
			fmt.Fprintf(&ss, "  %d              %5.1f%%  %5.1f%%\n", k, pair.Read, pair.Chance)
		}
	}
	return ss.String()
}
