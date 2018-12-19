package board

import (
	"log"
	"regexp"

	"github.com/domino14/macondo/alphabet"
)

var boardPlaintextRegex = regexp.MustCompile(`\|([[:print:]]+)\|`)

func (g *GameBoard) toDisplayText(alph *alphabet.Alphabet) string {
	var str string
	n := g.Dim()
	for i := 0; i < n; i++ {
		row := ""
		for j := 0; j < n; j++ {
			row = row + g.squares[i][j].DisplayString(alph) + " "
		}
		str = str + row + "\n"
	}
	return "\n" + str
}

func (g *GameBoard) SetFromPlaintext(qText string, alph *alphabet.Alphabet) {
	// Take a Quackle Plaintext Board and turn it into an internal structure.
	// (Another alternative later is to implement GCG)
	result := boardPlaintextRegex.FindAllStringSubmatch(qText, -1)
	if len(result) != 15 {
		panic("Wrongly implemented")
	}
	var err error
	for i := range result {
		// result[i][1] has the string
		for j, ch := range result[i][1] {
			if j%2 != 0 {
				continue
			}
			g.squares[i][j/2].letter, err = alph.Val(ch)
			if err != nil {
				// Ignore the error; we are passing in a space or another
				// board marker.
				g.squares[i][j/2].letter = alphabet.EmptySquareMarker
			} else {
				g.hasTiles = true
			}
		}
	}
}

func (b *GameBoard) SetRow(rowNum int8, letters string, alph *alphabet.Alphabet) {
	// Set the row in board to the passed in letters array.
	for idx := 0; idx < b.Dim(); idx++ {
		b.SetLetter(int(rowNum), idx, alphabet.EmptySquareMarker)
	}
	for idx, r := range letters {
		if r != ' ' {
			letter, err := alph.Val(r)
			if err != nil {
				log.Fatalf(err.Error())
			}
			b.SetLetter(int(rowNum), idx, letter)

		}
	}
}

// Two boards are equal if all the squares are equal. This includes anchors,
// letters, and cross-sets.
func (b *GameBoard) equals(b2 GameBoard) bool {
	if b.Dim() != b2.Dim() {
		return false
	}
	for row := 0; row < b.Dim(); row++ {
		for col := 0; col < b.Dim(); col++ {
			if !b.GetSquare(row, col).equals(b2.GetSquare(row, col)) {
				log.Printf("> Not equal, row %v col %v", row, col)
				return false
			}
		}
	}

	return true
}
