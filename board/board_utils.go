package board

import (
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
			row = row + g.squares[i][j].DisplayString(alph)
		}
		str = str + row + "\n\n"
	}
	return str
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
