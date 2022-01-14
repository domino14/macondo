package board

import (
	"fmt"
	"log"
	"regexp"
	"strings"

	"github.com/domino14/macondo/alphabet"
)

type TilesInPlay struct {
	OnBoard []alphabet.MachineLetter
	Rack1   []alphabet.MachineLetter
	Rack2   []alphabet.MachineLetter
}

var boardPlaintextRegex = regexp.MustCompile(`\|(.+)\|`)
var userRackRegex = regexp.MustCompile(`(?U).+\s+([A-Z\?]*)\s+-?[0-9]+`)

func (g *GameBoard) ToDisplayText(alph *alphabet.Alphabet) string {
	var str string
	n := g.Dim()
	row := "   "
	for i := 0; i < n; i++ {
		row = row + fmt.Sprintf("%c", 'A'+i) + " "
	}
	str = str + row + "\n"
	str = str + "   " + strings.Repeat("-", n*2) + "\n"
	for i := 0; i < n; i++ {
		row := fmt.Sprintf("%2d|", i+1)
		for j := 0; j < n; j++ {
			row = row + g.squares[i][j].DisplayString(alph) + " "
		}
		row = row + "|"
		str = str + row + "\n"
	}
	str = str + "   " + strings.Repeat("-", n*2) + "\n"
	return "\n" + str
}

// SetFromPlaintext sets the board from the given plaintext board.
// It returns a list of all played machine letters (tiles) so that the
// caller can reconcile the tile bag appropriately.
func (g *GameBoard) setFromPlaintext(qText string,
	alph *alphabet.Alphabet) *TilesInPlay {

	g.Clear()
	tilesInPlay := &TilesInPlay{}
	// Take a Quackle Plaintext Board and turn it into an internal structure.
	// (Another alternative later is to implement GCG)
	playedTiles := []alphabet.MachineLetter(nil)
	result := boardPlaintextRegex.FindAllStringSubmatch(qText, -1)
	if len(result) != 15 {
		panic("Wrongly implemented")
	}
	g.tilesPlayed = 0
	var err error
	var letter alphabet.MachineLetter
	for i := range result {
		// result[i][1] has the string
		j := -1
		for _, ch := range result[i][1] {
			j++
			if j%2 != 0 {
				continue
			}
			letter, err = alph.Val(ch)
			if err != nil {
				// Ignore the error; we are passing in a space or another
				// board marker.
				g.squares[i][j/2].letter = alphabet.EmptySquareMarker
			} else {
				g.squares[i][j/2].letter = letter
				g.tilesPlayed++
				playedTiles = append(playedTiles, letter)
			}
		}
	}
	userRacks := userRackRegex.FindAllStringSubmatch(qText, -1)
	for i := range userRacks {
		if i > 1 { // only the first two lines that match
			break
		}
		rack := userRacks[i][1]
		rackTiles := []alphabet.MachineLetter{}
		for _, ch := range rack {
			letter, err = alph.Val(ch)
			if err != nil {
				panic(err)
			}
			rackTiles = append(rackTiles, letter)
		}

		if i == 0 {
			tilesInPlay.Rack1 = rackTiles
		} else if i == 1 {
			tilesInPlay.Rack2 = rackTiles
		}
	}
	tilesInPlay.OnBoard = playedTiles
	return tilesInPlay
}

func (b *GameBoard) SetRow(rowNum int, letters string, alph *alphabet.Alphabet) []alphabet.MachineLetter {
	// Set the row in board to the passed in letters array.
	for idx := 0; idx < b.Dim(); idx++ {
		b.SetLetter(int(rowNum), idx, alphabet.EmptySquareMarker)
	}
	lettersPlayed := []alphabet.MachineLetter{}
	for idx, r := range letters {
		if r != ' ' {
			letter, err := alph.Val(r)
			if err != nil {
				log.Fatalf(err.Error())
			}
			b.SetLetter(int(rowNum), idx, letter)
			b.tilesPlayed++
			lettersPlayed = append(lettersPlayed, letter)
		}
	}
	return lettersPlayed
}

// Equals checks the boards for equality. Two boards are equal if all
// the squares are equal. This includes anchors, letters, and cross-sets.
func (g *GameBoard) Equals(g2 *GameBoard) bool {
	if g.Dim() != g2.Dim() {
		log.Printf("Dims don't match: %v %v", g.Dim(), g2.Dim())
		return false
	}
	if g.tilesPlayed != g2.tilesPlayed {
		log.Printf("Tiles played don't match: %v %v", g.tilesPlayed, g2.tilesPlayed)
		return false
	}
	for row := 0; row < g.Dim(); row++ {
		for col := 0; col < g.Dim(); col++ {
			if !g.GetSquare(row, col).equals(g2.GetSquare(row, col)) {
				log.Printf("> Not equal, row %v col %v", row, col)
				return false
			}
		}
	}

	return true
}
