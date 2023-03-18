package board

import (
	"fmt"
	"log"
	"regexp"
	"strings"

	"github.com/domino14/macondo/tilemapping"
)

type TilesInPlay struct {
	OnBoard []tilemapping.MachineLetter
	Rack1   []tilemapping.MachineLetter
	Rack2   []tilemapping.MachineLetter
}

var boardPlaintextRegex = regexp.MustCompile(`\|(.+)\|`)
var userRackRegex = regexp.MustCompile(`(?U).+\s+([A-Z\?]*)\s+-?[0-9]+`)

func (g *GameBoard) sqDisplayStr(row, col int, alph *tilemapping.TileMapping) string {
	pos := g.getSqIdx(row, col)
	var bonusdisp string
	if g.bonuses[pos] != ' ' {
		bonusdisp = g.bonuses[pos].displayString()
	} else {
		bonusdisp = " "
	}
	if g.squares[pos] == 0 {
		return bonusdisp
	}
	return string(g.squares[pos].UserVisible(alph, true))
}

func (g *GameBoard) ToDisplayText(alph *tilemapping.TileMapping) string {
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
			row = row + g.sqDisplayStr(i, j, alph) + " "
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
// Note: this does not work with multi-character tiles!
func (g *GameBoard) setFromPlaintext(qText string,
	alph *tilemapping.TileMapping) *TilesInPlay {

	g.Clear()
	tilesInPlay := &TilesInPlay{}
	// Take a Quackle Plaintext Board and turn it into an internal structure.
	playedTiles := []tilemapping.MachineLetter(nil)
	result := boardPlaintextRegex.FindAllStringSubmatch(qText, -1)
	if len(result) != 15 {
		panic("Wrongly implemented")
	}
	g.tilesPlayed = 0
	var err error
	var letter tilemapping.MachineLetter
	for i := range result {
		// result[i][1] has the string
		j := -1
		for _, ch := range result[i][1] {
			j++
			if j%2 != 0 {
				continue
			}
			letter, err = alph.Val(string(ch))
			pos := i*15 + (j / 2)
			if err != nil {
				// Ignore the error; we are passing in a space or another
				// board marker.

				g.squares[pos] = 0
			} else {
				g.squares[pos] = letter
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
		rackTiles := []tilemapping.MachineLetter{}
		for _, ch := range rack {
			letter, err = alph.Val(string(ch))
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

// SetRow sets the row in board to the passed-in letters array.
// Callers should use a " " (space character) for an empty space
func (b *GameBoard) SetRow(rowNum int, letters string, alph *tilemapping.TileMapping) []tilemapping.MachineLetter {
	// Set the row in board to the passed in letters array.
	for idx := 0; idx < b.Dim(); idx++ {
		b.SetLetter(int(rowNum), idx, 0)
	}
	lettersPlayed := []tilemapping.MachineLetter{}

	mls, err := tilemapping.ToMachineLetters(letters, alph)
	if err != nil {
		log.Fatalf(err.Error())
	}

	for idx, ml := range mls {
		b.SetLetter(int(rowNum), idx, ml)
		b.tilesPlayed++
		lettersPlayed = append(lettersPlayed, ml)
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
			pos := g.getSqIdx(row, col)
			if g.squares[pos] != g2.squares[pos] ||
				g.bonuses[pos] != g2.bonuses[pos] ||
				g.hCrossScores[pos] != g2.hCrossScores[pos] ||
				g.vCrossScores[pos] != g2.vCrossScores[pos] ||
				g.hCrossSets[pos] != g2.hCrossSets[pos] ||
				g.vCrossSets[pos] != g2.vCrossSets[pos] ||
				g.hAnchors[pos] != g2.hAnchors[pos] ||
				g.vAnchors[pos] != g2.vAnchors[pos] {
				return false
			}
		}
	}

	return true
}
