package rangefinder

import (
	"fmt"

	"github.com/domino14/macondo/alphabet"
)

// analyze the rack ranges
func Analyze(inferences [][]alphabet.MachineLetter, alph *alphabet.Alphabet,
	bagmap []uint8) string {
	// user visible

	totalCt := 0
	mlcts := map[alphabet.MachineLetter]int{}
	for _, inf := range inferences {
		for _, ml := range inf {
			mlcts[ml]++
			totalCt++
		}
	}
	inbag := uint8(0)
	for i := range bagmap {
		inbag += bagmap[i]
	}

	// bagcts is an array of ints from 0 to maxAlphabetLength, with the
	// values being the counts in the bag.

	// if we have 3 Es in bagcts for example we would expect 3/inbag to be
	// the ratio of Es in mlcts / totalCt.

	fmt.Println("Letter\tFound\tExpected\tInBag")
	for i := 0; i < int(alph.NumLetters()); i++ {
		fmt.Printf("%c\t", alph.Letter(alphabet.MachineLetter(i)))
		fmt.Printf("%5.3f\t", float64(mlcts[alphabet.MachineLetter(i)])/float64(totalCt))
		fmt.Printf("%5.3f\t", float64(bagmap[i])/float64(inbag))
		fmt.Printf("%d\n", bagmap[i])
	}
	fmt.Printf("?\t")
	fmt.Printf("%5.3f\t", float64(mlcts[alphabet.BlankMachineLetter])/float64(totalCt))
	fmt.Printf("%5.3f\t", float64(bagmap[alphabet.BlankMachineLetter])/float64(inbag))
	fmt.Printf("%d\n", bagmap[alphabet.BlankMachineLetter])

	fmt.Println()
	return ""
}
