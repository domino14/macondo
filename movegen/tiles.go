package movegen

// import (
// 	"math/rand"

// 	"github.com/domino14/macondo/alphabet"
// )

// // Bag - Tile distributions and scores. Hard-code for now, and load from
// // some sort of config file later.
// type Bag struct {
// 	numUniqueTiles uint8
// 	// distributions is a slice of ordered tile distributions, in machine
// 	// letter order. (0 to max size; max size corresponds to the blank)
// 	distributions []uint8
// 	// scores is a slice of ordered scores. The order is the machine letter
// 	// order.
// 	scores []int
// 	bag    []alphabet.MachineLetter
// }

// // InitializeTilesInBag initializes the actual bag of tiles with the correct
// // distribution and shuffles the bag.
// func (b *Bag) InitializeTilesInBag() {
// 	// Append all the tiles in the bag.
// 	var i, j uint8
// 	for i = 0; i < b.numUniqueTiles-1; i++ {
// 		for j = 0; j < b.distributions[i]; j++ {
// 			b.bag = append(b.bag, alphabet.MachineLetter(i))
// 		}
// 	}
// 	// the blank
// 	for j = 0; j < b.distributions[b.numUniqueTiles-1]; j++ {
// 		b.bag = append(b.bag, alphabet.BlankMachineLetter)
// 	}
// 	b.Shuffle()
// }

// func (b *Bag) score(ml alphabet.MachineLetter) int {
// 	if ml >= alphabet.BlankOffset {
// 		return b.scores[b.numUniqueTiles-1]
// 	}
// 	return b.scores[ml]
// }

// // Shuffle shuffles the bag.
// func (b *Bag) Shuffle() {
// 	rand.Shuffle(len(b.bag), func(i, j int) {
// 		b.bag[i], b.bag[j] = b.bag[j], b.bag[i]
// 	})
// }

// // Draw draws n tiles from the bag.
// func (b *Bag) Draw(n int) []alphabet.MachineLetter {
// 	if n > len(b.bag) {
// 		n = len(b.bag)
// 	}
// 	drawn := make([]alphabet.MachineLetter, n)
// 	for i := 0; i < n; i++ {
// 		drawn[i] = b.bag[i]
// 	}
// 	b.bag = b.bag[n:]
// 	return drawn
// }

// // Exchange exchanges the junk in your rack with new tiles.
// // It does not check how many tiles are left in the bag!
// func (b *Bag) Exchange(letters []alphabet.MachineLetter) []alphabet.MachineLetter {
// 	newTiles := b.Draw(len(letters))
// 	// put exchanged tiles back into the bag and re-shuffle
// 	b.bag = append(b.bag, letters...)
// 	b.Shuffle()
// 	return newTiles
// }

// // Pool returns the contents of the bag. Careful what you do with this
// // variable!
// func (b *Bag) Pool() []alphabet.MachineLetter {
// 	return b.bag
// }
