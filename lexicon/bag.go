package lexicon

import "fmt"

type Bag struct {
	tiles []rune
	idx   int
}

func (b *Bag) Draw() (rune, error) {
	if b.idx == len(b.tiles) {
		return 0, fmt.Errorf("Bag is empty!")
	}
	b.idx += 1
	return b.tiles[b.idx-1], nil
}
