package alphabet

// LetterDistribution encodes the tile distribution for the relevant game.
type LetterDistribution struct {
	Distribution map[rune]uint8
	PointValues  map[rune]uint8
	SortOrder    map[rune]int
	Vowels       []rune
	numLetters   int
}

func EnglishLetterDistribution() *LetterDistribution {
	dist := map[rune]uint8{
		'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
		'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
		'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
		'Y': 2, 'Z': 1, '?': 2,
	}
	ptValues := map[rune]uint8{
		'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
		'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
		'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
		'Y': 4, 'Z': 10, '?': 0,
	}
	return &LetterDistribution{dist, ptValues,
		makeSortMap("ABCDEFGHIJKLMNOPQRSTUVWXYZ?"),
		[]rune{'A', 'E', 'I', 'O', 'U'}, 100}
}

func SpanishLetterDistribution() *LetterDistribution {
	dist := map[rune]uint8{
		'1': 1, '2': 1, '3': 1, // 1: CH, 2: LL, 3: RR
		'A': 12, 'B': 2, 'C': 4, 'D': 5, 'E': 12, 'F': 1, 'G': 2, 'H': 2,
		'I': 6, 'J': 1, 'L': 4, 'M': 2, 'N': 5, 'Ñ': 1, 'O': 9, 'P': 2,
		'Q': 1, 'R': 5, 'S': 6, 'T': 4, 'U': 5, 'V': 1, 'X': 1, 'Y': 1,
		'Z': 1, '?': 2,
	}
	ptValues := map[rune]uint8{
		'1': 5, '2': 8, '3': 8, // 1: CH, 2: LL, 3: RR
		'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
		'I': 1, 'J': 8, 'L': 1, 'M': 3, 'N': 1, 'Ñ': 8, 'O': 1, 'P': 3,
		'Q': 5, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'X': 8, 'Y': 4,
		'Z': 10, '?': 0,
	}
	return &LetterDistribution{dist, ptValues,
		makeSortMap("ABC1DEFGHIJL2MNÑOPQR3STUVXYZ?"),
		[]rune{'A', 'E', 'I', 'O', 'U'}, 100}
}

func PolishLetterDistribution() *LetterDistribution {
	dist := map[rune]uint8{
		'A': 9, 'B': 2, 'C': 3, 'D': 3, 'E': 7, 'F': 1, 'G': 2, 'H': 2,
		'I': 8, 'J': 2, 'K': 3, 'L': 3, 'Ł': 2, 'M': 3, 'N': 5, 'O': 6,
		'P': 3, 'R': 4, 'S': 4, 'T': 3, 'U': 2, 'W': 4, 'Y': 4, 'Z': 5,
		'?': 2,
		'Ą': 1, 'Ę': 1, 'Ó': 1, 'Ś': 1, 'Ż': 1, 'Ć': 1, 'Ń': 1, 'Ź': 1,
	}
	ptValues := map[rune]uint8{
		'A': 1, 'B': 3, 'C': 2, 'D': 2, 'E': 1, 'F': 5, 'G': 3, 'H': 3,
		'I': 1, 'J': 3, 'K': 2, 'L': 2, 'Ł': 3, 'M': 2, 'N': 1, 'O': 1,
		'P': 2, 'R': 1, 'S': 1, 'T': 2, 'U': 3, 'W': 1, 'Y': 2, 'Z': 1,
		'?': 0,
		'Ą': 5, 'Ę': 5, 'Ó': 5, 'Ś': 5, 'Ż': 5, 'Ć': 6, 'Ń': 7, 'Ź': 9,
	}
	return &LetterDistribution{dist, ptValues,
		makeSortMap("AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ?"),
		[]rune{'A', 'Ą', 'E', 'Ę', 'I', 'O', 'Ó', 'U', 'Y'}, 100}
}

// MakeBag returns a bag of tiles.
func (ld *LetterDistribution) MakeBag(alph *Alphabet) *Bag {

	// Make an array of scores in alphabet order
	scores := make([]int, len(ld.Distribution))
	for rn, ptVal := range ld.PointValues {
		ml, err := alph.Val(rn)
		if err != nil {
			panic("Wrongly initialized")
		}
		if ml == BlankMachineLetter {
			ml = MachineLetter(len(ld.Distribution) - 1)
		}
		scores[ml] = int(ptVal)
	}
	b := NewBag(ld, len(ld.Distribution), alph, scores)
	b.Shuffle()

	return b
}

func makeSortMap(order string) map[rune]int {
	sortMap := make(map[rune]int)
	for idx, letter := range order {
		sortMap[letter] = idx
	}
	return sortMap
}
