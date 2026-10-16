package playability

import (
	"math"
	"math/rand/v2"
)

// The synthetic estimator re-deals a saved position so that the mover's rack
// holds enough of a target word's tiles, and weights the result by how likely
// that deal was. The deal is the mover's draw: d tiles, drawn uniformly from the
// pool the mover cannot see (everything not on the board and not in its own
// leave), added to the leave it kept.
//
// Tiles are grouped into categories: one per distinct letter of the target
// word, one for blanks, and one for everything else. A draw is then a count per
// category, and there are few enough of those to enumerate exactly.

// CondDraw is the set of draws, out of one pool, that put enough of a target
// word on the rack.
type CondDraw struct {
	letters []int // machine letter of each word-letter category
	pool    []int // unseen count per category: word letters..., blank, other
	other   []int // the "other" tiles, expanded, to fill draws from
	configs [][]int
	weights []float64
	cum     []float64
	// Prob is the probability the mover's actual draw lands in the event.
	Prob float64
}

func binom(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	r := 1.0
	for i := 0; i < k; i++ {
		r = r * float64(n-i) / float64(i+1)
	}
	return r
}

// NewCondDraw builds the conditioned draw.
//
//	unseen: count per machine letter (index 0 = blank) of the mover's unseen pool
//	leave:  count per machine letter of the tiles the mover kept
//	drawn:  how many tiles the mover drew
//	word:   the target word's letters (unblanked)
//	need:   how many of the word's letters the rack must cover (blanks count)
//	most:   the most it may cover (use a large number for no limit); a band
//	        [need, most] lets the caller split the event into disjoint strata
func NewCondDraw(unseen, leave []int, drawn int, word []byte, need, most int) *CondDraw {
	wc := map[int]int{}
	var letters []int
	for _, b := range word {
		if wc[int(b)] == 0 {
			letters = append(letters, int(b))
		}
		wc[int(b)]++
	}
	k := len(letters)
	cd := &CondDraw{letters: letters, pool: make([]int, k+2)}
	total := 0
	for _, n := range unseen {
		total += n
	}
	inWord := map[int]bool{}
	for j, ml := range letters {
		cd.pool[j] = unseen[ml]
		inWord[ml] = true
	}
	cd.pool[k] = unseen[0]
	for ml, n := range unseen {
		if ml == 0 || inWord[ml] {
			continue
		}
		cd.pool[k+1] += n
		for i := 0; i < n; i++ {
			cd.other = append(cd.other, ml)
		}
	}
	denom := binom(total, drawn)
	if denom == 0 {
		return cd
	}
	need_ := make([]int, k)
	have := make([]int, k)
	for j, ml := range letters {
		need_[j] = wc[ml]
		have[j] = leave[ml]
	}
	leaveBlanks := leave[0]

	x := make([]int, k+2)
	var rec func(c, left int, w float64)
	rec = func(c, left int, w float64) {
		if c == k+1 {
			if left > cd.pool[c] {
				return
			}
			x[c] = left
			w *= binom(cd.pool[c], left)
			// Is the word covered well enough?
			covered, short := 0, 0
			for j := 0; j < k; j++ {
				r := have[j] + x[j]
				if r >= need_[j] {
					covered += need_[j]
				} else {
					covered += r
					short += need_[j] - r
				}
			}
			covered += min(leaveBlanks+x[k], short)
			if covered >= need && covered <= most && w > 0 {
				cfg := make([]int, len(x))
				copy(cfg, x)
				cd.configs = append(cd.configs, cfg)
				cd.weights = append(cd.weights, w)
			}
			return
		}
		for n := 0; n <= min(left, cd.pool[c]); n++ {
			x[c] = n
			rec(c+1, left-n, w*binom(cd.pool[c], n))
		}
	}
	rec(0, drawn, 1)
	s := 0.0
	for _, w := range cd.weights {
		s += w
		cd.cum = append(cd.cum, s)
	}
	cd.Prob = s / denom
	return cd
}

// Sample returns one draw (machine letters) from the event, chosen with its
// conditional probability. It returns nil if the event is empty.
func (cd *CondDraw) Sample(rng *rand.Rand) []int {
	if len(cd.cum) == 0 {
		return nil
	}
	u := rng.Float64() * cd.cum[len(cd.cum)-1]
	i := 0
	for i < len(cd.cum)-1 && cd.cum[i] <= u {
		i++
	}
	cfg := cd.configs[i]
	k := len(cd.letters)
	var tiles []int
	for j, ml := range cd.letters {
		for n := 0; n < cfg[j]; n++ {
			tiles = append(tiles, ml)
		}
	}
	for n := 0; n < cfg[k]; n++ {
		tiles = append(tiles, 0)
	}
	// Other tiles uniformly without replacement.
	perm := rng.Perm(len(cd.other))
	for n := 0; n < cfg[k+1]; n++ {
		tiles = append(tiles, cd.other[perm[n]])
	}
	return tiles
}

// Need is how many of a word's letters the rack must cover for the synthetic
// estimator: the whole word up to a full rack, less slack.
func Need(wordLen, slack int) int {
	return int(math.Max(1, float64(min(wordLen, 7)-slack)))
}

// Combos is the number of ways to draw the alphagram's tiles out of a full bag,
// letting up to two blanks stand in (Zyzzyva's probability count).
// dist is the letter distribution's count per machine letter, blank at 0.
func Combos(dist []uint8, alpha []byte) float64 {
	cnt := map[byte]int{}
	var letters []byte
	for _, b := range alpha {
		if cnt[b] == 0 {
			letters = append(letters, b)
		}
		cnt[b]++
	}
	blanks := int(dist[0])
	total := 0.0
	var rec func(i, used int, w float64)
	rec = func(i, used int, w float64) {
		if i == len(letters) {
			total += w * binom(blanks, used)
			return
		}
		l := letters[i]
		for x := 0; x <= cnt[l] && used+x <= min(2, blanks); x++ {
			rec(i+1, used+x, w*binom(int(dist[l]), cnt[l]-x))
		}
	}
	rec(0, 0, 1)
	return total
}
