package playability

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
