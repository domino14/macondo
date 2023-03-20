package tilemapping

import (
	"fmt"
)

func SortMW(l MachineWord) {
	// sort in place. This should be fast enough for small arrays.
	ll := len(l)
	for i := 1; i < ll; i++ {
		for j := i; j > 0 && l[j-1] > l[j]; j-- {
			l[j-1], l[j] = l[j], l[j-1]
		}
	}
}

// Leave calculates the leave from the rack and the made play.
func Leave(rack MachineWord, play MachineWord, isExchange bool) (MachineWord, error) {

	rackletters := map[MachineLetter]int{}
	for _, l := range rack {
		rackletters[l]++
	}
	leave := make([]MachineLetter, 0)

	for _, t := range play {
		if t == 0 && !isExchange {
			// play-through
			continue
		}
		if t.IsBlanked() {
			if isExchange {
				return nil, fmt.Errorf("cannot exchange a designated blank")
			}
			// it's a blank
			t = 0
		}
		if rackletters[t] != 0 {
			rackletters[t]--
		} else {
			return nil, fmt.Errorf("tile in play but not in rack: %v", t)
		}
	}

	for k, v := range rackletters {
		if v > 0 {
			for i := 0; i < v; i++ {
				leave = append(leave, k)
			}
		}
	}
	SortMW(leave)
	return leave, nil

}
