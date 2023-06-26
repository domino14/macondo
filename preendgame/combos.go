package preendgame

import (
	"gonum.org/v1/gonum/stat/combin"

	"github.com/domino14/macondo/tilemapping"
)

type rackCombo struct {
	rack  *tilemapping.Rack
	bag   []tilemapping.MachineLetter
	count int
}

func permuteLeaves(letters []tilemapping.MachineLetter, maxLeaveLength int) [][]tilemapping.MachineLetter {
	leaveCombos := [][]tilemapping.MachineLetter{}

	cs := combin.Permutations(len(letters), maxLeaveLength)
	for _, c := range cs {
		leave := []tilemapping.MachineLetter{}
		for _, ll := range c {
			leave = append(leave, letters[ll])
		}

		leaveCombos = append(leaveCombos, leave)
	}

	return leaveCombos
}
