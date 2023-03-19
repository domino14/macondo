package equity

import (
	"encoding/binary"
	"io"

	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/tilemapping"
	"github.com/samber/lo"
)

// KLV is a Kurnia Leave Value file. It contains a KWG and a list of
// float32 leave values. We convert to float64 for our internal structures.
type KLV struct {
	kwg         *kwg.KWG
	leaveValues []float64
}

func ReadKLV(file io.Reader) (*KLV, error) {
	var kwgSize uint32
	var numLeaves uint32
	if err := binary.Read(file, binary.LittleEndian, &kwgSize); err != nil {
		return nil, err
	}

	k, err := kwg.ScanKWG(io.LimitReader(file, int64(kwgSize)*4))

	if err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.LittleEndian, &numLeaves); err != nil {
		return nil, err
	}
	leaveValues := make([]float32, numLeaves)
	if err := binary.Read(file, binary.LittleEndian, &leaveValues); err != nil {
		return nil, err
	}
	// Count words so we can figure out how to map leaves to indexes.
	k.CountWords()
	float64Leaves := lo.Map(leaveValues, func(item float32, idx int) float64 {
		return float64(item)
	})
	return &KLV{kwg: k, leaveValues: float64Leaves}, nil
}

func (k *KLV) LeaveValue(leave tilemapping.MachineWord) float64 {
	ll := len(leave)
	if ll == 0 {
		return 0.0
	}
	for i := 1; i < ll; i++ {
		for j := i; j > 0 && leave[j-1] > leave[j]; j-- {
			leave[j-1], leave[j] = leave[j], leave[j-1]
		}
	}

	idx := k.kwg.GetWordIndexOf(k.kwg.ArcIndex(0), leave)
	if idx != -1 {
		return k.leaveValues[idx]
	}
	return 0.0
}
