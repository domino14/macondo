package equity

import (
	"encoding/binary"
	"io"

	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/tilemapping"
)

const LeaveScale = 1 / 256.0

// KLV is a Kurnia Leave Value file. It contains a KWG and a list of
// int16s that represent scaled leave values.
type KLV struct {
	kwg         *kwg.KWG
	leaveValues []int16
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
	leaveValues := make([]int16, numLeaves)
	if err := binary.Read(file, binary.LittleEndian, &leaveValues); err != nil {
		return nil, err
	}
	// Count words so we can figure out how to map leaves to indexes.
	k.CountWords()
	return &KLV{kwg: k, leaveValues: leaveValues}, nil
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
		return float64(k.leaveValues[idx]) * LeaveScale
	}
	return 0.0
}
