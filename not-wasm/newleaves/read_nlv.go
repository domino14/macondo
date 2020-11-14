package newleaves

import (
	"encoding/binary"
	"fmt"
	"io"
	"sort"

	"github.com/domino14/macondo/alphabet"
)

type NewLeaves struct {
	leaveFloats []float32
	maxLength   uint32
	buf         []byte
}

func (nlv *NewLeaves) LeaveValue1(leave alphabet.MachineWord) float64 {
	// Same quirks as ExhaustiveLeaveStrategy.LeaveValue for now, to be fair.
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Recovered from panic; leave was %v\n", leave.UserVisible(alphabet.EnglishAlphabet()))
			// Panic anyway; the recover was just to figure out which leave did it.
			panic("panicking anyway")
		}
	}()
	if len(leave) == 0 {
		return 0
	}
	if len(leave) > 1 {
		sort.Slice(leave, func(i, j int) bool {
			return leave[i] < leave[j]
		})
	}
	if len(leave) <= int(nlv.maxLength) {
		leaveBytes := leave.Bytes()
		// If one key is a prefix of the other, sort the longer key first,
		// because we are going to effectively pad the shorter key with 0xff.
		// That choice, in turn, is because mw.Bytes() legitimately uses 0x00.

		var idx int
		idx = sort.Search(len(nlv.leaveFloats), func(i int) bool {
			// data[i] >= leaveBytes, which is
			// bytes.Compare(leaveBytes + 0xff padding, data[i]) <= 0
			ofs := i * int(nlv.maxLength)
			for j := 0; j < int(nlv.maxLength); j++ {
				leaveByte := byte(0xff)
				if j < len(leaveBytes) {
					leaveByte = leaveBytes[j]
				}
				if leaveByte < nlv.buf[ofs+j] {
					// a < b
					return true
				} else if leaveByte > nlv.buf[ofs+j] {
					// a > b
					return false
				}
			}
			// a == b
			return true
		})
		if idx < len(nlv.leaveFloats) {
			// bytes.Equal(leaveBytes + 0xff padding, data[idx])
			ofs := idx * int(nlv.maxLength)
			for j := 0; j < int(nlv.maxLength); j++ {
				leaveByte := byte(0xff)
				if j < len(leaveBytes) {
					leaveByte = leaveBytes[j]
				}
				if leaveByte != nlv.buf[ofs+j] {
					// a != b, so it was a false positive.
					panic("what")
					return 0.0
				}
			}
			// a == b
			return float64(nlv.leaveFloats[idx])
		}
		// Key not present.
		panic("what")
		return 0.0
	}
	// Only will happen if we have a pass. Passes are very rare and
	// we should ignore this a bit since there will be a negative
	// adjustment already from the fact that we're scoring 0.
	return float64(0)
}

func (nlv *NewLeaves) LeaveValue2(leave alphabet.MachineWord) float64 {
	// Same quirks as ExhaustiveLeaveStrategy.LeaveValue for now, to be fair.
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Recovered from panic; leave was %v\n", leave.UserVisible(alphabet.EnglishAlphabet()))
			// Panic anyway; the recover was just to figure out which leave did it.
			panic("panicking anyway")
		}
	}()
	if len(leave) == 0 {
		return 0
	}
	if len(leave) > 1 {
		sort.Slice(leave, func(i, j int) bool {
			return leave[i] < leave[j]
		})
	}
	if len(leave) <= int(nlv.maxLength) {
		leaveBytes := leave.Bytes()
		// If one key is a prefix of the other, sort the longer key first,
		// because we are going to effectively pad the shorter key with 0xff.
		// That choice, in turn, is because mw.Bytes() legitimately uses 0x00.

		var idx int
		lo := 0
		hi := len(nlv.leaveFloats)
		for j := 0; j < int(nlv.maxLength); j++ {
			leaveByte := byte(0xff)
			if j < len(leaveBytes) {
				leaveByte = leaveBytes[j]
			}
			lo += sort.Search(hi-lo, func(i int) bool {
				// data[i] >= leaveBytes, which is
				// bytes.Compare(leaveBytes + 0xff padding, data[i]) <= 0
				ofs := (lo + i) * int(nlv.maxLength)
				return leaveByte <= nlv.buf[ofs+j]
			})
			if leaveByte < byte(0xff) {
				// adjust hi too, so the remaining range shares the same prefix
				leaveByte++
				hi = lo + sort.Search(hi-lo, func(i int) bool {
					ofs := (lo + i) * int(nlv.maxLength)
					return leaveByte <= nlv.buf[ofs+j]
				})
			}
		}
		idx = lo
		if idx < len(nlv.leaveFloats) {
			// bytes.Equal(leaveBytes + 0xff padding, data[idx])
			ofs := idx * int(nlv.maxLength)
			for j := 0; j < int(nlv.maxLength); j++ {
				leaveByte := byte(0xff)
				if j < len(leaveBytes) {
					leaveByte = leaveBytes[j]
				}
				if leaveByte != nlv.buf[ofs+j] {
					// a != b, so it was a false positive.
					panic("what")
					return 0.0
				}
			}
			// a == b
			return float64(nlv.leaveFloats[idx])
		}
		// Key not present.
		panic("what")
		return 0.0
	}
	// Only will happen if we have a pass. Passes are very rare and
	// we should ignore this a bit since there will be a negative
	// adjustment already from the fact that we're scoring 0.
	return float64(0)
}

func ReadNewLeaves(file io.Reader) (*NewLeaves, error) {
	var lenLeaves uint32
	if err := binary.Read(file, binary.BigEndian, &lenLeaves); err != nil {
		return nil, err
	}
	leaveFloats := make([]float32, lenLeaves)
	if err := binary.Read(file, binary.BigEndian, &leaveFloats); err != nil {
		return nil, err
	}
	var maxLength uint32
	if err := binary.Read(file, binary.BigEndian, &maxLength); err != nil {
		return nil, err
	}
	buf := make([]byte, maxLength*lenLeaves)
	if err := binary.Read(file, binary.BigEndian, &buf); err != nil {
		return nil, err
	}

	return &NewLeaves{
		leaveFloats: leaveFloats,
		maxLength:   maxLength,
		buf:         buf,
	}, nil
}

type NewLeavesAlt struct {
	leaveFloats []float32
	maxLength   int
	bufAlt      []uint64
}

func TurnToAlt(maxLength int, b []byte) uint64 {
	// Necessarily big endian.
	v := uint64(0)
	for i := 0; i < int(maxLength); i++ {
		if i < len(b) {
			v = (v << 8) | uint64(b[i])
		} else {
			v = (v << 8) | 0xff
		}
	}
	return v
}

// Shares the leaveFloats.
func MakeNewLeavesAlt(nlv *NewLeaves) *NewLeavesAlt {
	bufAlt := make([]uint64, len(nlv.leaveFloats))
	maxLength := int(nlv.maxLength)
	for i := 0; i < len(nlv.leaveFloats); i++ {
		bufAlt[i] = TurnToAlt(maxLength, nlv.buf[i*maxLength:])
	}
	return &NewLeavesAlt{
		leaveFloats: nlv.leaveFloats,
		maxLength:   maxLength,
		bufAlt:      bufAlt,
	}
}

func (nlv *NewLeavesAlt) LeaveValue3(leave alphabet.MachineWord) float64 {
	// Same quirks as ExhaustiveLeaveStrategy.LeaveValue for now, to be fair.
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Recovered from panic; leave was %v\n", leave.UserVisible(alphabet.EnglishAlphabet()))
			// Panic anyway; the recover was just to figure out which leave did it.
			panic("panicking anyway")
		}
	}()
	if len(leave) == 0 {
		return 0
	}
	if len(leave) > 1 {
		sort.Slice(leave, func(i, j int) bool {
			return leave[i] < leave[j]
		})
	}
	if len(leave) <= nlv.maxLength {
		leaveBytes := leave.Bytes()
		leaveAlt := TurnToAlt(nlv.maxLength, leaveBytes)
		idx := sort.Search(len(nlv.leaveFloats), func(i int) bool {
			// data[i] >= leaveBytes
			return nlv.bufAlt[i] >= leaveAlt
		})
		if idx < len(nlv.leaveFloats) {
			// bytes.Equal(leaveBytes + 0xff padding, data[idx])
			if nlv.bufAlt[idx] == leaveAlt {
				return float64(nlv.leaveFloats[idx])
			}
		}
		// Key not present.
		panic("what")
		return 0.0
	}
	// Only will happen if we have a pass. Passes are very rare and
	// we should ignore this a bit since there will be a negative
	// adjustment already from the fact that we're scoring 0.
	return float64(0)
}

func (nlv *NewLeavesAlt) LeaveValue4(leave alphabet.MachineWord) float64 {
	ll := len(leave)
	for i := 1; i < ll; i++ {
		for j := i; j > 0 && leave[j-1] > leave[j]; j-- {
			leave[j-1], leave[j] = leave[j], leave[j-1]
		}
	}
	if len(leave) <= nlv.maxLength {
		leaveBytes := leave.Bytes()
		leaveAlt := TurnToAlt(nlv.maxLength, leaveBytes)
		idx := sort.Search(len(nlv.leaveFloats), func(i int) bool {
			// data[i] >= leaveBytes
			return nlv.bufAlt[i] >= leaveAlt
		})
		if idx < len(nlv.leaveFloats) {
			// bytes.Equal(leaveBytes + 0xff padding, data[idx])
			if nlv.bufAlt[idx] == leaveAlt {
				return float64(nlv.leaveFloats[idx])
			}
		}
		// Key not present.
		panic("what")
		return 0.0
	}
	// Only will happen if we have a pass. Passes are very rare and
	// we should ignore this a bit since there will be a negative
	// adjustment already from the fact that we're scoring 0.
	return float64(0)
}
