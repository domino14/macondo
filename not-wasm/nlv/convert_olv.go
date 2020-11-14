package main

import (
	"encoding/binary"
	"io"
	"os"
)

// idx is from CHD.Write()
func ConvertIdxToOlv(file io.Reader, w io.Writer) error {
	var rl uint32
	if err := binary.Read(file, binary.LittleEndian, &rl); err != nil {
		return err
	}
	r := make([]uint64, rl)
	if err := binary.Read(file, binary.LittleEndian, &r); err != nil {
		return err
	}
	var il uint32
	if err := binary.Read(file, binary.LittleEndian, &il); err != nil {
		return err
	}
	indices := make([]uint16, il)
	if err := binary.Read(file, binary.LittleEndian, &indices); err != nil {
		return err
	}
	var lenLeaves uint32
	if err := binary.Read(file, binary.LittleEndian, &lenLeaves); err != nil {
		return err
	}
	leaveFloats := make([]float32, lenLeaves)
	barr := make([][]byte, lenLeaves)
	maxLength := uint32(0)
	for i := 0; i < int(lenLeaves); i++ {
		var kl uint32
		if err := binary.Read(file, binary.LittleEndian, &kl); err != nil {
			return err
		}
		if kl > maxLength {
			maxLength = kl
		}
		var vl uint32
		if err := binary.Read(file, binary.LittleEndian, &vl); err != nil {
			return err
		}
		if vl != 4 {
			panic("unexpected")
		}
		barr[i] = make([]byte, kl)
		if err := binary.Read(file, binary.LittleEndian, &barr[i]); err != nil {
			return err
		}
		// This is BigEndian. To CHD, it's just []byte.
		if err := binary.Read(file, binary.BigEndian, &leaveFloats[i]); err != nil {
			return err
		}
	}
	buf := make([]byte, maxLength*lenLeaves)
	wp := 0
	for i := 0; i < int(lenLeaves); i++ {
		copy(buf[wp:], barr[i])
		for j := len(barr[i]); j < int(maxLength); j++ {
			buf[wp+j] = 0xff
		}
		wp += int(maxLength)
	}
	if wp != len(buf) {
		panic("oops")
	}

	if err := binary.Write(w, binary.LittleEndian, rl); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, r); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, il); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, indices); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, lenLeaves); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, leaveFloats); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, maxLength); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, buf); err != nil {
		return err
	}
	return nil
}

func main() {
	fi, err := os.Open("data.idx")
	if err != nil {
		panic(err)
	}
	defer fi.Close()
	fo, err := os.Create("data.olv")
	if err != nil {
		panic(err)
	}
	defer fo.Close()
	err = ConvertIdxToOlv(fi, fo)
	if err != nil {
		panic(err)
	}
}
