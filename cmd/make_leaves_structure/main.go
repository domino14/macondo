package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"sort"
	"strconv"
	"strings"

	"github.com/alecthomas/mph"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cache"
	"github.com/rs/zerolog/log"
)

func float32ToByte(f float32) []byte {
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.BigEndian, f)
	if err != nil {
		fmt.Println("binary.Write failed:", err)
	}
	return buf.Bytes()
}

func moveBlanksToEnd(letters string) string {
	blankCt := strings.Count(letters, "?")
	if strings.Contains(letters, "?") {
		letters = strings.ReplaceAll(letters, "?", "")
		letters += strings.Repeat("?", blankCt)
	}

	return letters
}

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

func getAlphabet(alphabetName string) *alphabet.Alphabet {
	if strings.EqualFold(alphabetName, "English") {
		return alphabet.EnglishAlphabet()
	} else if strings.EqualFold(alphabetName, "German") {
		return alphabet.GermanAlphabet()
	} else if strings.EqualFold(alphabetName, "Norwegian") {
		return alphabet.NorwegianAlphabet()
	} else if strings.EqualFold(alphabetName, "Polish") {
		return alphabet.PolishAlphabet()
	} else if strings.EqualFold(alphabetName, "Spanish") {
		return alphabet.SpanishAlphabet()
	} else if strings.EqualFold(alphabetName, "French") {
		return alphabet.FrenchAlphabet()
	}
	return nil
}

func parseIntoMPH(filename string, alph *alphabet.Alphabet) {

	hb := mph.Builder()
	fmt.Println(filename)
	file, err := cache.Open(filename)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	defer file.Close()

	r := csv.NewReader(file)
	records, err := r.ReadAll()
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}

	for _, record := range records {
		letters := moveBlanksToEnd(record[0])

		// These bytes can be put in hash table right away.
		mw, err := alphabet.ToMachineWord(letters, alph)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		if !sort.SliceIsSorted(mw, func(i, j int) bool { return mw[i] < mw[j] }) {
			fmt.Println(mw.Bytes(), "became")
			sort.Slice(mw, func(i, j int) bool { return mw[i] < mw[j] })
			fmt.Println(mw.Bytes())
		}
		leaveVal, err := strconv.ParseFloat(record[1], 32)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		hb.Add(mw.Bytes(), float32ToByte(float32(leaveVal)))
	}

	var bb bytes.Buffer
	var bb2 bytes.Buffer
	nTries := 0
	bestSize := int(^uint(0) >> 1)
	for nTries < 10 {
		bb.Reset()
		bb2.Reset()
		leaves, err := hb.Build()
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		log.Info().Msgf("Finished building MPH for leave file %v", filename)
		log.Info().Msgf("Size of MPH: %v", leaves.Len())
		bw := bufio.NewWriter(&bb)
		err = leaves.Write(bw)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		err = bw.Flush()
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		log.Info().Msgf("Size of file: %v", bb.Len())
		if bb.Len() < bestSize {
			by := bb.Bytes()
			err = ioutil.WriteFile("data.idx", by, 0644)
			if err != nil {
				log.Fatal().Err(err).Msg("")
			}
			log.Info().Msgf("Wrote index file to data.idx. Please copy to data directory in right place.")
			// Assumption: both files shrink/grow together.
			bw2 := bufio.NewWriter(&bb2)
			err = ConvertIdxToOlv(bytes.NewReader(by), bw2)
			if err != nil {
				log.Fatal().Err(err).Msg("")
			}
			err = bw2.Flush()
			if err != nil {
				log.Fatal().Err(err).Msg("")
			}
			err = ioutil.WriteFile("data.olv", bb2.Bytes(), 0644)
			if err != nil {
				log.Fatal().Err(err).Msg("")
			}
			log.Info().Msgf("Wrote compacted file to data.olv. Please copy to data directory in right place. (size %v)", bb2.Len())
			bestSize = bb.Len()
			nTries = 0
		} else {
			nTries++
			log.Info().Msgf("Not overwriting, because no improvement. (tries=%v)", nTries)
		}
	}
}

func main() {
	filename := flag.String("filename", "", "filename of the leaves .csv")
	alphabetName := flag.String("alphabet", "English", "example English, German, Norwegian")

	flag.Parse()

	alph := getAlphabet(*alphabetName)
	if alph == nil {
		panic("invalid alphabet: " + *alphabetName)
	}
	alph.Reconcile()

	parseIntoMPH(*filename, alph)
}
