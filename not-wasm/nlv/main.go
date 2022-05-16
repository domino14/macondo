package main

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cache"
	"github.com/rs/zerolog/log"
)

func moveBlanksToEnd(letters string) string {
	blankCt := strings.Count(letters, "?")
	if strings.Contains(letters, "?") {
		letters = strings.ReplaceAll(letters, "?", "")
		letters += strings.Repeat("?", blankCt)
	}

	return letters
}

func parseIntoNewLeaves(filename string, alphabetName string) {
	var alph *alphabet.Alphabet
	if alphabetName == "English" {
		// XXX: Support other alphabets in the future in other ways.
		alph = alphabet.EnglishAlphabet()
	}

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

	type LeaveEntry struct {
		k []byte
		v float32
	}

	var leaveEntries []LeaveEntry

	maxLength := 0

	for _, record := range records {
		letters := moveBlanksToEnd(record[0])

		// These bytes can be put in hash table right away.
		mw, err := alphabet.ToMachineWord(letters, alph)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		leaveVal, err := strconv.ParseFloat(record[1], 32)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}

		// Currently mw.Bytes() allocates.
		mwBytes := mw.Bytes()
		leaveEntries = append(leaveEntries, LeaveEntry{k: mwBytes, v: float32(leaveVal)})
		if len(mwBytes) > maxLength {
			maxLength = len(mwBytes)
		}
	}

	// If one key is a prefix of the other, sort the longer key first,
	// because we are going to effectively pad the shorter key with 0xff.
	// That choice, in turn, is because mw.Bytes() legitimately uses 0x00.
	sort.Slice(leaveEntries, func(i, j int) bool {
		a := leaveEntries[i].k
		b := leaveEntries[j].k
		lena := len(a)
		lenb := len(b)
		if lena <= lenb {
			cmp := bytes.Compare(a, b[:lena])
			if cmp != 0 {
				return cmp < 0
			}
			return false // !(lena > lenb), so !(a < b)
		}
		cmp := bytes.Compare(a[:lenb], b)
		if cmp != 0 {
			return cmp < 0
		}
		return true // lena > lenb, so a < b
	})

	leaves := make([]byte, 8+(maxLength+4)*len(leaveEntries))
	wp := 0
	binary.BigEndian.PutUint32(leaves[wp:], uint32(len(leaveEntries)))
	wp += 4
	for _, leaveEntry := range leaveEntries {
		binary.BigEndian.PutUint32(leaves[wp:], math.Float32bits(leaveEntry.v))
		wp += 4
	}
	binary.BigEndian.PutUint32(leaves[wp:], uint32(maxLength))
	wp += 4
	for _, leaveEntry := range leaveEntries {
		copy(leaves[wp:], leaveEntry.k)
		for j := len(leaveEntry.k); j < maxLength; j++ {
			leaves[wp+j] = 0xff
		}
		wp += maxLength
	}
	fmt.Println(wp, len(leaves))

	log.Info().Msgf("Finished building NewLeaves for leave file %v", filename)
	log.Info().Msgf("Size of NewLeaves: %v", len(leaves))
	w, err := os.Create("data.nlv")
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}
	defer w.Close()
	_, err = w.Write(leaves)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}
	log.Info().Msgf("Wrote index file to data.nlv. Please copy to data directory in right place.")
}

func main() {
	filename := flag.String("filename", "", "filename of the leaves .csv")

	flag.Parse()
	parseIntoNewLeaves(*filename, "English")
}
