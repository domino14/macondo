package main

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/alecthomas/mph"
	"github.com/domino14/macondo/alphabet"
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

func parseIntoMPH(filename string, alphabetName string) {

	var alph *alphabet.Alphabet
	if alphabetName == "English" {
		// XXX: Support other alphabets in the future in other ways.
		alph = alphabet.EnglishAlphabet()
	}

	hb := mph.Builder()
	fmt.Println(filename)
	file, err := os.Open(filename)
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
		leaveVal, err := strconv.ParseFloat(record[1], 32)
		if err != nil {
			log.Fatal().Err(err).Msg("")
		}
		hb.Add(mw.Bytes(), float32ToByte(float32(leaveVal)))
	}

	leaves, err := hb.Build()
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}
	log.Info().Msgf("Finished building MPH for leave file %v", filename)
	log.Info().Msgf("Size of MPH: %v", leaves.Len())
	w, err := os.Create("data.idx")
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}
	err = leaves.Write(w)
	if err != nil {
		log.Fatal().Err(err).Msg("")
	}
	log.Info().Msgf("Wrote index file to data.idx. Please copy to data directory in right place.")
}

func main() {
	filename := flag.String("filename", "", "filename of the leaves .csv")

	flag.Parse()
	parseIntoMPH(*filename, "English")
}
