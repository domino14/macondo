package gaddag

import (
	"encoding/binary"
	"errors"

	"os"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/rs/zerolog/log"
)

// SimpleDawg is basically a SimpleGaddag, but with only one pathway.
// The structures are otherwise totally compatible. The SimpleDawg should
// only be used by anagramming utilities due to its smaller size.
type SimpleDawg struct {
	SimpleGaddag
	reverse bool
}

// Ensure the magic number matches.
func compareMagicDawg(bytes [4]uint8) bool {
	cast := string(bytes[:])
	return cast == gaddagmaker.DawgMagicNumber || cast == gaddagmaker.ReverseDawgMagicNumber
}

// Reverse returns true if this is a "reverse" dawg
func (s *SimpleDawg) Reverse() bool {
	return s.reverse
}

func loadCommonDagStructure(file *os.File) ([]uint32, []alphabet.LetterSet,
	[]uint32, []byte) {

	var lexNameLen uint8
	binary.Read(file, binary.BigEndian, &lexNameLen)
	lexName := make([]byte, lexNameLen)
	binary.Read(file, binary.BigEndian, &lexName)
	log.Debug().Msgf("Read lexicon name: '%v'", string(lexName))

	var alphabetSize, lettersetSize, nodeSize uint32

	binary.Read(file, binary.BigEndian, &alphabetSize)
	log.Debug().Msgf("Alphabet size: %v", alphabetSize)
	alphabetArr := make([]uint32, alphabetSize)
	binary.Read(file, binary.BigEndian, &alphabetArr)

	binary.Read(file, binary.BigEndian, &lettersetSize)
	log.Debug().Msgf("LetterSet size: %v", lettersetSize)
	letterSets := make([]alphabet.LetterSet, lettersetSize)
	binary.Read(file, binary.BigEndian, letterSets)

	binary.Read(file, binary.BigEndian, &nodeSize)
	log.Debug().Msgf("Nodes size: %v", nodeSize)
	nodes := make([]uint32, nodeSize)
	binary.Read(file, binary.BigEndian, &nodes)
	file.Close()
	return nodes, letterSets, alphabetArr, lexName
}

// LoadDawg loads a dawg from a file and returns a *SimpleDawg
func LoadDawg(filename string) (*SimpleDawg, error) {
	log.Debug().Msgf("Loading %v ...", filename)
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	var magicStr [4]uint8
	binary.Read(file, binary.BigEndian, &magicStr)
	if !compareMagicDawg(magicStr) {
		return nil, errors.New("magic number does not match dawg or reverse dawg")
	}
	d := &SimpleDawg{}
	if string(magicStr[:]) == gaddagmaker.ReverseDawgMagicNumber {
		d.reverse = true
	}
	nodes, letterSets, alphabetArr, lexName := loadCommonDagStructure(file)
	d.Nodes = nodes
	d.LetterSets = letterSets
	d.alphabet = alphabet.FromSlice(alphabetArr)
	d.lexiconName = string(lexName)
	return d, nil
}
