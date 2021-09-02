package gaddag

import (
	"encoding/binary"
	"errors"
	"io"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/cache"
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

func (s *SimpleDawg) Type() GenericDawgType {
	return TypeDawg
}

func loadCommonDagStructure(stream io.Reader) ([]uint32, []alphabet.LetterSet,
	[]uint32, []byte) {

	var lexNameLen uint8
	binary.Read(stream, binary.BigEndian, &lexNameLen)
	lexName := make([]byte, lexNameLen)
	binary.Read(stream, binary.BigEndian, &lexName)
	log.Debug().Msgf("Read lexicon name: '%v'", string(lexName))

	var alphabetSize, lettersetSize, nodeSize uint32

	binary.Read(stream, binary.BigEndian, &alphabetSize)
	log.Debug().Msgf("Alphabet size: %v", alphabetSize)
	alphabetArr := make([]uint32, alphabetSize)
	binary.Read(stream, binary.BigEndian, &alphabetArr)

	binary.Read(stream, binary.BigEndian, &lettersetSize)
	log.Debug().Msgf("LetterSet size: %v", lettersetSize)
	letterSets := make([]alphabet.LetterSet, lettersetSize)
	binary.Read(stream, binary.BigEndian, letterSets)

	binary.Read(stream, binary.BigEndian, &nodeSize)
	log.Debug().Msgf("Nodes size: %v", nodeSize)
	nodes := make([]uint32, nodeSize)
	binary.Read(stream, binary.BigEndian, &nodes)
	return nodes, letterSets, alphabetArr, lexName
}

func ReadDawg(data io.Reader) (*SimpleDawg, error) {
	var magicStr [4]uint8
	binary.Read(data, binary.BigEndian, &magicStr)
	if !compareMagicDawg(magicStr) {
		log.Debug().Msgf("Magic number does not match")
		return nil, errors.New("magic number does not match dawg or reverse dawg")
	}
	d := &SimpleDawg{}
	if string(magicStr[:]) == gaddagmaker.ReverseDawgMagicNumber {
		d.reverse = true
	}
	nodes, letterSets, alphabetArr, lexName := loadCommonDagStructure(data)
	d.nodes = nodes
	d.letterSets = letterSets
	d.alphabet = alphabet.FromSlice(alphabetArr)
	d.lexiconName = string(lexName)
	return d, nil
}

// LoadDawg loads a dawg from a file and returns a *SimpleDawg
func LoadDawg(filename string) (*SimpleDawg, error) {
	log.Debug().Msgf("Loading %v ...", filename)
	file, err := cache.Open(filename)
	if err != nil {
		log.Debug().Msgf("Could not load %v", filename)
		return nil, err
	}
	defer file.Close()
	return ReadDawg(file)
}
