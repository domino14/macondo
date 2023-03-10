package kwg

import (
	"encoding/binary"
	"io"

	"github.com/domino14/macondo/alphabet"
	"github.com/rs/zerolog/log"
)

// A KWG is a Kurnia Word Graph. More information is available here:
// https://github.com/andy-k/wolges/blob/main/details.txt
// Thanks to Andy Kurnia.
type KWG struct {
	// Nodes is just a slice of 32-bit elements, the node array.
	nodes    []uint32
	alphabet *alphabet.Alphabet
}

func ScanKWG(data io.Reader) (*KWG, error) {
	nodes := []uint32{}
	var node uint32
	for {
		err := binary.Read(data, binary.LittleEndian, &node)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		nodes = append(nodes, node)
	}
	log.Debug().Int("num-nodes", len(nodes)).Msg("loaded-kwg")
	return &KWG{nodes: nodes}, nil
}

func (k *KWG) GetRootNodeIndex() uint32 {
	return k.arcIndex(1)
}

func (k *KWG) GetAlphabet() *alphabet.Alphabet {
	return k.alphabet
}

func (k *KWG) NextNodeIdx(nodeIdx uint32, letter alphabet.MachineLetter) uint32 {
	// we need to add 1 to an alphabet.MachineLetter to turn it into a Kurnia
	// machine letter.
	// XXX: let's fix this later and make them match.
	// Kurnia: 1-26 A-Z, 0: special GADDAG token
	// Cesar: 0-25 A-Z, 50: special GADDAG token
	var kletter uint8
	if letter != alphabet.SeparationMachineLetter {
		kletter = uint8(letter) + 1
	} // else already 0
	for i := nodeIdx; ; i++ {
		if k.tile(i) == kletter {
			return k.arcIndex(i)
		}
		if k.isEnd(i) {
			return 0
		}
	}
}

func (k *KWG) InLetterSet(letter alphabet.MachineLetter, nodeIdx uint32) bool {
	if letter >= alphabet.BlankOffset {
		letter -= alphabet.BlankOffset
	}
	var kletter uint8
	if letter != alphabet.SeparationMachineLetter {
		kletter = uint8(letter) + 1
	} // else already 0
	for i := nodeIdx; ; i++ {
		if k.tile(i) == kletter {
			return k.accepts(i)
		}
		if k.isEnd(i) {
			return false
		}
	}
}

func (k *KWG) GetLetterSet(nodeIdx uint32) alphabet.LetterSet {
	var ls alphabet.LetterSet
	for i := nodeIdx; ; i++ {
		t := k.tile(i)
		// XXX Ugly conversion to undo later:
		if t == 0 {
			t = alphabet.SeparationMachineLetter
		} else {
			t--
		}
		if k.accepts(i) {
			ls |= (1 << t)
		}
		if k.isEnd(i) {
			break
		}
	}
	return ls
}

func (k *KWG) isEnd(nodeIdx uint32) bool {
	return k.nodes[nodeIdx]&0x400000 != 0
}

func (k *KWG) accepts(nodeIdx uint32) bool {
	return k.nodes[nodeIdx]&0x800000 != 0
}

func (k *KWG) arcIndex(nodeIdx uint32) uint32 {
	return k.nodes[nodeIdx] & 0x3fffff
}

func (k *KWG) tile(nodeIdx uint32) uint8 {
	return uint8(k.nodes[nodeIdx] >> 24)
}

func (k *KWG) IterateSiblings(nodeIdx uint32, cb func(ml alphabet.MachineLetter, nnidx uint32)) {
	if k.isEnd(nodeIdx) {
		// no siblings.
		return
	}
	for i := nodeIdx + 1; ; i++ {
		t := k.tile(i)
		// XXX Ugly conversion to undo later:
		if t == 0 {
			t = alphabet.SeparationMachineLetter
		} else {
			t--
		}
		nn := k.arcIndex(i)
		cb(alphabet.MachineLetter(t), nn)
		if k.isEnd(i) {
			break
		}
	}
}
