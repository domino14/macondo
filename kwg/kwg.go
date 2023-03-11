package kwg

import (
	"encoding/binary"
	"io"

	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
)

// A KWG is a Kurnia Word Graph. More information is available here:
// https://github.com/andy-k/wolges/blob/main/details.txt
// Thanks to Andy Kurnia.
type KWG struct {
	// Nodes is just a slice of 32-bit elements, the node array.
	nodes       []uint32
	alphabet    *tilemapping.TileMapping
	lexiconName string
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

func (k *KWG) GetAlphabet() *tilemapping.TileMapping {
	return k.alphabet
}

func (k *KWG) LexiconName() string {
	return k.lexiconName
}

func (k *KWG) NextNodeIdx(nodeIdx uint32, letter tilemapping.MachineLetter) uint32 {
	for i := nodeIdx; ; i++ {
		if k.tile(i) == uint8(letter) {
			return k.arcIndex(i)
		}
		if k.isEnd(i) {
			return 0
		}
	}
}

func (k *KWG) InLetterSet(letter tilemapping.MachineLetter, nodeIdx uint32) bool {
	letter = letter.Unblank()
	for i := nodeIdx; ; i++ {
		if k.tile(i) == uint8(letter) {
			return k.accepts(i)
		}
		if k.isEnd(i) {
			return false
		}
	}
}

func (k *KWG) GetLetterSet(nodeIdx uint32) tilemapping.LetterSet {
	var ls tilemapping.LetterSet
	for i := nodeIdx; ; i++ {
		t := k.tile(i)
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

func (k *KWG) IterateSiblings(nodeIdx uint32, cb func(ml tilemapping.MachineLetter, nnidx uint32)) {
	if k.isEnd(nodeIdx) {
		// no siblings.
		return
	}
	for i := nodeIdx + 1; ; i++ {
		t := k.tile(i)
		nn := k.arcIndex(i)
		cb(tilemapping.MachineLetter(t), nn)
		if k.isEnd(i) {
			break
		}
	}
}
