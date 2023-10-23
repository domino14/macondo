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
	wordCounts  []int32
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
	return k.ArcIndex(1) // (1) for a GADDAG, (0) for a DAWG
}

func (k KWG) GetAlphabet() *tilemapping.TileMapping {
	return k.alphabet
}

func (k *KWG) LexiconName() string {
	return k.lexiconName
}

func (k *KWG) NextNodeIdx(nodeIdx uint32, letter tilemapping.MachineLetter) uint32 {
	for i := nodeIdx; ; i++ {
		if k.Tile(i) == uint8(letter) {
			return k.ArcIndex(i)
		}
		if k.IsEnd(i) {
			return 0
		}
	}
}

func (k *KWG) InLetterSet(letter tilemapping.MachineLetter, nodeIdx uint32) bool {
	letter = letter.Unblank()
	for i := nodeIdx; ; i++ {
		if k.Tile(i) == uint8(letter) {
			return k.Accepts(i)
		}
		if k.IsEnd(i) {
			return false
		}
	}
}

func (k *KWG) GetLetterSet(nodeIdx uint32) tilemapping.LetterSet {
	var ls tilemapping.LetterSet
	for i := nodeIdx; ; i++ {
		t := k.Tile(i)
		if k.Accepts(i) {
			ls |= (1 << t)
		}
		if k.IsEnd(i) {
			break
		}
	}
	return ls
}

func (k *KWG) IsEnd(nodeIdx uint32) bool {
	return k.nodes[nodeIdx]&0x400000 != 0
}

func (k *KWG) Accepts(nodeIdx uint32) bool {
	return k.nodes[nodeIdx]&0x800000 != 0
}

func (k *KWG) ArcIndex(nodeIdx uint32) uint32 {
	return k.nodes[nodeIdx] & 0x3fffff
}

func (k *KWG) Tile(nodeIdx uint32) uint8 {
	return uint8(k.nodes[nodeIdx] >> 24)
}

// I have no idea what is going on in these functions. See wolges kwg.rs
func (k *KWG) countWordsAt(p uint32) int {
	if p >= uint32(len(k.wordCounts)) {
		return 0
	}
	if k.wordCounts[p] == -1 {
		panic("unexpected -1")
	}
	if k.wordCounts[p] == 0 {
		k.wordCounts[p] = -1

		a := 0
		if k.Accepts(p) {
			a = 1
		}
		b := 0
		if k.ArcIndex(p) != 0 {
			b = k.countWordsAt(k.ArcIndex(p))
		}
		c := 0
		if !k.IsEnd(p) {
			c = k.countWordsAt(p + 1)
		}
		k.wordCounts[p] = int32(a + b + c)
	}
	return int(k.wordCounts[p])
}

func (k *KWG) CountWords() {
	k.wordCounts = make([]int32, len(k.nodes))
	for p := len(k.wordCounts) - 1; p >= 0; p-- {
		k.countWordsAt(uint32(p))
	}
}

func (k *KWG) GetWordIndexOf(nodeIdx uint32, letters tilemapping.MachineWord) int32 {
	idx := int32(0)
	lidx := 0

	for nodeIdx != 0 {
		idx += k.wordCounts[nodeIdx]
		for k.Tile(nodeIdx) != uint8(letters[lidx]) {
			if k.IsEnd(nodeIdx) {
				return -1
			}
			nodeIdx++
		}
		idx -= k.wordCounts[nodeIdx]
		lidx++
		if lidx > len(letters)-1 {
			if k.Accepts(nodeIdx) {
				return int32(idx)
			}
			return -1
		}
		if k.Accepts(nodeIdx) {
			idx += 1
		}
		nodeIdx = k.ArcIndex(nodeIdx)
	}
	return -1
}
