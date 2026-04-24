// Package wordprune implements a pruned KWG (word graph) for endgame solving.
//
// Before solving an endgame position, we enumerate all words that could
// potentially be played given the tiles remaining on both racks. We then
// build a compact KWG containing only those words and use it in place of
// the full dictionary during the search. This typically reduces endgame
// solve time by ~20% by letting the GADDAG traversal prune dead branches
// much earlier.
//
// The algorithm is a Go port of word_prune.c from the magpie engine
// (https://github.com/domino14/magpie).
package wordprune

import (
	"bytes"
	"encoding/binary"
	"sort"

	"github.com/domino14/macondo/board"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
)

// separationML is the GADDAG separator token (machine letter 0).
const separationML = tilemapping.MachineLetter(0)

// GeneratePrunedKWG builds a pruned KWG containing only words that can
// potentially be played in the current endgame position.  rack0 and rack1
// are the two players' racks; gd is the full dictionary KWG.  Returns nil
// (with no error) when the board is empty, which callers should treat as
// "use the full KWG".
func GeneratePrunedKWG(
	b *board.GameBoard,
	rack0, rack1 *tilemapping.Rack,
	gd *kwg.KWG,
) (*kwg.KWG, error) {
	words := generatePossibleWords(b, rack0, rack1, gd)
	if len(words) == 0 {
		return nil, nil
	}
	return buildGADDAGFromWords(words)
}

// generatePossibleWords returns a sorted, deduplicated list of all words
// that could be played in the current endgame position.
func generatePossibleWords(
	b *board.GameBoard,
	rack0, rack1 *tilemapping.Rack,
	gd *kwg.KWG,
) []tilemapping.MachineWord {
	ldSize := len(rack0.LetArr)

	// Combined tile pool: both racks together.
	metaRack := make([]int, ldSize)
	for i := 0; i < ldSize; i++ {
		metaRack[i] = rack0.LetArr[i] + rack1.LetArr[i]
	}

	rows := extractBoardRows(b)

	// Find the maximum number of consecutive empty squares not adjacent to
	// any board tile.  This bounds the length of standalone plays.
	maxNonPlaythrough := 0
	for _, row := range rows {
		n := maxNonPlaythroughSpaces(row, b.Dim())
		if n > maxNonPlaythrough {
			maxNonPlaythrough = n
		}
	}

	var tempWords []tilemapping.MachineWord

	// 1. Standalone words: can be formed entirely from meta-rack tiles,
	//    placed in an empty region of the board.
	word := make(tilemapping.MachineWord, b.Dim())
	dawgRoot := gd.ArcIndex(0)
	addWordsWithoutPlaythrough(gd, dawgRoot, metaRack, maxNonPlaythrough, word, 0, false, &tempWords)

	// 2. Playthrough words: extend or cross existing board tiles.
	for _, row := range rows {
		addPlaythroughWordsFromRow(row, gd, metaRack, b.Dim(), &tempWords)
	}

	sort.Slice(tempWords, func(i, j int) bool {
		return compareMachineWords(tempWords[i], tempWords[j]) < 0
	})
	return dedup(tempWords)
}

// boardRow stores one row (or column) of the board, unblanked.  0 = empty.
type boardRow []tilemapping.MachineLetter

// extractBoardRows extracts all unique rows and columns from the board.
func extractBoardRows(b *board.GameBoard) []boardRow {
	dim := b.Dim()
	rows := make([]boardRow, 0, dim*2)

	for r := 0; r < dim; r++ {
		row := make(boardRow, dim)
		for c := 0; c < dim; c++ {
			row[c] = b.GetLetter(r, c).Unblank()
		}
		rows = append(rows, row)
	}
	for c := 0; c < dim; c++ {
		row := make(boardRow, dim)
		for r := 0; r < dim; r++ {
			row[r] = b.GetLetter(r, c).Unblank()
		}
		rows = append(rows, row)
	}

	sort.Slice(rows, func(i, j int) bool {
		for k := 0; k < dim; k++ {
			if rows[i][k] < rows[j][k] {
				return true
			}
			if rows[i][k] > rows[j][k] {
				return false
			}
		}
		return false
	})

	// Deduplicate.
	if len(rows) == 0 {
		return rows
	}
	out := rows[:1]
	for _, r := range rows[1:] {
		last := out[len(out)-1]
		diff := false
		for k := 0; k < dim; k++ {
			if r[k] != last[k] {
				diff = true
				break
			}
		}
		if diff {
			out = append(out, r)
		}
	}
	return out
}

// maxNonPlaythroughSpaces returns the maximum number of consecutive empty
// squares in a row that are not directly adjacent to any board tile.
// A word placed entirely within such a region requires no playthroughs.
func maxNonPlaythroughSpaces(row boardRow, dim int) int {
	maxEmpty := 0
	emptyCount := 0
	for i := 0; i < dim; i++ {
		if row[i] == 0 {
			emptyCount++
		} else {
			// The square immediately before a tile can't be the end of a
			// standalone play (it's adjacent to the tile).
			emptyCount--
			if emptyCount > maxEmpty {
				maxEmpty = emptyCount
			}
			// The square immediately after a tile is similarly adjacent.
			emptyCount = -1
		}
	}
	// The trailing run of empty squares is bounded by the board edge, not
	// a tile, so no decrement needed.
	if emptyCount > maxEmpty {
		maxEmpty = emptyCount
	}
	return maxEmpty
}

// addWordsWithoutPlaythrough walks the DAWG starting at nodeIdx, collecting
// every word reachable from the given rack within maxLen tiles.
// Blanks substitute for any letter (prefer the natural tile when both exist,
// matching magpie's behaviour to avoid duplicate word entries).
func addWordsWithoutPlaythrough(
	gd *kwg.KWG,
	nodeIdx uint32,
	rack []int,
	maxLen int,
	word tilemapping.MachineWord,
	tilesPlayed int,
	accepts bool,
	result *[]tilemapping.MachineWord,
) {
	if accepts {
		w := make(tilemapping.MachineWord, tilesPlayed)
		copy(w, word)
		*result = append(*result, w)
	}
	if tilesPlayed == maxLen || nodeIdx == 0 {
		return
	}
	for i := nodeIdx; ; i++ {
		ml := tilemapping.MachineLetter(gd.Tile(i))
		if ml == separationML {
			if gd.IsEnd(i) {
				break
			}
			continue
		}
		nodeAccepts := gd.Accepts(i)
		nextIdx := gd.ArcIndex(i)
		if rack[ml] > 0 {
			rack[ml]--
			word[tilesPlayed] = ml
			addWordsWithoutPlaythrough(gd, nextIdx, rack, maxLen, word, tilesPlayed+1, nodeAccepts, result)
			rack[ml]++
		} else if rack[0] > 0 {
			rack[0]--
			word[tilesPlayed] = ml
			addWordsWithoutPlaythrough(gd, nextIdx, rack, maxLen, word, tilesPlayed+1, nodeAccepts, result)
			rack[0]++
		}
		if gd.IsEnd(i) {
			break
		}
	}
}

// addPlaythroughWordsFromRow generates all words that can be formed by
// extending or crossing the tiles in a single board row (or column).
func addPlaythroughWordsFromRow(row boardRow, gd *kwg.KWG, rack []int, dim int, result *[]tilemapping.MachineWord) {
	strip := make([]tilemapping.MachineLetter, dim)
	gaddagRoot := gd.GetRootNodeIndex()
	leftmostCol := 0

	for col := 0; col < dim; col++ {
		if row[col] == 0 {
			continue
		}
		// Advance to the rightmost tile in this contiguous group.
		for col+1 < dim && row[col+1] != 0 {
			col++
		}
		ml := row[col]

		// Find this tile at the GADDAG root to start the traversal.
		nextNodeIdx := uint32(0)
		for i := gaddagRoot; ; i++ {
			if gd.Tile(i) == uint8(ml) {
				nextNodeIdx = gd.ArcIndex(i)
				break
			}
			if gd.IsEnd(i) {
				break
			}
		}

		playthroughWordsGoOn(
			row, gd, rack, dim,
			col, col,
			ml, nextNodeIdx, false,
			col, col, leftmostCol, 0,
			strip, result,
		)
		leftmostCol = col + 2
	}
}

// playthroughWordsGoOn is the core recursive routine that tracks the
// current position in the word being assembled (moving left then right
// through the GADDAG).
func playthroughWordsGoOn(
	row boardRow, gd *kwg.KWG, rack []int, dim int,
	currentCol, anchorCol int,
	currentLetter tilemapping.MachineLetter,
	newNodeIdx uint32, accepts bool,
	leftstrip, rightstrip, leftmostCol, tilesPlayed int,
	strip []tilemapping.MachineLetter,
	result *[]tilemapping.MachineWord,
) {
	if currentCol <= anchorCol {
		// ── Left phase (including the anchor itself) ──────────────────────
		if row[currentCol] != 0 {
			strip[currentCol] = row[currentCol]
		} else {
			strip[currentCol] = currentLetter
		}
		leftstrip = currentCol

		if accepts && tilesPlayed > 0 {
			w := make(tilemapping.MachineWord, rightstrip-leftstrip+1)
			copy(w, strip[leftstrip:rightstrip+1])
			*result = append(*result, w)
		}
		if newNodeIdx == 0 {
			return
		}
		if currentCol > leftmostCol {
			playthroughWordsRecursiveGen(row, gd, rack, dim, currentCol-1, anchorCol, newNodeIdx,
				leftstrip, rightstrip, leftmostCol, tilesPlayed, strip, result)
		}

		noLetterDirectlyLeft := currentCol == 0 || row[currentCol-1] == 0
		separationNodeIdx := gd.NextNodeIdx(newNodeIdx, separationML)
		if separationNodeIdx != 0 && noLetterDirectlyLeft && anchorCol < dim-1 {
			playthroughWordsRecursiveGen(row, gd, rack, dim, anchorCol+1, anchorCol, separationNodeIdx,
				leftstrip, rightstrip, leftmostCol, tilesPlayed, strip, result)
		}
	} else {
		// ── Right phase ───────────────────────────────────────────────────
		if row[currentCol] != 0 {
			strip[currentCol] = row[currentCol]
		} else {
			strip[currentCol] = currentLetter
		}
		rightstrip = currentCol

		noLetterDirectlyRight := currentCol == dim-1 || row[currentCol+1] == 0
		if accepts && noLetterDirectlyRight && tilesPlayed > 0 {
			w := make(tilemapping.MachineWord, rightstrip-leftstrip+1)
			copy(w, strip[leftstrip:rightstrip+1])
			*result = append(*result, w)
		}
		if newNodeIdx != 0 && currentCol < dim-1 {
			playthroughWordsRecursiveGen(row, gd, rack, dim, currentCol+1, anchorCol, newNodeIdx,
				leftstrip, rightstrip, leftmostCol, tilesPlayed, strip, result)
		}
	}
}

// playthroughWordsRecursiveGen advances one square in the current direction,
// consuming either a board tile (no rack cost) or a rack tile.
func playthroughWordsRecursiveGen(
	row boardRow, gd *kwg.KWG, rack []int, dim int,
	col, anchorCol int, nodeIdx uint32,
	leftstrip, rightstrip, leftmostCol, tilesPlayed int,
	strip []tilemapping.MachineLetter,
	result *[]tilemapping.MachineWord,
) {
	currentLetter := row[col]
	if currentLetter != 0 {
		// This square has a board tile: find it in the KWG.
		nextNodeIdx := uint32(0)
		nodeAccepts := false
		for i := nodeIdx; ; i++ {
			if tilemapping.MachineLetter(gd.Tile(i)) == currentLetter {
				nextNodeIdx = gd.ArcIndex(i)
				nodeAccepts = gd.Accepts(i)
				break
			}
			if gd.IsEnd(i) {
				break
			}
		}
		playthroughWordsGoOn(row, gd, rack, dim, col, anchorCol, currentLetter, nextNodeIdx, nodeAccepts,
			leftstrip, rightstrip, leftmostCol, tilesPlayed, strip, result)
	} else {
		// Empty square: try each KWG arc using a rack tile or blank.
		for i := nodeIdx; ; i++ {
			ml := tilemapping.MachineLetter(gd.Tile(i))
			if ml != separationML {
				nextNodeIdx := gd.ArcIndex(i)
				nodeAccepts := gd.Accepts(i)
				if rack[ml] > 0 {
					rack[ml]--
					playthroughWordsGoOn(row, gd, rack, dim, col, anchorCol, ml, nextNodeIdx, nodeAccepts,
						leftstrip, rightstrip, leftmostCol, tilesPlayed+1, strip, result)
					rack[ml]++
				} else if rack[0] > 0 {
					rack[0]--
					playthroughWordsGoOn(row, gd, rack, dim, col, anchorCol, ml, nextNodeIdx, nodeAccepts,
						leftstrip, rightstrip, leftmostCol, tilesPlayed+1, strip, result)
					rack[0]++
				}
			}
			if gd.IsEnd(i) {
				break
			}
		}
	}
}

func compareMachineWords(a, b tilemapping.MachineWord) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return 0
}

func dedup(words []tilemapping.MachineWord) []tilemapping.MachineWord {
	if len(words) == 0 {
		return words
	}
	out := words[:1]
	for _, w := range words[1:] {
		if compareMachineWords(w, out[len(out)-1]) != 0 {
			out = append(out, w)
		}
	}
	return out
}

// ═══════════════════════════════════════════════════════════════════════════
// KWG builder: assembles a GADDAG from a sorted, deduplicated word list.
// ═══════════════════════════════════════════════════════════════════════════

// trieNode is a node in the GADDAG trie being built.
type trieNode struct {
	children []*trieNode            // sorted ascending by tile
	tile     tilemapping.MachineLetter
	accepts  bool
}

func (t *trieNode) insert(word []tilemapping.MachineLetter) {
	node := t
	for _, ml := range word {
		node = node.findOrAddChild(ml)
	}
	node.accepts = true
}

func (t *trieNode) findOrAddChild(ml tilemapping.MachineLetter) *trieNode {
	pos := sort.Search(len(t.children), func(i int) bool {
		return t.children[i].tile >= ml
	})
	if pos < len(t.children) && t.children[pos].tile == ml {
		return t.children[pos]
	}
	newNode := &trieNode{tile: ml}
	t.children = append(t.children, nil)
	copy(t.children[pos+1:], t.children[pos:])
	t.children[pos] = newNode
	return newNode
}

// buildGADDAGFromWords builds a KWG containing the GADDAG for the given
// word list.  words must be sorted and deduplicated.
func buildGADDAGFromWords(words []tilemapping.MachineWord) (*kwg.KWG, error) {
	root := &trieNode{}

	for _, word := range words {
		n := len(word)
		if n == 0 {
			continue
		}

		// Reversed word (no separator): for starting a play at the last
		// letter and extending left.
		rev := make(tilemapping.MachineWord, n)
		for i, ml := range word {
			rev[n-1-i] = ml
		}
		root.insert(rev)

		// Pivot forms with separator: for each pivot position k (the
		// rightmost tile played through), the GADDAG string is
		//   word[k-1], word[k-2], ..., word[0], SEP, word[k], ..., word[n-1]
		gstr := make(tilemapping.MachineWord, n+1)
		for sepPos := n - 1; sepPos >= 1; sepPos-- {
			for i := 0; i < sepPos; i++ {
				gstr[i] = word[sepPos-1-i]
			}
			gstr[sepPos] = separationML
			for i := sepPos; i < n; i++ {
				gstr[sepPos+1+(i-sepPos)] = word[i]
			}
			root.insert(gstr[:n+1])
		}
	}

	nodes := serializeTrie(root)
	return buildKWGFromNodes(nodes)
}

// serializeTrie converts the in-memory trie to the flat KWG node array.
//
// KWG format (one uint32 per node):
//   bits  0-21  arc index (first child in the output array)
//   bit   22    IsEnd – set on the last sibling in a group
//   bit   23    Accepts – set when this node ends a valid GADDAG string
//   bits 24-31  Tile – machine letter (0 = separator)
//
// Nodes 0 and 1 are root pointers:
//   node 0: DAWG root (we set it to 0 since we only emit the GADDAG)
//   node 1: GADDAG root – its ArcIndex points to the first child group
type queueEntry struct {
	node    *trieNode
	baseIdx uint32
}

func serializeTrie(root *trieNode) []uint32 {
	// Indices 0 and 1 are always reserved for the root pointers.
	nodes := make([]uint32, 2)
	nodes[0] = kwg.KWGNodeIsEndBit // no DAWG

	if len(root.children) == 0 {
		nodes[1] = kwg.KWGNodeIsEndBit
		return nodes
	}

	// Reserve contiguous slots for root's children.
	rootBase := uint32(len(nodes))
	nodes[1] = rootBase | kwg.KWGNodeIsEndBit
	for range root.children {
		nodes = append(nodes, 0)
	}

	queue := []queueEntry{{root, rootBase}}

	for len(queue) > 0 {
		entry := queue[0]
		queue = queue[1:]

		parent := entry.node
		base := entry.baseIdx

		for idx, child := range parent.children {
			outIdx := base + uint32(idx)

			node := uint32(child.tile) << kwg.KWGNodeTileShift
			if child.accepts {
				node |= kwg.KWGNodeAcceptsBit
			}
			if idx == len(parent.children)-1 {
				node |= kwg.KWGNodeIsEndBit
			}
			if len(child.children) > 0 {
				childBase := uint32(len(nodes))
				node |= childBase
				for range child.children {
					nodes = append(nodes, 0)
				}
				queue = append(queue, queueEntry{child, childBase})
			}
			nodes[outIdx] = node
		}
	}

	return nodes
}

// buildKWGFromNodes serialises the node array to little-endian bytes and
// loads it back via kwg.ScanKWG, which is the only public constructor.
func buildKWGFromNodes(nodes []uint32) (*kwg.KWG, error) {
	buf := &bytes.Buffer{}
	for _, n := range nodes {
		if err := binary.Write(buf, binary.LittleEndian, n); err != nil {
			return nil, err
		}
	}
	return kwg.ScanKWG(bytes.NewReader(buf.Bytes()), buf.Len())
}
