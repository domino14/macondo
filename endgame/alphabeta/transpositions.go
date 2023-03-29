package alphabeta

type nodeFlag int8

const (
	tValid nodeFlag = iota
	tLBound
	tUBound
)

type TNode struct {
	gameNode *GameNode
	flag     nodeFlag
	height   int8
}

func (s *Solver) retrieveFromTable(pos uint64) *TNode {
	// The position is a Zobrist hash. We'll need to handle collisions later.
	return s.ttable[pos]
}
