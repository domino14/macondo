// This has utility functions for minimizing the GADDAG.

package gaddagmaker

import (
	"github.com/rs/zerolog/log"
)

type NodeArr []*Node
type NodeBucket [][]NodeArr

const (
	MaxDepth      = 35
	LetterBuckets = 997
)

// Minimize is a method to minimize the passed-in GADDAG!
// It works by first calculating the depths and letter sums of all nodes.
// It uses these values to sort the nodes into a two-dimensional array, to
// greatly minimize the number of required comparisons.
func (g *Gaddag) Minimize() {
	log.Debug().Msg("Minimizing...")
	calculateDepth(g.Root)
	calculateSums(g.Root)
	// Two nodes are the same if they have the same letter sets, the same
	// arc letters, and all their children are the same. To narrow down the
	// number of direct comparisons we have to make, let's bucket our nodes.
	bucket := make(NodeBucket, MaxDepth) // Max depth
	for i := 0; i < MaxDepth; i++ {
		bucket[i] = make([]NodeArr, LetterBuckets)
	}
	visits := 0
	traverseTreeAndExecute(g.Root, func(node *Node) {
		key := node.letterSum % LetterBuckets
		if !node.visited {
			bucket[node.depth-1][key] = append(bucket[node.depth-1][key], node)
			visits++
		}
		node.visited = true
	})
	log.Debug().Msgf("Visited %v nodes", visits)
	for i := 0; i < MaxDepth; i++ {
		log.Debug().Msgf("Depth %v...", i)
		for j := 0; j < LetterBuckets; j++ {
			narr := bucket[i][j]
			nNodes := len(narr)
			if nNodes < 2 {
				// If there's only 1 or 0 nodes, nothing to compare against.
				continue
			}
			for idx1, n1 := range narr[:nNodes-1] {
				if n1.copyOf != nil {
					continue
				}
				for _, n2 := range narr[idx1+1:] {
					if n2.copyOf != nil {
						continue
					}
					if n1.Equals(n2) {
						n2.copyOf = n1
					}
				}
			}
		}
	}
	g.AllocArcs = uint32(0)
	nodeArr := []*Node{}
	nodesAppended := make(map[*Node]bool)
	traverseTreeAndExecute(g.Root, func(node *Node) {
		if node.copyOf != nil {
			if nodesAppended[node.copyOf] == false {
				nodeArr = append(nodeArr, node.copyOf)
				nodesAppended[node.copyOf] = true
			}
		} else {
			if nodesAppended[node] == false {
				nodeArr = append(nodeArr, node)
				nodesAppended[node] = true
			}
		}
	})

	for _, node := range nodeArr {
		g.AllocArcs += uint32(node.numArcs)
		// Look through arcs to see if any point to a node copy; point to
		// original if so.
		for _, arc := range node.arcs {
			if arc.destination.copyOf != nil {
				arc.destination = arc.destination.copyOf
				if arc.destination.copyOf != nil {
					panic("Chain of nodes - something went wrong!")
				}
			}
		}
	}
	g.Root = nodeArr[0]
	g.AllocStates = uint32(len(nodeArr))
	log.Debug().Msgf("Number of arcs, nodes now: %v, %v", g.AllocArcs, g.AllocStates)
	//874624 460900
}

// Equals compares two nodes. They are the same if they have the same
// letter sets, the same arc letters, and all their children are the same.
func (node *Node) Equals(other *Node) bool {
	if node.numArcs != other.numArcs {
		return false
	}
	if node.letterSet != other.letterSet {
		return false
	}
	if node.letterSum != other.letterSum {
		return false
	}
	if node.depth != other.depth {
		return false
	}
	for idx, arc1 := range node.arcs {
		if arc1.letter != other.arcs[idx].letter {
			return false
		}
		if !arc1.destination.Equals(other.arcs[idx].destination) {
			return false
		}
	}
	return true
}

// Calculates the depth of every node by doing a full recursive traversal.
func calculateDepth(node *Node) uint8 {
	maxDepth := uint8(0)
	for _, arc := range node.arcs {
		thisDepth := calculateDepth(arc.destination)
		if thisDepth > maxDepth {
			maxDepth = thisDepth
		}
	}
	node.depth = 1 + maxDepth
	return node.depth
}

// Calculates sums of all letters and those of children. This is done to
// bucket the nodes during minimization.
func calculateSums(node *Node) uint32 {
	if node.numArcs == 0 {
		node.letterSum = 0
		return 0
	}
	sum := uint32(0)
	for _, arc := range node.arcs {
		thisSum := uint32(arc.letter) + calculateSums(arc.destination)
		sum += thisSum
	}
	node.letterSum = sum
	return node.letterSum
}
