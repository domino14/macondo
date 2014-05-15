// This has utility functions for minimizing the GADDAG.

package gaddag

import "fmt"

type NodeArr []*Node
type NodeBucket [][]NodeArr

const (
	MAX_DEPTH      = 16
	LETTER_BUCKETS = 100
)

// Minimize is a method to minimize the passed-in GADDAG!
// It works by first calculating the depths and letter sums of all nodes.
// It uses these values to sort the nodes into a two-dimensional array, to
// greatly minimize the number of required comparisons.
func (g *Gaddag) Minimize() {
	calculateDepth(g.NodeArr[0])
	calculateSums(g.NodeArr[0])
	// Two nodes are the same if they have the same letter sets, the same
	// arc letters, and all their children are the same. To narrow down the
	// number of direct comparisons we have to make, let's bucket our nodes.
	bucket := make(NodeBucket, MAX_DEPTH) // Max depth
	for i := 0; i < MAX_DEPTH; i++ {
		bucket[i] = make([]NodeArr, LETTER_BUCKETS)
	}
	for idx, node := range g.NodeArr {
		key := node.letterSum % LETTER_BUCKETS
		bucket[node.depth-1][key] = append(bucket[node.depth-1][key], node)
		// Let's also number every node sequentially for later.
		node.index = uint32(idx)
	}
	for i := 0; i < MAX_DEPTH; i++ {
		fmt.Println("Depth", i, "...")
		for j := 0; j < LETTER_BUCKETS; j++ {
			narr := bucket[i][j]
			nNodes := len(narr)
			if nNodes < 2 {
				// If there's only 1 or 0 nodes, nothing to compare against.
				continue
			}
			fmt.Println("Depth", i, "Bucket", j, "nNodes:", nNodes)
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
	// Now that we have a lot of identical nodes, let's merge.
	newArr := []*Node{}
	indicesAppended := make(map[uint32]bool)
	for _, node := range g.NodeArr {
		if node.copyOf != nil {
			if indicesAppended[node.copyOf.index] == false {
				newArr = append(newArr, node.copyOf)
				indicesAppended[node.copyOf.index] = true
			}
		} else {
			newArr = append(newArr, node)
		}
	}
	// And renumber all nodes in newArr
	g.NodeArr = newArr
	g.AllocArcs = uint32(0)
	for idx, node := range g.NodeArr {
		node.index = uint32(idx)
		g.AllocArcs += uint32(node.NumArcs)
		// Look through arcs to see if any point to a node copy; point to
		// original if so.
		for _, arc := range node.Arcs {
			if arc.Destination.copyOf != nil {
				arc.Destination = arc.Destination.copyOf
				if arc.Destination.copyOf != nil {
					panic("There should not be a chain of copies!")
				}
			}
		}
	}
	g.AllocStates = uint32(len(g.NodeArr))

	fmt.Println("Number of arcs, nodes now:", g.AllocArcs, g.AllocStates)
}

// Equals compares two nodes. They are the same if they have the same
// letter sets, the same arc letters, and all their children are the same.
func (node *Node) Equals(other *Node) bool {
	if node.NumArcs != other.NumArcs {
		return false
	}
	if node.LetterSet != other.LetterSet {
		return false
	}
	if node.letterSum != other.letterSum {
		return false
	}
	if node.depth != other.depth {
		return false
	}
	for idx, arc1 := range node.Arcs {
		if arc1.Letter != other.Arcs[idx].Letter {
			return false
		}
		if !arc1.Destination.Equals(other.Arcs[idx].Destination) {
			return false
		}
	}
	return true
}

// Calculates the depth of every node by doing a full recursive traversal.
func calculateDepth(node *Node) uint8 {
	maxDepth := uint8(0)
	for _, arc := range node.Arcs {
		thisDepth := calculateDepth(arc.Destination)
		if thisDepth > maxDepth {
			maxDepth = thisDepth
		}
	}
	node.depth = 1 + maxDepth
	node.visited = true
	return node.depth
}

func calculateSums(node *Node) uint32 {
	if node.NumArcs == 0 {
		node.letterSum = 0
		return 0
	}
	sum := uint32(0)
	for _, arc := range node.Arcs {
		thisSum := uint32(arc.Letter) + calculateSums(arc.Destination)
		sum += thisSum
	}
	node.letterSum = sum
	return node.letterSum
}
