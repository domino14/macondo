// This has utility functions for minimizing the GADDAG.

package gaddag

import "fmt"

// Minimizes the goddamn in gaddag.NodeArr.
func minimizeGaddag() {
	calculateDepth(gaddag.NodeArr[0])
	for _, node := range gaddag.NodeArr {
		if node.visited == false {
			panic("A node was not visited!")
		}
	}
	fmt.Println("All nodes were visited :D")
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
