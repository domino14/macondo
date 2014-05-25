// Here we have utility functions for creating a GADDAG.
package gaddag

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
)

// Node is a temporary type used in the creation of a GADDAG.
// It will not be used when loading the GADDAG.
type Node struct {
	Arcs      []*Arc
	NumArcs   uint8
	LetterSet uint32
	// Utility fields, for minimizing GADDAG at the end:
	visited           bool
	copyOf            *Node
	depth             uint8
	letterSum         uint32
	indexInSerialized uint32
}

// Arc is also a temporary type.
type Arc struct {
	Letter      byte
	Destination *Node
}

// Gaddag is a temporary structure to hold the nodes in sequential order prior
// to writing them to file. It should not be used after making the gaddag.
type Gaddag struct {
	Root               *Node
	AllocStates        uint32
	AllocArcs          uint32
	SerializedElements []uint32
}

var gaddag Gaddag

type ArcPtrSlice []*Arc

func (a ArcPtrSlice) Len() int           { return len(a) }
func (a ArcPtrSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ArcPtrSlice) Less(i, j int) bool { return a[i].Letter < a[j].Letter }

func getWords(filename string) []string {
	words := []string{}
	file, err := os.Open(filename)
	if err != nil {
		log.Println("Filename", filename, "not found")
		return nil
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		// Split line into spaces.
		fields := strings.Fields(scanner.Text())
		if len(fields) > 0 {
			words = append(words, fields[0])
		}
	}
	file.Close()
	return words
}

// Create a new node and store it in the node array.
func (g *Gaddag) createNode() *Node {
	newNode := Node{}
	g.AllocStates++
	return &newNode
}

// Does the node contain the uppercase letter in its letter set?
func (node *Node) containsLetter(letter byte) bool {
	return node.LetterSet&(1<<(letter-'A')) != 0
}

// Does the Node contain an arc for the letter c? Return the arc if so.
func (state *Node) containsArc(c byte) *Arc {
	for i := uint8(0); i < state.NumArcs; i++ {
		if state.Arcs[i].Letter == c {
			return state.Arcs[i]
		}
	}
	return nil
}

// Creates an arc from node named "from", and returns the new node that
// this arc points to (or if to is not NULL, it returns that).
func (from *Node) createArcFrom(c byte, to *Node) *Node {
	var newNode *Node
	if to == nil {
		newNode = gaddag.createNode()
	} else {
		newNode = to
	}
	newArc := Arc{c, nil}
	gaddag.AllocArcs++
	from.Arcs = append(from.Arcs, &newArc)
	from.NumArcs++
	newArc.Destination = newNode
	return newNode
}

// Adds an arc from state for c (if one does not already exist) and
// resets state to the node this arc leads to. Every state has an array
// of Arc pointers. We need to create the array if it doesn't exist.
// Returns the created or existing *Node
func (state *Node) addArc(c byte) *Node {
	var nextNode *Node
	existingArc := state.containsArc(c)
	if existingArc == nil {
		nextNode = state.createArcFrom(c, nil)
	} else {
		nextNode = existingArc.Destination
	}
	return nextNode
}

// Add arc from state to c1 and add c2 to this arc's letter set.
func (state *Node) addFinalArc(c1 byte, c2 byte) *Node {
	nextNode := state.addArc(c1)
	if nextNode.containsLetter(c2) {
		log.Fatal("Containsletter", nextNode, c2)
	}
	bit := uint32(1 << (c2 - 'A'))
	nextNode.LetterSet |= bit
	return nextNode
}

// Add an arc from state to forceState for c (an error occurs if an arc
// from st for c already exists going to any other state).
func (state *Node) forceArc(c byte, forceState *Node) {
	arc := state.containsArc(c)
	if arc != nil {
		if arc.Destination != forceState {
			log.Fatal("Arc already existed pointing elsewhere")
		} else {
			// Don't create the arc if it already exists; redundant.
			return
		}
	}
	if state.createArcFrom(c, forceState) != forceState {
		log.Fatal("createArcFrom did not equal forceState")
	}
}

type nodeTraversalFn func(*Node)

func traverseTreeAndExecute(node *Node, fn nodeTraversalFn) {
	fn(node)
	for _, arc := range node.Arcs {
		traverseTreeAndExecute(arc.Destination, fn)
	}
}

// Serializes the elements of the gaddag into the SerializedElements array.
func (g *Gaddag) serializeElements() {
	fmt.Println("Serializing elements...")
	var serialized, letter uint32
	g.SerializedElements = []uint32{}
	traverseTreeAndExecute(g.Root, func(node *Node) {
		node.visited = false
	})
	fmt.Println("Root node parameters", g.Root.NumArcs, g.Root.LetterSet,
		g.Root.Arcs)
	count := uint32(0)
	missingElements := make(map[uint32]*Node)
	traverseTreeAndExecute(g.Root, func(node *Node) {
		if !node.visited {
			node.visited = true
			// Represent node as a 32-bit number
			serialized = node.LetterSet + uint32(node.NumArcs)<<NumArcsBitLoc
			g.SerializedElements = append(g.SerializedElements, serialized)
			node.indexInSerialized = count
			count++
			for _, arc := range node.Arcs {
				letter = uint32(arc.Letter - 'A')
				if letter == SeparationToken {
					// XXX: Hard-coded letter here, need lex params
					letter = 26
				}
				serialized = letter << LetterBitLoc
				missingElements[count] = arc.Destination
				count++
				g.SerializedElements = append(g.SerializedElements, serialized)
			}
		}
	})
	// Now go through the node pointers and assign SerializedElements properly.
	for idx, node := range missingElements {
		g.SerializedElements[idx] += node.indexInSerialized
	}
	fmt.Println("Assigned", len(missingElements), "missing elements.")
}

// Saves the GADDAG to a file.
func (g *Gaddag) Save(filename string) {
	g.serializeElements()
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Could not create file: ", err)
	}
	// Save it in a compressed format.
	binary.Write(file, binary.LittleEndian, uint32(len(g.SerializedElements)))
	fmt.Println("Writing serialized elements, first of which is", g.SerializedElements[0])
	binary.Write(file, binary.LittleEndian, g.SerializedElements)
	file.Close()
	fmt.Println("Saved gaddag to", filename)
}

func GenerateGaddag(filename string, minimize bool) {
	gaddag = Gaddag{}
	words := getWords(filename)
	if words == nil {
		return
	}
	gaddag.Root = gaddag.createNode()
	fmt.Println("Read", len(words), "words")
	for idx, word := range words {
		if idx%10000 == 0 {
			fmt.Printf("%d...\n", idx)
		}
		st := gaddag.Root
		// Create path for anan-1...a1:
		n := len(word)
		for j := n - 1; j >= 2; j-- {
			st = st.addArc(word[j])
		}
		st = st.addFinalArc(word[1], word[0])

		// Create path for an-1...a1^an
		st = gaddag.Root
		for j := n - 2; j >= 0; j-- {
			st = st.addArc(word[j])
		}
		st = st.addFinalArc(SeparationToken, word[n-1])

		// Partially minimize remaining paths.
		for m := n - 3; m >= 0; m-- {
			forceSt := st
			st = gaddag.Root
			for j := m; j >= 0; j-- {
				st = st.addArc(word[j])
			}
			st = st.addArc(SeparationToken)
			st.forceArc(word[m+1], forceSt)
		}
	}
	fmt.Printf("Allocated arcs: %d states: %d\n", gaddag.AllocArcs,
		gaddag.AllocStates)
	// We need to also sort the arcs alphabetically prior to minimization/
	// serialization.
	traverseTreeAndExecute(gaddag.Root, func(node *Node) {
		sort.Sort(ArcPtrSlice(node.Arcs))
	})
	if minimize {
		gaddag.Minimize()
	} else {
		fmt.Println("Not minimizing; saving to disk.")
	}
	gaddag.Save("out.gaddag")
}
