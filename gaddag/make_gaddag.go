// Here we have utility functions for creating a GADDAG.
package gaddag

import (
	"bufio"
	"encoding/binary"
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
	Letter      rune
	Destination *Node
}

// Gaddag is a temporary structure to hold the nodes in sequential order prior
// to writing them to file. It should not be used after making the gaddag.
type Gaddag struct {
	Root               *Node
	AllocStates        uint32
	AllocArcs          uint32
	SerializedElements []uint32
	Alphabet           *Alphabet
}

type ArcPtrSlice []*Arc

func (a ArcPtrSlice) Len() int           { return len(a) }
func (a ArcPtrSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ArcPtrSlice) Less(i, j int) bool { return a[i].Letter < a[j].Letter }

func getWords(filename string) ([]string, *Alphabet) {
	words := []string{}
	alphabet := Alphabet{}
	alphabet.Init()
	file, err := os.Open(filename)
	if err != nil {
		log.Println("[WARN] Filename", filename, "not found")
		return nil, &alphabet
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		// Split line into spaces.
		fields := strings.Fields(scanner.Text())
		if len(fields) > 0 {
			word := strings.ToUpper(fields[0])
			words = append(words, word)
			err := alphabet.update(word)
			if err != nil {
				panic(err)
			}
		}
	}
	file.Close()
	alphabet.Reconcile()
	return words, &alphabet
}

// Create a new node and store it in the node array.
func (g *Gaddag) createNode() *Node {
	newNode := Node{}
	g.AllocStates++
	return &newNode
}

// Does the node contain the letter in its letter set?
func (node *Node) containsLetter(letter rune, g *Gaddag) bool {
	return node.LetterSet&(1<<g.Alphabet.vals[letter]) != 0
}

// Does the Node contain an arc for the letter c? Return the arc if so.
func (state *Node) containsArc(c rune) *Arc {
	for i := uint8(0); i < state.NumArcs; i++ {
		if state.Arcs[i].Letter == c {
			return state.Arcs[i]
		}
	}
	return nil
}

// Creates an arc from node named "from", and returns the new node that
// this arc points to (or if to is not NULL, it returns that).
func (from *Node) createArcFrom(c rune, to *Node, g *Gaddag) *Node {
	var newNode *Node
	if to == nil {
		newNode = g.createNode()
	} else {
		newNode = to
	}
	newArc := Arc{c, nil}
	g.AllocArcs++
	from.Arcs = append(from.Arcs, &newArc)
	from.NumArcs++
	newArc.Destination = newNode
	return newNode
}

// Adds an arc from state for c (if one does not already exist) and
// resets state to the node this arc leads to. Every state has an array
// of Arc pointers. We need to create the array if it doesn't exist.
// Returns the created or existing *Node
func (state *Node) addArc(c rune, g *Gaddag) *Node {
	var nextNode *Node
	existingArc := state.containsArc(c)
	if existingArc == nil {
		nextNode = state.createArcFrom(c, nil, g)
	} else {
		nextNode = existingArc.Destination
	}
	return nextNode
}

// Add arc from state to c1 and add c2 to this arc's letter set.
func (state *Node) addFinalArc(c1 rune, c2 rune, g *Gaddag) *Node {
	nextNode := state.addArc(c1, g)
	if nextNode.containsLetter(c2, g) {
		log.Fatal("[ERROR] Containsletter", nextNode, c2)
	}
	bit := uint32(1 << g.Alphabet.vals[c2])
	nextNode.LetterSet |= bit
	return nextNode
}

// Add an arc from state to forceState for c (an error occurs if an arc
// from st for c already exists going to any other state).
func (state *Node) forceArc(c rune, forceState *Node, g *Gaddag) {
	arc := state.containsArc(c)
	if arc != nil {
		if arc.Destination != forceState {
			log.Fatal("[ERROR] Arc already existed pointing elsewhere")
		} else {
			// Don't create the arc if it already exists; redundant.
			return
		}
	}
	if state.createArcFrom(c, forceState, g) != forceState {
		log.Fatal("[ERROR] createArcFrom did not equal forceState")
	}
}

type nodeTraversalFn func(*Node)

func traverseTreeAndExecute(node *Node, fn nodeTraversalFn) {
	fn(node)
	for _, arc := range node.Arcs {
		traverseTreeAndExecute(arc.Destination, fn)
	}
}

func (g *Gaddag) serializeAlphabet() {
	// Append the alphabet.
	serializedAlphabet := g.Alphabet.Serialize()
	g.SerializedElements = append(g.SerializedElements, serializedAlphabet...)
}

// serializeLetterSets makes a map of all unique letter sets, serializes
// to the appropriate array, and returns the map for later use.
func (g *Gaddag) serializeLetterSets() map[uint32]uint32 {
	// Make a map of all unique letter sets. The value of the map is the
	// index in the serialized array.

	letterSets := make(map[uint32]uint32)
	letterSetIdx := uint32(0)
	serializedLetterSets := []uint32{}
	traverseTreeAndExecute(g.Root, func(node *Node) {
		node.visited = false
		if _, ok := letterSets[node.LetterSet]; !ok {
			letterSets[node.LetterSet] = letterSetIdx
			letterSetIdx++
			serializedLetterSets = append(serializedLetterSets, node.LetterSet)
		}
	})
	// Prepend the letter set count so we can parse the file later.
	serializedLetterSets = append([]uint32{uint32(len(letterSets))},
		serializedLetterSets...)
	log.Println("[INFO] Number of unique letter sets", len(letterSets))
	g.SerializedElements = append(g.SerializedElements, serializedLetterSets...)
	return letterSets
}

// Serializes the elements of the gaddag into the SerializedElements array.
func (g *Gaddag) serializeElements() {
	log.Println("[INFO] Serializing elements...")
	g.SerializedElements = []uint32{}
	g.serializeAlphabet()
	letterSets := g.serializeLetterSets()
	count := uint32(len(g.SerializedElements))
	//rootNodeIdx := count
	missingElements := make(map[uint32]*Node)
	traverseTreeAndExecute(g.Root, func(node *Node) {
		if !node.visited {
			var letterCode, serialized uint32
			node.visited = true
			// Represent node as a 32-bit number
			serialized = letterSets[node.LetterSet] +
				uint32(node.NumArcs)<<NumArcsBitLoc
			g.SerializedElements = append(g.SerializedElements, serialized)
			node.indexInSerialized = count
			count++
			for _, arc := range node.Arcs {
				if arc.Letter == SeparationToken {
					letterCode = MaxAlphabetSize
				} else {
					letterCode = g.Alphabet.vals[arc.Letter]
				}
				serialized = letterCode << LetterBitLoc
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
	log.Println("[INFO] Assigned", len(missingElements), "missing elements.")
}

// Saves the GADDAG to a file.
func (g *Gaddag) Save(filename string) {
	g.serializeElements()
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("[ERROR] Could not create file: ", err)
	}
	// Save it in a compressed format.
	binary.Write(file, binary.LittleEndian, uint32(len(g.SerializedElements)))
	log.Println("[INFO] Writing serialized elements, first of which is",
		g.SerializedElements[0])
	binary.Write(file, binary.LittleEndian, g.SerializedElements)
	file.Close()
	log.Println("[INFO] Saved gaddag to", filename)
}

// GenerateDawg makes a GADDAG with only one permutation of letters
// allowed per word, the spelled-out permutation. We still treat it for
// all intents and purposes as a GADDAG, but note that it only has one path!
func GenerateDawg(filename string, minimize bool, writeToFile bool) *Gaddag {
	gaddag := &Gaddag{}
	words, alphabet := getWords(filename)
	if words == nil {
		return gaddag
	}
	gaddag.Root = gaddag.createNode()
	gaddag.Alphabet = alphabet
	log.Println("[INFO] Read", len(words), "words")
	for idx, word := range words {
		if idx%10000 == 0 {
			log.Printf("[DEBUG] %d...\n", idx)
		}
		st := gaddag.Root
		// Create path for anan-1...a1:
		wordRunes := []rune(word)
		n := len(wordRunes)
		for j := 0; j < n-1; j++ {
			st = st.addArc(wordRunes[j], gaddag)
		}
		st = st.addFinalArc(wordRunes[n-2], wordRunes[n-1], gaddag)
	}
	log.Printf("[INFO] Allocated arcs: %d states: %d\n", gaddag.AllocArcs,
		gaddag.AllocStates)
	// We need to also sort the arcs alphabetically prior to minimization/
	// serialization.
	traverseTreeAndExecute(gaddag.Root, func(node *Node) {
		sort.Sort(ArcPtrSlice(node.Arcs))
	})
	if minimize {
		gaddag.Minimize()
	} else {
		log.Println("[INFO] Not minimizing.")
	}
	if writeToFile {
		gaddag.Save("out.dawg")
	}
	return gaddag
}

// GenerateGaddag makes a GADDAG out of the filename, and optionally
// minimizes it and/or writes it to file.
func GenerateGaddag(filename string, minimize bool, writeToFile bool) *Gaddag {
	gaddag := &Gaddag{}
	words, alphabet := getWords(filename)
	if words == nil {
		return gaddag
	}
	gaddag.Root = gaddag.createNode()
	gaddag.Alphabet = alphabet
	log.Println("[INFO] Read", len(words), "words")
	for idx, word := range words {
		if idx%10000 == 0 {
			log.Printf("[DEBUG] %d...\n", idx)
		}
		st := gaddag.Root
		// Create path for anan-1...a1:
		wordRunes := []rune(word)
		n := len(wordRunes)
		for j := n - 1; j >= 2; j-- {
			st = st.addArc(wordRunes[j], gaddag)
		}
		st = st.addFinalArc(wordRunes[1], wordRunes[0], gaddag)

		// Create path for an-1...a1^an
		st = gaddag.Root
		for j := n - 2; j >= 0; j-- {
			st = st.addArc(wordRunes[j], gaddag)
		}
		st = st.addFinalArc(SeparationToken, wordRunes[n-1], gaddag)

		// Partially minimize remaining paths.
		for m := n - 3; m >= 0; m-- {
			forceSt := st
			st = gaddag.Root
			for j := m; j >= 0; j-- {
				st = st.addArc(wordRunes[j], gaddag)
			}
			st = st.addArc(SeparationToken, gaddag)
			st.forceArc(wordRunes[m+1], forceSt, gaddag)
		}
	}
	log.Printf("[INFO] Allocated arcs: %d states: %d\n", gaddag.AllocArcs,
		gaddag.AllocStates)
	// We need to also sort the arcs alphabetically prior to minimization/
	// serialization.
	traverseTreeAndExecute(gaddag.Root, func(node *Node) {
		sort.Sort(ArcPtrSlice(node.Arcs))
	})
	if minimize {
		gaddag.Minimize()
	} else {
		log.Println("[INFO] Not minimizing.")
	}
	if writeToFile {
		gaddag.Save("out.gaddag")
	}
	return gaddag
}
