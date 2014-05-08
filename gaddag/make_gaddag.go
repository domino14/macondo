// Here we have utility functions for creating a GADDAG.
package gaddag

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"strings"
)

// Node is a temporary type used in the creation of a GADDAG.
// It will not be used when loading the GADDAG.
type Node struct {
	Arcs            []*Arc
	NumArcs         uint8
	ArcBitVector    uint32
	LetterSet       uint32
	SerializedIndex uint32
}

// Arc is also a temporary type.
type Arc struct {
	Letter      byte
	Destination *Node
	Source      *Node
}

func GenerateGaddag(string filename) {
	words := []string{}
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		// Split line into spaces.
		fields := strings.Fields(scanner.Text())
		if len(fields) > 0 {
			words = words.append(fields[0])
		}
	}
	file.close()
	fmt.Println(words)
}
