package movegen

import (
	"encoding/gob"
	"os"
)

// MachineWordString is what you get when you cast MachineWord to a string,
// i.e. string(mw) for a MachineWord mw. It's not a user-readable string and
// is only meant to be used as a lookup for maps, which don't allow us to use
// MachineWord as a key directly.
type MachineWordString string

// Load leaves into a map

// LeaveMap gets created from a binary file of leaves. The leave-generation
// code is in the xwordgame package.
type LeaveMap map[MachineWordString]float64

func loadLeaves(lexiconName string) LeaveMap {
	var lm map[MachineWordString]float64
	file, err := os.Open("./data/" + lexiconName + "/leaves.gob")
	if err != nil {
		// log.Printf("[ERROR] Data file not found for lexicon %v. Will not use leaves for evaluation!",
		// 	lexiconName)
		return LeaveMap(map[MachineWordString]float64{})
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	decoder.Decode(&lm)
	return LeaveMap(lm)
}
