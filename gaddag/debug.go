package gaddag

// Prints out the gaddag in a user-readable form of some sort.
// func (g SimpleGaddag) Analyzer() {
// 	state := "node"
// 	arcs := uint32(0)
// 	for i := uint32(0); i < uint32(len(g)); i++ {
// 		if state == "node" {
// 			numArcs := g[i] >> NumArcsBitLoc
// 			arcs = numArcs
// 			letterSet := g[i] & ((1 << NumArcsBitLoc) - 1)
// 			fmt.Printf("%d NODE\tArcs: %d\tLetterSet: %s\tRaw:%b (%d)\n", i, numArcs,
// 				letterSetToLetters(letterSet), g[i], g[i])
// 			if arcs != 0 {
// 				state = "arc"
// 			}
// 		} else if state == "arc" {
// 			arcs--
// 			// XXX: FIX CALL:
// 			nodeIdx, letter := g.ArcToIdxLetter(i)

// 			fmt.Printf("%d ARC\tNodeIdx: %d\tLetter: %s\tRaw:%b (%d)\n", i, nodeIdx,
// 				string(letter), g[i], g[i])

// 			if arcs == 0 {
// 				state = "node"
// 			}
// 		}
// 	}
// }

// Prints out the letter set in a user-friendly form.
// func letterSetToLetters(letterSet uint32) string {
// 	str := ""
// 	for i := uint32(0); i < 26; i++ {
// 		if (1<<i)&letterSet != 0 {
// 			str += string('A' + i)
// 		}
// 	}
// 	if str == "" {
// 		return "-"
// 	}
// 	return str
// }
