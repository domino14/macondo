package main

import (
	"bufio"
	"os"
	"strconv"
)

// NPlies is the number of plies to look ahead in the game assembler. We will predict
// the game spread after NPlies.
const NPlies = 5

// ─────────────────────────────────────────────────────────────────────────────
// text-based writer  → one line  per vector
// Format:  "0.000 1.000 0.125 …\n"
// ─────────────────────────────────────────────────────────────────────────────
func writeVectorText(w *bufio.Writer, vec []float32) error {
	for i, f := range vec {
		if i > 0 {
			if err := w.WriteByte(' '); err != nil { // space separator
				return err
			}
		}
		// strconv is ~5× faster than fmt for tight loops
		if _, err := w.WriteString(strconv.FormatFloat(float64(f), 'f', -1, 32)); err != nil {
			return err
		}
	}
	return w.WriteByte('\n') // newline terminator
}

// ─────────────────────────────────────────────────────────────────────────────
// main streaming loop
// ─────────────────────────────────────────────────────────────────────────────
func main() {
	const bufSize = 1 << 20 // 1 MiB buffered stdout
	out := bufio.NewWriterSize(os.Stdout, bufSize)

	engine := NewGameAssembler(NPlies)  // holds live games, emits vectors
	scanner := NewTurnScanner(os.Stdin) // feeds individual turns

	for scanner.Scan() {
		turn := scanner.Turn()
		engine.FeedTurn(turn)

		for engine.Ready() { // ≥1 vectors ready
			vec := engine.PopVector() // []float32
			if err := writeVectorText(out, vec); err != nil {
				panic(err) // production: handle/propagate
			}
		}
	}
	out.Flush() // flush any buffered lines
}
