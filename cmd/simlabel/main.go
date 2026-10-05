// simlabel labels positions with a Monte Carlo sim's verdict on every
// candidate play, for distilling the sim into the value net
// (pytorch/plan-sim-distill.md).
//
//	simlabel select -turns open.txt.gz -tag open -out positions.jsonl -val positions-val.jsonl
//	simlabel sim -in positions.jsonl -out labels.jsonl -threads 64
//
// select replays turn logs and keeps one position per game: the board
// before a chosen turn, as a CGP with the mover's rack (the opponent's rack
// left out, as a bot sees it), the move actually played and the mover's
// final result. sim simulates each position (one thread per position, many
// at once) and writes every candidate's sim statistics. Both write JSON
// lines; sim appends and skips positions already in its output, so it can
// be stopped and restarted.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: simlabel select|sim|check [flags]   (-h for flags)")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "select":
		selectMain(os.Args[2:])
	case "sim":
		simMain(os.Args[2:])
	case "check":
		checkMain(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q: want select, sim or check\n", os.Args[1])
		os.Exit(2)
	}
}
