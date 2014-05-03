package main

import (
	"fmt"
	"github.com/domino14/gorilla/anagrammer"
	"github.com/domino14/gorilla/gaddag"
	"os"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: %s [gaddagfile] {anagram|build} tiles\n",
		os.Args[0])
	os.Exit(2)
}

func main() {
	if len(os.Args) < 4 {
		usage()
	}

	root := gaddag.LoadGaddag(os.Args[1])
	anagrammer.Anagram(root, os.Args[3], os.Args[2])
}
