package main

import (
	"fmt"
	"github.com/domino14/gorilla/anagrammer"
	"github.com/domino14/gorilla/gaddag"
	"os"
)

func usage() {
	fmt.Fprintf(os.Stderr,
		"usage: %s [gaddagfile] {anagram|build} tiles\n"+
			"Use _ for blanks\n",
		os.Args[0])
	os.Exit(2)
}

func main() {
	if len(os.Args) < 4 {
		usage()
	}
	var mode uint8
	root := gaddag.LoadGaddag(os.Args[1])
	switch os.Args[2] {
	case "anagram":
		mode = anagrammer.ModeAnagram
	case "build":
		mode = anagrammer.ModeBuild
	default:
		panic("You must select a mode: anagram or build")
	}
	anagrammer.Anagram(root, os.Args[3], mode)
}
