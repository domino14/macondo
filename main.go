package main

import (
	"./gaddag"
	"fmt"
	"os"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: %s [inputfile]\n", os.Args[0])
	os.Exit(2)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}

	root := gaddag.LoadGaddag(os.Args[1])
	fmt.Println(len(root))
}
