package main

import (
	"flag"

	"github.com/domino14/macondo/gaddagmaker"
)

func main() {
	structtype := flag.String("type", "gaddag", "gaddag or dawg")
	minimize := flag.Bool("minimize", true, "minimize the gaddag/dawg")
	reverse := flag.Bool("reverse", false, "reverse the dawg (ignored for gaddags)")
	filename := flag.String("filename", "", "filename of the word list")

	flag.Parse()
	if *structtype == "gaddag" {
		gaddagmaker.GenerateGaddag(*filename, *minimize, true)
	} else if *structtype == "dawg" {
		gaddagmaker.GenerateDawg(*filename, *minimize, true, *reverse)
	} else {
		panic("Unsupported data structure " + *structtype)
	}
}
