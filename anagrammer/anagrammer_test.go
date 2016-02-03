package anagrammer

import "testing"
import "github.com/domino14/macondo/gaddag"

func TestAnagram(t *testing.T) {
	gaddag.GenerateDawg("/Users/cesar/coding/webolith/words/OWL2.txt", true,
		true)
	d := gaddag.LoadGaddag("out.dawg")
	Anagram("AEROLITH", gaddag.SimpleDawg(d))
}
