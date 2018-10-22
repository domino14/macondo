package gaddag

import (
	"log"
	"os"
	"testing"

	"github.com/domino14/macondo/gaddagmaker"
)

var simplefindPrefixTests = []testpair{
	{"N", true},
	{"NO", true},
	{"O", false},
	{"Z", false},
}

var simplefindWordTests = []testpair{
	{"NO", true},
	{"ON", false},
	{"OO", false},
	{"NN", false},
}

func TestSimpleFindPrefix(t *testing.T) {
	gaddagmaker.GenerateGaddag("../gaddagmaker/test_files/no.txt", true, true)
	os.Rename("out.gaddag", "/tmp/gen_no.gaddag")

	gd := LoadGaddag("/tmp/gen_no.gaddag")
	log.Println("Loaded gd:", gd)

	for _, pair := range simplefindPrefixTests {
		found := FindPrefix(gd, pair.prefix)
		if found != pair.found {
			t.Error("For", pair.prefix, "expected", pair.found, "got", found)
		}

	}
}
