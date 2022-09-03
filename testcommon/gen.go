//go:build ignore

package main

import (
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/testcommon"
)

var DefaultConfig = config.DefaultConfig()

func main() {
	testcommon.CreateGaddags(DefaultConfig, []string{
		"NWL20", "NWL18", "OSPS44", "CSW19", "CSW21",
		"America", "pseudo_twl1979",
	})
	// not actually sure which ones we need.
	testcommon.CreateDawgs(DefaultConfig, []string{
		"NWL20", "NWL18", "CSW19", "CSW21", "America",
	})
}
