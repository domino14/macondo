package kwg

import (
	"testing"

	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestLoadKWG(t *testing.T) {
	is := is.New(t)
	kwg, err := Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	is.Equal(len(kwg.nodes), 855967)
}
