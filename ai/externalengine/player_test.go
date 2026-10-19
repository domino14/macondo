package externalengine

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/move"
)

var DefaultConfig = config.DefaultConfig()

// fakeEngine writes a shell script that checks the arguments the player
// passes and then prints `output`. A wrong argument fails the script.
func fakeEngine(t *testing.T, output string) string {
	script := `#!/bin/sh
[ "$1" = "--extra" ] || { echo "want --extra first, got $1" >&2; exit 1; }
shift
for a in "$@"; do
	case "$a" in
	--gcg=*) [ -s "${a#--gcg=}" ] || { echo "empty gcg" >&2; exit 1; } ;;
	--lexicon=*) [ "$a" = "--lexicon=CSW19" ] || { echo "bad $a" >&2; exit 1; } ;;
	--rack=*) [ "$a" = "--rack=?AACDER" ] || { echo "bad $a" >&2; exit 1; } ;;
	*) echo "unexpected $a" >&2; exit 1 ;;
	esac
done
echo "engine chatter that is not JSON"
cat <<'JSON'
` + output + `
JSON
`
	path := filepath.Join(t.TempDir(), "engine.sh")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func newPlayer(t *testing.T, binary string) *ExternalEnginePlayer {
	is := is.New(t)
	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW19", board.CrosswordGameLayout,
		"English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	hist, err := gcgio.ParseGCG(DefaultConfig, "../../gcgio/testdata/vs_frentz.gcg")
	is.NoErr(err)
	// Before the first move: cesar to play ?AACDER.
	g, err := game.NewFromHistory(hist, rules, 0)
	is.NoErr(err)
	p, err := NewFromGame(g, DefaultConfig, Config{BinaryPath: binary, ExtraArgs: []string{"--extra"}})
	is.NoErr(err)
	return p
}

func TestBestPlayPlace(t *testing.T) {
	is := is.New(t)
	p := newPlayer(t, fakeEngine(t,
		`{"player":"cesar","rack":"?AACDER","action":"place","position":"8D","tiles":"CRAAlED","score":74}`))
	m, err := p.BestPlay(context.Background())
	is.NoErr(err)
	is.Equal(m.Action(), move.MoveTypePlay)
	is.Equal(m.ShortDescription(), " 8D CRAAlED")
	is.Equal(m.Score(), 74) // scored by macondo, not taken from the engine
	is.NoErr(p.PlayMove(m, false, 0))
}

func TestBestPlayExchangeAndPass(t *testing.T) {
	is := is.New(t)
	m, err := newPlayer(t, fakeEngine(t, `{"action":"exchange","tiles":"ac"}`)).
		BestPlay(context.Background())
	is.NoErr(err)
	is.Equal(m.Action(), move.MoveTypeExchange)
	is.Equal(m.TilesPlayed(), 2)

	m, err = newPlayer(t, fakeEngine(t, `{"action":"pass"}`)).BestPlay(context.Background())
	is.NoErr(err)
	is.Equal(m.Action(), move.MoveTypePass)
}

func TestBestPlayErrors(t *testing.T) {
	is := is.New(t)
	for _, tc := range []struct{ output, want string }{
		{`{"action":"resign"}`, "unknown action"},
		{`not json`, "parse response"},
		// Tiles the rack doesn't hold.
		{`{"action":"place","position":"8D","tiles":"ZZZ"}`, "parse move"},
	} {
		_, err := newPlayer(t, fakeEngine(t, tc.output)).BestPlay(context.Background())
		is.True(err != nil)
		is.True(strings.Contains(err.Error(), tc.want))
	}

	// The engine failing surfaces its stderr.
	path := filepath.Join(t.TempDir(), "broken.sh")
	is.NoErr(os.WriteFile(path, []byte("#!/bin/sh\necho no lexicon >&2\nexit 3\n"), 0o755))
	_, err := newPlayer(t, path).BestPlay(context.Background())
	is.True(err != nil)
	is.True(strings.Contains(err.Error(), "no lexicon"))
}
