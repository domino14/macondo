package automatic

import (
	"bufio"
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

func TestDrawOpeningPlies(t *testing.T) {
	is := is.New(t)
	rng := rand.New(rand.NewSource(1))
	n, sum, zeros := 20000, 0, 0
	for i := 0; i < n; i++ {
		k := drawOpeningPlies(2.0, rng)
		is.True(k >= 0)
		sum += k
		if k == 0 {
			zeros++
		}
	}
	mean := float64(sum) / float64(n)
	is.True(math.Abs(mean-2.0) < 0.1) // E[round(Exp(2))] ~ 2
	// P(round(Exp(2)) == 0) = P(X < 0.5) = 1 - e^-0.25 ~ 0.221
	zf := float64(zeros) / float64(n)
	is.True(math.Abs(zf-0.221) < 0.02)
	is.Equal(drawOpeningPlies(0, rng), 0)
}

func TestOpeningSeedIsDeterministic(t *testing.T) {
	is := is.New(t)
	runner := NewGameRunner(nil, DefaultConfig)
	runner.opening = OpeningConfig{Mean: 2, Temperature: 3, TopN: 50}
	var seed [32]byte
	seed[0] = 7
	runner.StartGameWithSeed(0, seed)
	k1 := runner.openingPlies
	runner.StartGameWithSeed(1, seed) // the pair's other half: same K
	is.Equal(k1, runner.openingPlies)
	// Over a few seeds the draw must vary.
	seen := map[int]bool{}
	for b := byte(1); b < 20; b++ {
		seed[0] = b
		runner.StartGameWithSeed(0, seed)
		seen[runner.openingPlies] = true
	}
	is.True(len(seen) > 1)
}

func TestSampleOpeningMoveIsLegalAndVaried(t *testing.T) {
	is := is.New(t)
	runner := NewGameRunner(nil, DefaultConfig)
	runner.opening = OpeningConfig{Mean: 2, Temperature: 3, TopN: 50}
	runner.openingRng = rand.New(rand.NewSource(3))
	runner.StartGame(0)
	best := runner.genBestStaticTurn(0).ShortDescription()
	seen := map[string]bool{}
	for i := 0; i < 40; i++ {
		m := runner.sampleOpeningMove(0)
		is.True(m.Action() == move.MoveTypePlay || m.Action() == move.MoveTypeExchange)
		seen[m.ShortDescription()] = true
	}
	// At 3 points of temperature over the top 50 the sampler must not
	// collapse onto the argmax.
	is.True(len(seen) > 1)
	is.True(seen[best] || len(seen) > 3)
	// Uniform draws come from the whole move list.
	runner.opening.UniformProb = 1
	for i := 0; i < 10; i++ {
		m := runner.sampleOpeningMove(0)
		is.True(m != nil)
	}
}

// TestOpeningGamesLogColumn plays a small batch with sampled openings and
// checks the turn log: every row carries the game's K, K is one value per
// game, some games have K > 0 and some K == 0, and the games finish.
func TestOpeningGamesLogColumn(t *testing.T) {
	is := is.New(t)
	out := filepath.Join(t.TempDir(), "open.txt")
	err := StartCompVCompStaticGames(
		context.Background(), DefaultConfig, 40, true, 4,
		out, "NWL20", "English",
		[]AutomaticRunnerPlayer{
			{BotCode: macondo.BotRequest_HASTY_BOT},
			{BotCode: macondo.BotRequest_HASTY_BOT},
		}, nil, OpeningConfig{Mean: 2, Temperature: 3, TopN: 50})
	is.NoErr(err)

	f, err := os.Open(out)
	is.NoErr(err)
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	is.True(sc.Scan())
	header := strings.Split(sc.Text(), ",")
	is.Equal(header[len(header)-1], "openingplies")
	kCol := len(header) - 1
	kByGame := map[string]int{}
	turnsByGame := map[string]int{}
	for sc.Scan() {
		f := strings.Split(sc.Text(), ",")
		is.Equal(len(f), len(header))
		k, err := strconv.Atoi(f[kCol])
		is.NoErr(err)
		if prev, ok := kByGame[f[1]]; ok {
			is.Equal(prev, k) // one K per game
		}
		kByGame[f[1]] = k
		turnsByGame[f[1]]++
	}
	is.Equal(len(kByGame), 40)
	zeros, positives := 0, 0
	for g, k := range kByGame {
		is.True(turnsByGame[g] > 10) // the game was played out
		if k == 0 {
			zeros++
		} else {
			positives++
		}
	}
	is.True(zeros > 0 && positives > 0)
}
