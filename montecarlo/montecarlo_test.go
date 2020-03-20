package montecarlo

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
	"github.com/matryer/is"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

const (
	LeaveFile = "leave_values_112719.idx.gz"

	Epsilon = 1e-6
)

func GaddagFromLexicon(lex string) (*gaddag.SimpleGaddag, error) {
	return gaddag.LoadGaddag(filepath.Join(LexiconDir, "gaddag", lex+".gaddag"))
}

func TestSimSingleIteration(t *testing.T) {
	is := is.New(t)
	plies := 2
	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	// This will deal a random rack to players:
	game.StartGame()
	strategy := strategy.NewExhaustiveLeaveStrategy(game.Bag(), gd.LexiconName(),
		gd.GetAlphabet(), LeaveFile)
	generator := movegen.NewGordonGenerator(game, strategy)
	generator.Reset()
	game.SetRackFor(0, alphabet.RackFromString("AAADERW", gd.GetAlphabet()))
	generator.GenAll(game.RackFor(0))
	plays := generator.Plays()[:10]

	simmer := &Simmer{}
	simmer.Init(generator, game)
	simmer.resetStats(plies, len(plays))
	simmer.plays = plays
	simmer.simSingleIteration(plays, plies)

	// Board should be reset back to empty after the simulation.
	is.True(game.Board().IsEmpty())
	fmt.Println(simmer.printStats())
	is.True(false)
}

func TestLongerSim(t *testing.T) {
	is := is.New(t)
	plies := 2
	gd, err := GaddagFromLexicon("NWL18")
	if err != nil {
		t.Errorf("Expected error to be nil, got %v", err)
	}
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)
	// This will deal a random rack to players:
	game.StartGame()
	strategy := strategy.NewExhaustiveLeaveStrategy(game.Bag(), gd.LexiconName(),
		gd.GetAlphabet(), LeaveFile)
	generator := movegen.NewGordonGenerator(game, strategy)
	generator.Reset()
	// This is the prototypical Maven sim rack. AWA should sim best.
	game.SetRackFor(0, alphabet.RackFromString("AAADERW", gd.GetAlphabet()))
	generator.GenAll(game.RackFor(0))
	plays := generator.Plays()[:10]

	simmer := &Simmer{}
	simmer.Init(generator, game)
	timeout, cancel := context.WithTimeout(
		context.Background(), 100*time.Second)
	defer cancel()
	simmer.Simulate(timeout, plays, plies)

	// Board should be reset back to empty after the simulation.
	is.True(game.Board().IsEmpty())
	fmt.Println(simmer.printStats())
	fmt.Println("Total iterations", simmer.iterationCount)
	is.True(false)
}

func fuzzyEqual(a, b float64) bool {
	return math.Abs(a-b) < Epsilon
}

func TestRunningStat(t *testing.T) {
	is := is.New(t)
	type tc struct {
		scores []int
		mean   float64
		stdev  float64
	}
	cases := []tc{
		tc{[]int{10, 12, 23, 23, 16, 23, 21, 16}, 18, 5.2372293656638},
		tc{[]int{14, 35, 71, 124, 10, 24, 55, 33, 87, 19}, 47.2, 36.937785531891},
		tc{[]int{1}, 1, 0},
		tc{[]int{}, 0, 0},
		tc{[]int{1, 1}, 1, 0},
	}
	for _, c := range cases {
		s := &Statistic{}
		for _, score := range c.scores {
			s.push(score)
		}
		is.True(fuzzyEqual(s.mean(), c.mean))
		is.True(fuzzyEqual(s.stdev(), c.stdev))

	}
}
