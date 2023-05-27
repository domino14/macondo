package negamax

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/cross_set"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
)

func TestMain(m *testing.M) {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	os.Exit(m.Run())
}

var DefaultConfig = config.DefaultConfig()

func setUpSolver(lex, distName string, bvs board.VsWho, plies int, rack1, rack2 string,
	p1pts, p2pts int, onTurn int) (*Solver, error) {

	rules, err := game.NewBasicGameRules(&DefaultConfig, lex, board.CrosswordGameLayout, distName, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		panic(err)
	}

	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player 1"},
		{Nickname: "p2", RealName: "Player 2"},
	}

	g, err := game.NewGame(rules, players)
	if err != nil {
		panic(err)
	}

	g.StartGame()
	g.SetBackupMode(game.SimulationMode)
	g.SetStateStackLength(plies)
	// Throw in the random racks dealt to our players.
	g.ThrowRacksIn()

	gd, err := kwg.Get(g.Config(), lex)
	if err != nil {
		panic(err)
	}

	dist := rules.LetterDistribution()
	gen := movegen.NewGordonGenerator(gd, g.Board(), dist)
	gen.SetPlayRecorder(movegen.AllMinimalPlaysRecorder)
	alph := g.Alphabet()

	tilesInPlay := g.Board().SetToGame(alph, bvs)
	err = g.Bag().RemoveTiles(tilesInPlay.OnBoard)
	if err != nil {
		panic(err)
	}
	cross_set.GenAllCrossSets(g.Board(), gd, dist)

	g.SetRacksForBoth([]*tilemapping.Rack{
		tilemapping.RackFromString(rack1, alph),
		tilemapping.RackFromString(rack2, alph),
	})
	g.SetPointsFor(0, p1pts)
	g.SetPointsFor(1, p2pts)
	g.SetPlayerOnTurn(onTurn)
	fmt.Println("Racks are", g.RackLettersFor(0), g.RackLettersFor(1))
	fmt.Println(g.Board().ToDisplayText(alph))

	s := new(Solver)
	err = s.Init(gen, g)
	if err != nil {
		panic(err)
	}
	return s, nil
}

func TestSolveStandard(t *testing.T) {
	// This endgame is solved with at least 3 plies. Most endgames should
	// start with 3 plies (so the first player can do an out in 2) and
	// then proceed with iterative deepening.
	plies := 7

	is := is.New(t)

	s, err := setUpSolver("NWL18", "english", board.VsCanik, plies, "DEHILOR", "BGIV", 389, 384,
		1)
	is.NoErr(err)
	v, moves, _ := s.Solve(context.Background(), plies)

	is.Equal(moves[0].ShortDescription(), " 1G VIG.")
	is.True(moves[1].ShortDescription() == " 4A HOER" ||
		moves[1].ShortDescription() == " 4A HEIR")
	// There are two spots for the final B that are both worth 9
	// and right now we don't generate these deterministically.
	is.Equal(moves[2].Score(), 9)
	is.Equal(v, float32(11))
}
