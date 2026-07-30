package montecarlo

import (
	"context"
	"os"
	"testing"

	"github.com/matryer/is"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/movegen"
)

// TestSimWordSmog runs a short simulation on a WordSmog position. The simmer
// builds its own turn players from game copies, so this covers the whole
// "does the variant survive being copied into the rollout players" path.
func TestSimWordSmog(t *testing.T) {
	is := is.New(t)
	if os.Getenv("MACONDO_DATA_PATH") == "" {
		t.Skip("MACONDO_DATA_PATH not set")
	}
	plies := 2

	players := []*pb.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, "CSW21",
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarWordSmog)
	if err != nil {
		t.Skip("could not build WordSmog rules (is CSW21.kad present?): " + err.Error())
	}
	g, err := game.NewGame(rules, players)
	is.NoErr(err)

	gd, err := kwg.GetKWG(g.Config().WGLConfig(), g.LexiconName())
	is.NoErr(err)

	generator := movegen.NewGordonGenerator(gd, g.Board(), rules.LetterDistribution())
	generator.SetWordSmog(rules.AlphaDawg())

	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, tilemapping.RackFromString("AAADERW", g.Alphabet()))
	generator.SetRecordNTopPlays(10)
	plays := generator.GenAll(g.RackFor(0), false)
	is.True(len(plays) > 0)

	simmer := &Simmer{}
	calcs, leaves := defaultSimCalculators("CSW21")
	simmer.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
	simmer.PrepareSim(plies, plays)
	simmer.simSingleIteration(context.Background(), plies, 0, 1, nil)

	is.True(simmer.gameCopies[0].Board().IsEmpty())
	is.Equal(simmer.gameCopies[0].RackFor(0).String(), "AAADERW")

	// Every simmed play must have accumulated score statistics for each ply.
	for _, p := range simmer.simmedPlays.plays {
		for _, s := range p.ScoreStatsNoLock() {
			is.True(s.Iterations() > 0)
		}
	}
}
