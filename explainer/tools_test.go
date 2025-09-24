package explainer

import (
	"testing"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/matryer/is"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func TestGetFuturePlayMetadata(t *testing.T) {
	is := is.New(t)
	cfg := config.DefaultConfig()
	// This is tested indirectly via TestExplainGame
	bpos := "PEC4D3QUAY/1EUOI2UG4V1/2MINX1ER3TI1/5UNTANGLeD1/8I3N2/6RONZ2T2/5HOPE1T1H2/3COMBES1R4/2BOW5A4/3V1ASEItIES2/3EEW4T4/3DAD9/5L9/15/15 AFLOORY/AAFGNRT 337/281 0 lex CSW24;"
	g, err := cgp.ParseCGP(config.DefaultConfig(), bpos)
	is.NoErr(err)
	g.RecalculateBoard() // to calculate cross-scores etc.
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)

	leavesFile := ""
	conf := &bot.BotConfig{Config: *cfg, LeavesFile: leavesFile}

	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_HASTY_BOT)
	is.NoErr(err)
	an := NewAnalyzer()
	an.game = tp
	an.winningPlay = "G12 OO"
	an.winningStats = `### Our follow-up play
Play                Needed Draw   Score    Count    % of time
 L9 L(E)AFERY       {E}           38       696      13.24
 L9 L(E)AFY                       26       575      10.94
 1A (PEC)KY         {K}           39       273      5.19
I12 LYRA                          33       186      3.54
H13 RAJ             {J}           33       164      3.12
 L9 R(E)IFY         {I}           26       105      2.00
14G FRIARLY         {IR}          71       85       1.62
 1A (PECK)Y                       22       60       1.14
I13 FLY                           35       60       1.14
I13 FAY                           34       56       1.07
14G FRAY                          24       53       1.01
14B FLAYS           {S}           37       45       0.86
15A FAY                           41       42       0.80
14G FLAYERS         {ES}          77       41       0.78
H13 YAK             {K}           36       39       0.74`

	f, err := an.GetFuturePlayMetadata("L9 L(E)AFERY")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "L9 L(E)AFERY",
		NeededDraw:         []string{"E"},
		Score:              38,
		ProbabilityPercent: 13.24,
		IsBingo:            false,
		RequiresOtherPlay:  "none",
	})

	f, err = an.GetFuturePlayMetadata("L9 L(E)AFY")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "L9 L(E)AFY",
		NeededDraw:         []string{},
		Score:              26,
		ProbabilityPercent: 10.94,
		IsBingo:            false,
		RequiresOtherPlay:  "none",
	})

	f, err = an.GetFuturePlayMetadata("14G FLAYERS")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "14G FLAYERS",
		NeededDraw:         []string{"E", "S"},
		Score:              77,
		ProbabilityPercent: 0.78,
		IsBingo:            true,
		RequiresOtherPlay:  "requires us to play G12 OO first",
	})

	f, err = an.GetFuturePlayMetadata("I12 LYRA")
	is.NoErr(err)
	is.Equal(f, &FuturePlayMetadata{
		Play:               "I12 LYRA",
		NeededDraw:         []string{},
		Score:              33,
		ProbabilityPercent: 3.54,
		IsBingo:            false,
		RequiresOtherPlay:  "requires opponent play",
	})
}
