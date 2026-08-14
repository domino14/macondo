package explainer

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/matryer/is"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// A real follow-up table, from a position with ?AEIKUW on our rack. Note the
// lowercase letters: those plays use the blank. ZWIEBACK and SAWLIKE can each
// be made in more than one way, so they are grouped.
const blankyFollowups = `### Our follow-up play
Play                Needed Draw   Score    Count    % of time
 7F W(E)AK                        23       273      11.11
 1H (Z)WIEBACK      {B|C|?}       116-134  165      6.72
  -  1H (Z)WIEBAcK  {B}           134      70       2.85
  -  1H (Z)WIEbACK  {C}           125      68       2.77
  -  1H (Z)WIEbAcK  {?}           116      27       1.10
 I8 SAWLIKE         {S|L}         80-81    146      5.94
  -  I8 SAWlIKE     {S}           81       82       3.34
  -  I8 sAWLIKE     {L}           80       64       2.60
F12 WAKE                          30       85       3.46
 I8 sKIWEAR         {R}           86       78       3.17
 2D KIT(TI)WAkE     {T}           69       55       2.24
12A (D)ISCOMBOB(U)LATE {E}        42       25       1.02
Bingo probability: 34.83%
`

func TestGetPlayMetadata(t *testing.T) {
	is := is.New(t)
	cfg := config.DefaultConfig()

	// Test with opening position
	bpos := "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 ADOPQRR/ 0/0 0 lex NWL23;"
	g, err := cgp.ParseCGP(cfg, bpos)
	is.NoErr(err)
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)

	leavesFile := ""
	conf := &bot.BotConfig{Config: *cfg, LeavesFile: leavesFile}

	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_HASTY_BOT)
	is.NoErr(err)
	an := NewAnalyzer()
	an.game = tp
	an.SetConfig(cfg)

	// Test exchange move with parentheses format (as reported in issue #425)
	md, err := an.GetPlayMetadata("(exch Q)")
	is.NoErr(err)
	is.Equal(md.Play, "(exch Q)")
	is.Equal(md.Score, 0)
	is.Equal(md.TilesUsed, 1) // Exchanging 1 tile
	is.Equal(md.IsBingo, false)

	// Test exchange move without parentheses
	md, err = an.GetPlayMetadata("exch Q")
	is.NoErr(err)
	is.Equal(md.Play, "exch Q")
	is.Equal(md.Score, 0)
	is.Equal(md.TilesUsed, 1)

	// Test pass move
	md, err = an.GetPlayMetadata("pass")
	is.NoErr(err)
	is.Equal(md.Play, "pass")
	is.Equal(md.Score, 0)
	is.Equal(md.TilesUsed, 0)
}

// newFollowupAnalyzer sets up an analyzer on a position whose best play is
// G12 OO, leaving AFLRY.
func newFollowupAnalyzer(t *testing.T) *Analyzer {
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
	return an
}

func TestGetFuturePlayMetadata(t *testing.T) {
	is := is.New(t)
	an := newFollowupAnalyzer(t)
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

// However the model names a grouped play, it gets the whole family back, so
// it can't mistake one route's chance for the chance of making the play.
func TestGroupedPlayAlwaysReturnsEveryWay(t *testing.T) {
	is := is.New(t)
	an := newFollowupAnalyzer(t)
	an.winningStats = `### Our follow-up play
Play                Needed Draw   Score    Count    % of time
 L9 L(E)AFERY       {E|E?}        34-38    750      14.26
  -  L9 L(E)AFERY   {E}           38       696      13.24
  -  L9 L(E)AFERy   {E?}          34       54       1.02
 1A (PEC)KY         {K}           39       273      5.19
`
	tool := NewGetOurFuturePlayMetadataTool(an)

	for _, q := range []string{"L9 L(E)AFERY", "L9 L(E)AFERy", "L9 LEAFERY"} {
		res, err := tool.Execute(context.Background(), `{"play_string": "`+q+`"}`)
		is.NoErr(err)

		var got struct {
			Note       string `json:"note"`
			AskedAbout string `json:"asked_about"`
			Family     struct {
				Play              string   `json:"play"`
				CombinedPercent   float64  `json:"combined_probability_percent"`
				ScoreRange        string   `json:"score_range"`
				NeededDrawOptions []string `json:"needed_draw_options"`
			} `json:"family"`
			Ways []struct {
				Play               string  `json:"play"`
				Score              int     `json:"score"`
				ProbabilityPercent float64 `json:"probability_percent"`
			} `json:"ways"`
		}
		is.NoErr(json.Unmarshal([]byte(res), &got))

		is.Equal(got.Family.Play, "L9 L(E)AFERY")
		is.Equal(got.Family.CombinedPercent, 14.26) // not 13.24, the top way's share
		is.Equal(got.Family.ScoreRange, "34-38")
		is.Equal(got.Family.NeededDrawOptions, []string{"E", "E?"})
		is.Equal(len(got.Ways), 2)
		is.Equal(got.Ways[1].Play, "L9 L(E)AFERy")
		is.Equal(got.Ways[1].Score, 34)
		is.True(strings.Contains(got.Note, "combined_probability_percent"))
	}

	// naming one route reports which one was asked about
	res, err := tool.Execute(context.Background(), `{"play_string": "L9 L(E)AFERy"}`)
	is.NoErr(err)
	is.True(strings.Contains(res, `"asked_about":"L9 L(E)AFERy"`))

	// a play with only one way keeps the flat shape
	res, err = tool.Execute(context.Background(), `{"play_string": "1A (PEC)KY"}`)
	is.NoErr(err)
	is.True(!strings.Contains(res, `"ways"`))
	is.True(strings.Contains(res, `"probability_percent":5.19`))
}

func TestParseFollowupFamilies(t *testing.T) {
	is := is.New(t)
	an := NewAnalyzer()
	an.winningStats = blankyFollowups

	fams, err := an.parseFollowupFamilies()
	is.NoErr(err)
	is.Equal(len(fams), 7)

	is.Equal(fams[0].row, followupRow{play: "7F W(E)AK", scoreLow: 23, scoreHigh: 23, percent: 11.11})
	is.Equal(len(fams[0].variants), 0)

	// a grouped play: combined count and percentage, score range, and the
	// alternative draws that unlock it
	is.Equal(fams[1].row, followupRow{play: "1H (Z)WIEBACK", neededDraw: "B|C|?",
		scoreLow: 116, scoreHigh: 134, percent: 6.72})
	is.Equal(len(fams[1].variants), 3)
	is.Equal(fams[1].variants[2], followupRow{play: "1H (Z)WIEbAcK", neededDraw: "?",
		scoreLow: 116, scoreHigh: 116, percent: 1.10})

	is.Equal(fams[4].row, followupRow{play: "I8 sKIWEAR", neededDraw: "R",
		scoreLow: 86, scoreHigh: 86, percent: 3.17})
	// a play too long for the play column still parses
	is.Equal(fams[6].row, followupRow{play: "12A (D)ISCOMBOB(U)LATE", neededDraw: "E",
		scoreLow: 42, scoreHigh: 42, percent: 1.02})
}

func TestMatchFollowup(t *testing.T) {
	is := is.New(t)
	an := NewAnalyzer()
	an.winningStats = blankyFollowups
	fams, err := an.parseFollowupFamilies()
	is.NoErr(err)

	match := func(q string) (string, string) {
		fam, variant := matchFollowup(fams, q)
		if fam == nil {
			return "", ""
		}
		if variant == nil {
			return fam.row.play, ""
		}
		return fam.row.play, variant.play
	}

	// exact match on a play with only one way to make it
	fam, variant := match("I8 sKIWEAR")
	is.Equal(fam, "I8 sKIWEAR")
	is.Equal(variant, "")

	// leading whitespace, as it appears in the table
	fam, _ = match(" I8 sKIWEAR")
	is.Equal(fam, "I8 sKIWEAR")

	// the model uppercased the blank
	fam, _ = match("I8 SKIWEAR")
	is.Equal(fam, "I8 sKIWEAR")

	// the grouped name resolves to the family, with no single way named
	fam, variant = match("1H (Z)WIEBACK")
	is.Equal(fam, "1H (Z)WIEBACK")
	is.Equal(variant, "")

	// naming one way still identifies the family it belongs to
	fam, variant = match("1H (Z)WIEbACK")
	is.Equal(fam, "1H (Z)WIEBACK")
	is.Equal(variant, "1H (Z)WIEbACK")

	// the model also guessed at the playthrough parens
	fam, _ = match("I8 S(K)IWEAR")
	is.Equal(fam, "I8 sKIWEAR")

	fam, _ = match("12A (D)ISCOMBOB(U)LATE")
	is.Equal(fam, "12A (D)ISCOMBOB(U)LATE")

	fam, _ = match("8H QUIXOTIC")
	is.Equal(fam, "")
}

func TestFuturePlayNotFoundIsNotAnError(t *testing.T) {
	is := is.New(t)
	an := NewAnalyzer()
	an.winningStats = blankyFollowups

	_, err := an.LookupFuturePlay("8H QUIXOTIC")
	var notFound *PlayNotFoundError
	is.True(errors.As(err, &notFound))
	is.Equal(len(notFound.Available), 12) // 7 families plus 5 individual ways

	// The tool hands the model a normal result listing the plays it can ask
	// about, rather than an error it will just retry.
	tool := NewGetOurFuturePlayMetadataTool(an)
	res, err := tool.Execute(context.Background(), `{"play_string": "8H QUIXOTIC"}`)
	is.NoErr(err)
	is.True(strings.Contains(res, "is not one of the follow-up plays"))
	is.True(strings.Contains(res, "- I8 sKIWEAR"))
	is.True(strings.Contains(res, "- 1H (Z)WIEbAcK"))
}
