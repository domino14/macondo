package rangefinder

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var DefaultConfig = config.DefaultConfig()

func defaultSimCalculators(lexiconName string) []equity.EquityCalculator {
	c, err := equity.NewCombinedStaticCalculator(
		lexiconName, DefaultConfig, "", equity.PEGAdjustmentFilename)
	if err != nil {
		panic(err)
	}
	return []equity.EquityCalculator{c}
}

func TestInferTilePlay(t *testing.T) {

	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)

	game.SetRackFor(0, tilemapping.RackFromString("PHEW", game.Alphabet()))
	_, err = game.PlayScoringMove("H6", "PHEW", true)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, DefaultConfig)
	// Single-stage: this test covers the posterior and its displays, and
	// giving round 0 the whole budget keeps what it sees independent of how
	// many refinement rounds happened to fit. TestInferRefinementRounds
	// covers the loop.
	rangeFinder.SetMaxRounds(0)

	f, err := os.Create("/tmp/inferlog")
	is.NoErr(err)
	defer f.Close()
	rangeFinder.logStream = f

	rangeFinder.PrepareFinder(nil)
	timeout, cancel := context.WithTimeout(
		context.Background(), 10*time.Second)
	defer cancel()

	err = rangeFinder.Infer(timeout)
	is.NoErr(err)
	fmt.Println(rangeFinder.iterationCount)
	fmt.Println(len(rangeFinder.inference.InferredRacks))
	fmt.Println("analyze inferences")
	fmt.Println(rangeFinder.AnalyzeInferences(true))
	fmt.Println(rangeFinder.AnalyzeInferences(false))

	// Query a specific leave from the complete posterior.
	is.True(rangeFinder.inference.Complete)

	// SetMaxRounds(0) means exactly that: everything measured here came from
	// round 0's prior sampling.
	is.Equal(rangeFinder.refinedCount, 0)
	is.Equal(len(rangeFinder.roundLog), 0)

	top := rangeFinder.inference.InferredRacks[0]
	topStr := tilemapping.MachineWord(top.Leave).UserVisible(game.Alphabet())
	analysis, err := rangeFinder.AnalyzeLeave(topStr)
	is.NoErr(err)
	fmt.Println(analysis)
	is.True(strings.Contains(analysis, "posterior"))

	// An imputed leave gets the full calculation walkthrough: per-subleave φ
	// table plus the chain from prior and ℓ̂ to the normalized weight. A
	// measured leave gets its own prior × mean likelihood chain.
	var imputedStr, measuredStr string
	for _, ir := range rangeFinder.inference.InferredRacks {
		str := tilemapping.MachineWord(ir.Leave).UserVisible(game.Alphabet())
		if ml := rangeFinder.measured[leaveKey(ir.Leave)]; ml == nil || ml.count == 0 {
			if imputedStr == "" {
				imputedStr = str
			}
		} else if measuredStr == "" {
			measuredStr = str
		}
		if imputedStr != "" && measuredStr != "" {
			break
		}
	}
	if measuredStr != "" {
		measuredAnalysis, err := rangeFinder.AnalyzeLeave(measuredStr)
		is.NoErr(err)
		fmt.Println(measuredAnalysis)
		is.True(strings.Contains(measuredAnalysis, "[MEASURED ×"))
		is.True(strings.Contains(measuredAnalysis, "measured: evaluated"))
		is.True(strings.Contains(measuredAnalysis, "weight = prior × mean likelihood ÷ max"))
		is.True(strings.Contains(measuredAnalysis, "imputation model comparison"))
		is.True(strings.Contains(measuredAnalysis, "measured/imputed likelihood ratio"))
	}
	if imputedStr != "" {
		imputedAnalysis, err := rangeFinder.AnalyzeLeave(imputedStr)
		is.NoErr(err)
		fmt.Println(imputedAnalysis)
		is.True(strings.Contains(imputedAnalysis, "[IMPUTED]"))
		is.True(strings.Contains(imputedAnalysis, "Σφ"))
		is.True(strings.Contains(imputedAnalysis, "weight = prior × ℓ̂ ÷ max"))
	}

	// Lowercase input is treated as regular (capitalized) tiles, not as
	// blank designations — `infer leave zit` used to panic.
	lower, err := rangeFinder.AnalyzeLeave(strings.ToLower(topStr))
	is.NoErr(err)
	is.Equal(lower, analysis)

	// Wrong leave length errors out.
	_, err = rangeFinder.AnalyzeLeave("A")
	is.True(err != nil)
}

// TestInferRefinementRounds covers the measure–impute–recalibrate loop
// end-to-end: leaves drawn from the imputed posterior, evaluated with real
// mini-sims, and folded back in as exact weights.
//
// The leave space has to be large for the assertions to be deterministic. In
// a small one, round 0 alone can measure nearly everything worth measuring,
// and refinement then correctly does nothing — it stops on "unmeasured mass
// below the floor" or finds no candidates left at all, and refinedCount is
// legitimately 0. Four tiles played out of a full bag leaves thousands of
// candidates, so there is always something for the loop to draw.
func TestInferRefinementRounds(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	lex := "NWL23"
	players := []*macondo.PlayerInfo{
		{Nickname: "p1", RealName: "Alice"},
		{Nickname: "p2", RealName: "Bob"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout,
		"English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	g, err := game.NewGame(rules, players)
	is.NoErr(err)
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, tilemapping.RackFromString("HELPXYZ", g.Alphabet()))
	_, err = g.PlayScoringMove("H8", "HELP", true)
	is.NoErr(err)

	rf := &RangeFinder{}
	rf.Init(g, defaultSimCalculators(lex), DefaultConfig)
	is.NoErr(rf.PrepareFinder(nil))
	is.Equal(rf.inference.RackLength, 3)
	is.True(countMultisets(rf.inferenceBagMap, rf.inference.RackLength) > 1000)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	is.NoErr(rf.Infer(ctx))
	fmt.Println(rf.AnalyzeInferences(false))

	is.True(rf.inference.Complete)
	is.True(rf.refinedCount > 0)
	is.True(len(rf.roundLog) > 0)

	for _, st := range rf.roundLog {
		is.True(st.drawn > 0)
		is.True(st.evaluated > 0)
		is.True(st.distinct <= st.drawn) // repeats fold into the weight
		is.True(st.seLogRatio >= 0)
		is.True(!math.IsNaN(st.logRatio))
		is.True(!math.IsNaN(st.seLogRatio))
		is.True(st.unmeasured >= 0 && st.unmeasured <= 1)
		// A batch too small to have a meaningful interval must never declare
		// convergence: with one leaf the ratio fits exactly, leaving a zero
		// residual and a zero standard error.
		if st.converged {
			is.True(st.evaluated >= refineMinBatch)
		}
	}

	// Refinement only ever draws leaves that are still unmeasured, so every
	// evaluation it makes lands on a distinct leave stamped with its round.
	refined := 0
	for _, ml := range rf.measured {
		if ml.round > 0 {
			is.True(ml.count > 0)
			is.True(ml.sumU > 0)
			refined++
		}
	}
	is.Equal(refined, rf.refinedCount)

	// A refined leave carries the model's pre-measurement prediction, which
	// is the only out-of-sample estimate for it that survives the refit.
	predicted := 0
	for _, ml := range rf.measured {
		if ml.round > 0 && ml.predicted > 0 {
			predicted++
		}
	}
	is.True(predicted > 0)

	// Those leaves carry exact weights in the posterior, and the display says
	// which round found them. Not every refined leave has a row: one measured
	// to be impossible is excluded outright, and one whose weight lands under
	// the negligible-mass cutoff is dropped, so this is a subset.
	rows := rf.rankedRacks()
	tagged := 0
	for _, row := range rows {
		if row.measured && row.round > 0 {
			is.True(strings.Contains(row.source(), roundLabel(row.round)))
			tagged++
		}
	}
	is.True(tagged > 0)
	is.True(tagged <= rf.refinedCount)
}

func TestInferExchange(t *testing.T) {

	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "Joel", RealName: "Joel"},
		{Nickname: "Nigel", RealName: "Nigel"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)

	game.SetRackFor(0, tilemapping.RackFromString("AENSTUU", game.Alphabet()))

	uu, err := tilemapping.ToMachineLetters("UU", game.Alphabet())
	is.NoErr(err)
	aenst, err := tilemapping.ToMachineLetters("AENST", game.Alphabet())
	is.NoErr(err)

	m := move.NewExchangeMove(uu, aenst, game.Alphabet())
	err = game.PlayMove(m, true, 0)
	is.NoErr(err)

	calcs := defaultSimCalculators(lex)

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, DefaultConfig)

	f, err := os.Create("/tmp/inferlog")
	is.NoErr(err)
	defer f.Close()
	rangeFinder.logStream = f

	// Nigel's rack was AELNOQT.
	aelnoqt, err := tilemapping.ToMachineLetters("AELNOQT", game.Alphabet())
	is.NoErr(err)
	err = rangeFinder.PrepareFinder(aelnoqt)
	is.NoErr(err)
	timeout, cancel := context.WithTimeout(
		context.Background(), 5*time.Second)
	defer cancel()

	err = rangeFinder.Infer(timeout)
	is.NoErr(err)
	fmt.Println(rangeFinder.iterationCount)
	fmt.Println(len(rangeFinder.inference.InferredRacks))
	fmt.Println("analyze inferences")
	fmt.Println(rangeFinder.AnalyzeInferences(true))
	fmt.Println(rangeFinder.AnalyzeInferences(false))
}

func BenchmarkInferTilePlay(b *testing.B) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	lex := "NWL18"
	players := []*macondo.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lex, board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		b.Fatal(err)
	}
	g, err := game.NewGame(rules, players)
	if err != nil {
		b.Fatal(err)
	}
	g.StartGame()
	g.SetPlayerOnTurn(0)
	g.SetRackFor(0, tilemapping.RackFromString("PHEW", g.Alphabet()))
	if _, err = g.PlayScoringMove("H6", "PHEW", true); err != nil {
		b.Fatal(err)
	}

	calcs := defaultSimCalculators(lex)
	rangeFinder := &RangeFinder{}
	rangeFinder.Init(g, calcs, DefaultConfig)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		if err = rangeFinder.PrepareFinder(nil); err != nil {
			b.Fatal(err)
		}
		b.StartTimer()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err = rangeFinder.Infer(ctx); err != nil {
			cancel()
			b.Fatal(err)
		}
		cancel()
	}

	b.ReportMetric(float64(rangeFinder.simCount.Load())/b.Elapsed().Seconds(), "sims/sec")
}

func TestInferSingle(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.DebugLevel)

	players := []*macondo.PlayerInfo{
		{Nickname: "JD", RealName: "Jesse"},
		{Nickname: "cesar", RealName: "César"},
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL20", board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)
	game, err := game.NewGame(rules, players)
	is.NoErr(err)

	// This will start the game and deal a random rack to players:
	game.StartGame()
	game.SetPlayerOnTurn(0)

	game.SetRackFor(0, tilemapping.RackFromString("PHEW", game.Alphabet()))
	_, err = game.PlayScoringMove("8F", "PHEW", true)
	is.NoErr(err)

	is.Equal(game.PlayerOnTurn(), 1)

	calcs := defaultSimCalculators("NWL20")

	rangeFinder := &RangeFinder{}
	rangeFinder.Init(game, calcs, DefaultConfig)
	rangeFinder.PrepareFinder(nil)

	is.Equal(rangeFinder.gameCopies[0].PlayerOnTurn(), 0)

	_, err = rangeFinder.inferSingle(0, 0, nil)
	is.NoErr(err)
}
