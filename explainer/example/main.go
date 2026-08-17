// Command example runs a real simulation on a fixed position and asks the
// configured LLM to explain the best play. It is the smallest end-to-end
// exercise of the explainer outside the shell.
//
// Set MACONDO_NO_LLM=1 to print the assembled prompt instead of calling an
// API, which is the quickest way to see what a position actually sends.
package main

import (
	"context"
	"fmt"
	stdlog "log"
	"os"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/explainer"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// The position from docs/manual/ai-explainability.md: we're 40 points down
// with ACDEPQU, and the best play sets up 15G PREAD(JUST).
const position = "COXA2B2ROPING/3T1DINGY5/3t2ZO1AINEE1/3A3V2FOWTH/3S3A2FOE1U/" +
	"1LEKE2T1MY3T/2RELATIVE4I/ARED3N5BA/7g5R1/12WO1/12HA1/12ID1/12ME1/13N1/" +
	"11JUST ACDEPQU/ 400/440 0 lex CSW24;"

const (
	numPlays   = 30
	plies      = 5
	iterations = 1500
)

func main() {
	ctx := context.Background()

	macondoConfig := config.DefaultConfig()
	if err := macondoConfig.Load(nil); err != nil {
		stdlog.Fatalf("Failed to load config: %v", err)
	}

	tp, err := newPlayer(macondoConfig, position)
	if err != nil {
		stdlog.Fatalf("Failed to set up the position: %v", err)
	}

	simmer, simStats, err := runSim(macondoConfig, tp)
	if err != nil {
		stdlog.Fatalf("Failed to simulate: %v", err)
	}
	defer simmer.CleanupTempFile()

	if os.Getenv("MACONDO_NO_LLM") != "1" {
		provider := macondoConfig.GetString("genai-provider")
		fmt.Printf("Generating explanation using the %s provider...\n", provider)
	}

	service := explainer.NewService(macondoConfig)
	result, err := service.Explain(ctx, &explainer.ExplainInput{
		Game: tp, Simmer: simmer, SimStats: simStats,
	})
	if err != nil {
		stdlog.Fatalf("Failed to generate explanation: %v", err)
	}

	fmt.Printf("\nConcept cards this position pulled in: %v\n", result.Concepts)
	fmt.Println("\n=== Explanation ===")
	fmt.Println(result.Explanation)

	if result.InputTokens > 0 {
		fmt.Printf("\nInput tokens: %d\n", result.InputTokens)
		fmt.Printf("Output tokens: %d\n", result.OutputTokens)
	}
}

func newPlayer(cfg *config.Config, cgpStr string) (*bot.BotTurnPlayer, error) {
	g, err := cgp.ParseCGP(cfg, cgpStr)
	if err != nil {
		return nil, err
	}
	// Without this the cross-sets are empty and move generation finds almost
	// nothing on a loaded board.
	g.RecalculateBoard()
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	return bot.NewBotTurnPlayerFromGame(g.Game, &bot.BotConfig{Config: *cfg}, pb.BotRequest_HASTY_BOT)
}

func runSim(cfg *config.Config, tp *bot.BotTurnPlayer) (
	*montecarlo.Simmer, *stats.SimStats, error) {

	calc, err := equity.NewCombinedStaticCalculator(tp.LexiconName(), cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, nil, err
	}
	simmer := &montecarlo.Simmer{}
	simmer.Init(tp.Game, []equity.EquityCalculator{calc}, calc, cfg)
	// The follow-up tables and the lane breakdown both come out of the heat
	// map log, so the sim has to be told to keep one.
	if err := simmer.SetCollectHeatmap(true); err != nil {
		return nil, nil, err
	}
	if err := simmer.PrepareSim(plies, tp.GenerateMoves(numPlays)); err != nil {
		return nil, nil, err
	}
	fmt.Printf("Simulating %d plays for %d iterations...\n", numPlays, iterations)
	simmer.SimSingleThread(iterations, plies)

	return simmer, stats.NewSimStats(simmer, tp), nil
}
