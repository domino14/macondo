package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-lambda-go/lambda"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var cfg *config.Config

const HardTimeLimit = 180 // max time per turn in seconds

type MyEvent struct {
	CGP          string `json:"cgp"`
	BotType      int    `json:"botType"`
	ReplyChannel string `json:"replyChannel"`
}

func HandleRequest(ctx context.Context, evt MyEvent) (string, error) {
	// Return something but we have to block till we're done.
	g, err := cgp.ParseCGP(cfg, evt.CGP)
	if err != nil {
		return "", err
	}
	botTime := HardTimeLimit
	if tmr, ok := g.Opcodes["tmr"]; ok {
		tmrs := strings.Split(tmr, "/")
		if len(tmrs) > 0 {
			botTime, err = strconv.Atoi(tmrs[0])
			botTime /= 1000 // convert from milliseconds
		}
	} else {
		log.Warn().Msg("no timer found in CGP")
	}

	// Estimate: bot plays 4.25 tiles per turn. Divide bag by 2 because bot will only
	// play half the tiles left.
	estimatedTurnsLeft := float64(g.Bag().TilesRemaining()/2+int(g.RackFor(g.PlayerOnTurn()).NumTiles())) / 4.25
	maxTimeShouldTake := min(float64(botTime-5)/estimatedTurnsLeft, HardTimeLimit)
	log.Info().Float64("bot-estimated-turns-left", estimatedTurnsLeft).
		Int("inbag", g.Bag().TilesRemaining()).
		Float64("max-time-should-take", maxTimeShouldTake).Msg("time-management")

	var cancel context.CancelFunc
	ctx, cancel = context.WithTimeout(ctx, time.Duration(maxTimeShouldTake)*time.Second)

	lexicon := g.History().Lexicon
	if lexicon == "" {
		lexicon = cfg.DefaultLexicon
		log.Info().Msgf("cgp file had no lexicon, so using default lexicon %v",
			lexicon)
	}
	conf := &bot.BotConfig{Config: *cfg, MinSimPlies: 5, UseOppRacksInAnalysis: true}
	tp, err := bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_BotCode(evt.BotType))
	if err != nil {
		cancel()
		return "", err
	}
	tp.SetBackupMode(game.InteractiveGameplayMode)
	tp.SetStateStackLength(1)
	tp.RecalculateBoard()

	m, err := tp.BestPlay(ctx)
	if err != nil {
		cancel()
		return "", err
	}
	// It doesn't fully matter what we return here. We will be sending the play
	// on the reply channel in NATS, and that's what liwords should hopefully
	// be listening to.
	cancel()
	return m.ShortDescription(), nil

	// Add bot data structures
	// c, err := equity.NewCombinedStaticCalculator(
	// 	tp.LexiconName(),
	// 	cfg, "", equity.PEGAdjustmentFilename)
	// if err != nil {
	// 	return "", err
	// }

	// gd, err := kwg.Get(cfg, tp.LexiconName())
	// if err != nil {
	// 	return "", err
	// }

	// return fmt.Sprintf("Hello %s!", name.Name), nil
}

func main() {
	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}
	exPath := filepath.Dir(ex)

	cfg = &config.Config{}
	args := os.Args[1:]
	cfg.Load(args)
	log.Info().Msgf("Loaded config: %v", cfg)
	cfg.AdjustRelativePaths(exPath)
	if cfg.Debug {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	lambda.Start(HandleRequest)
}
