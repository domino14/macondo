package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/avast/retry-go"
	"github.com/aws/aws-lambda-go/lambda"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	aibot "github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/bot"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var cfg *config.Config
var nc *nats.Conn

const HardTimeLimit = 180 // max time per turn in seconds

func HandleRequest(ctx context.Context, evt bot.LambdaEvent) (string, error) {
	// Return something but we have to block till we're done.

	logger := log.With().
		Str("gameID", evt.GameID).
		Logger()

	g, err := cgp.ParseCGP(cfg, evt.CGP)
	if err != nil {
		return "", err
	}
	botTime := HardTimeLimit
	if tmr, ok := g.Opcodes["tmr"]; ok {
		tmrs := strings.Split(tmr, "/")
		if len(tmrs) > 0 {
			botTime, err = strconv.Atoi(tmrs[0])
			if err != nil {
				return "", err
			}
			botTime /= 1000 // convert from milliseconds
		}
	} else {
		logger.Warn().Msg("no timer found in CGP")
	}

	// Estimate: bot plays 4.25 tiles per turn. Divide bag by 2 because bot will only
	// play half the tiles left.
	ourCount := int(g.RackFor(g.PlayerOnTurn()).NumTiles())
	unseen := g.Bag().TilesRemaining() + int(g.RackFor(g.NextPlayer()).NumTiles())
	actuallyInBag := max(unseen-game.RackTileLimit, 0)

	estimatedTurnsLeft := (float64(actuallyInBag)/2 + float64(ourCount)) / 4.25
	maxTimeShouldTake := min(float64(botTime-5)/estimatedTurnsLeft, HardTimeLimit)
	logger.Info().Float64("bot-estimated-turns-left", estimatedTurnsLeft).
		Int("inbag", actuallyInBag).
		Str("cgp", evt.CGP).
		Float64("max-time-should-take", maxTimeShouldTake).Msg("time-management")

	var cancel context.CancelFunc
	ctx, cancel = context.WithTimeout(ctx, time.Duration(maxTimeShouldTake)*time.Second)

	lexicon := g.History().Lexicon
	if lexicon == "" {
		lexicon = cfg.DefaultLexicon
		logger.Info().Msgf("cgp file had no lexicon, so using default lexicon %v",
			lexicon)
	}
	conf := &aibot.BotConfig{Config: *cfg, MinSimPlies: 5, UseOppRacksInAnalysis: true}
	tp, err := aibot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_BotCode(evt.BotType))
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

	gevt := tp.EventFromMove(m)
	resp := &pb.BotResponse{
		Response: &pb.BotResponse_Move{Move: gevt},
		GameId:   evt.GameID,
	}
	data, err := proto.Marshal(resp)
	if err != nil {
		return "", err
	}
	if evt.ReplyChannel != "" {
		logger.Info().Msg("move-success-sending-via-nats")
		err = retry.Do(
			func() error {
				_, err := nc.Request(evt.ReplyChannel, data, 3*time.Second)
				if err != nil {
					return err
				}
				// We're just waiting for an acknowledgement. The actual
				// data doesn't matter.
				return nil
			},
			retry.DelayType(func(n uint, err error, config *retry.Config) time.Duration {
				logger.Err(err).Uint("n", n).
					Msg("did-not-receive-ack-try-again")
				return retry.BackOffDelay(n, err, config)
			}),
		)
		if err != nil {
			logger.Err(err).Msg("bot-move-failed")
		}
	}
	logger.Info().Msg("exiting-fn")
	return m.ShortDescription(), nil
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

	nc, err = nats.Connect(cfg.NatsURL)
	if err != nil {
		log.Fatal().AnErr("natsConnectErr", err).Msg(":(")
	}

	lambda.Start(HandleRequest)
}
