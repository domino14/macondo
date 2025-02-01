package bot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/avast/retry-go/v4"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/lambda"
	"github.com/aws/aws-sdk-go-v2/service/lambda/types"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/board"
	mcfg "github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

type LambdaEvent struct {
	CGP          string `json:"cgp"`
	BotType      int    `json:"botType"`
	ReplyChannel string `json:"replyChannel"`
	GameID       string `json:"gameId"`
}

var (
	ErrNeedSimmingBot = errors.New("need simming bot")
)

const (
	StarPlayThreshold = 10.0 // equity
)

func debugWriteln(msg string) {
	io.WriteString(os.Stderr, msg)
	io.WriteString(os.Stderr, "\n")
}

type Bot struct {
	config  *mcfg.Config
	options *turnplayer.GameOptions

	game         *bot.BotTurnPlayer
	awscfg       aws.Config
	lambdaClient *lambda.Client
}

func NewBot(cfg *mcfg.Config, options *turnplayer.GameOptions) *Bot {
	bot := &Bot{}
	bot.config = cfg
	bot.options = options
	bot.game = nil

	ctx := context.Background()
	var err error
	bot.awscfg, err = config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Err(err).Msg("loading-aws-default-cfg")
	} else {
		bot.lambdaClient = lambda.NewFromConfig(bot.awscfg)
	}

	return bot
}

func (b *Bot) newGame() error {
	players := []*pb.PlayerInfo{
		{Nickname: "self", RealName: "Macondo Bot"},
		{Nickname: "opponent", RealName: "Arthur Dent"},
	}

	conf := &bot.BotConfig{Config: *b.config}
	game, err := bot.NewBotTurnPlayer(conf, b.options, players, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return err
	}
	b.game = game
	return nil
}

func errorResponse(message string, err error) *pb.BotResponse {
	msg := message
	if err != nil {
		msg = fmt.Sprintf("%s: %s", msg, err.Error())
	}
	return &pb.BotResponse{
		Response: &pb.BotResponse_Error{Error: msg},
	}
}

func (bot *Bot) Deserialize(data []byte) (*game.Game, *pb.BotRequest, error) {
	req := &pb.BotRequest{}
	err := proto.Unmarshal(data, req)
	if err != nil {
		return nil, nil, err
	}
	history := req.GameHistory
	boardLayout, ldName, variant := game.HistoryToVariant(history)
	rules, err := game.NewBasicGameRules(bot.config, history.Lexicon, boardLayout, ldName, game.CrossScoreAndSet, variant)
	if err != nil {
		return nil, nil, err
	}
	nturns := len(history.Events)
	ng, err := game.NewFromHistory(history, rules, 0)
	if err != nil {
		return nil, nil, err
	}
	ng.PlayToTurn(nturns)
	// debugWriteln(ng.ToDisplayText())
	return ng, req, nil
}

func evalSingleMove(g *bot.BotTurnPlayer, evtIdx int) *pb.SingleEvaluation {
	evts := g.History().Events
	playedEvt := evts[evtIdx]

	g.PlayToTurn(evtIdx)
	moves := g.GenerateMoves(100000)
	// find the played move in the list of moves
	topEquity := moves[0].Equity()
	topIsBingo := moves[0].TilesPlayed() == game.RackTileLimit && moves[0].Action() == move.MoveTypePlay
	foundEquity := float64(0)
	playedBingo := false
	hasStarPlay := false
	if len(moves) > 1 && moves[1].Equity() < topEquity-StarPlayThreshold {
		hasStarPlay = true
	}
	missedStarPlay := false
	onturnRack := g.RackLettersFor(g.PlayerOnTurn())
	for idx, m := range moves {
		evt := g.EventFromMove(m, onturnRack)
		if evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE || evt.Type == pb.GameEvent_EXCHANGE {
			if evt.PlayedTiles == playedEvt.PlayedTiles &&
				evt.Exchanged == playedEvt.Exchanged &&
				evt.Score == playedEvt.Score {

				if idx > 0 && hasStarPlay {
					// A star play is a stand-alone play that is better than anything else.
					missedStarPlay = true
				}
				// Same move
				foundEquity = m.Equity()
				playedBingo = m.TilesPlayed() == game.RackTileLimit && m.Action() == move.MoveTypePlay
				break
			}
		}
	}
	// if we don't find the move it means the user played a phony. This is ok. In the
	// absence of a better metric, we can evaluate the phony as a 0.
	return &pb.SingleEvaluation{
		EquityLoss:       foundEquity - topEquity,
		TopIsBingo:       topIsBingo,
		MissedBingo:      topIsBingo && !playedBingo,
		PossibleStarPlay: hasStarPlay,
		MissedStarPlay:   missedStarPlay,
	}
}

func (bot *Bot) evaluationResponse(req *pb.EvaluationRequest) *pb.BotResponse {

	evts := bot.game.History().Events
	players := bot.game.History().Players
	evals := []*pb.SingleEvaluation{}

	for idx, evt := range evts {
		evtNickname := players[evt.PlayerIndex].Nickname
		userMatches := strings.EqualFold(evtNickname, req.User)
		if userMatches && (evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE || evt.Type == pb.GameEvent_EXCHANGE) {
			eval := evalSingleMove(bot.game, idx)
			evals = append(evals, eval)
		}
	}
	evaluation := &pb.Evaluation{PlayEval: evals}
	log.Info().Interface("eval", evaluation).Msg("evaluation")

	return &pb.BotResponse{
		Response: nil,
		Eval:     evaluation,
	}
}

func (b *Bot) handle(data []byte) *pb.BotResponse {
	ng, req, err := b.Deserialize(data)
	if err != nil {
		return errorResponse("Could not parse request", err)
	}
	evalReq := req.EvaluationRequest
	botType := req.BotType

	leavesFile := ""
	if ng.History().BoardLayout == board.SuperCrosswordGameLayout {
		leavesFile = "super-leaves.klv2"
	}

	conf := &bot.BotConfig{Config: *b.config, LeavesFile: leavesFile}
	g, err := bot.NewBotTurnPlayerFromGame(ng, conf, botType)
	if err != nil {
		return errorResponse("Could not create AI player", err)
	}
	b.game = g

	if evalReq != nil {
		// We are asking it to evaluate the last play in the position
		// that we passed in.
		// Generate all possible moves.
		return b.evaluationResponse(evalReq)
	}
	isWordSmog := g.Rules().Variant() == game.VarWordSmog || g.Rules().Variant() == game.VarWordSmogSuper
	// See if we need to challenge the last move
	valid := true
	if g.LastEvent() != nil && g.LastEvent().Type == pb.GameEvent_TILE_PLACEMENT_MOVE {
		err = g.ValidateWords(g.Lexicon(), g.LastWordsFormed())
		valid = (err == nil)
	}

	var m *move.Move

	if !valid {
		m, _ = g.NewChallengeMove(g.PlayerOnTurn())
	} else if g.IsPlaying() {
		if g.Game.Playing() == pb.PlayState_WAITING_FOR_FINAL_PASS {
			m, _ = g.NewPassMove(g.PlayerOnTurn())
		} else {
			// Redirect to simming bot if needed.
			if botType == pb.BotRequest_SIMMING_BOT || botType == pb.BotRequest_SIMMING_INFER_BOT {
				return errorResponse(ErrNeedSimmingBot.Error(), nil)
			}

			var moves []*move.Move
			if !isWordSmog {
				moves = b.game.GenerateMoves(1)
			} else {
				moves, err = wolgesAnalyze(b.config, b.game)
				if err != nil {
					log.Err(err).Msg("wolges-analyze-error")
					// Just generate a move using the regular generator.
					moves = b.game.GenerateMoves(1)
				}
			}
			m = moves[0]
		}
	} else {
		m, _ = g.NewPassMove(g.PlayerOnTurn())
	}
	log.Info().Msgf("Generated move: %s", m.ShortDescription())
	rack := g.RackLettersFor(g.PlayerOnTurn())
	evt := b.game.EventFromMove(m, rack)
	return &pb.BotResponse{
		Response: &pb.BotResponse_Move{Move: evt},
		GameId:   g.Uid(),
	}
}

func (b *Bot) asyncLambdaCall(data []byte) {
	ng, req, err := b.Deserialize(data)
	if err != nil {
		return
	}
	cgp := ng.ToCGP(true)
	if req.MillisRemaining > 0 {
		// Opp MS remaining doesn't matter for now.
		cgp += fmt.Sprintf(" tmr %d/%d;", req.MillisRemaining, 1000)
	}
	lfn := b.config.GetString(mcfg.ConfigLambdaFunctionName)
	replyChannel := "bot.publish_event." + ng.Uid()
	ctx := context.Background()

	lambdaPayload := &LambdaEvent{
		CGP:          cgp,
		BotType:      int(req.BotType),
		ReplyChannel: replyChannel,
		GameID:       ng.Uid(),
	}
	bts, err := json.Marshal(lambdaPayload)
	if err != nil {
		log.Err(err).Msg("marshalling-json-payload")
		return
	}
	log.Info().Interface("payload", lambdaPayload).Msg("sending-lambda-payload")

	out, err := b.lambdaClient.Invoke(ctx, &lambda.InvokeInput{
		FunctionName:   aws.String(lfn),
		InvocationType: types.InvocationTypeEvent,
		Payload:        bts,
	})
	if err != nil {
		log.Err(err).Msg("lambda-invocation-error")
		return
	}
	log.Info().Int32("status", out.StatusCode).Msg("lambda-invocation-output")
}

// Main implements a NATS listener that takes data from a channel.
// It does not respond on a reply channel, because that implies the
// caller would need to block and wait for a response. Instead, we must
// send a bot move on a separate per-game-id channel when ready.
func Main(channel string, bot *Bot) {
	err := bot.newGame()
	if err != nil {
		log.Fatal().AnErr("newGameErr", err).Msg(":(")
	}
	nc, err := nats.Connect(bot.config.GetString(mcfg.ConfigNatsURL))
	if err != nil {
		log.Fatal().AnErr("natsConnectErr", err).Msg(":(")
	}
	// A user of the bot should send the data to only one bot instance.
	// Using a QueueSubscribe guarantees that only one listening bot will
	// receive a message.
	nc.QueueSubscribe(channel, "bot_queue", func(m *nats.Msg) {
		log.Info().Str("replyChannel", m.Reply).Msgf("RECV: %d bytes", len(m.Data))
		resp := bot.handle(m.Data)
		respErr := resp.GetError()
		if respErr != "" {
			if respErr == ErrNeedSimmingBot.Error() {
				// Handle this with an async call to an external Lambda function.
				bot.asyncLambdaCall(m.Data)
				return
			}
		}

		// debugWriteln(proto.MarshalTextString(resp))
		data, err := proto.Marshal(resp)
		if err != nil {
			// Should never happen, ideally, but we need to do something sensible here.
			m.Respond([]byte(err.Error()))
		} else {
			if m.Reply != "" {
				err = m.Respond(data)
				if err != nil {
					log.Err(err).Msg("error-responding")
				}
			} else {
				err := retry.Do(
					func() error {
						_, err := nc.Request(
							"bot.publish_event."+resp.GameId, data, 3*time.Second)
						if err != nil {
							return err
						}
						// We're just waiting for an acknowledgement. The actual
						// data doesn't matter.
						return nil
					},
					retry.DelayType(func(n uint, err error, config *retry.Config) time.Duration {
						log.Err(err).Uint("n", n).Str("gameID", resp.GameId).
							Msg("did-not-receive-ack-try-again")
						return retry.BackOffDelay(n, err, config)
					}),
				)
				if err != nil {
					log.Err(err).Msg("bot-move-failed")
				}
			}
		}
	})
	nc.Flush()

	if err := nc.LastError(); err != nil {
		log.Fatal()
	}
	log.Info().Msgf("Listening on [%s]", channel)

	runtime.Goexit()
	fmt.Println("exiting")

}
