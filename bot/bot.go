package bot

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
)

const (
	StarPlayThreshold = 10.0 // equity
)

func debugWriteln(msg string) {
	io.WriteString(os.Stderr, msg)
	io.WriteString(os.Stderr, "\n")
}

type Bot struct {
	config  *config.Config
	options *runner.GameOptions

	game *airunner.AIGameRunner
}

func NewBot(config *config.Config, options *runner.GameOptions) *Bot {
	bot := &Bot{}
	bot.config = config
	bot.options = options
	bot.game = nil
	return bot
}

func (bot *Bot) newGame() error {
	players := []*pb.PlayerInfo{
		{Nickname: "self", RealName: "Macondo Bot"},
		{Nickname: "opponent", RealName: "Arthur Dent"},
	}

	game, err := airunner.NewAIGameRunner(bot.config, bot.options, players, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return err
	}
	bot.game = game
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

func (bot *Bot) Deserialize(data []byte) (*game.Game, *pb.EvaluationRequest, pb.BotRequest_BotCode, error) {
	var err error

	req := pb.BotRequest{}
	err = proto.Unmarshal(data, &req)
	if err != nil {
		return nil, nil, 0, err
	}
	history := req.GameHistory
	boardLayout, ldName, variant := game.HistoryToVariant(history)
	rules, err := airunner.NewAIGameRules(bot.config, boardLayout, variant, history.Lexicon, ldName)
	if err != nil {
		return nil, nil, 0, err
	}
	var ng *game.Game

	if history.StartingCgp != "" {
		if len(history.Events) > 0 {
			return nil, nil, 0, errors.New("histories with a starting CGP cannot currently contain additional events")
		}
		ng, err = cgp.ParseCGP(bot.config, history.StartingCgp)
		if err != nil {
			return nil, nil, 0, err
		}
	} else {
		nturns := len(history.Events)
		ng, err = game.NewFromHistory(history, rules, 0)
		if err != nil {
			return nil, nil, 0, err
		}
		ng.PlayToTurn(nturns)
	}
	// debugWriteln(ng.ToDisplayText())
	return ng, req.EvaluationRequest, req.BotType, nil
}

func evalSingleMove(g *airunner.AIGameRunner, evtIdx int) *pb.SingleEvaluation {
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
	for idx, m := range moves {
		evt := g.EventFromMove(m)
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
		if evt.Nickname != "" {
			// remove -- deprecated
			evtNickname = evt.Nickname
		}
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

func (bot *Bot) handle(data []byte) *pb.BotResponse {
	ng, evalReq, botType, err := bot.Deserialize(data)
	if err != nil {
		return errorResponse("Could not parse request", err)
	}
	g, err := airunner.NewAIGameRunnerFromGame(ng, bot.config, botType)
	if err != nil {
		return errorResponse("Could not create AI player", err)
	}
	bot.game = g

	if evalReq != nil {
		// We are asking it to evaluate the last play in the position
		// that we passed in.
		// Generate all possible moves.
		return bot.evaluationResponse(evalReq)
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
			var moves []*move.Move
			if !isWordSmog {
				moves = bot.game.GenerateMoves(1)
			} else {
				moves, err = wolgesAnalyze(bot.config, bot.game)
				if err != nil {
					log.Err(err).Msg("wolges-analyze-error")
					// Just generate a move using the regular generator.
					moves = bot.game.GenerateMoves(1)
				}
			}
			m = moves[0]
		}
	} else {
		m, _ = g.NewPassMove(g.PlayerOnTurn())
	}
	log.Info().Msgf("Generated move: %s", m.ShortDescription())
	evt := bot.game.EventFromMove(m)
	return &pb.BotResponse{
		Response: &pb.BotResponse_Move{Move: evt},
	}
}

func Main(channel string, bot *Bot) {
	bot.newGame()
	nc, err := nats.Connect(bot.config.NatsURL)
	if err != nil {
		log.Fatal()
	}
	// Simple Async Subscriber
	nc.Subscribe(channel, func(m *nats.Msg) {
		log.Info().Msgf("RECV: %d bytes", len(m.Data))
		resp := bot.handle(m.Data)
		// debugWriteln(proto.MarshalTextString(resp))
		data, err := proto.Marshal(resp)
		if err != nil {
			// Should never happen, ideally, but we need to do something sensible here.
			m.Respond([]byte(err.Error()))
		} else {
			m.Respond(data)
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
