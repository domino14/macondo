package bot

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/golang/protobuf/proto"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"

	airunner "github.com/domino14/macondo/ai/runner"
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
	req := pb.BotRequest{}
	err := proto.Unmarshal(data, &req)
	if err != nil {
		return nil, nil, 0, err
	}
	history := req.GameHistory
	boardLayout, ldName, _ := game.HistoryToVariant(history)
	rules, err := airunner.NewAIGameRules(bot.config, boardLayout, history.Lexicon, ldName)
	if err != nil {
		return nil, nil, 0, err
	}
	nturns := len(history.Events)
	ng, err := game.NewFromHistory(history, rules, 0)
	if err != nil {
		return nil, nil, 0, err
	}
	ng.PlayToTurn(nturns)
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
	topIsBingo := moves[0].TilesPlayed() == 7 && moves[0].Action() == move.MoveTypePlay
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
				playedBingo = m.TilesPlayed() == 7 && m.Action() == move.MoveTypePlay
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

	evals := []*pb.SingleEvaluation{}

	for idx, evt := range evts {

		if strings.ToLower(evt.Nickname) == strings.ToLower(req.User) && (evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE ||
			evt.Type == pb.GameEvent_EXCHANGE) {
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

	// See if we need to challenge the last move
	valid := true
	if g.LastEvent() != nil &&
		g.LastEvent().Type == pb.GameEvent_TILE_PLACEMENT_MOVE {
		for _, word := range g.LastWordsFormed() {
			if !g.Lexicon().HasWord(word) {
				valid = false
				break
			}
		}
	}

	var m *move.Move

	if !valid {
		m, _ = g.NewChallengeMove(g.PlayerOnTurn())
	} else if g.IsPlaying() {
		moves := bot.game.GenerateMoves(1)
		m = moves[0]
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
