package bot

import (
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/golang/protobuf/proto"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/runner"
)

func debugWriteln(msg string) {
	io.WriteString(os.Stderr, msg)
	io.WriteString(os.Stderr, "\n")
}

type Bot struct {
	config  *config.Config
	options *runner.GameOptions

	game *runner.AIGameRunner
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

	game, err := runner.NewAIGameRunner(bot.config, bot.options, players)
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

func (bot *Bot) Deserialize(data []byte) (*game.Game, error) {
	req := pb.BotRequest{}
	err := proto.Unmarshal(data, &req)
	if err != nil {
		return nil, err
	}
	history := req.GameHistory
	rules, err := bot.game.Rules(bot.config, history)
	if err != nil {
		return nil, err
	}
	nturns := len(history.Events)
	ng, err := game.NewFromHistory(history, rules, 0)
	if err != nil {
		return nil, err
	}
	ng.PlayToTurn(nturns)
	// debugWriteln(ng.ToDisplayText())
	return ng, nil
}

func (bot *Bot) handle(data []byte) *pb.BotResponse {
	ng, err := bot.Deserialize(data)
	if err != nil {
		return errorResponse("Could not parse request", err)
	}
	g, err := runner.NewAIGameRunnerFromGame(ng, bot.config)
	if err != nil {
		return errorResponse("Could not create AI player", err)
	}
	bot.game = g

	moves := bot.game.GenerateMoves(1)
	m := moves[0]
	evt := bot.game.EventFromMove(m)
	return &pb.BotResponse{
		Response: &pb.BotResponse_Move{Move: evt},
	}
}

func Main(channel string, bot *Bot) {
	bot.newGame()
	nc, err := nats.Connect(nats.DefaultURL)
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
