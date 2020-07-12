package bot

import (
	"errors"
	"fmt"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
	"github.com/domino14/macondo/shell"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"
	"runtime"
	"strconv"
	"strings"
)

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

func (bot *Bot) setRack(rack string) error {
	return bot.game.SetCurrentRack(rack)
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

func (bot *Bot) acceptMove(args []string) error {
	var move *move.Move
	var err error
	playerid := bot.game.PlayerOnTurn()
	coords, word := args[0], args[1]
	if coords == "exchange" {
		move, err = bot.game.NewExchangeMove(playerid, word)
	} else {
		move, err = bot.game.NewPlacementMove(playerid, coords, word)
	}
	if err != nil {
		return err
	}
	err = bot.game.PlayMove(move, true, 0)
	if err != nil {
		return err
	}
	return nil
}

func (bot *Bot) commitAIMove() error {
	moves := bot.game.GenerateMoves(1)
	m := moves[0]
	return bot.game.PlayMove(m, true, 0)
}

func (bot *Bot) dispatch(data string) (*shell.Response, error) {
	var err error
	fields := strings.Fields(data)
	cmd := fields[0]
	args := fields[1:]
	switch cmd {
	case "new":
		err = bot.newGame()
	case "play":
		err = bot.acceptMove(args)
	case "setrack":
		err = bot.setRack(args[0])
	case "aiplay":
		err = bot.commitAIMove()
	default:
		msg := fmt.Sprintf("command %v not found", strconv.Quote(cmd))
		log.Info().Msg(msg)
		err = errors.New(msg)
	}
	return nil, err
}

func Main(channel string, bot *Bot) {
	nc, err := nats.Connect(nats.DefaultURL)
	if err != nil {
		log.Fatal()
	}
	// Simple Async Subscriber
	nc.Subscribe(channel, func(m *nats.Msg) {
		data := string(m.Data)
		log.Info().Msgf("RECV: %s\n", data)
		_, err := bot.dispatch(data)
		if err != nil {
			m.Respond([]byte(err.Error()))
		} else {
			m.Respond([]byte(bot.game.ToDisplayText()))
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
