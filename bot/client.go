package bot

import (
	"errors"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

type Client struct {
	// NATS connection
	nc      *nats.Conn
	channel string
}

func MakeRequest(game *bot.BotTurnPlayer, cfg *config.Config) ([]byte, error) {
	history := game.GenerateSerializableHistory()
	if history.Lexicon == "" {
		history.Lexicon = cfg.GetString(config.ConfigDefaultLexicon)
	}
	req := pb.BotRequest{GameHistory: history}
	return proto.Marshal(&req)
}

// Send a game to the bot and get a move back.
func (c *Client) RequestMove(game *bot.BotTurnPlayer, config *config.Config) (*move.Move, error) {
	data, err := MakeRequest(game, config)
	if err != nil {
		return nil, err
	}
	res, err := c.nc.Request(c.channel, data, 10*time.Second)
	if err != nil {
		if c.nc.LastError() != nil {
			log.Error().Msgf("%v for request", c.nc.LastError())
		}
		log.Error().Msgf("%v for request", err)
		return nil, err
	}
	log.Debug().Msgf("res: %v", string(res.Data))

	resp := pb.BotResponse{}
	err = proto.Unmarshal(res.Data, &resp)
	if err != nil {
		return nil, err
	}
	switch r := resp.Response.(type) {
	case *pb.BotResponse_Move:
		return game.MoveFromEvent(r.Move)
	case *pb.BotResponse_Error:
		return nil, errors.New("Bot returned: " + r.Error)
	default:
		return nil, errors.New("should never happen")
	}
}
