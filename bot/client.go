package bot

import (
	"errors"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
)

type Client struct {
	// NATS connection
	nc      *nats.Conn
	channel string
}

func MakeRequest(game *runner.GameRunner, config *config.Config) ([]byte, error) {
	history := game.History()
	if history.Lexicon == "" {
		history.Lexicon = config.DefaultLexicon
	}
	req := pb.BotRequest{GameHistory: history}
	return proto.Marshal(&req)
}

// Send a game to the bot and get a move back.
func (c *Client) RequestMove(game *runner.GameRunner, config *config.Config) (*move.Move, error) {
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
