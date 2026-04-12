package automatic

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"time"

	"google.golang.org/protobuf/encoding/protojson"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// LoadAutoplayConfig reads an AutoplayConfig from a protojson file.
func LoadAutoplayConfig(path string) (*pb.AutoplayConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading autoplay config: %w", err)
	}
	cfg := &pb.AutoplayConfig{}
	if err := protojson.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing autoplay config: %w", err)
	}
	return cfg, nil
}

// ResolveExperimentID returns the experiment ID from the config, or generates
// one from the current timestamp and 6 random hex characters.
func ResolveExperimentID(cfg *pb.AutoplayConfig) string {
	if cfg.ExperimentId != "" {
		return cfg.ExperimentId
	}
	ts := time.Now().Format("20060102-150405")
	b := make([]byte, 3)
	rand.Read(b)
	return fmt.Sprintf("%s-%s", ts, hex.EncodeToString(b))
}

// PlayersFromConfig converts an AutoplayConfig's player configs to
// AutomaticRunnerPlayer structs.
func PlayersFromConfig(cfg *pb.AutoplayConfig) []AutomaticRunnerPlayer {
	return []AutomaticRunnerPlayer{
		playerFromProto(cfg.Player1),
		playerFromProto(cfg.Player2),
	}
}

func playerFromProto(p *pb.AutoplayPlayerConfig) AutomaticRunnerPlayer {
	if p == nil {
		return AutomaticRunnerPlayer{BotCode: pb.BotRequest_HASTY_BOT}
	}
	return AutomaticRunnerPlayer{
		BotCode:              p.BotCode,
		LeaveFile:            p.LeaveFile,
		PEGFile:              p.PegFile,
		MinSimPlies:          int(p.MinSimPlies),
		SimThreads:           int(p.SimThreads),
		StochasticStaticEval: p.StochasticStaticEval,
		InferenceTau:         p.InferenceTau,
	}
}
