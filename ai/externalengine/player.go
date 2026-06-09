package externalengine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/rs/zerolog/log"
)

// ExternalEnginePlayer calls an external engine binary per turn to get moves.
// It satisfies the AITurnPlayer interface.
type ExternalEnginePlayer struct {
	aiturnplayer.AIStaticTurnPlayer
	cfg Config
}

// NewFromGame creates an ExternalEnginePlayer for the given game.
func NewFromGame(g *game.Game, appCfg *config.Config, ec Config) (*ExternalEnginePlayer, error) {
	if ec.BinaryPath == "" {
		return nil, fmt.Errorf("externalengine: BinaryPath required")
	}
	c, err := equity.NewExhaustiveLeaveCalculator(g.LexiconName(), appCfg, "")
	if err != nil {
		return nil, err
	}
	aip, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, appCfg, []equity.EquityCalculator{c})
	if err != nil {
		return nil, err
	}
	return &ExternalEnginePlayer{AIStaticTurnPlayer: *aip, cfg: ec}, nil
}

func (p *ExternalEnginePlayer) GetBotType() pb.BotRequest_BotCode {
	return pb.BotRequest_CUSTOM_BOT
}

// SetLastMoves is a no-op; the engine is stateless and reads full GCG each turn.
func (p *ExternalEnginePlayer) SetLastMoves([]*move.Move) {}

// AddLastMove is a no-op; the engine reads full GCG each turn.
func (p *ExternalEnginePlayer) AddLastMove(*move.Move) {}

func (p *ExternalEnginePlayer) MoveGenerator() movegen.MoveGenerator {
	return p.AIStaticTurnPlayer.MoveGenerator()
}

// BestPlay writes the current game state to a temp GCG file, calls the external
// engine binary, parses the JSON response, and returns the chosen move.
func (p *ExternalEnginePlayer) BestPlay(ctx context.Context) (*move.Move, error) {
	playerid := p.PlayerOnTurn()
	rack := p.RackLettersFor(playerid)
	if rack == "" {
		return nil, fmt.Errorf("externalengine: empty rack for player %d", playerid)
	}

	gcg, err := gcgio.GameHistoryToGCG(p.History(), false)
	if err != nil {
		return nil, fmt.Errorf("externalengine: gcg export: %w", err)
	}

	f, err := os.CreateTemp("", "macondo-ext-*.gcg")
	if err != nil {
		return nil, fmt.Errorf("externalengine: temp file: %w", err)
	}
	defer os.Remove(f.Name())
	if _, err := f.WriteString(gcg); err != nil {
		f.Close()
		return nil, fmt.Errorf("externalengine: writing gcg: %w", err)
	}
	if err := f.Close(); err != nil {
		return nil, fmt.Errorf("externalengine: closing gcg: %w", err)
	}

	args := make([]string, 0, len(p.cfg.ExtraArgs)+3)
	args = append(args, p.cfg.ExtraArgs...)
	args = append(args,
		"--gcg="+f.Name(),
		"--lexicon="+p.LexiconName(),
		"--rack="+rack,
	)
	cmd := exec.CommandContext(ctx, p.cfg.BinaryPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	log.Info().Str("rack", rack).Str("binary", p.cfg.BinaryPath).Msg("externalengine: invoking")
	if err := cmd.Run(); err != nil {
		stderrSnip := stderr.String()
		if len(stderrSnip) > 512 {
			stderrSnip = stderrSnip[:512]
		}
		return nil, fmt.Errorf("externalengine: subprocess: %w; stderr: %s", err, stderrSnip)
	}

	log.Info().Str("stdout", strings.TrimSpace(stdout.String())).Msg("externalengine: stdout")

	resp, err := parseLastJSONLine(stdout.Bytes())
	if err != nil {
		return nil, fmt.Errorf("externalengine: parse response: %w", err)
	}

	log.Info().Str("action", resp.Action).Str("position", resp.Position).Str("tiles", resp.Tiles).Int("score", resp.Score).Msg("externalengine: response")

	return p.responseToMove(playerid, resp)
}

// parseLastJSONLine extracts the last non-empty line from stdout and unmarshals it.
func parseLastJSONLine(data []byte) (*EngineResponse, error) {
	lines := bytes.Split(bytes.TrimRight(data, "\n"), []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		line := bytes.TrimSpace(lines[i])
		if len(line) == 0 {
			continue
		}
		var resp EngineResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			return nil, fmt.Errorf("json unmarshal: %w (line: %q)", err, line)
		}
		return &resp, nil
	}
	return nil, fmt.Errorf("no JSON output from engine")
}

func (p *ExternalEnginePlayer) responseToMove(playerid int, resp *EngineResponse) (*move.Move, error) {
	var fields []string
	switch resp.Action {
	case "pass":
		fields = []string{"pass"}
	case "exchange":
		fields = []string{"exchange", strings.ToUpper(resp.Tiles)}
	case "place":
		fields = []string{resp.Position, resp.Tiles}
	default:
		return nil, fmt.Errorf("externalengine: unknown action %q", resp.Action)
	}
	// lowercase=false: uppercase letters are tiles, lowercase are blanks.
	// This matches Quackle and similar engines' conventions.
	m, err := p.ParseMove(playerid, false, fields, false)
	if err != nil {
		return nil, fmt.Errorf("externalengine: parse move %v: %w", fields, err)
	}
	return m, nil
}
