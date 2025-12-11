package explainer

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
)

// PlayMetadata represents metadata about a Scrabble play
type PlayMetadata struct {
	Play           string `json:"play"`
	Score          int    `json:"score"`
	TilesUsed      int    `json:"tiles_used"`
	IsBingo        bool   `json:"is_bingo"`
	VowelsInLeave  int    `json:"vowels_in_leave"`
	ConsonantsLeft int    `json:"consonants_in_leave"`
	LeaveBalance   string `json:"leave_balance"` // "balanced", "vowel-heavy", "consonant-heavy"
}

// FuturePlayMetadata represents metadata about a potential future play
type FuturePlayMetadata struct {
	Play               string   `json:"play"`
	Score              int      `json:"score"`
	IsBingo            bool     `json:"is_bingo"`
	NeededDraw         []string `json:"needed_draw"`         // tiles needed from bag
	RequiresOtherPlay  string   `json:"requires_opp_play"`   // opponent play needed first
	ProbabilityPercent float64  `json:"probability_percent"` // likelihood of this play
}

// GetOurPlayMetadataTool analyzes metadata for a current play
type GetOurPlayMetadataTool struct {
	analyzer *Analyzer
}

func NewGetOurPlayMetadataTool(analyzer *Analyzer) *GetOurPlayMetadataTool {
	return &GetOurPlayMetadataTool{analyzer: analyzer}
}

func (t *GetOurPlayMetadataTool) Name() string {
	return "get_our_play_metadata"
}

func (t *GetOurPlayMetadataTool) Description() string {
	return "Get metadata about a play including score, tiles used, vowel/consonant balance, and whether it's a bingo"
}

func (t *GetOurPlayMetadataTool) Parameters() map[string]interfaces.ParameterSpec {
	return map[string]interfaces.ParameterSpec{
		"play_string": {
			Type:        "string",
			Description: "The play string to analyze (e.g., '5D (S)PIC(A)')",
			Required:    true,
		},
	}
}

func (t *GetOurPlayMetadataTool) Run(ctx context.Context, args string) (string, error) {
	return t.Execute(ctx, args)
}

func (t *GetOurPlayMetadataTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		PlayString string `json:"play_string"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	metadata, err := t.analyzer.GetPlayMetadata(params.PlayString)
	if err != nil {
		return "", err
	}

	result, err := json.Marshal(metadata)
	if err != nil {
		return "", err
	}
	return string(result), nil
}

// GetOurFuturePlayMetadataTool analyzes metadata for a potential future play
type GetOurFuturePlayMetadataTool struct {
	analyzer *Analyzer
}

func NewGetOurFuturePlayMetadataTool(analyzer *Analyzer) *GetOurFuturePlayMetadataTool {
	return &GetOurFuturePlayMetadataTool{analyzer: analyzer}
}

func (t *GetOurFuturePlayMetadataTool) Name() string {
	return "get_our_future_play_metadata"
}

func (t *GetOurFuturePlayMetadataTool) Description() string {
	return "Get metadata about a potential future play including required tile draws, setup requirements, and probability"
}

func (t *GetOurFuturePlayMetadataTool) Parameters() map[string]interfaces.ParameterSpec {
	return map[string]interfaces.ParameterSpec{
		"play_string": {
			Type:        "string",
			Description: "The future play string to analyze (e.g., '8H QUIXOTIC')",
			Required:    true,
		},
	}
}

func (t *GetOurFuturePlayMetadataTool) Run(ctx context.Context, args string) (string, error) {
	return t.Execute(ctx, args)
}

func (t *GetOurFuturePlayMetadataTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		PlayString string `json:"play_string"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	metadata, err := t.analyzer.GetFuturePlayMetadata(params.PlayString)
	if err != nil {
		return "", err
	}

	result, err := json.Marshal(metadata)
	if err != nil {
		return "", err
	}
	return string(result), nil
}

// EvaluateLeaveTool evaluates the value of a leave
type EvaluateLeaveTool struct {
	analyzer *Analyzer
}

func NewEvaluateLeaveTool(analyzer *Analyzer) *EvaluateLeaveTool {
	return &EvaluateLeaveTool{analyzer: analyzer}
}

func (t *EvaluateLeaveTool) Name() string {
	return "evaluate_leave"
}

func (t *EvaluateLeaveTool) Description() string {
	return "Evaluate the value of a leave (tiles remaining on rack after a play). Returns a numerical value where +2 to +3 is decent, and +8 or above is really strong."
}

func (t *EvaluateLeaveTool) Parameters() map[string]interfaces.ParameterSpec {
	return map[string]interfaces.ParameterSpec{
		"leave": {
			Type:        "string",
			Description: "The leave tiles to evaluate (e.g., 'AEINRT')",
			Required:    true,
		},
	}
}

func (t *EvaluateLeaveTool) Run(ctx context.Context, args string) (string, error) {
	return t.Execute(ctx, args)
}

func (t *EvaluateLeaveTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		Leave string `json:"leave"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("failed to parse parameters: %w", err)
	}

	value, err := t.analyzer.EvaluateLeave(params.Leave)
	if err != nil {
		return "", err
	}

	return strconv.FormatFloat(value, 'f', 3, 64), nil
}

// Analyzer provides the actual analysis logic
type Analyzer struct {
	// This will eventually integrate with the game state and simulation
	gameState                 string
	simResults                string
	simDetails                string
	winningPlay               string
	winningStats              string
	game                      *bot.BotTurnPlayer
	exhaustiveLeaveCalculator *equity.ExhaustiveLeaveCalculator
	config                    *config.Config
}

func NewAnalyzer() *Analyzer {
	return &Analyzer{}
}

// SetConfig sets the configuration for the analyzer
func (a *Analyzer) SetConfig(cfg *config.Config) {
	a.config = cfg
}

// SetGameContext sets the current game context for analysis
func (a *Analyzer) SetGameContext(gameState, simResults, simDetails, winningPlay, winningStats string) {
	a.gameState = gameState
	a.simResults = simResults
	a.simDetails = simDetails
	a.winningPlay = winningPlay
	a.winningStats = winningStats
}

// EvaluateLeave evaluates the value of a leave
func (a *Analyzer) EvaluateLeave(leave string) (float64, error) {
	log.Info().Str("leave", leave).Msg("evaluating leave")
	if a.exhaustiveLeaveCalculator == nil {
		if a.config == nil {
			return 0, fmt.Errorf("config not set")
		}
		if a.game == nil {
			return 0, fmt.Errorf("game not set")
		}

		lexiconName := a.game.LexiconName()
		leavesFile := ""
		if a.game.Board().Dim() == 21 {
			leavesFile = "super-leaves.klv2"
		}

		elc, err := equity.NewExhaustiveLeaveCalculator(lexiconName, a.config, leavesFile)
		if err != nil {
			return 0, fmt.Errorf("failed to create exhaustive leave calculator: %w", err)
		}
		a.exhaustiveLeaveCalculator = elc
	}

	// Convert the leave to machine word
	dist := a.game.Bag().LetterDistribution()
	machineLeave, err := tilemapping.ToMachineWord(leave, dist.TileMapping())
	if err != nil {
		return 0, fmt.Errorf("failed to convert leave: %w", err)
	}

	value := a.exhaustiveLeaveCalculator.LeaveValue(machineLeave)
	log.Info().Str("leave", leave).Float64("value", value).Msg("evaluated leave")
	return value, nil
}

// GetPlayMetadata analyzes a play and returns metadata
func (a *Analyzer) GetPlayMetadata(playString string) (*PlayMetadata, error) {
	// Check if this is an exchange or pass move
	trimmed := strings.TrimSpace(playString)
	var dottedPlayStr string

	// Handle exchange moves like "(exch Q)" or "exch Q" or "exchange Q"
	if strings.HasPrefix(trimmed, "(exch ") || strings.HasPrefix(trimmed, "(exchange ") {
		// Remove outer parentheses for exchange moves
		dottedPlayStr = strings.Trim(trimmed, "()")
	} else if trimmed == "pass" || strings.HasPrefix(trimmed, "exch ") || strings.HasPrefix(trimmed, "exchange ") {
		// Already in correct format
		dottedPlayStr = trimmed
	} else {
		// For placement moves, convert characters inside parentheses to dots
		// (parentheses indicate tiles already on the board)
		var sb strings.Builder
		inParens := false
		for _, ch := range playString {
			if ch == '(' {
				inParens = true
				continue
			} else if ch == ')' {
				inParens = false
				continue
			}
			if !inParens {
				sb.WriteRune(ch)
			} else {
				sb.WriteRune('.')
			}
		}
		dottedPlayStr = sb.String()
	}

	m, err := a.game.ParseMove(a.game.PlayerOnTurn(), false, strings.Fields(dottedPlayStr), false)
	if err != nil {
		return nil, err
	}

	// Parse basic info from play string
	isBingo := m.BingoPlayed()
	tilesUsed := m.TilesPlayed()

	vwlct := 0
	cstct := 0
	blanks := 0
	for _, t := range m.Leave() {
		if t.IsVowel(a.game.Bag().LetterDistribution()) {
			vwlct++
		} else if t != 0 {
			cstct++
		} else if t == 0 {
			blanks++
		}
	}
	leaveBalance := "balanced"
	if vwlct >= cstct+2 {
		leaveBalance = "vowel-heavy"
	} else if cstct >= vwlct+2 {
		leaveBalance = "consonant-heavy"
	}
	if len(m.Leave()) == 0 {
		leaveBalance = "N/A"
	}

	md := &PlayMetadata{
		Play:           playString,
		Score:          m.Score(),
		TilesUsed:      tilesUsed,
		IsBingo:        isBingo,
		VowelsInLeave:  vwlct,
		ConsonantsLeft: cstct,
		LeaveBalance:   leaveBalance,
	}
	log.Info().Interface("metadata", md).Msg("analyzed play metadata")
	return md, nil
}

// GetFuturePlayMetadata analyzes a potential future play by parsing winningStats
func (a *Analyzer) GetFuturePlayMetadata(playString string) (*FuturePlayMetadata, error) {
	if a.winningStats == "" {
		return nil, fmt.Errorf("no winning stats available")
	}
	log.Info().Str("play", playString).Msg("analyzing future play metadata")
	// get only the Our follow-up play section
	sections := strings.Split(a.winningStats, "### Our follow-up play")
	if len(sections) < 2 {
		return nil, fmt.Errorf("no 'Our follow-up play' section found in winning stats")
	}
	followupSection := sections[1]

	lines := strings.Split(followupSection, "\n")
	normalizedBestPlay := stats.Normalize(a.winningPlay)

	// Skip header lines and find the play
	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Skip empty lines and header lines
		if line == "" || strings.HasPrefix(line, "Play") || strings.HasPrefix(line, "---") || strings.HasPrefix(line, "Bingo probability") {
			continue
		}

		// Parse the line format: "Play    Needed Draw   Score    Count    % of time"
		// The play might be multiple words like "J7 QAT" or "D11 (U)MIAQ"

		// line has a very specific format. %-20s%-14s%-9s%-9s%-16s
		// see playStatsStr in heatmap.go
		// This is very fragile, we just need tests.

		playName := strings.TrimSpace(line[0:19])

		fields := strings.Fields(line[19:])
		if len(fields) == 3 {
			// needed draw is empty.
			fields = append([]string{""}, fields...)
		}
		if len(fields) != 4 {
			return nil, fmt.Errorf("unexpected format in winning stats line: %s", line)
		}

		neededDraw := fields[0]
		score := fields[1]
		// count := fields[2]
		percent := fields[3]

		p := strings.TrimSpace(playName)
		// Check if this matches our requested play (trim spaces)
		if p != strings.TrimSpace(playString) {
			continue
		}
		normalizedPlay := stats.Normalize(playString)
		drawLetters := []string{}
		reqOtherPlay := "none"
		isBingo := false

		scoreInt, err := strconv.Atoi(score)
		if err != nil {
			return nil, fmt.Errorf("failed to parse score: %w", err)
		}

		if strings.HasPrefix(p, "exchange ") || p == "pass" {

		} else {
			drawStr := strings.Trim(neededDraw, "{}")
			// Split individual letters
			for _, char := range drawStr {
				if char >= 'A' && char <= 'Z' {
					drawLetters = append(drawLetters, string(char))
				}
			}

			// Determine if it's a bingo by counting tiles played (ignoring parentheses content)
			fields := strings.Fields(normalizedPlay)
			if len(fields) < 2 {
				return nil, fmt.Errorf("invalid play string format: %s", playString)
			}
			tilesUsed := len(fields[1]) - strings.Count(fields[1], ".")
			isBingo = tilesUsed == 7

			// Check if there's a required opponent play
			npfields := strings.Fields(normalizedPlay)

			// Try to create the move to see if it's valid. Temporarily play
			// move and set our rack to imagine we drew the required letters.

			ourBestPlay, err := a.game.ParseMove(a.game.PlayerOnTurn(), false, strings.Fields(normalizedBestPlay), false)
			if err != nil {
				return nil, fmt.Errorf("failed to parse our best play %s: %w", normalizedBestPlay, err)
			}
			gcopy := a.game.Copy()
			err = gcopy.PlayMove(ourBestPlay, false, 0)
			if err != nil {
				return nil, fmt.Errorf("failed to play our best play %s: %w", normalizedBestPlay, err)
			}

			rackLetters := gcopy.RackLettersFor(1 - gcopy.PlayerOnTurn())
			for _, l := range drawLetters {
				rackLetters += l
			}

			m, err := gcopy.CreateAndScorePlacementMove(npfields[0], npfields[1], rackLetters, false)
			if err != nil || m.Score() != scoreInt {
				mscore := 0
				if m != nil {
					mscore = m.Score()
				}
				log.Err(err).Str("fields", npfields[0]+","+npfields[1]).
					Int("score", mscore).
					Msg("failed to parse move or score mismatch, assuming opponent play needed")
				reqOtherPlay = "requires opponent play"
			}

			// Try to parse this move on the original game state. If it fails, this means that it
			// requires US to play first to begin with (potential setup play)
			m, err = a.game.CreateAndScorePlacementMove(npfields[0], npfields[1], rackLetters, false)
			if err != nil || m.Score() != scoreInt {
				mscore := 0
				if m != nil {
					mscore = m.Score()
				}
				log.Err(err).Str("fields", npfields[0]+","+npfields[1]).
					Int("score", mscore).
					Msg("failed to parse move or score mismatch, assuming our best play needed")
				if reqOtherPlay == "none" {
					reqOtherPlay = "requires us to play " + normalizedBestPlay + " first"
				}
			}

		}

		probability, err := strconv.ParseFloat(percent, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse probability: %w", err)
		}

		log.Info().Str("play", playString).Interface("needed_draw", drawLetters).
			Int("score", scoreInt).Float64("probability", probability).Bool("is_bingo", isBingo).
			Str("requires_other_play", reqOtherPlay).
			Msg("analyzed future play metadata")

		return &FuturePlayMetadata{
			Play:               playString,
			Score:              scoreInt,
			IsBingo:            isBingo,
			NeededDraw:         drawLetters,
			RequiresOtherPlay:  reqOtherPlay,
			ProbabilityPercent: probability,
		}, nil
	}

	return nil, fmt.Errorf("play %s not found in winning stats", playString)
}

// BuildPrompt constructs the full prompt with the game situation
func (a *Analyzer) BuildPrompt(templatePath, quirkyPath string) (string, error) {
	// Read templates
	mainPrompt, err := os.ReadFile(templatePath)
	if err != nil {
		return "", fmt.Errorf("failed to read main prompt: %w", err)
	}

	situationTemplate, err := os.ReadFile("explainer/situation_template.md")
	if err != nil {
		return "", fmt.Errorf("failed to read situation template: %w", err)
	}

	quirkyPrompt := ""
	if quirkyPath != "" {
		quirkyBytes, _ := os.ReadFile(quirkyPath)
		quirkyPrompt = string(quirkyBytes)
	}

	// Build situation text from template
	situation := string(situationTemplate)
	situation = strings.ReplaceAll(situation, "{game_state}", a.gameState)
	situation = strings.ReplaceAll(situation, "{sim_results}", a.simResults)
	situation = strings.ReplaceAll(situation, "{sim_details}", a.simDetails)
	situation = strings.ReplaceAll(situation, "{best_play}", a.winningPlay)
	situation = strings.ReplaceAll(situation, "{winning_play_stats}", a.winningStats)

	// Replace placeholders in main prompt
	prompt := string(mainPrompt)
	prompt = strings.ReplaceAll(prompt, "{situation}", situation)
	prompt = strings.ReplaceAll(prompt, "{quirky}", quirkyPrompt)
	prompt = strings.ReplaceAll(prompt, "{best_play}", a.winningPlay)

	return prompt, nil
}
