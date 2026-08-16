package explainer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
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
	IsSetup            bool     `json:"is_setup"`
}

// FuturePlayFamily describes a play that can be made in more than one way,
// because a blank can stand in for different tiles. The combined percentage is
// the one to quote: each individual way is only part of the chance of making
// the play.
type FuturePlayFamily struct {
	Play              string   `json:"play"`
	CombinedPercent   float64  `json:"combined_probability_percent"`
	ScoreRange        string   `json:"score_range"`
	NeededDrawOptions []string `json:"needed_draw_options"` // any one of these unlocks the play
	IsSetup           bool     `json:"is_setup"`
}

// FuturePlayLookup is the result of looking a follow-up play up in the sim
// stats: every way of making it, plus the combined figures when there is more
// than one.
type FuturePlayLookup struct {
	Asked  string // the specific way the model named, if it named one
	Family *FuturePlayFamily
	Ways   []*FuturePlayMetadata
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
	return "Get metadata about a play including score, tiles used, vowel/consonant balance, and whether it's a bingo. " +
		"Use this whenever you want to talk about a play's leave balance, vowel/consonant counts, or tiles used - " +
		"do not count or calculate those yourself."
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
	return "Get metadata about one of our potential follow-up plays: the tile draws it needs, whether anything has " +
		"to be played first, whether it is a setup, and how likely it is. " +
		"Only plays listed in the 'Our follow-up play' table can be looked up, and they must be spelled exactly as they " +
		"appear there (a lowercase letter means that tile is the blank)."
}

func (t *GetOurFuturePlayMetadataTool) Parameters() map[string]interfaces.ParameterSpec {
	return map[string]interfaces.ParameterSpec{
		"play_string": {
			Type: "string",
			Description: "The future play string to analyze, copied exactly from the 'Our follow-up play' table " +
				"(e.g., '8H QUIXOTIC', or '1H (Z)WIEBAcK' where the lowercase c is the blank)",
			Required: true,
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

	lookup, err := t.analyzer.LookupFuturePlay(params.PlayString)
	var notFound *PlayNotFoundError
	if errors.As(err, &notFound) {
		// Not an error as far as the model is concerned: tell it which plays
		// it can actually ask about so it doesn't guess at the spelling again.
		log.Info().Str("play", params.PlayString).Msg("future play not in follow-up table")
		return notFound.ToolMessage(), nil
	}
	if err != nil {
		return "", err
	}

	var result []byte
	if lookup.Family == nil {
		result, err = json.Marshal(lookup.Ways[0])
	} else {
		// There is more than one way to make this play, differing in which
		// tile is the blank. Hand back all of them however the model asked,
		// so it can't mistake one way's chance for the play's chance.
		result, err = json.Marshal(struct {
			Note       string                `json:"note"`
			AskedAbout string                `json:"asked_about,omitempty"`
			Family     *FuturePlayFamily     `json:"family"`
			Ways       []*FuturePlayMetadata `json:"ways"`
		}{
			Note: fmt.Sprintf("There are %d different ways to make this play, differing in which tile is "+
				"played as the blank. Each way scores differently and needs a different draw. The chance of "+
				"making the play at all is the family's combined_probability_percent (%.2f%%) - quote that, "+
				"not an individual way's probability_percent. Name a specific way only by its exact play "+
				"string, where a lowercase letter is the blank.",
				len(lookup.Ways), lookup.Family.CombinedPercent),
			AskedAbout: lookup.Asked,
			Family:     lookup.Family,
			Ways:       lookup.Ways,
		})
	}
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

// Analyzer holds the game and the facts we computed about the position, and
// answers the tools' questions from them. Nothing here parses the prompt.
type Analyzer struct {
	game                      *bot.BotTurnPlayer
	facts                     *PositionFacts
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

// SetGame sets the position under analysis.
func (a *Analyzer) SetGame(tp *bot.BotTurnPlayer) {
	a.game = tp
	a.facts = nil
	a.exhaustiveLeaveCalculator = nil
}

// Facts returns the fact pack built for the current position, if any.
func (a *Analyzer) Facts() *PositionFacts {
	return a.facts
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

// DottedPlay converts a play from the playthrough notation the tables use,
// 5D (S)PIC(A), into the dotted form the move parser wants, 5D .PIC. - the
// parenthesized tiles are the ones already on the board. Exchanges and passes
// come back unchanged apart from their wrapping parentheses. Input that is
// already dotted is left alone.
func DottedPlay(playString string) string {
	trimmed := strings.TrimSpace(playString)

	// Handle exchange moves like "(exch Q)" or "exch Q" or "exchange Q"
	if strings.HasPrefix(trimmed, "(exch ") || strings.HasPrefix(trimmed, "(exchange ") {
		// Remove outer parentheses for exchange moves
		return strings.Trim(trimmed, "()")
	}
	if trimmed == "pass" || trimmed == "(Pass)" ||
		strings.HasPrefix(trimmed, "exch ") || strings.HasPrefix(trimmed, "exchange ") {
		if trimmed == "(Pass)" {
			return "pass"
		}
		return trimmed
	}

	var sb strings.Builder
	inParens := false
	for _, ch := range trimmed {
		switch {
		case ch == '(':
			inParens = true
		case ch == ')':
			inParens = false
		case inParens:
			sb.WriteRune('.')
		default:
			sb.WriteRune(ch)
		}
	}
	return sb.String()
}

// GetPlayMetadata analyzes a play and returns metadata
func (a *Analyzer) GetPlayMetadata(playString string) (*PlayMetadata, error) {
	m, err := a.game.ParseMove(a.game.PlayerOnTurn(), false, strings.Fields(DottedPlay(playString)), false)
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

// PlayNotFoundError is returned when a play is not one of the sampled
// follow-up plays. It carries the plays we do know about so that the tool can
// hand them back to the model instead of letting it guess at spellings.
type PlayNotFoundError struct {
	Play      string
	Available []string
}

func (e *PlayNotFoundError) Error() string {
	return fmt.Sprintf("play %s not found in winning stats (available: %s)",
		e.Play, strings.Join(e.Available, ", "))
}

// ToolMessage is what we hand back to the LLM. It is deliberately not an
// error; a plain result that lists the valid plays lets the model correct
// itself in a single turn instead of retrying the same lookup.
func (e *PlayNotFoundError) ToolMessage() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "%q is not one of the follow-up plays sampled by the simulation, "+
		"so there is no data for it. Only the plays listed below can be looked up; "+
		"copy one of them exactly as written. Note that a lowercase letter means that "+
		"tile is the blank, so it is a different play from its uppercase spelling.\n",
		strings.TrimSpace(e.Play))
	for _, p := range e.Available {
		fmt.Fprintf(&sb, "  - %s\n", p)
	}
	return sb.String()
}

var parenRepl = strings.NewReplacer("(", "", ")", "", " ", "")

// foldKey canonicalizes a play string for fuzzy matching. Case is meaningful
// in play notation (a lowercase letter is the blank), so folding it can match
// several distinct plays; callers must be prepared to get more than one row.
func foldKey(play string, dropParens bool) string {
	key := strings.ToUpper(strings.TrimSpace(play))
	if dropParens {
		return parenRepl.Replace(key)
	}
	return strings.Join(strings.Fields(key), " ")
}

// matchFollowup finds the play the model is asking about. It returns the
// family it belongs to, plus the index of the exact way if the model named one
// rather than the grouped play. An exact match wins outright; otherwise we
// relax the comparison, first on case (the model tends to uppercase the blank,
// turning sKIWEAR into SKIWEAR) and then on playthrough parentheses.
func matchFollowup(fams []*FollowupFact, playString string) (*FollowupFact, int) {
	for _, tier := range []func(string) string{
		strings.TrimSpace,
		func(p string) string { return foldKey(p, false) },
		func(p string) string { return foldKey(p, true) },
	} {
		want := tier(playString)
		for _, fam := range fams {
			if tier(fam.Play) == want {
				return fam, -1
			}
			for j, way := range fam.Ways {
				if tier(way.Play) == want {
					return fam, j
				}
			}
		}
	}
	return nil, -1
}

// GetFuturePlayMetadata analyzes a potential future play. If the play can be
// made in more than one way it returns the most likely one; use
// LookupFuturePlay to see every way.
func (a *Analyzer) GetFuturePlayMetadata(playString string) (*FuturePlayMetadata, error) {
	lookup, err := a.LookupFuturePlay(playString)
	if err != nil {
		return nil, err
	}
	return lookup.Ways[0], nil
}

// LookupFuturePlay looks a follow-up play up in the facts computed for this
// position. It returns a *PlayNotFoundError if the play isn't one the
// simulation sampled.
func (a *Analyzer) LookupFuturePlay(playString string) (*FuturePlayLookup, error) {
	log.Info().Str("play", playString).Msg("analyzing future play metadata")
	if a.facts == nil {
		return nil, errors.New("no analysis available for this position")
	}
	fams := a.facts.Followups

	fam, wayIdx := matchFollowup(fams, playString)
	if fam == nil {
		available := []string{}
		for _, f := range fams {
			available = append(available, f.Play)
			if f.Grouped() {
				for _, w := range f.Ways {
					available = append(available, w.Play)
				}
			}
		}
		return nil, &PlayNotFoundError{Play: playString, Available: available}
	}

	lookup := &FuturePlayLookup{}
	if wayIdx >= 0 {
		lookup.Asked = fam.Ways[wayIdx].Play
	}
	if fam.Grouped() {
		// A grouped play has no notation of its own - the blank has to land
		// somewhere - so the ways, not the family, are what we can analyze.
		lookup.Family = &FuturePlayFamily{
			Play:              fam.Play,
			CombinedPercent:   fam.Pct,
			ScoreRange:        scoreRange(fam.MinScore, fam.MaxScore),
			NeededDrawOptions: fam.NeededDraws,
			IsSetup:           fam.IsSetup,
		}
	}
	for i, way := range fam.Ways {
		req := "none"
		if i < len(fam.WayRequirements) {
			req = fam.WayRequirements[i]
		}
		lookup.Ways = append(lookup.Ways, &FuturePlayMetadata{
			Play:               way.Play,
			Score:              way.Score,
			IsBingo:            way.Bingo,
			NeededDraw:         splitDraw(way.NeededDraw),
			RequiresOtherPlay:  req,
			ProbabilityPercent: way.Pct,
			IsSetup:            fam.IsSetup,
		})
	}
	return lookup, nil
}

// splitDraw turns a draw like "SE" into the individual tiles that make it up.
// A ? is the blank.
func splitDraw(draw string) []string {
	out := []string{}
	for _, char := range draw {
		if (char >= 'A' && char <= 'Z') || char == '?' {
			out = append(out, string(char))
		}
	}
	return out
}
