package explainer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
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

// FuturePlayFamily describes a play that can be made in more than one way,
// because a blank can stand in for different tiles. The combined percentage is
// the one to quote: each individual way is only part of the chance of making
// the play.
type FuturePlayFamily struct {
	Play              string   `json:"play"`
	CombinedPercent   float64  `json:"combined_probability_percent"`
	ScoreRange        string   `json:"score_range"`
	NeededDrawOptions []string `json:"needed_draw_options"` // any one of these unlocks the play
}

// FuturePlayLookup is the result of looking a follow-up play up in the sim
// stats: every way of making it, plus the combined figures when there is more
// than one way.
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
	return "Get metadata about a potential future play including required tile draws, setup requirements, and probability. " +
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

// followupRow is one parsed row of the "Our follow-up play" table.
type followupRow struct {
	play       string
	neededDraw string // brace contents; a family row may list "B|C|?" alternatives
	scoreLow   int
	scoreHigh  int // equal to scoreLow unless the row shows a range
	percent    float64
}

// followupFamily is a play in the table together with the individual ways of
// making it, when there is more than one. The variants differ only in which
// tile is the blank; see playFamily in montecarlo/stats/heatmap.go.
type followupFamily struct {
	row      followupRow
	variants []followupRow
}

// followupRowRe matches a table row generated by playStatsStr in
// montecarlo/stats/heatmap.go: a play, an optional {NEEDED DRAW}, then score
// (possibly a range), count and percentage. Anchoring on the trailing numeric
// columns keeps this working for plays that overflow the play column.
var followupRowRe = regexp.MustCompile(`^(.*?)\s*(?:\{([^}]*)\})?\s+(\d+)(?:-(\d+))?\s+(\d+)\s+(\d+(?:\.\d+)?)$`)

// parseFollowupRow parses one row. It returns false for anything that isn't a
// table row.
func parseFollowupRow(line string) (followupRow, bool) {
	groups := followupRowRe.FindStringSubmatch(line)
	if groups == nil {
		return followupRow{}, false
	}
	scoreLow, err := strconv.Atoi(groups[3])
	if err != nil {
		return followupRow{}, false
	}
	scoreHigh := scoreLow
	if groups[4] != "" {
		if scoreHigh, err = strconv.Atoi(groups[4]); err != nil {
			return followupRow{}, false
		}
	}
	percent, err := strconv.ParseFloat(groups[6], 64)
	if err != nil {
		return followupRow{}, false
	}
	return followupRow{
		play:       strings.TrimSpace(groups[1]),
		neededDraw: groups[2],
		scoreLow:   scoreLow,
		scoreHigh:  scoreHigh,
		percent:    percent,
	}, true
}

// parseFollowupFamilies pulls the rows out of the "### Our follow-up play"
// section of the winning play's stats. Rows indented with a "-" marker are the
// individual ways of making the play above them. Rows we can't parse are
// skipped rather than failing the whole lookup.
func (a *Analyzer) parseFollowupFamilies() ([]followupFamily, error) {
	if a.winningStats == "" {
		return nil, errors.New("no winning stats available")
	}
	_, section, found := strings.Cut(a.winningStats, "### Our follow-up play")
	if !found {
		return nil, errors.New("no 'Our follow-up play' section found in winning stats")
	}

	fams := []followupFamily{}
	for _, line := range strings.Split(section, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "###") {
			// another section started; the follow-up table is done.
			break
		}
		if line == "" || strings.HasPrefix(line, "Play") || strings.HasPrefix(line, "---") ||
			strings.HasPrefix(line, "Bingo probability") {
			continue
		}
		isVariant := strings.HasPrefix(line, "- ")
		row, ok := parseFollowupRow(strings.TrimPrefix(line, "- "))
		if !ok {
			log.Warn().Str("line", line).Msg("unparseable line in follow-up play stats, skipping")
			continue
		}
		if isVariant && len(fams) > 0 {
			fams[len(fams)-1].variants = append(fams[len(fams)-1].variants, row)
			continue
		}
		fams = append(fams, followupFamily{row: row})
	}
	return fams, nil
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
// family it belongs to, plus the exact variant if the model named one rather
// than the grouped play. An exact match wins outright; otherwise we relax the
// comparison, first on case (the model tends to uppercase the blank, turning
// sKIWEAR into SKIWEAR) and then on playthrough parentheses.
func matchFollowup(fams []followupFamily, playString string) (*followupFamily, *followupRow) {
	for _, tier := range []func(string) string{
		func(p string) string { return strings.TrimSpace(p) },
		func(p string) string { return foldKey(p, false) },
		func(p string) string { return foldKey(p, true) },
	} {
		want := tier(playString)
		for i := range fams {
			if tier(fams[i].row.play) == want {
				return &fams[i], nil
			}
			for j := range fams[i].variants {
				if tier(fams[i].variants[j].play) == want {
					return &fams[i], &fams[i].variants[j]
				}
			}
		}
	}
	return nil, nil
}

// GetFuturePlayMetadata analyzes a potential future play by parsing
// winningStats. If the play can be made in more than one way it returns the
// most likely one; use LookupFuturePlay to see every way.
func (a *Analyzer) GetFuturePlayMetadata(playString string) (*FuturePlayMetadata, error) {
	lookup, err := a.LookupFuturePlay(playString)
	if err != nil {
		return nil, err
	}
	return lookup.Ways[0], nil
}

// LookupFuturePlay analyzes a potential future play by parsing winningStats.
// It returns a *PlayNotFoundError if the play isn't in the table.
func (a *Analyzer) LookupFuturePlay(playString string) (*FuturePlayLookup, error) {
	log.Info().Str("play", playString).Msg("analyzing future play metadata")

	fams, err := a.parseFollowupFamilies()
	if err != nil {
		return nil, err
	}
	fam, variant := matchFollowup(fams, playString)
	if fam == nil {
		available := []string{}
		for _, f := range fams {
			available = append(available, f.row.play)
			for _, v := range f.variants {
				available = append(available, v.play)
			}
		}
		return nil, &PlayNotFoundError{Play: playString, Available: available}
	}

	lookup := &FuturePlayLookup{}
	if variant != nil {
		lookup.Asked = variant.play
	}
	// A grouped play has no notation of its own - the blank has to land
	// somewhere - so the ways, not the family row, are what we can analyze.
	rows := fam.variants
	if len(rows) == 0 {
		rows = []followupRow{fam.row}
	} else {
		// Take the alternatives from the ways themselves rather than the
		// family row, whose cell may have been truncated to fit its column.
		opts := []string{}
		seen := map[string]bool{}
		for _, v := range fam.variants {
			if !seen[v.neededDraw] {
				seen[v.neededDraw] = true
				opts = append(opts, v.neededDraw)
			}
		}
		lookup.Family = &FuturePlayFamily{
			Play:              fam.row.play,
			CombinedPercent:   fam.row.percent,
			ScoreRange:        scoreRangeStr(fam.row),
			NeededDrawOptions: opts,
		}
	}
	for _, row := range rows {
		md, err := a.futurePlayMetadata(row)
		if err != nil {
			return nil, err
		}
		lookup.Ways = append(lookup.Ways, md)
	}
	return lookup, nil
}

func scoreRangeStr(row followupRow) string {
	if row.scoreLow == row.scoreHigh {
		return strconv.Itoa(row.scoreHigh)
	}
	return fmt.Sprintf("%d-%d", row.scoreLow, row.scoreHigh)
}

// futurePlayMetadata builds the metadata for a single row of the follow-up
// play table.
func (a *Analyzer) futurePlayMetadata(row followupRow) (*FuturePlayMetadata, error) {
	bestPlay := strings.TrimSpace(a.winningPlay)
	normalizedBestPlay := stats.Normalize(bestPlay)
	normalizedPlay := stats.Normalize(row.play)
	drawLetters := []string{}
	reqOtherPlay := "none"
	isBingo := false

	if !strings.HasPrefix(row.play, "exchange ") && row.play != "pass" {
		// Split individual letters. A ? is the blank.
		for _, char := range row.neededDraw {
			if (char >= 'A' && char <= 'Z') || char == '?' {
				drawLetters = append(drawLetters, string(char))
			}
		}

		npfields := strings.Fields(normalizedPlay)
		if len(npfields) < 2 {
			return nil, fmt.Errorf("invalid play string format: %s", row.play)
		}

		// Determine if it's a bingo by counting tiles played (ignoring
		// playthrough, which Normalize turned into dots)
		tilesUsed := len(npfields[1]) - strings.Count(npfields[1], ".")
		isBingo = tilesUsed == 7

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
		if err != nil || m.Score() != row.scoreLow {
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
		if err != nil || m.Score() != row.scoreLow {
			mscore := 0
			if m != nil {
				mscore = m.Score()
			}
			log.Err(err).Str("fields", npfields[0]+","+npfields[1]).
				Int("score", mscore).
				Msg("failed to parse move or score mismatch, assuming our best play needed")
			if reqOtherPlay == "none" {
				reqOtherPlay = "requires us to play " + bestPlay + " first"
			}
		}
	}

	md := &FuturePlayMetadata{
		Play:               row.play,
		Score:              row.scoreLow,
		IsBingo:            isBingo,
		NeededDraw:         drawLetters,
		RequiresOtherPlay:  reqOtherPlay,
		ProbabilityPercent: row.percent,
	}
	log.Info().Interface("metadata", md).Msg("analyzed future play metadata")
	return md, nil
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
