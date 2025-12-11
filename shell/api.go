package shell

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/explainer"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/preendgame"
)

const defaultEndgamePlies = 4

//go:embed render_template.html
var renderTemplateHTML string

// RenderTemplateData holds all data passed to the 3D render template
type RenderTemplateData struct {
	FEN            string
	Rack           string
	RemainingTiles template.JS // JSON of remaining tiles
	TileColor      string
	BoardColor     string
	Player0Name    string
	Player1Name    string
	Player0Score   int
	Player1Score   int
	Heatmap        template.JS // JSON of heatmap data
	HeatmapActive  bool        // Whether heatmap mode is active
	HeatmapPlay    string      // The play being analyzed for heatmap
	HeatmapPly     int         // The ply index for heatmap
	PlayerOnTurn   int         // 0 or 1 - which player is on turn
	LastPlay       string      // Last play summary
	AlphabetScores template.JS // JSON map of letter → score for the current alphabet
}

type Response struct {
	message string
}

type CmdOptions map[string][]string

func (c CmdOptions) String(key string) string {
	v := c[key]
	if len(v) > 0 {
		return v[0]
	}
	return ""
}

func (c CmdOptions) Int(key string) (int, error) {
	v := c[key]
	if len(v) == 0 {
		return 0, errors.New(key + " not found in options")
	}
	return strconv.Atoi(v[0])
}
func (c CmdOptions) IntDefault(key string, defaultI int) (int, error) {
	v := c[key]
	if len(v) == 0 {
		return defaultI, nil
	}
	return strconv.Atoi(v[0])
}

func (c CmdOptions) Bool(key string) bool {
	v := c[key]
	if len(v) == 0 {
		return false
	}
	return strings.ToLower(v[0]) == "true"
}

func (c CmdOptions) StringArray(key string) []string {
	return c[key]
}

func msg(message string) *Response {
	return &Response{message: message}
}

func (sc *ShellController) set(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return msg(sc.options.ToDisplayText()), nil
	}
	opt := cmd.args[0]
	if len(cmd.args) == 1 {
		_, val := sc.options.Show(opt)
		return msg(val), nil
	}
	values := cmd.args[1:]
	ret, err := sc.Set(opt, values)
	if err != nil {
		return nil, err
	}
	return msg("set " + opt + " to " + ret), nil
}

func (sc *ShellController) setConfig(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil || len(cmd.args) < 2 {
		return nil, errors.New("usage: setconfig <key> <value>")
	}

	key := cmd.args[0]
	value := cmd.args[1]

	// Set the configuration value
	sc.config.Set(key, value)

	// Save the configuration to file
	err := sc.config.Write()
	if err != nil {
		return nil, fmt.Errorf("failed to save config: %w", err)
	}

	return msg(fmt.Sprintf("set config %s to %s and saved to file", key, value)), nil
}

func (sc *ShellController) gid(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("no currently loaded game")
	}
	gid := sc.game.History().Uid
	if gid != "" {
		idauth := sc.game.History().IdAuth
		fullID := strings.TrimSpace(idauth + " " + gid)
		return msg(fullID), nil
	}
	return nil, errors.New("no ID set for this game")
}

func (sc *ShellController) newGame(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	players := []*pb.PlayerInfo{
		{Nickname: "arcadio", RealName: "José Arcadio Buendía"},
		{Nickname: "úrsula", RealName: "Úrsula Iguarán Buendía"},
	}
	if frand.Intn(2) == 1 {
		players[0], players[1] = players[1], players[0]
	}

	opts := sc.options.GameOptions
	leavesFile := ""
	if opts.BoardLayoutName == board.SuperCrosswordGameLayout {
		leavesFile = "super-leaves.klv2"
	}

	conf := &bot.BotConfig{Config: *sc.config, LeavesFile: leavesFile}

	g, err := bot.NewBotTurnPlayer(conf, &opts, players, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return nil, err
	}
	sc.game = g
	err = sc.initGameDataStructures()
	if err != nil {
		return nil, err
	}
	sc.curTurnNum = 0
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) solving() bool {
	return (sc.endgameSolver != nil && sc.endgameSolver.IsSolving()) ||
		(sc.preendgameSolver != nil && sc.preendgameSolver.IsSolving()) ||
		(sc.simmer != nil && sc.simmer.IsSimming()) ||
		(sc.rangefinder != nil && sc.rangefinder.IsBusy()) ||
		sc.botBusy
}

func (sc *ShellController) load(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need arguments for load")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	if cmd.args[0] == "cgp" {
		if len(cmd.args) < 2 {
			return nil, errors.New("need to provide a cgp string")
		}
		cgpStr := strings.Join(cmd.args[1:], " ")
		err := sc.loadCGP(cgpStr)
		if err != nil {
			return nil, err
		}

	} else {
		err := sc.loadGCG(cmd.args)
		if err != nil {
			return nil, err
		}
	}
	sc.curTurnNum = 0
	// Initialize a fresh variation tree for the newly loaded game
	sc.initializeVariationTree()
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) unload(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	sc.game = nil
	return msg("No active game."), nil
}

func (sc *ShellController) show(cmd *shellcmd) (*Response, error) {
	output := sc.game.ToDisplayText()

	// Add variation info
	if sc.currentVariation != nil {
		varLabel := "main line"
		if sc.currentVariation.variationID != 0 {
			varLabel = fmt.Sprintf("Variation %d", sc.currentVariation.variationID)
		}
		output += fmt.Sprintf("\n[%s, turn %d]", varLabel, sc.curTurnNum)
	}

	return msg(output), nil
}

func (sc *ShellController) list(cmd *shellcmd) (*Response, error) {
	res := sc.genDisplayMoveList()
	return msg(res), nil
}

func (sc *ShellController) next(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	err := sc.setToTurn(sc.curTurnNum + 1)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) last(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	err := sc.setToTurn(len(sc.game.History().Events))
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) prev(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	err := sc.setToTurn(sc.curTurnNum - 1)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) name(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) < 3 {
		return nil, errors.New("need 3 arguments for name")
	}
	if sc.game == nil {
		return nil, errors.New("no game is loaded")
	}
	p, err := strconv.Atoi(cmd.args[0])
	if err != nil {
		return nil, err
	}
	p -= 1
	if p < 0 || p >= len(sc.game.History().Players) {
		return nil, errors.New("player index not in range")
	}
	err = sc.game.RenamePlayer(p, &pb.PlayerInfo{
		Nickname: cmd.args[1],
		RealName: strings.Join(cmd.args[2:], " "),
	})
	if err != nil {
		return nil, err
	}

	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) note(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) < 1 {
		return nil, errors.New("need at least one argument for note")
	}
	if sc.game == nil {
		return nil, errors.New("no game is loaded")
	}
	if sc.game.Turn() == 0 {
		return nil, errors.New("there must be at least one turn that has been played")
	}
	note := strings.Join(cmd.args, " ")
	err := sc.game.AddNote(note)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) turn(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need argument for turn")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}
	t, err := strconv.Atoi(cmd.args[0])
	if err != nil {
		return nil, err
	}
	err = sc.setToTurn(t)
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) rack(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need argument for rack")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}
	rack := cmd.args[0]
	err := sc.addRack(strings.ToUpper(rack))
	if err != nil {
		return nil, err
	}
	return msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) gameState(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	if sc.game == nil {
		return nil, errors.New("no game is loaded")
	}
	inbag := sc.game.Bag().TilesRemaining()
	onopprack := sc.game.RackFor(sc.game.NextPlayer()).NumTiles()
	rack := sc.game.RackFor(sc.game.PlayerOnTurn()).TilesOn().UserVisible(sc.game.Alphabet())
	spread := sc.game.CurrentSpread()

	var spreadFriendly string
	if spread == 0 {
		spreadFriendly = "The game is tied."
	} else if spread > 0 {
		spreadFriendly = fmt.Sprintf("We are ahead by %d points.", spread)
	} else {
		spreadFriendly = fmt.Sprintf("We are behind by %d points.", -spread)
	}
	var bagStats string

	vowels := 0
	consonants := 0
	blanks := 0
	powerTiles := ""

	if inbag > 0 {
		bagTiles := sc.game.Bag().Peek()
		oppTiles := sc.game.RackFor(sc.game.NextPlayer()).TilesOn()
		ld := sc.game.Bag().LetterDistribution()

		combined := append(bagTiles, oppTiles...)
		for i := range combined {
			tile := combined[i]
			if tile.IsVowel(ld) {
				vowels++
			} else if tile == 0 {
				blanks++
			} else {
				consonants++
			}
			if ld.Score(tile) > 5 || tile == 0 || tile.UserVisible(ld.TileMapping(), false) == "S" {
				powerTiles += fmt.Sprintf("%s ", tile.UserVisible(ld.TileMapping(), false))
			}
		}

		bagStats = fmt.Sprintf(" In the bag: %d vowels, %d consonants, %d blanks.", vowels, consonants, blanks)
		if powerTiles != "" {
			bagStats += fmt.Sprintf(" Unseen power tiles: %s", powerTiles)
		}
	}

	return msg(fmt.Sprintf("Our rack is %s and there are %d tiles unseen to us (so %d in our opponent's rack, and %d in the bag). %s%s",
		rack, inbag+int(onopprack), onopprack,
		inbag, spreadFriendly, bagStats)), nil

}

func (sc *ShellController) generate(cmd *shellcmd) (*Response, error) {
	var numPlays int
	var err error

	if sc.game == nil {
		return nil, errors.New("please load or create a game first")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	if cmd.args == nil {
		numPlays = 15
	} else {
		numPlays, err = strconv.Atoi(cmd.args[0])
		if err != nil {
			return nil, err
		}
	}
	return msg(sc.genMovesAndDescription(numPlays)), nil
}

func (sc *ShellController) autoplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.handleAutoplay(cmd.args, cmd.options)
}

func (sc *ShellController) sim(cmd *shellcmd) (*Response, error) {
	return sc.handleSim(cmd.args, cmd.options)
}

func (sc *ShellController) add(cmd *shellcmd) (*Response, error) {
	return nil, sc.addPlay(cmd.args)
}

func (sc *ShellController) commit(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitPlay(cmd.args)
}

func (sc *ShellController) variation(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first")
	}

	if len(cmd.args) == 0 {
		// Default to "var list"
		return sc.variationList()
	}

	subcommand := cmd.args[0]
	switch subcommand {
	case "list", "ls":
		return sc.variationList()
	case "info":
		return sc.variationInfo()
	case "delete", "del", "rm":
		if len(cmd.args) < 2 {
			return nil, errors.New("var delete requires a variation ID")
		}
		return sc.variationDelete(cmd.args[1])
	case "promote":
		if len(cmd.args) < 2 {
			return nil, errors.New("var promote requires a variation ID")
		}
		return sc.variationPromote(cmd.args[1])
	case "main":
		return sc.variationSwitch(0)
	default:
		// Try to parse as variation ID
		varID, err := strconv.Atoi(subcommand)
		if err != nil {
			return nil, fmt.Errorf("unknown variation subcommand or invalid ID: %s", subcommand)
		}
		return sc.variationSwitch(varID)
	}
}

func (sc *ShellController) eliteplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitAIMove()
}

func (sc *ShellController) hastyplay(cmd *shellcmd) (*Response, error) {
	return nil, sc.commitHastyMove()
}

func (sc *ShellController) selftest(cmd *shellcmd) (*Response, error) {
	_, err := sc.newGame(cmd)
	if err != nil {
		return nil, err
	}
	for sc.IsPlaying() {
		err := sc.commitAIMove()
		if err != nil {
			return nil, err
		}
	}
	return nil, nil
}

func (sc *ShellController) challenge(cmd *shellcmd) (*Response, error) {
	if sc.solving() {
		return nil, errMacondoSolving
	}
	fields := cmd.args

	if len(fields) > 0 {
		addlBonus, err := strconv.Atoi(fields[0])
		if err != nil {
			return nil, err
		}
		// Set it to single to have a base bonus of 0, and add passed-in bonus.
		sc.game.SetChallengeRule(pb.ChallengeRule_SINGLE)
		_, err = sc.game.ChallengeEvent(addlBonus, 0)
		if err != nil {
			return nil, err
		}
		sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	} else {
		// Do double-challenge.
		_, err := sc.game.ChallengeEvent(0, 0)
		if err != nil {
			return nil, err
		}
	}
	sc.curTurnNum = sc.game.Turn()
	return msg(sc.game.ToDisplayText()), nil
}

// endgameParams holds parsed parameters for endgame solving
type endgameParams struct {
	plies      int
	maxtime    int
	maxthreads int
	enableFW   bool // first-win optimization
}

// endgamePrepare parses options and initializes the endgame solver.
// Returns the params needed for running.
func (sc *ShellController) endgamePrepare(cmd *shellcmd) (*endgameParams, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	var err error
	params := &endgameParams{
		maxthreads: runtime.NumCPU(),
	}

	if params.plies, err = cmd.options.IntDefault("plies", defaultEndgamePlies); err != nil {
		return nil, err
	}
	if params.maxtime, err = cmd.options.IntDefault("maxtime", 0); err != nil {
		return nil, err
	}
	if params.maxthreads, err = cmd.options.IntDefault("threads", params.maxthreads); err != nil {
		return nil, err
	}
	multipleVars, err := cmd.options.IntDefault("multiple-vars", 1)
	if err != nil {
		return nil, err
	}
	parallelAlgo := cmd.options.String("parallel-algo")
	if parallelAlgo == "" {
		parallelAlgo = negamax.ParallelAlgoAuto
	}
	disableID := cmd.options.Bool("disable-id")
	disableTT := cmd.options.Bool("disable-tt")
	params.enableFW = cmd.options.Bool("first-win-optim")
	preventSR := cmd.options.Bool("prevent-slowroll")
	disableNegascout := cmd.options.Bool("disable-negascout")
	nullWindow := cmd.options.Bool("null-window")

	// clear out the last value of this endgame node; gc should delete the tree.
	sc.endgameSolver = new(negamax.Solver)

	if cmd.options.Bool("log") {
		sc.endgameLogFile, err = os.Create(EndgameLog)
		if err != nil {
			return nil, err
		}
		sc.endgameSolver.SetLogStream(sc.endgameLogFile)
		sc.showMessage("endgame will log to " + EndgameLog)
	}

	sc.showMessage(fmt.Sprintf(
		"plies %v, maxtime %v, threads %v",
		params.plies, params.maxtime, params.maxthreads))

	sc.game.SetBackupMode(game.SimulationMode)

	sc.endgameCtx, sc.endgameCancel = context.WithCancel(context.Background())
	if params.maxtime > 0 {
		sc.endgameCtx, sc.endgameCancel = context.WithTimeout(sc.endgameCtx, time.Duration(params.maxtime)*time.Second)
	}

	gd, err := kwg.GetKWG(sc.game.Config().WGLConfig(), sc.game.LexiconName())
	if err != nil {
		return nil, err
	}

	mg := movegen.NewGordonGenerator(gd, sc.game.Board(), sc.game.Bag().LetterDistribution())

	err = sc.endgameSolver.Init(mg, sc.game.Game)
	if err != nil {
		return nil, err
	}

	sc.endgameSolver.SetIterativeDeepening(!disableID)
	sc.endgameSolver.SetTranspositionTableOptim(!disableTT)
	sc.endgameSolver.SetThreads(params.maxthreads)
	if err = sc.endgameSolver.SetParallelAlgorithm(parallelAlgo); err != nil {
		return nil, err
	}
	sc.endgameSolver.SetFirstWinOptim(params.enableFW)
	sc.endgameSolver.SetNullWindowOptim(nullWindow)
	sc.endgameSolver.SetSolveMultipleVariations(multipleVars)
	sc.endgameSolver.SetPreventSlowroll(preventSR)
	sc.endgameSolver.SetNegascoutOptim(!disableNegascout)

	return params, nil
}

// endgameRunSync runs the endgame solver synchronously and returns the result.
func (sc *ShellController) endgameRunSync(params *endgameParams) (string, error) {
	defer func() {
		sc.game.SetBackupMode(game.InteractiveGameplayMode)
		sc.game.SetStateStackLength(1)
	}()

	val, seq, err := sc.endgameSolver.Solve(sc.endgameCtx, params.plies)
	if err != nil {
		return "", err
	}

	var result strings.Builder
	if !params.enableFW {
		result.WriteString(fmt.Sprintf("Best sequence has a spread difference (value) of %+d\n", val))
	} else {
		if val+int16(sc.game.CurrentSpread()) > 0 {
			result.WriteString("Win found!\n")
		} else {
			result.WriteString("Win was not found.\n")
		}
		result.WriteString(fmt.Sprintf("Spread diff: %+d. Note: this sequence may not be correct. Turn off first-win-optim to search more accurately.\n", val))
	}
	result.WriteString(fmt.Sprintf("Final spread after seq: %+d\n", val+int16(sc.game.CurrentSpread())))
	result.WriteString("Best move: ")
	result.WriteString(seq[0].ShortDescription() + "\n")
	// Format sequence using ShortDescription which includes score
	result.WriteString("Best sequence:\n")
	for idx, m := range seq {
		result.WriteString(fmt.Sprintf("%d) %v (%d)\n", idx+1, m.ShortDescription(), m.Score()))
	}

	variations := sc.endgameSolver.Variations()
	if len(variations) > 1 {
		result.WriteString("Other variations:\n")
		for i := range variations[1:] {
			result.WriteString(fmt.Sprintf("%d) %s\n", i+2, variations[i+1].NLBString()))
		}
	}

	return result.String(), nil
}

// endgameSync runs endgame synchronously and returns the result.
// This is the preferred method for scripts.
func (sc *ShellController) endgameSync(cmd *shellcmd) (*Response, error) {
	params, err := sc.endgamePrepare(cmd)
	if err != nil {
		return nil, err
	}

	sc.showMessage(sc.game.ToDisplayText())

	result, err := sc.endgameRunSync(params)
	if err != nil {
		return nil, err
	}

	return msg(result), nil
}

// endgame runs endgame asynchronously (for interactive shell use).
func (sc *ShellController) endgame(cmd *shellcmd) (*Response, error) {
	// Handle subcommands first
	if len(cmd.args) > 0 && cmd.args[0] == "stop" {
		if sc.endgameSolver != nil && sc.endgameSolver.IsSolving() {
			sc.endgameCancel()
		} else {
			return nil, errors.New("no endgame to cancel")
		}
		return msg(""), nil
	}

	if len(cmd.args) > 0 && cmd.args[0] == "metrics" {
		if sc.endgameSolver == nil {
			return nil, errors.New("no endgame has been run yet")
		}
		return msg(sc.endgameSolver.GetMetrics()), nil
	}

	if len(cmd.args) > 0 && cmd.args[0] == "output" {
		if sc.endgameSolver == nil {
			return nil, errors.New("no endgame has been run yet")
		}
		return msg(sc.endgameSolver.ShortDetails()), nil
	}

	params, err := sc.endgamePrepare(cmd)
	if err != nil {
		return nil, err
	}

	sc.showMessage(sc.game.ToDisplayText())

	go func() {
		result, err := sc.endgameRunSync(params)
		if err != nil {
			sc.showError(err)
			return
		}
		sc.showMessage(result)
	}()

	return msg(""), nil
}

// pegParams holds parsed parameters for pre-endgame solving
type pegParams struct {
	maxsolutions int
}

// pegPrepare parses options and initializes the pre-endgame solver.
func (sc *ShellController) pegPrepare(cmd *shellcmd) (*pegParams, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	var maxtime int
	var maxthreads = 0
	var err error
	endgamePlies := 4

	params := &pegParams{
		maxsolutions: 30,
	}

	knownOppRack := cmd.options.String("opprack")

	if endgamePlies, err = cmd.options.IntDefault("endgameplies", defaultEndgamePlies); err != nil {
		return nil, err
	}
	if maxtime, err = cmd.options.IntDefault("maxtime", 0); err != nil {
		return nil, err
	}
	if maxthreads, err = cmd.options.IntDefault("threads", 0); err != nil {
		return nil, err
	}
	if params.maxsolutions, err = cmd.options.IntDefault("maxsolutions", params.maxsolutions); err != nil {
		return nil, err
	}
	skipNonEmptying := cmd.options.Bool("skip-non-emptying")
	skipLoss := cmd.options.Bool("skip-loss")
	earlyCutoff := cmd.options.Bool("early-cutoff")
	skipTiebreaker := cmd.options.Bool("skip-tiebreaker")
	disableIterativeDeepening := cmd.options.Bool("disable-id")
	movesToSolveStrs := cmd.options.StringArray("only-solve")
	movesToSolve := []*move.Move{}

	for _, ms := range movesToSolveStrs {
		m, err := sc.game.ParseMove(
			sc.game.PlayerOnTurn(),
			sc.options.lowercaseMoves,
			strings.Fields(ms),
			false)
		if err != nil {
			return nil, err
		}
		movesToSolve = append(movesToSolve, m)
	}
	sc.showMessage(fmt.Sprintf(
		"endgameplies %v, maxtime %v, threads %v",
		endgamePlies, maxtime, maxthreads))

	gd, err := kwg.GetKWG(sc.game.Config().WGLConfig(), sc.game.LexiconName())
	if err != nil {
		return nil, err
	}

	sc.preendgameSolver = new(preendgame.Solver)
	sc.preendgameSolver.Init(sc.game.Game, gd)

	if maxthreads != 0 {
		sc.preendgameSolver.SetThreads(maxthreads)
	}

	if cmd.options.Bool("log") {
		sc.pegLogFile, err = os.Create(PEGLog)
		if err != nil {
			return nil, err
		}
		sc.preendgameSolver.SetLogStream(sc.pegLogFile)
		sc.showMessage("peg will log to " + PEGLog)
	}

	if knownOppRack != "" {
		knownOppRack = strings.ToUpper(knownOppRack)
		r, err := tilemapping.ToMachineLetters(knownOppRack, sc.game.Alphabet())
		if err != nil {
			return nil, err
		}
		sc.preendgameSolver.SetKnownOppRack(r)
	}
	sc.preendgameSolver.SetEndgamePlies(endgamePlies)
	sc.preendgameSolver.SetEarlyCutoffOptim(earlyCutoff)
	sc.preendgameSolver.SetSkipNonEmptyingOptim(skipNonEmptying)
	sc.preendgameSolver.SetSkipTiebreaker(skipTiebreaker)
	sc.preendgameSolver.SetSkipLossOptim(skipLoss)
	sc.preendgameSolver.SetIterativeDeepening(!disableIterativeDeepening)
	sc.preendgameSolver.SetSolveOnly(movesToSolve)

	sc.pegCtx, sc.pegCancel = context.WithCancel(context.Background())
	if maxtime > 0 {
		sc.pegCtx, sc.pegCancel = context.WithTimeout(sc.pegCtx, time.Duration(maxtime)*time.Second)
	}

	return params, nil
}

// pegRunSync runs the pre-endgame solver synchronously and returns the result.
func (sc *ShellController) pegRunSync(params *pegParams) (string, error) {
	moves, err := sc.preendgameSolver.Solve(sc.pegCtx)
	if err != nil {
		return "", err
	}

	maxsolutions := params.maxsolutions
	if len(moves) < maxsolutions {
		maxsolutions = len(moves)
	}

	if sc.pegLogFile != nil {
		err := sc.pegLogFile.Close()
		if err != nil {
			log.Err(err).Msg("closing-log-file")
		}
	}

	return "Winner: " + moves[0].Play.ShortDescription() + "\n" + sc.preendgameSolver.SolutionStats(maxsolutions), nil
}

// preendgameSync runs pre-endgame synchronously and returns the result.
// This is the preferred method for scripts.
func (sc *ShellController) preendgameSync(cmd *shellcmd) (*Response, error) {
	params, err := sc.pegPrepare(cmd)
	if err != nil {
		return nil, err
	}

	sc.showMessage(sc.game.ToDisplayText())

	result, err := sc.pegRunSync(params)
	if err != nil {
		return nil, err
	}

	return msg(result), nil
}

// preendgame runs pre-endgame asynchronously (for interactive shell use).
func (sc *ShellController) preendgame(cmd *shellcmd) (*Response, error) {
	// Handle subcommands first
	if len(cmd.args) > 0 && cmd.args[0] == "stop" {
		if sc.preendgameSolver != nil && sc.preendgameSolver.IsSolving() {
			sc.pegCancel()
		} else {
			return nil, errors.New("no pre-endgame to cancel")
		}
		return msg(""), nil
	}

	if len(cmd.args) > 0 && cmd.args[0] == "output" {
		if sc.preendgameSolver == nil {
			return nil, errors.New("no pre-endgame has been run yet")
		}
		return msg(sc.preendgameSolver.ShortDetails()), nil
	}

	params, err := sc.pegPrepare(cmd)
	if err != nil {
		return nil, err
	}

	sc.showMessage(sc.game.ToDisplayText())

	go func() {
		result, err := sc.pegRunSync(params)
		if err != nil {
			sc.showError(err)
			return
		}
		sc.showMessage(result)
	}()

	return msg(""), nil
}

// inferParams holds parsed parameters for inference
type inferParams struct {
	timesec int
	ctx     context.Context
}

// inferPrepare parses options and prepares the rangefinder for inference.
func (sc *ShellController) inferPrepare(cmd *shellcmd) (*inferParams, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	var err error
	var threads, timesec int

	for opt := range cmd.options {
		switch opt {
		case "threads":
			threads, err = cmd.options.Int(opt)
			if err != nil {
				return nil, err
			}

		case "time":
			timesec, err = cmd.options.Int(opt)
			if err != nil {
				return nil, err
			}

		default:
			return nil, errors.New("option " + opt + " not recognized")
		}
	}

	if threads != 0 {
		sc.rangefinder.SetThreads(threads)
	}
	if timesec == 0 {
		timesec = 60
	}

	err = sc.rangefinder.PrepareFinder(sc.game.RackFor(sc.game.PlayerOnTurn()).TilesOn())
	if err != nil {
		return nil, err
	}

	ctx, _ := context.WithTimeout(
		context.Background(), time.Duration(timesec*int(time.Second)))

	return &inferParams{
		timesec: timesec,
		ctx:     ctx,
	}, nil
}

// inferRunSync runs inference synchronously and returns the result.
func (sc *ShellController) inferRunSync(params *inferParams) (string, error) {
	err := sc.rangefinder.Infer(params.ctx)
	if err != nil {
		return "", err
	}
	return sc.rangefinder.AnalyzeInferences(false), nil
}

// inferSync runs inference synchronously and returns the result.
// This is the preferred method for scripts.
func (sc *ShellController) inferSync(cmd *shellcmd) (*Response, error) {
	params, err := sc.inferPrepare(cmd)
	if err != nil {
		return nil, err
	}

	sc.showMessage("Rangefinding started. Please wait until it is done.")

	result, err := sc.inferRunSync(params)
	if err != nil {
		return nil, err
	}

	return msg(result), nil
}

// infer runs inference asynchronously (for interactive shell use).
func (sc *ShellController) infer(cmd *shellcmd) (*Response, error) {
	// Handle subcommands first
	if len(cmd.args) > 0 {
		var err error
		switch cmd.args[0] {
		case "log":
			sc.rangefinderFile, err = os.Create(InferLog)
			if err != nil {
				return nil, err
			}
			sc.rangefinder.SetLogStream(sc.rangefinderFile)
			sc.showMessage("inference engine will log to " + InferLog)

		case "details":
			return msg(sc.rangefinder.AnalyzeInferences(true)), nil

		case "output":
			return msg(sc.rangefinder.AnalyzeInferences(false)), nil

		default:
			return nil, errors.New("don't recognize " + cmd.args[0])
		}

		return nil, nil
	}

	params, err := sc.inferPrepare(cmd)
	if err != nil {
		return nil, err
	}

	sc.showMessage("Rangefinding started. Please wait until it is done.")
	sc.showMessage("Note that the default infer timeout has been increased to 60 seconds for more accuracy. See `help infer` for more information.")

	go func() {
		result, err := sc.inferRunSync(params)
		if err != nil {
			sc.showError(err)
			return
		}
		sc.showMessage(result)
		log.Debug().Msg("inference thread exiting...")
	}()

	return nil, nil
}

func (sc *ShellController) help(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return usage("standard")
	} else {
		helptopic := cmd.args[0]
		return usageTopic(helptopic)
	}
}

func (sc *ShellController) alias(cmd *shellcmd) (*Response, error) {
	// No arguments - list all aliases
	if cmd.args == nil || len(cmd.args) == 0 {
		if len(sc.aliases) == 0 {
			return msg("No aliases defined"), nil
		}

		// Sort by alias name for consistent output
		names := make([]string, 0, len(sc.aliases))
		for name := range sc.aliases {
			names = append(names, name)
		}
		sort.Strings(names)

		var result strings.Builder
		result.WriteString("Defined aliases:\n")
		for _, name := range names {
			result.WriteString(fmt.Sprintf("  %s = %s\n", name, sc.aliases[name]))
		}
		return msg(result.String()), nil
	}

	subcommand := cmd.args[0]

	switch subcommand {
	case "set":
		// alias set <name> <command>
		if len(cmd.args) < 3 {
			return nil, errors.New("usage: alias set <name> <command>")
		}
		name := cmd.args[1]

		// Reconstruct the full command from args and options
		commandParts := cmd.args[2:]
		for opt, values := range cmd.options {
			for _, val := range values {
				commandParts = append(commandParts, "-"+opt, val)
			}
		}
		command := strings.Join(commandParts, " ")

		sc.aliases[name] = command

		// Save to config
		sc.config.Set(config.ConfigAliases, sc.aliases)
		err := sc.config.Write()
		if err != nil {
			return nil, fmt.Errorf("failed to save alias: %w", err)
		}

		return msg(fmt.Sprintf("Alias '%s' set to: %s", name, command)), nil

	case "delete", "remove", "rm":
		// alias delete <name>
		if len(cmd.args) < 2 {
			return nil, errors.New("usage: alias delete <name>")
		}
		name := cmd.args[1]

		if _, exists := sc.aliases[name]; !exists {
			return nil, fmt.Errorf("alias '%s' not found", name)
		}

		delete(sc.aliases, name)

		// Save to config
		sc.config.Set(config.ConfigAliases, sc.aliases)
		err := sc.config.Write()
		if err != nil {
			return nil, fmt.Errorf("failed to save config: %w", err)
		}

		return msg(fmt.Sprintf("Alias '%s' deleted", name)), nil

	case "show":
		// alias show <name>
		if len(cmd.args) < 2 {
			return nil, errors.New("usage: alias show <name>")
		}
		name := cmd.args[1]

		if command, exists := sc.aliases[name]; exists {
			return msg(fmt.Sprintf("%s = %s", name, command)), nil
		}
		return nil, fmt.Errorf("alias '%s' not found", name)

	case "list":
		// Same as calling with no arguments
		return sc.alias(&shellcmd{cmd: "alias", args: nil, options: nil})

	default:
		return nil, fmt.Errorf("unknown subcommand '%s'. Valid: set, delete, show, list", subcommand)
	}
}

func (sc *ShellController) setMode(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return msg("Current mode: " + modeToStr(sc.curMode)), nil
	}
	mode := cmd.args[0]
	m, err := modeFromStr(mode)
	if err != nil {
		return nil, err
	}
	sc.curMode = m
	return msg("Setting current mode to " + mode), nil
}

func (sc *ShellController) export(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("please provide a filename to save to")
	}
	filename := cmd.args[0]
	contents, err := gcgio.GameHistoryToGCG(sc.game.History(), true)
	if err != nil {
		return nil, err
	}
	f, err := os.Create(filename)
	if err != nil {
		return nil, err
	}
	log.Debug().Interface("game-history", sc.game.History()).Msg("converted game history to gcg")
	f.WriteString(contents)
	f.Close()
	return msg("gcg written to " + filename), nil
}

func (sc *ShellController) render3D(cmd *shellcmd) (*Response, error) {
	if sc.game == nil {
		return nil, errors.New("please load a game first with the `load` command")
	}

	// Get color options
	tileColor := cmd.options.String("tile-color")
	if tileColor == "" {
		tileColor = "orange"
	}
	boardColor := cmd.options.String("board-color")
	if boardColor == "" {
		boardColor = "jade"
	}

	// Get heatmap option
	heatmapPlay := cmd.options.String("heatmap")
	heatmapPly := 0
	if plyStr := cmd.options.String("ply"); plyStr != "" {
		var err error
		heatmapPly, err = strconv.Atoi(plyStr)
		if err != nil {
			return nil, fmt.Errorf("invalid ply value: %w", err)
		}
	}

	var heatmapData [][]float64
	var heatmapMove *move.Move
	if heatmapPlay != "" {
		if sc.simStats == nil {
			return nil, errors.New("no simulation stats available - run sim with -collect-heatmap true first")
		}

		// Parse and validate the move
		var err error
		heatmapMove, err = sc.game.ParseMove(
			sc.game.PlayerOnTurn(), sc.options.lowercaseMoves, strings.Fields(heatmapPlay), false)
		if err != nil {
			return nil, fmt.Errorf("invalid heatmap move: %w", err)
		}

		heatmap, err := sc.simStats.CalculateHeatmap(heatmapPlay, heatmapPly)
		if err != nil {
			return nil, fmt.Errorf("failed to calculate heatmap: %w", err)
		}
		// Extract fractionOfMax values as 2D array
		heatmapData = make([][]float64, len(heatmap.Squares()))
		for i, row := range heatmap.Squares() {
			heatmapData[i] = make([]float64, len(row))
			for j, heat := range row {
				heatmapData[i][j] = heat.FractionOfMax()
			}
		}
	}

	// Get FEN from the board
	board := sc.game.Board()
	alph := sc.game.Alphabet()

	// If rendering a heatmap, temporarily place the move on a copy of the board
	var fen string
	if heatmapMove != nil {
		// Create a copy of the board
		tempBoard := board.Copy()
		tempBoard.PlaceMoveTiles(heatmapMove)
		fen = tempBoard.ToFEN(alph)
	} else {
		fen = board.ToFEN(alph)
	}

	// Get rack if available
	rack := ""
	if sc.game.RackFor(sc.game.PlayerOnTurn()) != nil {
		currentRack := sc.game.RackFor(sc.game.PlayerOnTurn())

		// If we're rendering a heatmap with a move, remove the played tiles from the rack
		if heatmapMove != nil && (heatmapMove.Action() == move.MoveTypePlay || heatmapMove.Action() == move.MoveTypeExchange) {
			// Make a copy of the rack and remove the tiles
			rackCopy := currentRack.Copy()
			for _, t := range heatmapMove.Tiles() {
				if t != 0 {
					rackCopy.Take(t.IntrinsicTileIdx())
				}
			}
			rack = rackCopy.String()
		} else {
			rack = currentRack.String()
		}
	}

	// Get player names and scores
	history := sc.game.History()
	var player0Name, player1Name string
	if history != nil && len(history.Players) >= 2 {
		player0Name = history.Players[0].Nickname
		player1Name = history.Players[1].Nickname
	}
	if player0Name == "" {
		player0Name = "Player 1"
	}
	if player1Name == "" {
		player1Name = "Player 2"
	}
	player0Score := sc.game.PointsFor(0)
	player1Score := sc.game.PointsFor(1)

	// Get last play summary - use same logic as ToDisplayText (game/display.go:109-111)
	lastPlay := ""
	if history != nil && sc.curTurnNum-1 >= 0 && len(history.Events) > sc.curTurnNum-1 {
		evt := history.Events[sc.curTurnNum-1]
		who := history.Players[evt.PlayerIndex].Nickname

		// Match the summary logic from game/turn.go:247
		switch evt.Type {
		case pb.GameEvent_TILE_PLACEMENT_MOVE:
			lastPlay = fmt.Sprintf("%s played %s %s for %d pts from a rack of %s",
				who, evt.Position, evt.PlayedTiles, evt.Score, evt.Rack)
		case pb.GameEvent_PASS:
			lastPlay = fmt.Sprintf("%s passed, holding a rack of %s",
				who, evt.Rack)
		case pb.GameEvent_EXCHANGE:
			lastPlay = fmt.Sprintf("%s exchanged %s from a rack of %s",
				who, evt.Exchanged, evt.Rack)
		case pb.GameEvent_CHALLENGE:
			lastPlay = fmt.Sprintf("%s challenged, holding a rack of %s",
				who, evt.Rack)
		case pb.GameEvent_UNSUCCESSFUL_CHALLENGE_TURN_LOSS:
			lastPlay = fmt.Sprintf("%s challenged unsuccessfully, holding a rack of %s",
				who, evt.Rack)
		}
	}

	// Get unseen tiles (bag + opponent's rack)
	bag := sc.game.Bag()
	remainingTiles := make(map[string]int)

	// Add tiles from bag
	for _, tile := range bag.Tiles() {
		letter := tile.UserVisible(alph, false)
		remainingTiles[letter]++
	}

	// Add opponent's rack tiles
	playerOnTurn := sc.game.PlayerOnTurn()
	opponentIdx := (playerOnTurn + 1) % 2
	opponentRack := sc.game.RackFor(opponentIdx)
	if opponentRack != nil {
		for _, tile := range opponentRack.TilesOn() {
			letter := tile.UserVisible(alph, false)
			remainingTiles[letter]++
		}
	}

	// Convert remaining tiles to JSON
	remainingTilesJSON, err := json.Marshal(remainingTiles)
	if err != nil {
		remainingTilesJSON = []byte("{}")
	}

	// Convert heatmap to JSON
	heatmapJSON := "null"
	if heatmapData != nil {
		heatmapBytes, err := json.Marshal(heatmapData)
		if err != nil {
			heatmapJSON = "null"
		} else {
			heatmapJSON = string(heatmapBytes)
		}
	}

	// Generate alphabet scores map
	dist := sc.game.Bag().LetterDistribution()
	alphabetScores := make(map[string]int)

	// Iterate through all letters in the alphabet
	for i := tilemapping.MachineLetter(0); i < tilemapping.MachineLetter(alph.NumLetters()); i++ {
		letter := i.UserVisible(alph, false)
		score := dist.Score(i)
		// Strip brackets from digraphs (e.g., "[CH]" -> "CH") for JavaScript lookup
		cleanLetter := strings.ReplaceAll(strings.ReplaceAll(letter, "[", ""), "]", "")
		alphabetScores[cleanLetter] = int(score)
	}

	// Add blank tile explicitly
	alphabetScores["?"] = 0

	// Convert alphabet scores to JSON
	alphabetScoresJSON, err := json.Marshal(alphabetScores)
	if err != nil {
		alphabetScoresJSON = []byte("{}")
	}

	// Prepare template data
	data := RenderTemplateData{
		FEN:            fen,
		Rack:           rack,
		RemainingTiles: template.JS(remainingTilesJSON),
		TileColor:      tileColor,
		BoardColor:     boardColor,
		Player0Name:    player0Name,
		Player1Name:    player1Name,
		Player0Score:   player0Score,
		Player1Score:   player1Score,
		Heatmap:        template.JS(heatmapJSON),
		HeatmapActive:  heatmapData != nil,
		HeatmapPlay:    heatmapPlay,
		HeatmapPly:     heatmapPly,
		PlayerOnTurn:   sc.game.PlayerOnTurn(),
		LastPlay:       lastPlay,
		AlphabetScores: template.JS(alphabetScoresJSON),
	}

	// Parse and execute template
	tmpl, err := template.New("render").Parse(renderTemplateHTML)
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	var htmlBuffer bytes.Buffer
	err = tmpl.Execute(&htmlBuffer, data)
	if err != nil {
		return nil, fmt.Errorf("failed to execute template: %w", err)
	}

	// Write to temporary file
	timestamp := time.Now().Format("20060102-150405")
	tmpFile := filepath.Join(os.TempDir(), fmt.Sprintf("macondo-render-%s.html", timestamp))
	err = os.WriteFile(tmpFile, htmlBuffer.Bytes(), 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to write render file: %w", err)
	}

	// Open in browser
	var browserCmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		browserCmd = exec.Command("open", tmpFile)
	case "linux":
		// Try common browsers instead of xdg-open to avoid opening with wrong app
		browsers := []string{"google-chrome", "chromium", "chromium-browser", "firefox"}
		var browserFound bool
		for _, browser := range browsers {
			if _, err := exec.LookPath(browser); err == nil {
				browserCmd = exec.Command(browser, tmpFile)
				browserFound = true
				break
			}
		}
		if !browserFound {
			return msg(fmt.Sprintf("3D board saved to %s\nNo browser found. Please open manually.", tmpFile)), nil
		}
	case "windows":
		browserCmd = exec.Command("cmd", "/c", "start", tmpFile)
	default:
		return msg(fmt.Sprintf("3D board saved to %s\nPlease open manually in your browser", tmpFile)), nil
	}

	err = browserCmd.Start()
	if err != nil {
		return msg(fmt.Sprintf("3D board saved to %s\nFailed to open automatically: %v", tmpFile, err)), nil
	}

	return msg(fmt.Sprintf("3D board rendered and opened in browser: %s", tmpFile)), nil
}

func (sc *ShellController) autoAnalyze(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("please provide a filename to analyze")
	}
	filename := cmd.args[0]
	options := cmd.options
	if options.String("export") != "" {
		// Sanitize the gameID to make it filesystem-safe
		// Replace characters that are problematic in filenames
		safeGameID := strings.ReplaceAll(options.String("export"), ":", "-")
		safeGameID = strings.ReplaceAll(safeGameID, "/", "_")
		safeGameID = strings.ReplaceAll(safeGameID, "+", "-")

		f, err := os.Create(safeGameID + ".gcg")
		if err != nil {
			return nil, err
		}
		ld := options.String("letterdist")
		lex := options.String("lex")
		if ld == "" {
			ld = sc.config.GetString(config.ConfigDefaultLetterDistribution)
		}
		if lex == "" {
			lex = sc.config.GetString(config.ConfigDefaultLexicon)
		}

		err = automatic.ExportGCG(
			sc.config, filename, ld, lex,
			options.String("boardlayout"), options.String("export"), f)
		if err != nil {
			ferr := os.Remove(safeGameID + ".gcg")
			if ferr != nil {
				log.Err(ferr).Msg("removing gcg output file")
			}
			return nil, err
		}
		err = f.Close()
		if err != nil {
			return nil, err
		}
		return msg("exported to " + safeGameID + ".gcg"), nil
	}
	analysis, err := automatic.AnalyzeLogFile(filename)
	if err != nil {
		return nil, err
	}
	return msg(analysis), nil
}

func (sc *ShellController) leave(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) != 1 {
		return nil, errors.New("please provide a leave")
	}
	if sc.exhaustiveLeaveCalculator == nil {
		err := sc.setExhaustiveLeaveCalculator()
		if err != nil {
			return nil, err
		}
	}
	ldName := sc.config.GetString(config.ConfigDefaultLetterDistribution)
	dist, err := tilemapping.GetDistribution(sc.config.WGLConfig(), ldName)
	if err != nil {
		return nil, err
	}
	leave, err := tilemapping.ToMachineWord(cmd.args[0], dist.TileMapping())
	if err != nil {
		return nil, err
	}
	res := sc.exhaustiveLeaveCalculator.LeaveValue(leave)
	return msg(strconv.FormatFloat(res, 'f', 3, 64)), nil
}

func (sc *ShellController) cgp(cmd *shellcmd) (*Response, error) {
	cgpstr := sc.game.ToCGP(false)
	return msg(cgpstr), nil
}

func (sc *ShellController) check(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) == 0 {
		return nil, errors.New("please provide a word or space-separated list of words to check")
	}
	dist, err := tilemapping.GetDistribution(sc.config.WGLConfig(),
		sc.config.GetString(config.ConfigDefaultLetterDistribution))
	if err != nil {
		return nil, err
	}
	k, err := kwg.GetKWG(sc.config.WGLConfig(), sc.config.GetString(config.ConfigDefaultLexicon))
	if err != nil {
		return nil, err
	}
	lex := kwg.Lexicon{KWG: *k}

	playValid := true
	wordsFriendly := []string{}

	for _, w := range cmd.args {
		wordFriendly := strings.Trim(strings.ToUpper(w), ",")
		wordsFriendly = append(wordsFriendly, wordFriendly)

		word, err := tilemapping.ToMachineWord(wordFriendly, dist.TileMapping())
		if err != nil {
			return nil, err
		}
		valid := lex.HasWord(word)
		if !valid {
			playValid = false
		}
	}
	validStr := "VALID"
	if !playValid {
		validStr = "INVALID"
	}

	return msg(fmt.Sprintf("The play (%v) is %v in %v", strings.Join(wordsFriendly, ","), validStr, sc.config.GetString(config.ConfigDefaultLexicon))), nil

}

func (sc *ShellController) winpct(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) != 2 {
		return nil, errors.New("please provide a spread and tiles remaining to check win percentage")
	}

	spread, err := strconv.Atoi(cmd.args[0])
	if err != nil {
		return nil, fmt.Errorf("invalid spread: %v", err)
	}
	tilesRemaining, err := strconv.Atoi(cmd.args[1])
	if err != nil {
		return nil, fmt.Errorf("invalid tiles remaining: %v", err)
	}
	if tilesRemaining < 0 || tilesRemaining > 93 {
		return nil, fmt.Errorf("tiles remaining must be between 0 and 93, got %d", tilesRemaining)
	}

	if spread > equity.MaxRepresentedWinSpread {
		spread = equity.MaxRepresentedWinSpread
	} else if spread < -equity.MaxRepresentedWinSpread {
		spread = -equity.MaxRepresentedWinSpread
	}
	wpct := sc.winpcts[int(equity.MaxRepresentedWinSpread-spread)][tilesRemaining]

	return msg(fmt.Sprintf("Win percentage: %.2f%%", wpct*100)), nil

}

func (sc *ShellController) mleval(cmd *shellcmd) (*Response, error) {
	playerid := sc.game.PlayerOnTurn()

	if sc.exhaustiveLeaveCalculator == nil {
		err := sc.setExhaustiveLeaveCalculator()
		if err != nil {
			return nil, err
		}
	}
	// If no arguments are provided, evaluate all move

	if len(cmd.args) == 0 {
		// evaluate all moves
		evals, err := sc.game.MLEvaluateMoves(sc.curPlayList, sc.exhaustiveLeaveCalculator, nil)
		if err != nil {
			return nil, err
		}

		// Create a slice of move-evaluation pairs
		type moveEval struct {
			move         *move.Move
			eval         float32
			oppBingoProb float32
			totalPts     float32
			oppNextScore float32
			idx          int
		}

		pairs := make([]moveEval, len(sc.curPlayList))
		for i, m := range sc.curPlayList {
			pairs[i] = moveEval{
				move: m,
				eval: evals.Value[i],
				// oppBingoProb: evals.BingoProb[i],
				// totalPts:     evals.Points[i],
				// oppNextScore: evals.OppScore[i],
				idx: i + 1, // Store original index for reference
			}
		}

		// Sort by evaluation in descending order
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].eval > pairs[j].eval
		})

		// Display sorted moves
		for i, p := range pairs {
			sc.showMessage(fmt.Sprintf("%d) %s: %.6f (was #%d) (opp-bingo-prob %.3f, total-pts %.3f, opp-next-score %.3f)",
				i+1, p.move.ShortDescription(), p.eval, p.idx, p.oppBingoProb, p.totalPts, p.oppNextScore))
		}

		return msg("MLEval for all moves completed."), nil
	} else {
		m, err := sc.game.ParseMove(playerid, sc.options.lowercaseMoves, cmd.args, false)
		if err != nil {
			return nil, err
		}
		eval, err := sc.game.MLEvaluateMove(m, sc.exhaustiveLeaveCalculator, nil)
		if err != nil {
			return nil, err
		}
		return msg(fmt.Sprintf("MLEval for %s: %.3f",
			m.ShortDescription(), eval.Value[0])), nil
	}
}

func (sc *ShellController) explain(cmd *shellcmd) (*Response, error) {
	// explain with genai
	if sc.game == nil {
		return nil, errors.New("please load or create a game first")
	}

	// Check if we're in endgame (too few tiles)
	unseenTiles := sc.game.Bag().TilesRemaining() + int(sc.game.RackFor(sc.game.NextPlayer()).NumTiles())
	if unseenTiles <= 8 {
		return nil, errors.New("GenAI explainability is currently only available for 2 or more tiles in the bag")
	}

	// Initialize explainer if needed
	if sc.aiexplainer == nil {
		sc.aiexplainer = explainer.NewService(sc.config)
	}
	sc.aiexplainer.SetGame(sc.game)

	// First generate moves if we haven't already
	if len(sc.curPlayList) == 0 {
		// Generate 40 moves like the Lua script does
		_ = sc.genMovesAndDescription(40)
	}

	// Parse optional sim parameters

	simOptions := cmd.options

	// Set default simulation parameters matching the Lua script
	if _, exists := simOptions["plies"]; !exists {
		simOptions["plies"] = []string{"5"}
	}
	if _, exists := simOptions["stop"]; !exists {
		simOptions["stop"] = []string{"99"}
	}
	simOptions["collect-heatmap"] = []string{"true"}

	// Run the simulation
	sc.showMessage("Running simulation for AI explanation... (sim options " + fmt.Sprint(simOptions) + ")")
	err := sc.runSimulationForExplain(simOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to run simulation: %w", err)
	}

	// Wait for simulation to complete
	for sc.simmer.IsSimming() {
		time.Sleep(100 * time.Millisecond)
	}

	// Collect the game state
	gameStateResp, err := sc.gameState(&shellcmd{})
	if err != nil {
		return nil, fmt.Errorf("failed to get game state: %w", err)
	}
	gameStateStr := gameStateResp.message

	sc.simmer.TrimBottom(35)
	// Get simulation results (top 5 plays)
	simResults := sc.simmer.EquityStats()

	// Get simulation details
	simDetails := sc.simmer.ScoreDetails()

	// Get the winning play
	winningPlay := sc.simmer.WinningPlay()
	if winningPlay == nil {
		return nil, errors.New("no winning play found in simulation")
	}
	winningPlayStr := winningPlay.Move().ShortDescription()

	// Get detailed stats for the winning play
	winningPlayStats, err := sc.simStats.CalculatePlayStats(winningPlayStr)
	if err != nil {
		return nil, fmt.Errorf("failed to get play stats: %w", err)
	}

	// Call the explainer service
	ctx := context.Background()
	result, err := sc.aiexplainer.Explain(
		ctx,
		gameStateStr,
		simResults,
		simDetails,
		winningPlayStr,
		winningPlayStats,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate explanation: %w", err)
	}

	// Show token usage if available
	if result.InputTokens > 0 {
		sc.showMessage(fmt.Sprintf("Input tokens: %d", result.InputTokens))
		sc.showMessage(fmt.Sprintf("Output tokens: %d", result.OutputTokens))
	}

	return msg("\n**Explanation**:\n\n" + result.Explanation), nil
}

// runSimulationForExplain runs a simulation with the given options
func (sc *ShellController) runSimulationForExplain(options CmdOptions) error {
	if sc.simmer == nil {
		return errors.New("simmer not initialized")
	}
	if sc.simmer.IsSimming() {
		// Stop any existing simulation
		sc.simCancel()
		for sc.simmer.IsSimming() {
			time.Sleep(10 * time.Millisecond)
		}
	}

	// Prepare simulation parameters
	var plies, threads int
	var err error

	if plies, err = options.IntDefault("plies", 5); err != nil {
		return err
	}
	if threads, err = options.IntDefault("threads", 0); err != nil {
		return err
	}

	stoppingCondition := montecarlo.StopNone
	if stopVal, err := options.IntDefault("stop", 99); err == nil {
		switch stopVal {
		case 90:
			stoppingCondition = montecarlo.Stop90
		case 95:
			stoppingCondition = montecarlo.Stop95
		case 98:
			stoppingCondition = montecarlo.Stop98
		case 99:
			stoppingCondition = montecarlo.Stop99
		case 999:
			stoppingCondition = montecarlo.Stop999
		}
	}

	// Handle known opponent rack if specified
	var kr []tilemapping.MachineLetter
	if knownOppRack := options.String("opprack"); knownOppRack != "" {
		knownOppRack = strings.ToUpper(knownOppRack)
		kr, err = tilemapping.ToMachineLetters(knownOppRack, sc.game.Alphabet())
		if err != nil {
			return err
		}
	}

	// Set up simulation parameters
	params := simParams{
		threads:           threads,
		plies:             plies,
		stoppingCondition: stoppingCondition,
		knownOppRack:      kr,
	}

	// Configure the simmer
	err = sc.setSimmerParams(sc.simmer, params)
	if err != nil {
		return err
	}

	// Enable heatmap collection
	sc.simmer.SetCollectHeatmap(true)

	// Run simulation synchronously
	ctx := context.Background()
	err = sc.simmer.Simulate(ctx)
	if err != nil {
		return err
	}

	return nil
}

// parseSimOptions parses command-line style options into CmdOptions
func parseSimOptions(fields []string) CmdOptions {
	fmt.Println("PARSESIMOPTIONS", fields)
	options := make(CmdOptions)
	lastWasOption := false
	lastOption := ""

	for _, field := range fields {
		// Only treat as option if it starts with '-' and is not a negative number and is not a single dash
		if strings.HasPrefix(field, "-") && len(field) > 1 && (field[1] < '0' || field[1] > '9') {
			// option
			lastWasOption = true
			lastOption = field[1:]
			continue
		}
		if lastWasOption {
			lastWasOption = false
			options[lastOption] = append(options[lastOption], field)
		}
	}

	return options
}
