package shell

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/chzyer/readline"
	"github.com/domino14/word-golib/cache"
	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/kballard/go-shellquote"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/explainer"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/preendgame"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/turnplayer"
)

const (
	SimLog     = "./macondo-simlog"
	InferLog   = "./macondo-inferlog"
	PEGLog     = "./macondo-peglog"
	EndgameLog = "./macondo-endgamelog"
)

const eliteBotShellTimeout = time.Duration(1) * time.Minute

var (
	errNoData         = errors.New("no data in this line")
	errMacondoSolving = errors.New("macondo is busy working on a solution to a position")
)

var backtickRegex = regexp.MustCompile("`([^`]*)`")

// Options to configure the interactve shell
type ShellOptions struct {
	turnplayer.GameOptions
	lowercaseMoves bool
}

func NewShellOptions() *ShellOptions {
	return &ShellOptions{
		GameOptions: turnplayer.GameOptions{
			Lexicon:       nil,
			ChallengeRule: pb.ChallengeRule_DOUBLE,
		},
		lowercaseMoves: false,
	}
}

func (opts *ShellOptions) Show(key string) (bool, string) {
	switch key {
	case "lexicon":
		return true, opts.Lexicon.ToDisplayString()
	case "lower":
		return true, fmt.Sprintf("%v", opts.lowercaseMoves)
	case "challenge":
		rule := turnplayer.ShowChallengeRule(opts.ChallengeRule)
		return true, fmt.Sprintf("%v", rule)
	case "board":
		return true, opts.BoardLayoutName
	default:
		return false, "No such option: " + key
	}
}

func (opts *ShellOptions) ToDisplayText() string {
	keys := []string{"lexicon", "challenge", "lower", "board"}
	out := strings.Builder{}
	out.WriteString("Settings:\n")
	for _, key := range keys {
		_, val := opts.Show(key)
		out.WriteString("  " + key + ": ")
		out.WriteString(val + "\n")
	}
	return out.String()
}

// VariationNode represents a position in the variation tree.
// The tree structure allows exploring different move sequences from the same position.
type VariationNode struct {
	gameSnapshot *bot.BotTurnPlayer // Full game copy at this position
	parent       *VariationNode     // Parent node (nil for root)
	children     []*VariationNode   // Child variations
	move         *move.Move         // Move that led to this position (nil for root)
	variationID  int                // 0 = main line, >0 = variation number
	turnNumber   int                // Turn number at this position
}

type ShellController struct {
	l        *readline.Instance
	config   *config.Config
	execPath string

	options *ShellOptions
	aliases map[string]string

	game *bot.BotTurnPlayer

	simmer        *montecarlo.Simmer
	simCtx        context.Context
	simCancel     context.CancelFunc
	simTicker     *time.Ticker
	simTickerDone chan bool
	simLogFile    *os.File

	rangefinder     *rangefinder.RangeFinder
	rangefinderFile *os.File

	gameRunnerCtx     context.Context
	gameRunnerCancel  context.CancelFunc
	gameRunnerRunning bool
	gameRunnerTicker  *time.Ticker

	curTurnNum       int
	gen              movegen.MoveGenerator
	backupgen        movegen.MoveGenerator // used for endgame engine
	curMode          Mode
	endgameSolver    *negamax.Solver
	preendgameSolver *preendgame.Solver

	endgameCtx     context.Context
	endgameCancel  context.CancelFunc
	endgameLogFile *os.File
	pegCtx         context.Context
	pegCancel      context.CancelFunc
	pegLogFile     *os.File

	curPlayList  []*move.Move
	elitebot     *bot.BotTurnPlayer
	botCtx       context.Context
	botCtxCancel context.CancelFunc
	botBusy      bool

	exhaustiveLeaveCalculator *equity.ExhaustiveLeaveCalculator

	simStats *stats.SimStats
	winpcts  [][]float32 // win percentages for each spread, indexed by int(equity.MaxRepresentedWinSpread - spread)

	// Variation tree for exploring different lines of play
	variationRoot    *VariationNode // Root of the variation tree (initial game position)
	currentVariation *VariationNode // Current position in the variation tree
	nextVariationID  int            // Counter for assigning variation IDs

	macondoVersion string
	aiexplainer    *explainer.Service
}

type Mode int

const (
	StandardMode Mode = iota
	EndgameDebugMode
	InvalidMode
)

func filterInput(r rune) (rune, bool) {
	switch r {
	// block CtrlZ feature
	case readline.CharCtrlZ:
		return r, false
	}
	return r, true
}

func writeln(msg string, w io.Writer) {
	io.WriteString(w, msg)
	io.WriteString(w, "\n")
}

func formatMessage(msg string) string {
	formattedMsg := backtickRegex.ReplaceAllStringFunc(msg, func(match string) string {
		content := match[1 : len(match)-1] // Remove the backticks
		return underlineText(content)
	})
	return formattedMsg
}

func (sc *ShellController) showMessage(msg string) {
	formattedMsg := formatMessage(msg)
	writeln(formattedMsg, os.Stdout)
}

func underlineText(text string) string {
	return "\x1b[4m" + text + "\x1b[0m"
}

func (sc *ShellController) showError(err error) {
	repr := "Error: " + err.Error()
	if !board.ColorSupport {
		sc.showMessage(repr)
		return
	}
	sc.showMessage(fmt.Sprintf("\033[1;31m%s\033[0m", repr))
}

func NewShellController(cfg *config.Config, execPath, gitVersion string) *ShellController {
	execPath = config.FindBasePath(execPath)
	opts := NewShellOptions()
	opts.SetDefaults(cfg)

	// fix hardcoding later
	winpct, err := cache.Load(cfg.WGLConfig(), "winpctfile:NWL20:winpct.csv", equity.WinPCTLoadFunc)
	if err != nil {
		panic(err)
	}
	var ok bool
	winPcts, ok := winpct.([][]float32)
	if !ok {
		panic("win percentages not correct type")
	}

	// Load aliases from config
	aliases := make(map[string]string)
	if aliasesFromConfig := cfg.GetStringMapString(config.ConfigAliases); aliasesFromConfig != nil {
		aliases = aliasesFromConfig
	}

	// Create a partial ShellController so we can create the autocompleter
	sc := &ShellController{
		config:         cfg,
		execPath:       execPath,
		options:        opts,
		aliases:        aliases,
		macondoVersion: gitVersion,
		winpcts:        winPcts,
	}

	// Create the autocompleter with reference to the controller
	completer := NewShellCompleter(sc)

	// Now create the readline instance with autocomplete enabled
	prompt := "macondo>"
	if os.Getenv("NO_COLOR") == "" {
		prompt = fmt.Sprintf("\033[31m%s\033[0m", prompt)
	}
	l, err := readline.NewEx(&readline.Config{
		Prompt:          prompt + " ",
		HistoryFile:     "/tmp/readline.tmp",
		EOFPrompt:       "exit",
		InterruptPrompt: "^C",

		HistorySearchFold:   true,
		FuncFilterInputRune: filterInput,
		AutoComplete:        completer,
	})

	if err != nil {
		panic(err)
	}

	// Set the readline instance on the controller
	sc.l = l

	return sc
}

func (sc *ShellController) Cleanup() {
	if sc.simmer != nil {
		sc.simmer.CleanupTempFile()
	}
}

func (sc *ShellController) setExhaustiveLeaveCalculator() error {
	ldName := sc.config.GetString(config.ConfigDefaultLetterDistribution)
	leaves := ""
	if strings.HasSuffix(ldName, "_super") {
		leaves = "super-leaves.klv2"
	}
	els, err := equity.NewExhaustiveLeaveCalculator(sc.config.GetString(config.ConfigDefaultLexicon),
		sc.config, leaves)
	if err != nil {
		return err
	}
	sc.exhaustiveLeaveCalculator = els
	return nil
}

func (sc *ShellController) Set(key string, args []string) (string, error) {
	var err error
	var ret string
	switch key {
	case "lexicon":
		if sc.IsPlaying() {
			msg := "Cannot change the lexicon while a game is active (try `unload` to quit game)"
			err = errors.New(msg)
		} else {
			err = sc.options.SetLexicon(args, sc.config.WGLConfig())
			if err == nil {
				// Overwrite the config options since other parts of the code
				// use these to determine the lexicon
				sc.config.Set(config.ConfigDefaultLexicon, sc.options.Lexicon.Name)
				sc.config.Set(config.ConfigDefaultLetterDistribution, sc.options.Lexicon.Distribution)
				err = sc.config.Write()
				if err != nil {
					log.Err(err).Msg("error-writing-config")
				}
				err = sc.setExhaustiveLeaveCalculator()
				if err != nil {
					log.Err(err).Msg("error-setting-exhaustive-leave-calculator")
				}
			}
			_, ret = sc.options.Show("lexicon")
		}
	case "board":
		if sc.IsPlaying() {
			msg := "Cannot change the board layout while a game is active (try `unload` to quit game)"
			err = errors.New(msg)
		} else {
			err = sc.options.SetBoardLayoutName(args[0])
			if err == nil {
				sc.config.Set(config.ConfigDefaultBoardLayout, sc.options.BoardLayoutName)
				err = sc.config.Write()
				if err != nil {
					log.Err(err).Msg("error-writing-config")
				}
			}
			_, ret = sc.options.Show("board")
		}
	case "challenge":
		err = sc.options.SetChallenge(args[0])
		_, ret = sc.options.Show("challenge")
		// Allow the challenge rule to be changed in the middle of a game.
		if err != nil && sc.game != nil {
			sc.game.SetChallengeRule(sc.options.ChallengeRule)
		}
	case "lower":
		val, err := strconv.ParseBool(args[0])
		if err == nil {
			sc.options.lowercaseMoves = val
			ret = strconv.FormatBool(val)
		} else {
			err = errors.New("Valid options: 'true', 'false'")
		}
	default:
		err = errors.New("No such option: " + key)
	}
	if err == nil {
		return ret, nil
	} else {
		return "", err
	}
}

func (sc *ShellController) initGameDataStructures() error {
	if sc.simmer != nil {
		sc.simmer.CleanupTempFile()
	}
	sc.simmer = &montecarlo.Simmer{}
	sc.simStats = stats.NewSimStats(sc.simmer, sc.game)

	c, err := equity.NewCombinedStaticCalculator(
		sc.game.LexiconName(),
		sc.config, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return err
	}
	sc.simmer.Init(sc.game.Game, []equity.EquityCalculator{c}, c, sc.config)
	sc.gen = sc.game.MoveGenerator()

	gd, err := kwg.GetKWG(sc.config.WGLConfig(), sc.game.LexiconName())
	if err != nil {
		return err
	}

	sc.backupgen = movegen.NewGordonGenerator(gd, sc.game.Board(), sc.game.Bag().LetterDistribution())

	sc.rangefinder = &rangefinder.RangeFinder{}
	sc.rangefinder.Init(sc.game.Game, []equity.EquityCalculator{c}, sc.config)

	// initialize the elite bot

	leavesFile := ""
	if sc.game.Board().Dim() == 21 { // ghetto
		leavesFile = "super-leaves.klv2"
	}

	conf := &bot.BotConfig{Config: *sc.config, MinSimPlies: 5, LeavesFile: leavesFile,
		UseOppRacksInAnalysis: false}
	tp, err := bot.NewBotTurnPlayerFromGame(sc.game.Game, conf, pb.BotRequest_BotCode(pb.BotRequest_SIMMING_BOT))
	if err != nil {
		return err
	}
	tp.SetBackupMode(game.InteractiveGameplayMode)
	tp.SetStateStackLength(1)
	sc.elitebot = tp

	// Initialize variation tree with root node (main line)
	sc.initializeVariationTree()

	return nil
}

// findNodeAtTurn walks up the variation tree to find a node at the specified turn number.
// Returns nil if no node is found at that turn.
func (sc *ShellController) findNodeAtTurn(turnNum int) *VariationNode {
	// Start from current variation and walk backwards
	node := sc.currentVariation
	for node != nil {
		if node.turnNumber == turnNum {
			return node
		}
		node = node.parent
	}
	return nil
}

// variationList displays all variations available from the current position
func (sc *ShellController) variationList() (*Response, error) {
	// Walk back to find the branching point (node with multiple children)
	node := sc.currentVariation
	for node != nil && len(node.children) <= 1 {
		node = node.parent
	}

	if node == nil {
		return msg("No variations exist yet."), nil
	}

	var output strings.Builder
	fmt.Fprintf(&output, "Variations from turn %d:\n", node.turnNumber)

	// Show all children of this branching node
	for _, child := range node.children {
		// Get move description - from the move object or from history
		var moveStr string
		if child.move != nil {
			moveStr = child.move.ShortDescription()
		} else if child.gameSnapshot != nil && child.gameSnapshot.History() != nil {
			// Get the event at this child's turn number from the history
			history := child.gameSnapshot.History()
			// child.turnNumber is 1-indexed, history.Events is 0-indexed
			eventIdx := child.turnNumber - 1
			if eventIdx >= 0 && eventIdx < len(history.Events) {
				evt := history.Events[eventIdx]
				moveStr = fmt.Sprintf("%s %s", evt.Position, evt.PlayedTiles)
			} else if child.turnNumber == 0 {
				moveStr = "(start)"
			} else {
				moveStr = "(unknown)"
			}
		} else {
			moveStr = "(unknown)"
		}

		isCurrent := child == sc.currentVariation || sc.isAncestor(child, sc.currentVariation)

		varLabel := "main"
		if child.variationID != 0 {
			varLabel = fmt.Sprintf("%d", child.variationID)
		}

		// Count how many moves deep this variation goes
		depth := sc.countVariationDepth(child)

		if isCurrent {
			fmt.Fprintf(&output, "* %s: %s (%d turn%s)\n", varLabel, moveStr, depth, pluralize(depth))
		} else {
			fmt.Fprintf(&output, "  %s: %s (%d turn%s)\n", varLabel, moveStr, depth, pluralize(depth))
		}
	}

	return msg(output.String()), nil
}

// variationInfo shows information about the current variation
func (sc *ShellController) variationInfo() (*Response, error) {
	varLabel := "main line"
	if sc.currentVariation.variationID != 0 {
		varLabel = fmt.Sprintf("variation %d", sc.currentVariation.variationID)
	}

	return msg(fmt.Sprintf("Currently on %s, turn %d", varLabel, sc.currentVariation.turnNumber)), nil
}

// variationSwitch switches to a different variation by ID
func (sc *ShellController) variationSwitch(varID int) (*Response, error) {
	// Find the branching point (parent with multiple children)
	branchNode := sc.currentVariation
	for branchNode != nil && len(branchNode.children) <= 1 {
		branchNode = branchNode.parent
	}

	if branchNode == nil {
		return nil, errors.New("no branching point found")
	}

	// Find the child with the requested variation ID
	var targetChild *VariationNode
	for _, child := range branchNode.children {
		if child.variationID == varID {
			targetChild = child
			break
		}
	}

	if targetChild == nil {
		return nil, fmt.Errorf("variation %d not found at this branching point", varID)
	}

	// Find the deepest node in this variation to get the complete history
	deepestNode := targetChild
	for len(deepestNode.children) > 0 {
		foundChild := false
		for _, child := range deepestNode.children {
			if child.variationID == varID {
				deepestNode = child
				foundChild = true
				break
			}
		}
		if !foundChild {
			break
		}
	}

	// Use the deepest node's snapshot (which has complete variation history)
	// but we'll position it at the branching point for display
	// This lets you use 'n' to step through the continuation

	// Create a fresh copy of the snapshot's game with independent history
	// Use the deepest node's snapshot to get complete history
	leavesFile := ""
	if deepestNode.gameSnapshot.Board().Dim() == 21 {
		leavesFile = "super-leaves.klv2"
	}
	conf := &bot.BotConfig{Config: *sc.config, LeavesFile: leavesFile}

	gameCopy := deepestNode.gameSnapshot.Game.CopyWithHistory()
	newBot, err := bot.NewBotTurnPlayerFromGame(gameCopy, conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return nil, fmt.Errorf("failed to create bot from snapshot: %w", err)
	}
	newBot.SetBackupMode(game.InteractiveGameplayMode)
	newBot.SetStateStackLength(1)

	// Position at the branching point (first move of variation)
	newBot.PlayToTurn(targetChild.turnNumber)

	// Update shell state
	oldVariationRoot := sc.variationRoot
	oldNextID := sc.nextVariationID

	sc.game = newBot
	sc.curTurnNum = targetChild.turnNumber
	sc.curPlayList = nil

	// Reinitialize game data structures (but NOT the variation tree)
	if sc.simmer != nil {
		sc.simmer.CleanupTempFile()
	}
	sc.simmer = &montecarlo.Simmer{}
	sc.simStats = stats.NewSimStats(sc.simmer, sc.game)

	c, err := equity.NewCombinedStaticCalculator(
		sc.game.LexiconName(),
		sc.config, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	sc.simmer.Init(sc.game.Game, []equity.EquityCalculator{c}, c, sc.config)
	sc.gen = sc.game.MoveGenerator()

	sc.rangefinder = &rangefinder.RangeFinder{}
	sc.rangefinder.Init(sc.game.Game, []equity.EquityCalculator{c}, sc.config)

	// Restore variation tree (don't reinitialize it!)
	sc.variationRoot = oldVariationRoot
	sc.currentVariation = targetChild
	sc.nextVariationID = oldNextID

	varLabel := "main line"
	if varID != 0 {
		varLabel = fmt.Sprintf("variation %d", varID)
	}

	sc.showMessage(fmt.Sprintf("[Switched to %s, turn %d]", varLabel, targetChild.turnNumber))
	return msg(sc.game.ToDisplayText()), nil
}

// variationDelete removes a variation branch
func (sc *ShellController) variationDelete(varIDStr string) (*Response, error) {
	varID, err := strconv.Atoi(varIDStr)
	if err != nil {
		return nil, fmt.Errorf("invalid variation ID: %s", varIDStr)
	}

	if varID == 0 {
		return nil, errors.New("cannot delete the main line")
	}

	// Find the node with this variation ID
	targetNode := sc.findVariationByID(sc.variationRoot, varID)
	if targetNode == nil {
		return nil, fmt.Errorf("variation %d not found", varID)
	}

	// Check if we're currently on this variation or one of its descendants
	if targetNode == sc.currentVariation || sc.isAncestor(targetNode, sc.currentVariation) {
		return nil, errors.New("cannot delete the current variation - switch to a different variation first")
	}

	// Remove this node from its parent's children
	parent := targetNode.parent
	if parent != nil {
		for i, child := range parent.children {
			if child == targetNode {
				parent.children = append(parent.children[:i], parent.children[i+1:]...)
				break
			}
		}
	}

	return msg(fmt.Sprintf("Variation %d deleted", varID)), nil
}

// variationPromote promotes a variation to become the main line
func (sc *ShellController) variationPromote(varIDStr string) (*Response, error) {
	varID, err := strconv.Atoi(varIDStr)
	if err != nil {
		return nil, fmt.Errorf("invalid variation ID: %s", varIDStr)
	}

	if varID == 0 {
		return nil, errors.New("variation is already the main line")
	}

	// Find the branching point
	branchNode := sc.currentVariation
	for branchNode != nil && len(branchNode.children) <= 1 {
		branchNode = branchNode.parent
	}

	if branchNode == nil {
		return nil, errors.New("no branching point found")
	}

	// Find both the target variation and the main line at this branch
	var targetChild *VariationNode
	var mainChild *VariationNode

	for _, child := range branchNode.children {
		if child.variationID == varID {
			targetChild = child
		} else if child.variationID == 0 {
			mainChild = child
		}
	}

	if targetChild == nil {
		return nil, fmt.Errorf("variation %d not found at this branching point", varID)
	}

	if mainChild == nil {
		return nil, errors.New("main line not found at branching point")
	}

	// Swap variation IDs: recursively update all descendant nodes
	sc.updateVariationID(targetChild, varID, 0)
	sc.updateVariationID(mainChild, 0, varID)

	return msg(fmt.Sprintf("Variation %d promoted to main line", varID)), nil
}

// updateVariationID recursively updates variation IDs for a node and all its descendants
func (sc *ShellController) updateVariationID(node *VariationNode, oldID, newID int) {
	if node == nil {
		return
	}

	if node.variationID == oldID {
		node.variationID = newID
	}

	for _, child := range node.children {
		sc.updateVariationID(child, oldID, newID)
	}
}

// Helper functions

func (sc *ShellController) findVariationByID(node *VariationNode, varID int) *VariationNode {
	if node == nil {
		return nil
	}
	if node.variationID == varID {
		return node
	}
	for _, child := range node.children {
		if found := sc.findVariationByID(child, varID); found != nil {
			return found
		}
	}
	return nil
}

func (sc *ShellController) isAncestor(ancestor, descendant *VariationNode) bool {
	node := descendant
	for node != nil {
		if node == ancestor {
			return true
		}
		node = node.parent
	}
	return false
}

func (sc *ShellController) countVariationDepth(node *VariationNode) int {
	if len(node.children) == 0 {
		return 1
	}
	maxDepth := 0
	for _, child := range node.children {
		depth := sc.countVariationDepth(child)
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	return maxDepth + 1
}

func pluralize(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// initializeVariationTree creates the root node of the variation tree from the current game state.
// If the game has existing history, it builds nodes for all existing moves on the main line.
func (sc *ShellController) initializeVariationTree() {
	// Build nodes for all existing moves in the game history
	history := sc.game.History()

	// If there's no history (e.g., CGP loaded directly), create a simple root node
	// without calling PlayToTurn which would clear the board
	if history == nil || len(history.Events) == 0 {
		gameCopy := sc.game.Game.CopyWithHistory()
		botCopy, _ := bot.NewBotTurnPlayerFromGame(gameCopy, &bot.BotConfig{Config: *sc.config}, pb.BotRequest_HASTY_BOT)

		sc.variationRoot = &VariationNode{
			gameSnapshot: botCopy,
			parent:       nil,
			children:     []*VariationNode{},
			move:         nil,
			variationID:  0,
			turnNumber:   0,
		}
		sc.currentVariation = sc.variationRoot
		sc.nextVariationID = 1
		return
	}

	// Save the current turn number so we can restore it
	originalTurn := sc.game.Turn()

	// Create root node at turn 0 (main line, variation ID = 0)
	// For the root and all main line nodes, we keep the full history
	// but position the game at the appropriate turn
	sc.game.PlayToTurn(0)
	gameCopy := sc.game.Game.CopyWithHistory()
	botCopy, _ := bot.NewBotTurnPlayerFromGame(gameCopy, &bot.BotConfig{Config: *sc.config}, pb.BotRequest_HASTY_BOT)

	sc.variationRoot = &VariationNode{
		gameSnapshot: botCopy,
		parent:       nil,
		children:     []*VariationNode{},
		move:         nil,
		variationID:  0,
		turnNumber:   0,
	}

	currentNode := sc.variationRoot

	// For each turn in the main line, create a node with the full history
	for turnNum := 0; turnNum < len(history.Events); turnNum++ {
		// Create a copy of the full game (with complete history)
		sc.game.PlayToTurn(originalTurn) // Restore full history position
		fullGameCopy := sc.game.Game.CopyWithHistory() // Deep copy with independent history
		botCopy, _ := bot.NewBotTurnPlayerFromGame(fullGameCopy, &bot.BotConfig{Config: *sc.config}, pb.BotRequest_HASTY_BOT)

		// Position this copy at the specific turn
		botCopy.PlayToTurn(turnNum + 1)

		newNode := &VariationNode{
			gameSnapshot: botCopy,
			parent:       currentNode,
			children:     []*VariationNode{},
			move:         nil, // Move info is in snapshot's history
			variationID:  0,   // Main line
			turnNumber:   turnNum + 1,
		}

		currentNode.children = append(currentNode.children, newNode)
		currentNode = newNode
	}

	// Set current variation to the last node created
	sc.currentVariation = currentNode

	// Restore the original turn
	sc.game.PlayToTurn(originalTurn)

	sc.nextVariationID = 1 // First variation will be ID 1
}

func (sc *ShellController) IsPlaying() bool {
	return sc.game != nil && sc.game.IsPlaying()
}

func (sc *ShellController) loadGCG(args []string) error {
	var err error
	var history *pb.GameHistory
	// Try to parse filepath as a network path.
	if args[0] == "xt" {
		if len(args) < 2 {
			return errors.New("need to provide a cross-tables game id")
		}
		history, err = sc.loadGameHistoryFromCrossTables(args[1])
		if err != nil {
			return err
		}
	} else if args[0] == "woogles" {
		if len(args) < 2 {
			return errors.New("need to provide a woogles game id")
		}
		history, err = sc.loadGameHistoryFromWoogles(args[1])
		if err != nil {
			return err
		}
	} else if args[0] == "web" {
		if len(args) < 2 {
			return errors.New("need to provide a web URL")
		}
		history, err = sc.loadGameHistoryFromWeb(args[1])
		if err != nil {
			return err
		}
	} else {
		history, err = sc.loadGameHistoryFromFile(args[0])
		if err != nil {
			return err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", history.Players)
	lexicon := history.Lexicon
	if lexicon == "" {
		lexicon = sc.config.GetString(config.ConfigDefaultLexicon)
		log.Info().Msgf("gcg file had no lexicon, so using default lexicon %v",
			lexicon)
	}
	boardLayout, ldName, variant := game.HistoryToVariant(history)
	rules, err := game.NewBasicGameRules(sc.config, lexicon, boardLayout, ldName, game.CrossScoreAndSet, variant)
	if err != nil {
		return err
	}
	g, err := game.NewFromHistory(history, rules, 0)
	if err != nil {
		return err
	}
	leavesFile := ""
	if strings.HasSuffix(ldName, "_super") {
		leavesFile = "super-leaves.klv2"
	}

	conf := &bot.BotConfig{Config: *sc.config, LeavesFile: leavesFile}
	sc.game, err = bot.NewBotTurnPlayerFromGame(g, conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return err
	}
	sc.game.SetBackupMode(game.InteractiveGameplayMode)
	sc.game.SetStateStackLength(1)

	// Set challenge rule to double by default. This can be overridden.
	sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)

	return sc.initGameDataStructures()
}

// loadGameHistoryFromWoogles loads a game from Woogles by game ID
func (sc *ShellController) loadGameHistoryFromWoogles(gameID string) (*pb.GameHistory, error) {
	path := "https://woogles.io/api/game_service.GameMetadataService/GetGCG"
	reader := strings.NewReader(`{"gameId": "` + gameID + `"}`)

	resp, err := http.Post(path, "application/json", reader)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	type gcgstruct struct {
		Gcg string `json:"gcg"`
	}
	var gcgObj gcgstruct

	err = json.Unmarshal(body, &gcgObj)
	if err != nil {
		return nil, err
	}

	return gcgio.ParseGCGFromReader(sc.config, strings.NewReader(gcgObj.Gcg))
}

// loadGameHistoryFromCrossTables loads a game from Cross-tables by game ID
func (sc *ShellController) loadGameHistoryFromCrossTables(gameIDStr string) (*pb.GameHistory, error) {
	id, err := strconv.Atoi(gameIDStr)
	if err != nil {
		return nil, errors.New("badly formatted game ID")
	}

	prefix := strconv.Itoa(id / 100)
	xtpath := "https://www.cross-tables.com/annotated/selfgcg/" + prefix +
		"/anno" + gameIDStr + ".gcg"

	log.Info().Str("xtpath", xtpath).Msg("fetching")
	req, err := http.NewRequest("GET", xtpath, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", fmt.Sprintf("Macondo / v%v", sc.macondoVersion))
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, errors.New("bad status code: " + resp.Status)
	}
	defer resp.Body.Close()

	return gcgio.ParseGCGFromReader(sc.config, resp.Body)
}

// loadGameHistoryFromWeb loads a game from a web URL
func (sc *ShellController) loadGameHistoryFromWeb(url string) (*pb.GameHistory, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return gcgio.ParseGCGFromReader(sc.config, resp.Body)
}

// loadGameHistoryFromFile loads a game from a local GCG file
func (sc *ShellController) loadGameHistoryFromFile(path string) (*pb.GameHistory, error) {
	if strings.HasPrefix(path, "~/") {
		usr, err := user.Current()
		if err != nil {
			return nil, err
		}
		dir := usr.HomeDir
		path = filepath.Join(dir, path[2:])
	}

	return gcgio.ParseGCG(sc.config, path)
}

func (sc *ShellController) loadCGP(cgpstr string) error {
	g, err := cgp.ParseCGP(sc.config, cgpstr)
	if err != nil {
		return err
	}
	lexicon := g.History().Lexicon
	if lexicon == "" {
		lexicon = sc.config.GetString(config.ConfigDefaultLexicon)
		log.Info().Msgf("cgp file had no lexicon, so using default lexicon %v",
			lexicon)
	}

	leavesFile := ""
	if g.History().BoardLayout == board.SuperCrosswordGameLayout {
		leavesFile = "super-leaves.klv2"
	}

	conf := &bot.BotConfig{Config: *sc.config, LeavesFile: leavesFile}
	sc.game, err = bot.NewBotTurnPlayerFromGame(g.Game, conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return err
	}
	sc.game.SetBackupMode(game.InteractiveGameplayMode)
	sc.game.SetStateStackLength(1)

	// Set challenge rule to double by default. This can be overridden.
	// XXX: can read from cgp file.
	sc.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)

	sc.game.RecalculateBoard()
	return sc.initGameDataStructures()
}

func (sc *ShellController) setToTurn(turnnum int) error {

	if sc.game == nil {
		return errors.New("please load a game first with the `load` command")
	}
	err := sc.game.PlayToTurn(turnnum)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Set to turn %v", turnnum)
	sc.curPlayList = nil
	sc.simmer.Reset()
	sc.rangefinder.Reset()
	sc.curTurnNum = sc.game.Turn()
	if sc.curTurnNum != turnnum {
		return errors.New("unexpected turn number")
	}

	// Update currentVariation to match the turn we're at
	// Walk up/down the tree to find the node at this turn in our current variation path
	if sc.currentVariation != nil {
		targetNode := sc.findNodeAtTurnInPath(turnnum)
		if targetNode != nil {
			sc.currentVariation = targetNode
		}
	}

	return nil
}

// findNodeAtTurnInPath finds a node at the specified turn number by walking
// up the tree from current position, then walking down if needed
func (sc *ShellController) findNodeAtTurnInPath(turnNum int) *VariationNode {
	// First, walk up to find a node at or before this turn
	node := sc.currentVariation
	for node != nil && node.turnNumber > turnNum {
		node = node.parent
	}

	if node == nil {
		return nil
	}

	// If we found the exact turn, return it
	if node.turnNumber == turnNum {
		return node
	}

	// Otherwise, walk down the first child path until we reach the turn
	for node != nil && node.turnNumber < turnNum {
		if len(node.children) > 0 {
			// Follow the first child (main line or first variation)
			node = node.children[0]
		} else {
			break
		}
	}

	if node != nil && node.turnNumber == turnNum {
		return node
	}

	return nil
}

func moveTableHeader() string {
	return "     Move                Leave  Score Equity"
}

func MoveTableRow(idx int, m *move.Move, alph *tilemapping.TileMapping) string {
	return fmt.Sprintf("%3d: %-20s%-7s%5d %6.2f", idx+1,
		m.ShortDescription(), m.Leave().UserVisible(alph), m.Score(), m.Equity())
}

func (sc *ShellController) printEndgameSequence(moves []*move.Move) {
	sc.showMessage("Best sequence:")
	for idx, move := range moves {
		sc.showMessage(fmt.Sprintf("%d) %v (%d)", idx+1, move.ShortDescription(), move.Score()))
	}
}

func (sc *ShellController) genMovesAndDescription(numPlays int) string {
	sc.genMoves(numPlays)
	return sc.genDisplayMoveList()
}

func (sc *ShellController) genMoves(numPlays int) {
	sc.curPlayList = sc.game.GenerateMoves(numPlays)
}

func (sc *ShellController) genDisplayMoveList() string {
	var s strings.Builder
	s.WriteString(moveTableHeader() + "\n")
	for i, p := range sc.curPlayList {
		s.WriteString(MoveTableRow(i, p, sc.game.Alphabet()) + "\n")
	}
	return s.String()
}

func modeFromStr(mode string) (Mode, error) {
	mode = strings.TrimSpace(mode)
	switch mode {
	case "standard":
		return StandardMode, nil
	case "endgamedebug":
		return EndgameDebugMode, nil
	}
	return InvalidMode, errors.New("mode " + mode + " is not a valid choice")
}

func modeToStr(mode Mode) string {
	switch mode {
	case StandardMode:
		return "standard"
	case EndgameDebugMode:
		return "endgamedebug"
	default:
		return "invalid"
	}
}

func (sc *ShellController) addRack(rack string) error {
	// Set current player on turn's rack.
	if sc.game == nil {
		return errors.New("please start a game first")
	}
	return sc.game.SetCurrentRack(rack)
}

func (sc *ShellController) addMoveToList(playerid int, m *move.Move) error {
	opp := (playerid + 1) % sc.game.NumPlayers()
	oppRack := sc.game.RackFor(opp)
	sc.game.AssignEquity([]*move.Move{m}, sc.game.Board(), sc.game.Bag(), oppRack)
	sc.curPlayList = append(sc.curPlayList, m)
	sort.Slice(sc.curPlayList, func(i, j int) bool {
		return sc.curPlayList[j].Equity() < sc.curPlayList[i].Equity()
	})
	sc.showMessage(sc.genDisplayMoveList())
	return nil
}

func (sc *ShellController) getMoveFromList(playerid int, play string) (*move.Move, error) {
	// Add play that was generated.
	playID, err := strconv.Atoi(play[1:])
	if err != nil {
		return nil, err
	}

	idx := playID - 1 // since playID starts from 1
	if idx < 0 || idx > len(sc.curPlayList)-1 {
		return nil, errors.New("play outside range")
	}
	return sc.curPlayList[idx], nil
}

func (sc *ShellController) commitMove(m *move.Move) error {
	// Check if we're creating a variation by detecting if:
	// 1. We're at a node that already has children (branching from existing position)
	// 2. We're committing at a turn number earlier than our current variation node
	//    (this happens when we used p/n/turn to go back in history)
	currentTurnBeforeCommit := sc.curTurnNum
	isCreatingVariation := len(sc.currentVariation.children) > 0 ||
		currentTurnBeforeCommit < sc.currentVariation.turnNumber

	// If we went back in time, we need to find the right parent node in the tree
	var parentNode *VariationNode
	if currentTurnBeforeCommit < sc.currentVariation.turnNumber {
		// Walk back up the tree to find the node at currentTurnBeforeCommit
		parentNode = sc.findNodeAtTurn(currentTurnBeforeCommit)
		if parentNode == nil {
			// Fallback to current if we can't find the right node
			parentNode = sc.currentVariation
		}
		// Check if this node already has children
		if len(parentNode.children) > 0 {
			isCreatingVariation = true
		}
	} else {
		parentNode = sc.currentVariation
	}

	// Play the actual move on the board, draw tiles, etc.
	err := sc.game.PlayMove(m, true, 0)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Added turn at turn num %v", sc.curTurnNum)
	sc.curTurnNum = sc.game.Turn()

	// Create variation node after the move is played
	var newVariationID int
	if isCreatingVariation {
		// This is a new variation branch
		newVariationID = sc.nextVariationID
		sc.nextVariationID++
	} else {
		// Continuing existing line, inherit parent's variation ID
		newVariationID = parentNode.variationID
	}

	// Create a snapshot of the game state after this move
	gameCopy := sc.game.Game.CopyWithHistory()
	botCopy, _ := bot.NewBotTurnPlayerFromGame(gameCopy, &bot.BotConfig{Config: *sc.config}, pb.BotRequest_HASTY_BOT)

	newNode := &VariationNode{
		gameSnapshot: botCopy,
		parent:       parentNode,
		children:     []*VariationNode{},
		move:         m,
		variationID:  newVariationID,
		turnNumber:   sc.curTurnNum,
	}

	// Add as child of parent node
	parentNode.children = append(parentNode.children, newNode)
	sc.currentVariation = newNode

	if isCreatingVariation {
		sc.showMessage(fmt.Sprintf("[Variation %d created from turn %d] (type 'var' to display variations)",
			newVariationID, parentNode.turnNumber))
	}

	sc.curPlayList = nil
	sc.simmer.Reset()
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) parseAddMove(playerid int, fields []string) (*move.Move, error) {
	// Check that the user hasn't confused "add" and
	// "commit" to play a move from the list.
	if len(fields) == 1 &&
		strings.HasPrefix(fields[0], "#") {
		errmsg := "cannot use this option with the `add` command, " +
			"you may have wanted to use the `commit` command instead"
		return nil, errors.New(errmsg)
	}
	return sc.game.ParseMove(playerid, sc.options.lowercaseMoves, fields, false)
}

func (sc *ShellController) parseCommitMove(playerid int, fields []string) (*move.Move, error) {
	if len(fields) == 1 && strings.HasPrefix(fields[0], "#") {
		return sc.getMoveFromList(playerid, fields[0])
	}
	// Other than the `commit #id` command, `commit` and
	// `add` take the same arguments
	return sc.parseAddMove(playerid, fields)
}

func (sc *ShellController) commitPlay(fields []string) error {
	if sc.solving() {
		return errMacondoSolving
	}
	playerid := sc.game.PlayerOnTurn()
	m, err := sc.parseCommitMove(playerid, fields)
	if err != nil {
		return err
	}
	return sc.commitMove(m)
}

func (sc *ShellController) addPlay(fields []string) error {
	if sc.solving() {
		return errMacondoSolving
	}
	playerid := sc.game.PlayerOnTurn()
	m, err := sc.parseAddMove(playerid, fields)
	if err != nil {
		return err
	}
	return sc.addMoveToList(playerid, m)
}

func (sc *ShellController) commitHastyMove() error {
	if !sc.IsPlaying() {
		return errors.New("game is over")
	}
	if sc.solving() {
		return errMacondoSolving
	}
	sc.genMoves(15)
	m := sc.curPlayList[0]
	return sc.commitMove(m)
}

func (sc *ShellController) commitAIMove() error {
	if !sc.IsPlaying() {
		return errors.New("game is over")
	}
	if sc.solving() {
		return errMacondoSolving
	}

	go func() {
		log.Info().Msgf("Please wait, thinking for up to %v...", eliteBotShellTimeout)
		sc.botCtx, sc.botCtxCancel = context.WithTimeout(context.Background(),
			eliteBotShellTimeout)
		sc.botBusy = true
		defer func() {
			sc.botBusy = false
		}()
		m, err := sc.elitebot.BestPlay(sc.botCtx)
		if err != nil {
			log.Err(err).Msg("error with eliteplay")
		}
		sc.commitMove(m)
	}()

	return nil
}

func (sc *ShellController) handleAutoplay(args []string, options CmdOptions) error {
	var logfile, lexicon, letterDistribution, leavefile1, leavefile2, pegfile1, pegfile2 string
	var seedfile string
	var numgames, numthreads int
	var block, genseeds, deterministic bool
	var botcode1, botcode2 pb.BotRequest_BotCode
	var minsimplies1, minsimplies2 int
	var stochastic1, stochastic2 bool
	var err error
	if options.String("logfile") == "" {
		logfile = "/tmp/autoplay.txt"
	} else {
		logfile = options.String("logfile")
	}
	if options.String("lexicon") == "" {
		lexicon = sc.config.GetString(config.ConfigDefaultLexicon)
	} else {
		lexicon = options.String("lexicon")
	}
	if options.String("letterdistribution") == "" {
		letterDistribution = sc.config.GetString(config.ConfigDefaultLetterDistribution)
	} else {
		letterDistribution = options.String("letterdistribution")
	}
	leavefile1 = options.String("leavefile1")
	leavefile2 = options.String("leavefile2")
	pegfile1 = options.String("pegfile1")
	pegfile2 = options.String("pegfile2")

	if options.String("botcode1") == "" {
		botcode1 = pb.BotRequest_HASTY_BOT
	} else {
		botcode1Value, exists := pb.BotRequest_BotCode_value[options.String("botcode1")]
		if !exists {
			return fmt.Errorf("bot code %s does not exist", options.String("botcode1"))
		}
		botcode1 = pb.BotRequest_BotCode(botcode1Value)
	}
	if options.String("botcode2") == "" {
		botcode2 = pb.BotRequest_HASTY_BOT
	} else {
		botcode2Value, exists := pb.BotRequest_BotCode_value[options.String("botcode2")]
		if !exists {
			return fmt.Errorf("bot code %s does not exist", options.String("botcode2"))
		}
		botcode2 = pb.BotRequest_BotCode(botcode2Value)
	}
	if minsimplies1, err = options.IntDefault("minsimplies1", 0); err != nil {
		return err
	}
	if minsimplies2, err = options.IntDefault("minsimplies2", 0); err != nil {
		return err
	}
	if numgames, err = options.IntDefault("numgames", 1e9); err != nil {
		return err
	}
	stochastic1 = options.Bool("stochastic1")
	stochastic2 = options.Bool("stochastic2")
	block = options.Bool("block")
	genseeds = options.Bool("genseeds")
	deterministic = options.Bool("deterministic")
	seedfile = options.String("seedfile")
	if numthreads, err = options.IntDefault("threads", runtime.NumCPU()); err != nil {
		return err
	}
	if numthreads < 1 {
		return errors.New("need at least one thread")
	}
	if len(args) == 1 {
		if args[0] == "stop" {
			if !sc.gameRunnerRunning {
				return errors.New("automatic game runner is not running")
			}
			sc.gameRunnerCancel()
			sc.gameRunnerRunning = false
			return nil
		} else {
			return errors.New("argument not recognized")
		}
	}
	if sc.gameRunnerRunning {
		return errors.New("please stop automatic game runner before running another one")
	}
	if sc.solving() {
		return errMacondoSolving
	}

	sc.showMessage("automatic game runner will log to " + logfile)

	// Handle deterministic mode
	var detConfig *automatic.DeterministicConfig
	if deterministic || genseeds || seedfile != "" {
		detConfig = &automatic.DeterministicConfig{
			SeedFile: seedfile,
			NumGames: numgames,
		}

		if genseeds {
			// Just generate seeds and exit
			if seedfile == "" {
				return errors.New("-genseeds requires -seedfile")
			}
			seeds, err := automatic.GenerateSeeds(numgames)
			if err != nil {
				return fmt.Errorf("failed to generate seeds: %w", err)
			}
			err = automatic.SaveSeeds(seeds, seedfile)
			if err != nil {
				return fmt.Errorf("failed to save seeds: %w", err)
			}
			sc.showMessage(fmt.Sprintf("Generated and saved %d seeds to %s", numgames, seedfile))
			return nil
		}

		if deterministic {
			// Load seeds from file for deterministic games
			if seedfile == "" {
				return errors.New("-deterministic requires -seedfile")
			}
			seeds, err := automatic.LoadSeeds(seedfile)
			if err != nil {
				return fmt.Errorf("failed to load seeds: %w", err)
			}
			detConfig.Seeds = seeds
			sc.showMessage(fmt.Sprintf("Loaded %d seeds from %s for deterministic play", len(seeds), seedfile))
		}
	}

	sc.gameRunnerCtx, sc.gameRunnerCancel = context.WithCancel(context.Background())
	err = automatic.StartCompVCompStaticGames(
		sc.gameRunnerCtx, sc.config, numgames, block, numthreads,
		logfile, lexicon, letterDistribution,
		[]automatic.AutomaticRunnerPlayer{
			{LeaveFile: leavefile1,
				PEGFile:              pegfile1,
				BotCode:              botcode1,
				MinSimPlies:          minsimplies1,
				StochasticStaticEval: stochastic1},
			{LeaveFile: leavefile2,
				PEGFile:              pegfile2,
				BotCode:              botcode2,
				MinSimPlies:          minsimplies2,
				StochasticStaticEval: stochastic2},
		}, detConfig)

	if err != nil {
		return err
	}
	sc.gameRunnerRunning = true
	sc.showMessage("Started automatic game runner...")
	return nil
}

type shellcmd struct {
	cmd     string
	args    []string
	options CmdOptions
}

func extractFields(line string) (*shellcmd, error) {
	fields, err := shellquote.Split(line)
	if err != nil {
		return nil, err
	}

	if len(fields) == 0 {
		return nil, errNoData
	}
	cmd := fields[0]
	var args []string
	options := CmdOptions{}
	// handle options

	lastWasOption := false
	lastOption := ""
	for idx := 1; idx < len(fields); idx++ {

		// Only treat as option if it starts with '-' and is not a negative number and is not a single dash
		if strings.HasPrefix(fields[idx], "-") && len(fields[idx]) > 1 && (fields[idx][1] < '0' || fields[idx][1] > '9') {
			// If we had a previous option waiting for a value, treat it as a boolean flag
			if lastWasOption {
				options[lastOption] = append(options[lastOption], "true")
			}
			// option
			lastWasOption = true
			lastOption = fields[idx][1:]
			continue
		}
		if lastWasOption {
			lastWasOption = false
			options[lastOption] = append(options[lastOption], fields[idx])
		} else {
			args = append(args, fields[idx])
		}
	}
	// Handle boolean flags (flags without values at the end)
	if lastWasOption {
		options[lastOption] = append(options[lastOption], "true")
	}
	log.Debug().Msgf("cmd: %v, args: %v, options: %v", args, options, cmd)

	return &shellcmd{
		cmd:     cmd,
		args:    args,
		options: options,
	}, nil
}

const maxAliasDepth = 10 // Prevent infinite recursion in alias chaining

// expandAliases recursively expands aliases in the command line.
// It handles chaining (aliases that call other aliases) and prevents infinite loops.
func (sc *ShellController) expandAliases(line string) (string, error) {
	return sc.expandAliasesHelper(line, make(map[string]bool), 0)
}

func (sc *ShellController) expandAliasesHelper(line string, seen map[string]bool, depth int) (string, error) {
	if depth > maxAliasDepth {
		return "", fmt.Errorf("alias chain too deep (possible circular reference)")
	}

	// Parse the command to get the first word
	cmd, err := extractFields(line)
	if err != nil {
		return line, nil // If we can't parse, just return original
	}

	// Check if this command is an alias
	aliasValue, isAlias := sc.aliases[cmd.cmd]
	if !isAlias {
		return line, nil // Not an alias, return as-is
	}

	// Check for circular reference
	if seen[cmd.cmd] {
		return "", fmt.Errorf("circular alias reference detected: %s", cmd.cmd)
	}

	// Mark this alias as seen
	seen[cmd.cmd] = true

	// Build the expanded command
	// Start with the alias value
	expandedParts := []string{aliasValue}

	// Add any additional arguments from the original command
	if len(cmd.args) > 0 {
		expandedParts = append(expandedParts, cmd.args...)
	}

	// Add options
	for opt, values := range cmd.options {
		for _, val := range values {
			expandedParts = append(expandedParts, "-"+opt, val)
		}
	}

	// Join into a single command line
	expanded := strings.Join(expandedParts, " ")

	// Recursively expand in case the alias value contains another alias
	return sc.expandAliasesHelper(expanded, seen, depth+1)
}

func (sc *ShellController) standardModeSwitch(line string, sig chan os.Signal) (*Response, error) {
	// Expand any aliases first
	expandedLine, err := sc.expandAliases(line)
	if err != nil {
		return nil, err
	}

	cmd, err := extractFields(expandedLine)
	if err != nil {
		return nil, err
	}
	switch cmd.cmd {
	case "exit":
		sig <- syscall.SIGINT
		return nil, errors.New("sending quit signal")
	case "help":
		return sc.help(cmd)
	case "alias":
		return sc.alias(cmd)
	case "new":
		return sc.newGame(cmd)
	case "load":
		return sc.load(cmd)
	case "unload":
		return sc.unload(cmd)
	case "last":
		return sc.last(cmd)
	case "n":
		return sc.next(cmd)
	case "p":
		return sc.prev(cmd)
	case "s":
		return sc.show(cmd)
	case "name":
		return sc.name(cmd)
	case "note":
		return sc.note(cmd)
	case "turn":
		return sc.turn(cmd)
	case "rack":
		return sc.rack(cmd)
	case "set":
		return sc.set(cmd)
	case "setconfig":
		return sc.setConfig(cmd)
	case "gen":
		return sc.generate(cmd)
	case "autoplay":
		return sc.autoplay(cmd)
	case "sim":
		return sc.sim(cmd)
	case "infer":
		return sc.infer(cmd)
	case "add":
		return sc.add(cmd)
	case "challenge":
		return sc.challenge(cmd)
	case "commit":
		return sc.commit(cmd)
	case "aiplay":
		return sc.eliteplay(cmd)
	case "hastyplay":
		return sc.hastyplay(cmd)
	case "selftest":
		return sc.selftest(cmd)
	case "list":
		return sc.list(cmd)
	case "var", "variation", "variations":
		return sc.variation(cmd)
	case "endgame":
		return sc.endgame(cmd)
	case "peg":
		return sc.preendgame(cmd)
	case "mode":
		return sc.setMode(cmd)
	case "export":
		return sc.export(cmd)
	case "render":
		return sc.render3D(cmd)
	case "analyze":
		return sc.analyze(cmd)
	case "analyze-batch":
		return sc.analyzeBatch(cmd)
	case "autoanalyze":
		return sc.autoAnalyze(cmd)
	case "script":
		return sc.script(cmd)
	case "gid":
		return sc.gid(cmd)
	case "leave":
		return sc.leave(cmd)
	case "cgp":
		return sc.cgp(cmd)
	case "check":
		return sc.check(cmd)
	case "update":
		return sc.update(cmd)
	case "gamestate":
		return sc.gameState(cmd)
	case "mleval":
		return sc.mleval(cmd)
	case "winpct":
		return sc.winpct(cmd)
	case "explain":
		return sc.explain(cmd)
	default:
		msg := fmt.Sprintf("command %v not found", strconv.Quote(cmd.cmd))
		log.Info().Msg(msg)
		return nil, errors.New(msg)
	}
}

func (sc *ShellController) Loop(sig chan os.Signal) {

	defer sc.l.Close()

	for {
		if sc.game != nil && sc.game.History() != nil {
			log.Debug().Msgf("loop-lastknownracks %v", sc.game.History().LastKnownRacks)
		}
		line, err := sc.l.Readline()
		if err == readline.ErrInterrupt {
			if len(line) == 0 {
				sig <- syscall.SIGINT
				break
			} else {
				continue
			}
		} else if err == io.EOF {
			sig <- syscall.SIGINT
			break
		}
		line = strings.TrimSpace(line)

		if sc.curMode == StandardMode {
			resp, err := sc.standardModeSwitch(line, sig)
			if err != nil {
				sc.showError(err)
			} else if resp != nil {
				sc.showMessage(resp.message)
			}
		} else if sc.curMode == EndgameDebugMode {
			err := sc.endgameDebugModeSwitch(line, sig)
			if err != nil {
				sc.showError(err)
			}
		}

	}
	log.Debug().Msgf("Exiting readline loop...")
}

func (sc *ShellController) Execute(sig chan os.Signal, line string) {
	defer sc.l.Close()
	if sc.curMode == StandardMode {
		resp, err := sc.standardModeSwitch(line, sig)
		if err != nil {
			sc.showError(err)
		} else if resp != nil {
			sc.showMessage(resp.message)
		}
	} else if sc.curMode == EndgameDebugMode {
		err := sc.endgameDebugModeSwitch(line, sig)
		if err != nil {
			sc.showError(err)
		}
	}
}
