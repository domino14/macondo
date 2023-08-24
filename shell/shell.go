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
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/chzyer/readline"
	"github.com/kballard/go-shellquote"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/preendgame"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/tilemapping"
	"github.com/domino14/macondo/turnplayer"
)

const (
	SimLog   = "/tmp/simlog"
	InferLog = "/tmp/inferlog"
)

var (
	errNoData = errors.New("no data in this line")
)

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
	keys := []string{"lexicon", "challenge", "lower"}
	out := strings.Builder{}
	out.WriteString("Settings:\n")
	for _, key := range keys {
		_, val := opts.Show(key)
		out.WriteString("  " + key + ": ")
		out.WriteString(val + "\n")
	}
	return out.String()
}

type ShellController struct {
	l        *readline.Instance
	config   *config.Config
	execPath string

	options *ShellOptions

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
	curPlayList      []*move.Move
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

func (sc *ShellController) showMessage(msg string) {
	writeln(msg, sc.l.Stderr())
}

func (sc *ShellController) showError(err error) {
	sc.showMessage("Error: " + err.Error())
}

func NewShellController(cfg *config.Config, execPath string) *ShellController {
	l, err := readline.NewEx(&readline.Config{
		Prompt:          "\033[31mmacondo>\033[0m ",
		HistoryFile:     "/tmp/readline.tmp",
		EOFPrompt:       "exit",
		InterruptPrompt: "^C",

		HistorySearchFold:   true,
		FuncFilterInputRune: filterInput,
	})

	if err != nil {
		panic(err)
	}
	execPath = config.FindBasePath(execPath)
	opts := NewShellOptions()
	opts.SetDefaults(cfg)

	return &ShellController{l: l, config: cfg, execPath: execPath, options: opts}
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
			err = sc.options.SetLexicon(args)
			if err == nil {
				// Overwrite the config options since other parts of the code
				// use these to determine the lexicon
				sc.config.DefaultLexicon = sc.options.Lexicon.Name
				sc.config.DefaultLetterDistribution = sc.options.Lexicon.Distribution
			}
			_, ret = sc.options.Show("lexicon")
		}
	case "board":
		if sc.IsPlaying() {
			msg := "Cannot change the board layout while a game is active (try `unload` to quit game)"
			err = errors.New(msg)
		} else {
			err = sc.options.SetBoardLayoutName(args[0])
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
	sc.simmer = &montecarlo.Simmer{}
	c, err := equity.NewCombinedStaticCalculator(
		sc.game.LexiconName(),
		sc.config, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return err
	}
	sc.simmer.Init(sc.game.Game, []equity.EquityCalculator{c}, c, sc.config)
	sc.gen = sc.game.MoveGenerator()

	gd, err := kwg.Get(sc.config, sc.game.LexiconName())
	if err != nil {
		return err
	}

	sc.backupgen = movegen.NewGordonGenerator(gd, sc.game.Board(), sc.game.Bag().LetterDistribution())

	sc.rangefinder = &rangefinder.RangeFinder{}
	sc.rangefinder.Init(sc.game.Game, []equity.EquityCalculator{c}, sc.config)
	return nil
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
		idstr := args[1]
		id, err := strconv.Atoi(idstr)
		if err != nil {
			return errors.New("badly formatted game ID")
		}
		prefix := strconv.Itoa(id / 100)
		xtpath := "https://www.cross-tables.com/annotated/selfgcg/" + prefix +
			"/anno" + idstr + ".gcg"
		resp, err := http.Get(xtpath)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		history, err = gcgio.ParseGCGFromReader(sc.config, resp.Body)
		if err != nil {
			return err
		}

	} else if args[0] == "woogles" {
		if len(args) < 2 {
			return errors.New("need to provide a woogles game id")
		}
		idstr := args[1]
		path := "https://woogles.io/twirp/game_service.GameMetadataService/GetGCG"

		reader := strings.NewReader(`{"gameId": "` + idstr + `"}`)

		resp, err := http.Post(path, "application/json", reader)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		type gcgstruct struct {
			Gcg string `json:"gcg"`
		}
		var gcgObj gcgstruct

		err = json.Unmarshal(body, &gcgObj)
		if err != nil {
			return err
		}

		history, err = gcgio.ParseGCGFromReader(sc.config, strings.NewReader(gcgObj.Gcg))
		if err != nil {
			return err
		}
	} else if args[0] == "web" {
		if len(args) < 2 {
			return errors.New("need to provide a web URL")
		}
		path := args[1]

		resp, err := http.Get(path)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		history, err = gcgio.ParseGCGFromReader(sc.config, resp.Body)
		if err != nil {
			return err
		}
	} else {
		path := args[0]
		if strings.HasPrefix(path, "~/") {
			usr, err := user.Current()
			if err != nil {
				return err
			}
			dir := usr.HomeDir
			path = filepath.Join(dir, path[2:])
		}
		history, err = gcgio.ParseGCG(sc.config, path)
		if err != nil {
			return err
		}
	}
	log.Debug().Msgf("Loaded game repr; players: %v", history.Players)
	lexicon := history.Lexicon
	if lexicon == "" {
		lexicon = sc.config.DefaultLexicon
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
	conf := &bot.BotConfig{Config: *sc.config}
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

func (sc *ShellController) loadCGP(cgpstr string) error {
	g, err := cgp.ParseCGP(sc.config, cgpstr)
	if err != nil {
		return err
	}
	lexicon := g.History().Lexicon
	if lexicon == "" {
		lexicon = sc.config.DefaultLexicon
		log.Info().Msgf("cgp file had no lexicon, so using default lexicon %v",
			lexicon)
	}
	conf := &bot.BotConfig{Config: *sc.config}
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
	// Play the actual move on the board, draw tiles, etc.
	err := sc.game.PlayMove(m, true, 0)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Added turn at turn num %v", sc.curTurnNum)
	sc.curTurnNum = sc.game.Turn()
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
	return sc.game.ParseMove(playerid, sc.options.lowercaseMoves, fields)
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
	playerid := sc.game.PlayerOnTurn()
	m, err := sc.parseCommitMove(playerid, fields)
	if err != nil {
		return err
	}
	return sc.commitMove(m)
}

func (sc *ShellController) addPlay(fields []string) error {
	playerid := sc.game.PlayerOnTurn()
	m, err := sc.parseAddMove(playerid, fields)
	if err != nil {
		return err
	}
	return sc.addMoveToList(playerid, m)
}

func (sc *ShellController) commitAIMove() error {
	if !sc.IsPlaying() {
		return errors.New("game is over")
	}
	sc.genMoves(15)
	m := sc.curPlayList[0]
	return sc.commitMove(m)
}

func (sc *ShellController) handleAutoplay(args []string, options map[string]string) error {
	var logfile, lexicon, letterDistribution, leavefile1, leavefile2, pegfile1, pegfile2 string
	var numgames, numthreads int
	var block bool
	var botcode1, botcode2 pb.BotRequest_BotCode
	var minsimplies1, minsimplies2 int
	var err error
	if options["logfile"] == "" {
		logfile = "/tmp/autoplay.txt"
	} else {
		logfile = options["logfile"]
	}
	if options["lexicon"] == "" {
		lexicon = sc.config.DefaultLexicon
	} else {
		lexicon = options["lexicon"]
	}
	if options["letterdistribution"] == "" {
		letterDistribution = sc.config.DefaultLetterDistribution
	} else {
		letterDistribution = options["letterdistribution"]
	}
	if options["leavefile1"] == "" {
		leavefile1 = ""
	} else {
		leavefile1 = options["leavefile1"]
	}
	if options["leavefile2"] == "" {
		leavefile2 = ""
	} else {
		leavefile2 = options["leavefile2"]
	}
	if options["pegfile1"] == "" {
		pegfile1 = ""
	} else {
		pegfile1 = options["pegfile1"]
	}
	if options["pegfile2"] == "" {
		pegfile2 = ""
	} else {
		pegfile2 = options["pegfile2"]
	}
	if options["botcode1"] == "" {
		botcode1 = pb.BotRequest_HASTY_BOT
	} else {
		botcode1String := options["botcode1"]
		botcode1Value, exists := pb.BotRequest_BotCode_value[botcode1String]
		if !exists {
			return fmt.Errorf("bot code %s does not exist", botcode1String)
		}
		botcode1 = pb.BotRequest_BotCode(botcode1Value)
	}
	if options["botcode2"] == "" {
		botcode2 = pb.BotRequest_HASTY_BOT
	} else {
		botcode2String := options["botcode2"]
		botcode2Value, exists := pb.BotRequest_BotCode_value[botcode2String]
		if !exists {
			return fmt.Errorf("bot code %s does not exist", botcode2String)
		}
		botcode2 = pb.BotRequest_BotCode(botcode2Value)
	}

	if options["minsimplies1"] == "" {
		minsimplies1 = 0
	} else {
		minsimplies1, err = strconv.Atoi(options["minsimplies1"])
		if err != nil {
			return err
		}
	}
	if options["minsimplies2"] == "" {
		minsimplies2 = 0
	} else {
		minsimplies2, err = strconv.Atoi(options["minsimplies2"])
		if err != nil {
			return err
		}
	}

	if options["numgames"] == "" {
		numgames = 1e9
	} else {
		var err error
		numgames, err = strconv.Atoi(options["numgames"])
		if err != nil {
			return err
		}
	}
	if options["block"] == "" {
		block = false
	} else {
		block = true
	}

	if options["threads"] == "" {
		numthreads = runtime.NumCPU()
	} else {
		var err error
		numthreads, err = strconv.Atoi(options["threads"])
		if err != nil {
			return err
		}
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

	sc.showMessage("automatic game runner will log to " + logfile)
	sc.gameRunnerCtx, sc.gameRunnerCancel = context.WithCancel(context.Background())
	err = automatic.StartCompVCompStaticGames(
		sc.gameRunnerCtx, sc.config, numgames, block, numthreads,
		logfile, lexicon, letterDistribution,
		[]automatic.AutomaticRunnerPlayer{
			{LeaveFile: leavefile1, PEGFile: pegfile1, BotCode: botcode1, MinSimPlies: minsimplies1},
			{LeaveFile: leavefile2, PEGFile: pegfile2, BotCode: botcode2, MinSimPlies: minsimplies2},
		})

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
	options map[string]string
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
	options := map[string]string{}
	// handle options

	lastWasOption := false
	lastOption := ""
	for idx := 1; idx < len(fields); idx++ {
		if strings.HasPrefix(fields[idx], "-") {
			// option
			lastWasOption = true
			lastOption = fields[idx][1:]
			continue
		}
		if lastWasOption {
			lastWasOption = false
			options[lastOption] = fields[idx]
		} else {
			args = append(args, fields[idx])
		}
	}
	log.Debug().Msgf("cmd: %v, args: %v, options: %v", args, options, cmd)

	return &shellcmd{
		cmd:     cmd,
		args:    args,
		options: options,
	}, nil
}

func (sc *ShellController) standardModeSwitch(line string, sig chan os.Signal) (*Response, error) {
	cmd, err := extractFields(line)
	if err != nil {
		return nil, err
	}
	switch cmd.cmd {
	case "exit":
		sig <- syscall.SIGINT
		return nil, errors.New("sending quit signal")
	case "help":
		return sc.help(cmd)
	case "new":
		return sc.newGame(cmd)
	case "load":
		return sc.load(cmd)
	case "unload":
		return sc.unload(cmd)
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
		return sc.aiplay(cmd)
	case "selftest":
		return sc.selftest(cmd)
	case "list":
		return sc.list(cmd)
	case "endgame":
		return sc.endgame(cmd)
	case "peg":
		return sc.preendgame(cmd)
	case "mode":
		return sc.setMode(cmd)
	case "export":
		return sc.export(cmd)
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
	default:
		msg := fmt.Sprintf("command %v not found", strconv.Quote(cmd.cmd))
		log.Info().Msg(msg)
		return nil, errors.New(msg)
	}
}

func (sc *ShellController) Loop(sig chan os.Signal) {

	defer sc.l.Close()

	for {

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
