package bot

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"

	"github.com/chzyer/readline"
	"github.com/nats-io/nats.go"
	"github.com/rs/zerolog/log"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
)

const (
	SelfPlayer = 0
	BotPlayer  = 1
)

var (
	errNoData            = errors.New("no data in this line")
	errWrongOptionSyntax = errors.New("wrong format; all options need arguments")
)

// Options to configure the interactve shell
type ShellOptions struct {
	runner.GameOptions
	lowercaseMoves bool
}

func NewShellOptions() *ShellOptions {
	return &ShellOptions{
		GameOptions: runner.GameOptions{
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
		rule := runner.ShowChallengeRule(opts.ChallengeRule)
		return true, fmt.Sprintf("%v", rule)
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

type Response struct {
	message string
	send    bool
}

func Msg(message string) *Response {
	return &Response{message: message, send: false}
}

// Keep Send in case we need bot debug messages, but for regular bot-to-play
// the shell now automatically sends a message.
func Send(message string) *Response {
	return &Response{message: message, send: true}
}

type ShellController struct {
	l        *readline.Instance
	config   *config.Config
	execPath string
	options  *ShellOptions
	game     *airunner.AIGameRunner
	client   Client
}

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

func (sc *ShellController) IsPlaying() bool {
	return sc.game != nil && sc.game.IsPlaying()
}

func (sc *ShellController) IsBotOnTurn() bool {
	return sc.game != nil && sc.game.PlayerOnTurn() == BotPlayer
}

func (sc *ShellController) getMove() error {
	sc.showMessage("Requesting move from bot")
	m, err := sc.client.RequestMove(&sc.game.GameRunner, sc.config)
	if err != nil {
		sc.showMessage("Bot returned error: " + err.Error())
		return err
	} else {
		sc.showMessage("Bot returned move: " + m.ShortDescription())
	}
	err = sc.game.PlayMove(m, true, 0)
	if err != nil {
		return err
	}
	sc.showMessage(sc.game.ToDisplayText())
	return nil
}

func (sc *ShellController) newGame() (*Response, error) {
	players := []*pb.PlayerInfo{
		{Nickname: "self", RealName: "Arthur Dent"},
		{Nickname: "opponent", RealName: "Macondo Bot"},
	}

	opts := sc.options.GameOptions
	g, err := airunner.NewAIGameRunner(sc.config, &opts, players, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return nil, err
	}
	sc.game = g
	if g.PlayerOnTurn() == SelfPlayer {
		return Msg(sc.game.ToDisplayText()), nil
	} else {
		return Msg("Opponent goes first"), nil
	}
}

func (sc *ShellController) show() (*Response, error) {
	return Msg(sc.game.ToDisplayText()), nil
}

func (sc *ShellController) play(args []string) (*Response, error) {
	if len(args) != 2 {
		return nil, errors.New("play <coords> <word>")
	}
	coords, word := args[0], args[1]
	m, err := sc.game.NewPlacementMove(SelfPlayer, coords, word)
	if err != nil {
		return nil, err
	}
	return sc.commit(m)
}

func (sc *ShellController) exchange(args []string) (*Response, error) {
	tiles := args[0]
	m, err := sc.game.NewExchangeMove(SelfPlayer, tiles)
	if err != nil {
		return nil, err
	}
	return sc.commit(m)
}

func (sc *ShellController) pass() (*Response, error) {
	m, err := sc.game.NewPassMove(SelfPlayer)
	if err != nil {
		return nil, err
	}
	return sc.commit(m)
}

func (sc *ShellController) commit(m *move.Move) (*Response, error) {
	sc.showMessage("Committing move: " + m.ShortDescription())
	err := sc.game.PlayMove(m, true, 0)
	if err != nil {
		return nil, err
	}
	msg := sc.game.ToDisplayText()
	return Msg(msg), nil
}

func (sc *ShellController) aiplay() (*Response, error) {
	if !sc.IsPlaying() {
		return nil, errors.New("game is over")
	}
	moves := sc.game.GenerateMoves(1)
	m := moves[0]
	return sc.commit(m)
}

func (sc *ShellController) handle(line string) (*Response, error) {
	fields := strings.Fields(line)
	cmd := fields[0]
	args := fields[1:]
	switch cmd {
	case "new", "n":
		return sc.newGame()
	case "show", "s", "b":
		return sc.show()
	case "play", "pl", "p":
		return sc.play(args)
	case "exchange", "exch", "ex", "x":
		return sc.exchange(args)
	case "pass", "pa":
		return sc.pass()
	case "aiplay", "ai", "a":
		return sc.aiplay()
	default:
		msg := fmt.Sprintf("command %v not found", strconv.Quote(cmd))
		log.Info().Msg(msg)
		return nil, errors.New(msg)
	}
}

func (sc *ShellController) Loop(channel string, sig chan os.Signal) {

	defer sc.l.Close()

	// Initialize the NATS client
	nc, err := nats.Connect(nats.DefaultURL)
	if err != nil {
		log.Fatal()
	}
	defer nc.Close()
	sc.client = Client{nc: nc, channel: channel}

	// Run the readline loop
	for {
		if sc.IsBotOnTurn() {
			err = sc.getMove()
			if err != nil {
				sc.showError(err)
			}
			continue
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

		if line == "exit" {
			sig <- syscall.SIGINT
			break
		} else {
			resp, err := sc.handle(line)
			if err != nil {
				sc.showError(err)
			} else if resp != nil {
				sc.showMessage(resp.message)
			}
		}
	}
	log.Debug().Msgf("Exiting readline loop...")
}
