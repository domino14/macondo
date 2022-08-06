package tui

import (
	"os"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/alphabeta"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

type model struct {
	keys keyMap
	help help.Model

	config *config.Config

	game   *airunner.AIGameRunner
	simmer *montecarlo.Simmer

	gen           movegen.MoveGenerator
	endgameSolver *alphabeta.Solver
	curPlayList   []*move.Move
	err           error // some displayed error
}

type keyMap struct {
	// Blank mode
	NewGame  key.Binding
	LoadGame key.Binding

	// In-Game
	Sim          key.Binding // Sim can be toggled
	Gen          key.Binding
	DelPlay      key.Binding
	AiPlay       key.Binding
	NextPlay     key.Binding
	PreviousPlay key.Binding
	Endgame      key.Binding
	SetRack      key.Binding
	Commit       key.Binding
	AddPlay      key.Binding
	Challenge    key.Binding
	Up           key.Binding
	Down         key.Binding
	// Other?
	Help key.Binding
	Quit key.Binding
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Help, k.Quit}
}

func (k keyMap) FullHelp() [][]key.Binding {

	return [][]key.Binding{
		{k.NewGame, k.LoadGame}, // First column
		{k.Gen, k.Sim, k.NextPlay, k.PreviousPlay, k.SetRack},
		{k.Help, k.Quit},
	}
}

var keys = keyMap{
	NewGame: key.NewBinding(
		key.WithKeys("n"),
		key.WithHelp("n", "new game"),
	),
	LoadGame: key.NewBinding(
		key.WithKeys("l"),
		key.WithHelp("l", "load game"),
	),

	Sim: key.NewBinding(
		key.WithKeys("s"),
		key.WithHelp("s", "toggle sim start/stop"),
	),
	Gen: key.NewBinding(
		key.WithKeys("g"),
		key.WithHelp("g", "generate play table"),
	),
	DelPlay: key.NewBinding(
		key.WithKeys("d", "delete"),
		key.WithHelp("d/delete", "delete play from table"),
	),
	NextPlay: key.NewBinding(
		key.WithKeys("right"),
		key.WithHelp("→", "go forward one play"),
	),
	PreviousPlay: key.NewBinding(
		key.WithKeys("left"),
		key.WithHelp("←", "go back one play"),
	),
	Up: key.NewBinding(
		key.WithKeys("up"),
		key.WithHelp("↑", "select from play table (up)"),
	),
	Down: key.NewBinding(
		key.WithKeys("down"),
		key.WithHelp("↓", "select from play table (down)"),
	),
	Commit: key.NewBinding(
		key.WithKeys("c"),
		key.WithHelp("c", "commit selected play"),
	),
	SetRack: key.NewBinding(
		key.WithKeys("r"),
		key.WithHelp("r", "set rack"),
	),

	Help: key.NewBinding(
		key.WithKeys("?"),
		key.WithHelp("?", "toggle help"),
	),
	Quit: key.NewBinding(
		key.WithKeys("q"),
		key.WithHelp("q", "quit Macondo"),
	),
}

func InitialModel() model {
	return model{
		keys: keys,
		help: help.New(),
	}
}

type errMsg struct{ err error }
type cfgMsg *config.Config

// For messages that contain errors it's often handy to also implement the
// error interface on the message.
func (e errMsg) Error() string { return e.err.Error() }

func loadConfig(args []string) tea.Cmd {
	return func() tea.Msg {
		cfg := &config.Config{}
		err := cfg.Load(args)
		if err != nil {
			return errMsg{err}
		}
		return cfgMsg(cfg)
	}
}

func (m model) Init() tea.Cmd {
	return loadConfig(os.Args[1:])
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.help.Width = msg.Width

	case tea.KeyMsg:

		switch {
		case key.Matches(msg, m.keys.NewGame):

		case key.Matches(msg, m.keys.LoadGame):

		case key.Matches(msg, m.keys.Help):
			m.help.ShowAll = !m.help.ShowAll
		case key.Matches(msg, m.keys.Quit):
			return m, tea.Quit
		}

	case cfgMsg:
		m.config = (*config.Config)(msg)

	case errMsg:
		m.err = msg.err

	}

	return m, nil
}

func (m model) View() string {
	// if m.game == nil {
	// 	retur
	// }
	helpview := m.help.View(m.keys)
	return helpview
}
