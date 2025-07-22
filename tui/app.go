package tui

import (
	"io"
	
	"github.com/rivo/tview"
	"github.com/gdamore/tcell/v2"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/shell"
)

type TUIApp struct {
	app        *tview.Application
	shell      *shell.ShellController
	layout     *tview.Flex
	boardPanel *BoardPanel
	rackPanel  *RackPanel
	bagPanel   *BagPanel
	controls   *ControlsPanel
	statusBar  *tview.TextView
	logCapture *LogCapture
}

func NewTUIApp(cfg *config.Config, execPath, gitVersion string) *TUIApp {
	app := tview.NewApplication()
	// Enable mouse support for clicking buttons
	app.EnableMouse(true)

	tuiApp := &TUIApp{
		app: app,
	}
	
	// Re-enable logging to debug hanging issue
	tuiApp.initLogging()
	
	shellController := shell.NewShellControllerWithWriter(cfg, execPath, gitVersion, io.Discard)
	tuiApp.shell = shellController

	tuiApp.setupLayout()
	tuiApp.setupKeyBindings()

	return tuiApp
}

func (t *TUIApp) setupLayout() {
	// Create the main panels
	t.boardPanel = NewBoardPanel(t)
	t.rackPanel = NewRackPanel(t)
	t.bagPanel = NewBagPanel(t)
	t.controls = NewControlsPanel(t)
	t.statusBar = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWrap(false)
	t.statusBar.SetBorder(true).SetTitle("Status")

	// Create left panel (board + rack)
	leftPanel := tview.NewFlex().SetDirection(tview.FlexRow)
	leftPanel.AddItem(t.boardPanel.GetView(), 0, 1, false)
	leftPanel.AddItem(t.rackPanel.GetView(), 5, 0, false) // Fixed height for rack

	// Create right panel (bag + controls)
	rightPanel := tview.NewFlex().SetDirection(tview.FlexRow)
	rightPanel.AddItem(t.bagPanel.GetView(), 0, 1, false)
	rightPanel.AddItem(t.controls.GetView(), 12, 0, false) // Fixed height for controls

	// Create main layout
	mainLayout := tview.NewFlex().SetDirection(tview.FlexColumn)
	mainLayout.AddItem(leftPanel, 0, 2, true)  // Board side takes 2/3
	mainLayout.AddItem(rightPanel, 0, 1, false) // Right side takes 1/3

	// Add status bar at bottom
	t.layout = tview.NewFlex().SetDirection(tview.FlexRow)
	t.layout.AddItem(mainLayout, 0, 1, true)
	t.layout.AddItem(t.statusBar, 3, 0, false)
}

func (t *TUIApp) setupKeyBindings() {
	t.app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		// Only handle global shortcuts when not in an input field
		// Check if the current focus is an InputField
		if _, ok := t.app.GetFocus().(*tview.InputField); ok {
			// Let input fields handle their own key events
			return event
		}
		
		switch event.Rune() {
		case 'q', 'Q':
			t.app.Stop()
			return nil
		case 'n':
			t.updateStatus("Next turn pressed")
			t.nextTurn()
			return nil
		case 'p':
			t.updateStatus("Previous turn pressed")
			t.prevTurn()
			return nil
		case 'l':
			log.Debug().Msg("TUI: 'l' key pressed - showing load dialog")
			t.updateStatus("Load dialog pressed")
			t.showLoadDialog()
			return nil
		case 's':
			t.updateStatus("Simulation pressed")
			t.startSimulation()
			return nil
		case 'e':
			t.updateStatus("Endgame pressed")
			t.startEndgame()
			return nil
		case 'g':
			t.updateStatus("Generate moves pressed")
			t.generateMoves()
			return nil
		case 'd':
			t.showLogViewer()
			return nil
		}
		return event
	})
}

func (t *TUIApp) Run() error {
	// Set up initial display text for all panels
	if t.boardPanel != nil && t.boardPanel.view != nil {
		t.boardPanel.view.SetText("Macondo TUI Board Panel\nPress 'l' to load a game")
	}
	if t.rackPanel != nil && t.rackPanel.view != nil {
		t.rackPanel.view.SetText("Macondo TUI Rack Panel\nNo game loaded")
	}
	if t.bagPanel != nil && t.bagPanel.view != nil {
		t.bagPanel.view.SetText("Macondo TUI Bag Panel\nNo game loaded")
	}
	t.updateStatus("Welcome to Macondo TUI! Press 'l' to load a game, 'd' for debug log, 'q' to quit.")
	
	return t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView()).Run()
}

func (t *TUIApp) Cleanup() {
	if t.shell != nil {
		t.shell.Cleanup()
	}
}

func (t *TUIApp) updateStatus(message string) {
	t.statusBar.SetText(message)
}

func (t *TUIApp) GetShell() *shell.ShellController {
	return t.shell
}

func (t *TUIApp) GetApp() *tview.Application {
	return t.app
}

// Game control methods
func (t *TUIApp) nextTurn() {
	if !t.shell.IsPlaying() {
		t.updateStatus("No game loaded. Press 'l' to load a game.")
		return
	}
	
	err := t.shell.NextTurn()
	if err != nil {
		t.updateStatus("Error: " + err.Error())
		return
	}
	t.updateStatus("Moved to next turn")
	t.refresh()
}

func (t *TUIApp) prevTurn() {
	if !t.shell.IsPlaying() {
		t.updateStatus("No game loaded. Press 'l' to load a game.")
		return
	}
	
	err := t.shell.PrevTurn()
	if err != nil {
		t.updateStatus("Error: " + err.Error())
		return
	}
	t.updateStatus("Moved to previous turn")
	t.refresh()
}

func (t *TUIApp) generateMoves() {
	if !t.shell.IsPlaying() {
		t.updateStatus("No game loaded. Press 'l' to load a game.")
		return
	}
	
	err := t.shell.GenerateMoves(15)
	if err != nil {
		t.updateStatus("Error generating moves: " + err.Error())
		return
	}
	t.updateStatus("Generated 15 moves")
	t.refresh()
}

func (t *TUIApp) startSimulation() {
	if !t.shell.IsPlaying() {
		t.updateStatus("No game loaded. Press 'l' to load a game.")
		return
	}
	t.updateStatus("Simulation feature coming soon...")
}

func (t *TUIApp) startEndgame() {
	if !t.shell.IsPlaying() {
		t.updateStatus("No game loaded. Press 'l' to load a game.")
		return
	}
	t.updateStatus("Endgame solver feature coming soon...")
}


func (t *TUIApp) refresh() {
	log.Debug().Msg("TUI: Starting refresh - refreshing board panel")
	// Refresh all panels with current game state
	t.boardPanel.Refresh()
	log.Debug().Msg("TUI: Board panel refreshed - refreshing rack panel")
	t.rackPanel.Refresh()
	log.Debug().Msg("TUI: Rack panel refreshed - refreshing bag panel")
	t.bagPanel.Refresh()
	log.Debug().Msg("TUI: Bag panel refreshed - refreshing controls panel")
	t.controls.Refresh()
	log.Debug().Msg("TUI: All panels refreshed - refresh method completed")
	// No nested QueueUpdateDraw needed since we're already inside one
}