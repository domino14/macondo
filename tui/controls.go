package tui

import (
	"github.com/rivo/tview"
)

type ControlsPanel struct {
	view       *tview.Flex
	tuiApp     *TUIApp
	buttons    map[string]*tview.Button
	movesList  *tview.TextView
}

func NewControlsPanel(tuiApp *TUIApp) *ControlsPanel {
	panel := &ControlsPanel{
		tuiApp:  tuiApp,
		buttons: make(map[string]*tview.Button),
	}

	panel.setupControls()
	return panel
}

func (cp *ControlsPanel) setupControls() {
	// Create button layout
	buttonGrid := tview.NewFlex().SetDirection(tview.FlexColumn)

	// Navigation buttons
	cp.buttons["prev"] = tview.NewButton("◀ Prev (p)").SetSelectedFunc(func() {
		cp.tuiApp.prevTurn()
	})
	cp.buttons["next"] = tview.NewButton("Next ▶ (n)").SetSelectedFunc(func() {
		cp.tuiApp.nextTurn()
	})

	// Action buttons  
	cp.buttons["load"] = tview.NewButton("Load (l)").SetSelectedFunc(func() {
		cp.tuiApp.showLoadDialog()
	})
	cp.buttons["gen"] = tview.NewButton("Gen (g)").SetSelectedFunc(func() {
		cp.tuiApp.generateMoves()
	})

	// Analysis buttons
	cp.buttons["sim"] = tview.NewButton("Sim (s)").SetSelectedFunc(func() {
		cp.tuiApp.startSimulation()
	})
	cp.buttons["endgame"] = tview.NewButton("Endgame (e)").SetSelectedFunc(func() {
		cp.tuiApp.startEndgame()
	})

	// Debug button
	cp.buttons["debug"] = tview.NewButton("Debug (d)").SetSelectedFunc(func() {
		cp.tuiApp.showLogViewer()
	})

	// Add buttons to grid (first row)
	buttonGrid.AddItem(cp.buttons["load"], 0, 1, false)
	buttonGrid.AddItem(cp.buttons["prev"], 0, 1, false)
	buttonGrid.AddItem(cp.buttons["next"], 0, 1, false)
	buttonGrid.AddItem(cp.buttons["gen"], 0, 1, false)
	buttonGrid.AddItem(cp.buttons["sim"], 0, 1, false)
	buttonGrid.AddItem(cp.buttons["endgame"], 0, 1, false)
	buttonGrid.AddItem(cp.buttons["debug"], 0, 1, false)

	// Create moves list display
	cp.movesList = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWrap(false).
		SetScrollable(true)
	cp.movesList.SetBorder(true).SetTitle("Moves")
	cp.movesList.SetText("Generate moves to see them here.")

	// Create main controls layout
	cp.view = tview.NewFlex().SetDirection(tview.FlexRow)
	cp.view.SetBorder(true).SetTitle("Controls")
	cp.view.AddItem(buttonGrid, 3, 0, true) // Fixed height for buttons
	cp.view.AddItem(cp.movesList, 0, 1, false) // Remaining space for moves
}

func (cp *ControlsPanel) GetView() tview.Primitive {
	return cp.view
}

func (cp *ControlsPanel) Refresh() {
	// Update moves list if there's a current game
	if !cp.tuiApp.shell.IsPlaying() {
		cp.movesList.SetText("No game loaded.")
		return
	}

	// Get current moves from shell controller
	game := cp.tuiApp.shell.GetGame()
	if game == nil {
		cp.movesList.SetText("No game loaded.")
		return
	}

	// Check if there are generated moves to display
	moveList := cp.tuiApp.shell.GetCurPlayList()
	if len(moveList) == 0 {
		cp.movesList.SetText("Press 'Gen' to generate moves\nPress 'Sim' for simulation\nPress 'Endgame' for solver")
	} else {
		// Display the move list using the shell's formatter
		movesText := cp.tuiApp.shell.GetGenDisplayMoveList()
		cp.movesList.SetText(movesText)
	}
}