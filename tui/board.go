package tui

import (
	"github.com/rivo/tview"
)

type BoardPanel struct {
	view   *tview.TextView
	tuiApp *TUIApp
}

func NewBoardPanel(tuiApp *TUIApp) *BoardPanel {
	panel := &BoardPanel{
		tuiApp: tuiApp,
	}

	panel.view = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWrap(false).
		SetScrollable(false)

	panel.view.SetBorder(true).SetTitle("Board")
	panel.view.SetText("No game loaded. Press 'l' to load a game.")

	return panel
}

func (bp *BoardPanel) GetView() tview.Primitive {
	return bp.view
}

func (bp *BoardPanel) Refresh() {
	if !bp.tuiApp.shell.IsPlaying() {
		bp.view.SetText("No game loaded. Press 'l' to load a game.")
		return
	}

	game := bp.tuiApp.shell.GetGame()
	if game == nil {
		bp.view.SetText("No game loaded.")
		return
	}

	// Use the game's built-in board display method
	boardText := game.Board().ToDisplayText(game.Alphabet())
	bp.view.SetText(boardText)
}

