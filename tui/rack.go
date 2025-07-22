package tui

import (
	"fmt"
	"strings"

	"github.com/rivo/tview"
)

type RackPanel struct {
	view   *tview.TextView
	tuiApp *TUIApp
}

func NewRackPanel(tuiApp *TUIApp) *RackPanel {
	panel := &RackPanel{
		tuiApp: tuiApp,
	}

	panel.view = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWrap(false)

	panel.view.SetBorder(true).SetTitle("Player Rack")
	panel.view.SetText("No game loaded.")

	return panel
}

func (rp *RackPanel) GetView() tview.Primitive {
	return rp.view
}

func (rp *RackPanel) Refresh() {
	if !rp.tuiApp.shell.IsPlaying() {
		rp.view.SetText("No game loaded.")
		return
	}

	game := rp.tuiApp.shell.GetGame()
	if game == nil {
		rp.view.SetText("No game loaded.")
		return
	}

	var sb strings.Builder
	
	// Show current player info
	currentPlayer := game.PlayerOnTurn()
	playerInfo := game.History().Players[currentPlayer]
	sb.WriteString(fmt.Sprintf("Player: [yellow]%s[white] (%s)\n", 
		playerInfo.Nickname, playerInfo.RealName))
	
	// Show current rack
	rack := game.RackFor(currentPlayer)
	rackTiles := rack.TilesOn()
	if len(rackTiles) == 0 {
		sb.WriteString("Rack: [red]Empty[white]")
	} else {
		sb.WriteString("Rack: ")
		for i, tile := range rackTiles {
			if i > 0 {
				sb.WriteString(" ")
			}
			letter := tile.UserVisible(game.Alphabet(), false)
			if tile == 0 { // blank tile
				sb.WriteString(fmt.Sprintf("[yellow]%s[white]", strings.ToLower(letter)))
			} else {
				sb.WriteString(fmt.Sprintf("[white]%s[white]", letter))
			}
		}
	}
	
	// Show scores
	sb.WriteString("\n\nScores:\n")
	for i, player := range game.History().Players {
		score := game.PointsFor(i)
		if i == currentPlayer {
			sb.WriteString(fmt.Sprintf("→ [yellow]%s: %d[white]\n", player.Nickname, score))
		} else {
			sb.WriteString(fmt.Sprintf("  %s: %d\n", player.Nickname, score))
		}
	}
	
	// Show turn info
	sb.WriteString(fmt.Sprintf("\nTurn: %d", game.Turn()))

	rp.view.SetText(sb.String())
}