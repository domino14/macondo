package tui

import (
	"fmt"
	"strings"

	"github.com/rivo/tview"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/word-golib/tilemapping"
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
	log.Debug().Msg("TUI: Refreshing board panel")
	if !bp.tuiApp.shell.IsPlaying() {
		bp.view.SetText("No game loaded. Press 'l' to load a game.")
		return
	}

	game := bp.tuiApp.shell.GetGame()
	if game == nil {
		bp.view.SetText("No game loaded.")
		return
	}

	// Use our custom TUI board renderer
	boardText := bp.renderBoardForTUI(game.Board(), game.Alphabet())
	bp.view.SetText(boardText)
}

// renderBoardForTUI creates a board display using tview color tags instead of ANSI codes
func (bp *BoardPanel) renderBoardForTUI(gameBoard *board.GameBoard, alphabet *tilemapping.TileMapping) string {
	var sb strings.Builder
	
	// Header with column letters - use consistent 4-character width per column
	sb.WriteString("    ") // 4 spaces to align with row numbers
	dim := gameBoard.Dim()
	for col := 0; col < dim; col++ {
		sb.WriteString(fmt.Sprintf(" %c  ", 'A'+col)) // 4 chars: space + letter + 2 spaces
	}
	sb.WriteString("\n")
	
	// Add a separator line for better visual separation
	sb.WriteString("    ")
	for col := 0; col < dim; col++ {
		sb.WriteString("────") // 4 dashes per column
	}
	sb.WriteString("\n")
	
	// Board rows with more vertical spacing
	for row := 0; row < dim; row++ {
		// Row number - use 3 characters + space for alignment
		sb.WriteString(fmt.Sprintf("%2d │", row+1))
		
		// Board squares - each square is exactly 4 characters wide
		for col := 0; col < dim; col++ {
			squareStr := bp.renderSquareForTUI(gameBoard, row, col, alphabet)
			sb.WriteString(squareStr)
		}
		sb.WriteString("\n")
		
		// Add vertical spacing between rows for better readability
		if row < dim-1 {
			sb.WriteString("   │")
			for col := 0; col < dim; col++ {
				sb.WriteString("    ") // 4 spaces per column
			}
			sb.WriteString("\n")
		}
	}
	
	return sb.String()
}

// renderSquareForTUI renders a single board square with TUI color formatting
// Each square is exactly 4 characters wide for perfect alignment
func (bp *BoardPanel) renderSquareForTUI(gameBoard *board.GameBoard, row, col int, alphabet *tilemapping.TileMapping) string {
	tile := gameBoard.GetLetter(row, col)
	
	// If there's a tile on this square
	if tile != 0 {
		tileStr := bp.getTileDisplayString(tile, alphabet)
		// Use white background and black text for tiles - exactly 4 chars
		return fmt.Sprintf("[black:white:b] %s  [-:-:-]", tileStr)
	}
	
	// Empty square - show bonus square with appropriate color
	bonusSquare := gameBoard.GetBonus(row, col)
	bonusStr, bgColor := bp.getBonusDisplayString(bonusSquare)
	
	if bgColor != "" {
		// Bonus squares - exactly 4 chars
		return fmt.Sprintf("[white:%s]%s[-:-]", bgColor, bonusStr)
	}
	
	// Regular empty square - exactly 4 chars
	return " ·  "
}

// getTileDisplayString formats a tile for display
func (bp *BoardPanel) getTileDisplayString(tile tilemapping.MachineLetter, alphabet *tilemapping.TileMapping) string {
	if tile == 0 {
		return " "
	}
	
	// Handle blank tiles
	if tile.IsBlanked() {
		// Show the letter that the blank represents in lowercase
		letter := tile.UserVisible(alphabet, true)
		return string(letter)
	}
	
	// Regular tiles
	letter := tile.UserVisible(alphabet, false)
	return string(letter)
}

// getBonusDisplayString returns the display string and background color for bonus squares
// Each string is exactly 4 characters for consistent alignment
func (bp *BoardPanel) getBonusDisplayString(bonus board.BonusSquare) (string, string) {
	switch bonus {
	case board.Bonus4WS: // Quadruple word score
		return " 4W ", "yellow"
	case board.Bonus3WS: // Triple word score
		return " 3W ", "red"
	case board.Bonus2WS: // Double word score
		return " 2W ", "purple"
	case board.Bonus4LS: // Quadruple letter score
		return " 4L ", "orange"
	case board.Bonus3LS: // Triple letter score
		return " 3L ", "blue"
	case board.Bonus2LS: // Double letter score
		return " 2L ", "cyan"
	case board.NoBonus:
		return " ·  ", ""
	default:
		return " ?  ", ""
	}
}