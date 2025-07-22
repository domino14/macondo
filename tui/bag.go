package tui

import (
	"fmt"
	"sort"
	"strings"

	"github.com/rivo/tview"
	"github.com/domino14/word-golib/tilemapping"
)

type BagPanel struct {
	view   *tview.TextView
	tuiApp *TUIApp
}

func NewBagPanel(tuiApp *TUIApp) *BagPanel {
	panel := &BagPanel{
		tuiApp: tuiApp,
	}

	panel.view = tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWrap(true)

	panel.view.SetBorder(true).SetTitle("Bag & Unseen Tiles")
	panel.view.SetText("No game loaded.")

	return panel
}

func (bp *BagPanel) GetView() tview.Primitive {
	return bp.view
}

func (bp *BagPanel) Refresh() {
	if !bp.tuiApp.shell.IsPlaying() {
		bp.view.SetText("No game loaded.")
		return
	}

	game := bp.tuiApp.shell.GetGame()
	if game == nil {
		bp.view.SetText("No game loaded.")
		return
	}

	var sb strings.Builder
	
	bag := game.Bag()
	tilesInBag := bag.TilesRemaining()
	
	sb.WriteString(fmt.Sprintf("Tiles in bag: [yellow]%d[white]\n\n", tilesInBag))
	
	if tilesInBag > 0 {
		// Get bag contents and opponent rack tiles (unseen to current player)
		bagTiles := bag.Peek()
		nextPlayer := game.NextPlayer()
		oppRack := game.RackFor(nextPlayer)
		oppTiles := oppRack.TilesOn()
		
		// Combine unseen tiles
		unseenTiles := append(bagTiles, oppTiles...)
		
		// Count tiles by letter
		tileCounts := make(map[tilemapping.MachineLetter]int)
		for _, tile := range unseenTiles {
			tileCounts[tile]++
		}
		
		// Sort letters for consistent display
		var letters []tilemapping.MachineLetter
		for letter := range tileCounts {
			letters = append(letters, letter)
		}
		sort.Slice(letters, func(i, j int) bool {
			return letters[i] < letters[j]
		})
		
		sb.WriteString("Unseen tile distribution:\n")
		for _, letter := range letters {
			count := tileCounts[letter]
			letterStr := letter.UserVisible(game.Alphabet(), false)
			if letter == 0 { // blank tile
				sb.WriteString(fmt.Sprintf("[yellow]?[white]:%d ", count))
			} else if letter.IsVowel(bag.LetterDistribution()) {
				sb.WriteString(fmt.Sprintf("[lightblue]%s[white]:%d ", letterStr, count))
			} else {
				sb.WriteString(fmt.Sprintf("[white]%s[white]:%d ", letterStr, count))
			}
		}
		
		// Show vowel/consonant breakdown
		vowels := 0
		consonants := 0
		blanks := 0
		for _, tile := range unseenTiles {
			if tile == 0 { // blank tile is represented as 0
				blanks++
			} else if tile.IsVowel(bag.LetterDistribution()) {
				vowels++
			} else {
				consonants++
			}
		}
		
		sb.WriteString(fmt.Sprintf("\n\nBreakdown:\n"))
		sb.WriteString(fmt.Sprintf("Vowels: [lightblue]%d[white]\n", vowels))
		sb.WriteString(fmt.Sprintf("Consonants: [white]%d[white]\n", consonants))
		sb.WriteString(fmt.Sprintf("Blanks: [yellow]%d[white]\n", blanks))
		
		// Show opponent rack info
		oppRackSize := oppRack.NumTiles()
		sb.WriteString(fmt.Sprintf("\nOpp rack: %d tiles\n", oppRackSize))
		
	} else {
		sb.WriteString("Bag is empty!")
	}

	bp.view.SetText(sb.String())
}