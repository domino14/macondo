package tui

import (
	"github.com/rivo/tview"
	"github.com/rs/zerolog/log"
)

func (t *TUIApp) showLoadDialog() {
	modal := tview.NewModal().
		SetText("Load Game:\n\n1. GCG file: /path/to/file.gcg\n2. CGP string: cgp <position>\n3. Cross-tables: xt <game_id>\n4. Woogles: woogles <game_id>\n5. Create new game: new").
		AddButtons([]string{"File", "CGP", "Cross-tables", "Woogles", "New Game", "Cancel"}).
		SetDoneFunc(func(buttonIndex int, buttonLabel string) {
			log.Debug().Int("buttonIndex", buttonIndex).Str("buttonLabel", buttonLabel).Msg("TUI: Load dialog button selected")
			t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
			switch buttonIndex {
			case 0: // File
				t.showFileDialog()
			case 1: // CGP
				t.showCGPDialog()
			case 2: // Cross-tables
				t.showXTDialog()
			case 3: // Woogles
				log.Debug().Msg("TUI: Showing Woogles dialog")
				t.showWooglesDialog()
			case 4: // New Game
				t.createNewGame()
			case 5: // Cancel
				return
			}
		})

	t.app.SetRoot(modal, false).SetFocus(modal)
}

func (t *TUIApp) showFileDialog() {
	form := tview.NewForm()
	form.AddInputField("File path", "", 50, nil, nil)
	form.AddButton("Load", func() {
		path := form.GetFormItem(0).(*tview.InputField).GetText()
		if path != "" {
			err := t.shell.LoadGCGFile(path)
			if err != nil {
				t.updateStatus("Error loading file: " + err.Error())
			} else {
				t.updateStatus("Loaded file: " + path)
				t.refresh()
			}
		}
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})
	form.AddButton("Cancel", func() {
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})

	form.SetBorder(true).SetTitle("Load GCG File").SetTitleAlign(tview.AlignLeft)
	t.app.SetRoot(form, true).SetFocus(form)
}

func (t *TUIApp) showCGPDialog() {
	form := tview.NewForm()
	form.AddInputField("CGP string", "", 80, nil, nil)
	form.AddButton("Load", func() {
		cgp := form.GetFormItem(0).(*tview.InputField).GetText()
		if cgp != "" {
			err := t.shell.LoadCGPString(cgp)
			if err != nil {
				t.updateStatus("Error loading CGP: " + err.Error())
			} else {
				t.updateStatus("Loaded CGP position")
				t.refresh()
			}
		}
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})
	form.AddButton("Cancel", func() {
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})

	form.SetBorder(true).SetTitle("Load CGP Position").SetTitleAlign(tview.AlignLeft)
	t.app.SetRoot(form, true).SetFocus(form)
}

func (t *TUIApp) showXTDialog() {
	form := tview.NewForm()
	form.AddInputField("Cross-tables Game ID", "", 20, nil, nil)
	form.AddButton("Load", func() {
		gameID := form.GetFormItem(0).(*tview.InputField).GetText()
		if gameID != "" {
			err := t.shell.LoadXTGame(gameID)
			if err != nil {
				t.updateStatus("Error loading XT game: " + err.Error())
			} else {
				t.updateStatus("Loaded Cross-tables game: " + gameID)
				t.refresh()
			}
		}
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})
	form.AddButton("Cancel", func() {
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})

	form.SetBorder(true).SetTitle("Load Cross-tables Game").SetTitleAlign(tview.AlignLeft)
	t.app.SetRoot(form, true).SetFocus(form)
}

func (t *TUIApp) showWooglesDialog() {
	form := tview.NewForm()
	form.AddInputField("Woogles Game ID", "", 20, nil, nil)
	form.AddButton("Load", func() {
		log.Debug().Msg("TUI: Load button clicked in Woogles dialog")
		gameID := form.GetFormItem(0).(*tview.InputField).GetText()
		log.Debug().Str("gameID", gameID).Msg("TUI: Retrieved game ID from input field")
		if gameID != "" {
			log.Debug().Msg("TUI: Game ID is not empty, proceeding with load")
			t.updateStatus("Loading Woogles game " + gameID + "...")
			// Do the loading in a goroutine to avoid blocking the UI
			go func() {
				log.Debug().Str("gameID", gameID).Msg("TUI: Starting goroutine for Woogles load")
				err := t.shell.LoadWooglesGame(gameID)
				log.Debug().Err(err).Msg("TUI: LoadWooglesGame returned")
				// Use QueueUpdateDraw to safely update UI from goroutine
				log.Debug().Msg("TUI: Queueing UI update")
				t.app.QueueUpdateDraw(func() {
					log.Debug().Msg("TUI: Inside QueueUpdateDraw callback")
					if err != nil {
						log.Debug().Err(err).Msg("TUI: Updating status with error")
						t.updateStatus("Error loading Woogles game: " + err.Error())
					} else {
						log.Debug().Msg("TUI: Updating status with success and refreshing")
						t.updateStatus("Loaded Woogles game: " + gameID)
						t.refresh()
						log.Debug().Msg("TUI: Refresh completed")
					}
					log.Debug().Msg("TUI: QueueUpdateDraw callback completed")
				})
			}()
		}
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})
	form.AddButton("Cancel", func() {
		t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
	})

	form.SetBorder(true).SetTitle("Load Woogles Game").SetTitleAlign(tview.AlignLeft)
	t.app.SetRoot(form, true).SetFocus(form)
}

func (t *TUIApp) createNewGame() {
	err := t.shell.NewGame()
	if err != nil {
		t.updateStatus("Error creating new game: " + err.Error())
	} else {
		t.updateStatus("Created new game")
		t.refresh()
	}
}