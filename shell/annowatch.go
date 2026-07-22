package shell

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gameanalysis"
	"github.com/domino14/macondo/turnplayer"
	wmppkg "github.com/domino14/macondo/wmp"
	"github.com/domino14/macondo/worker"
)

// AnnowatchPollInterval is how often annowatch polls the Woogles API for the
// current position of the game it is watching.
const AnnowatchPollInterval = 10 * time.Second

// annowatch handles the annowatch command. `annowatch <gameID>` starts
// polling a live (typically annotated) Woogles game every ~10s and, whenever
// the position or on-turn rack changes, analyzes the current position with
// whatever rack is currently assigned and prints the result. `annowatch stop`
// stops it — immediately if idle, or after the in-flight analysis completes
// if one is running (a running analysis is never cut off).
func (sc *ShellController) annowatch(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) > 0 && cmd.args[0] == "stop" {
		return sc.stopAnnowatch()
	}
	if len(cmd.args) == 0 {
		return nil, errors.New("usage: annowatch <gameID>  or  annowatch stop")
	}
	return sc.startAnnowatch(cmd.args[0])
}

// startAnnowatch begins watching the given game ID in the background.
func (sc *ShellController) startAnnowatch(gameID string) (*Response, error) {
	sc.annoMu.Lock()
	alreadyWatching := sc.annoMode
	sc.annoMu.Unlock()
	if alreadyWatching {
		return msg(fmt.Sprintf("Already watching game %s. Use 'annowatch stop' to exit.", sc.annoGameID)), nil
	}

	// Require unload first (no active interactive game), mirroring volunteer mode.
	if sc.game != nil {
		return msg("Please unload the current game first with 'unload' before using annowatch."), nil
	}

	sc.annoMu.Lock()
	sc.annoMode = true
	sc.annoStop = false
	sc.annoBusy = false
	sc.annoLastCGP = ""
	sc.annoPendingCGP = ""
	sc.annoMu.Unlock()
	sc.annoGameID = gameID

	sc.annoCtx, sc.annoCancel = context.WithCancel(context.Background())

	go sc.annowatchLoop()

	return msg(fmt.Sprintf(
		"Watching game %s. Polling every %s; analyzing the position whenever it changes.\n"+
			"Use 'annowatch stop' to exit.", gameID, AnnowatchPollInterval)), nil
}

// stopAnnowatch stops annowatch, immediately if idle or after the current
// analysis if one is in progress. The actual "Annowatch stopped." announcement
// always comes from cleanupAnnowatch (triggered here for the idle case via
// annoCancel, or by the analysis worker once it notices annoStop) — that
// keeps a single source of truth instead of printing it here too and racing
// with (or duplicating) cleanupAnnowatch's own message.
func (sc *ShellController) stopAnnowatch() (*Response, error) {
	sc.annoMu.Lock()

	if !sc.annoMode {
		sc.annoMu.Unlock()
		return msg("Not watching any game."), nil
	}
	if sc.annoBusy {
		sc.annoStop = true
		sc.annoMu.Unlock()
		return msg("Stopping annowatch after the current analysis completes..."), nil
	}
	sc.annoMu.Unlock()

	sc.annoCancel()
	return msg("Stopping annowatch..."), nil
}

// annowatchLoop polls the game's current position on a ticker. Whenever the
// position changes it hands the new CGP off to the analysis worker. Polling
// continues independently of the (possibly slow) analysis worker; a change
// observed while an analysis is running is simply remembered as "pending" —
// the running analysis is never cancelled, and the worker picks up the
// latest pending position as soon as it finishes.
func (sc *ShellController) annowatchLoop() {
	wooglesURL := sc.config.GetString(config.ConfigWooglesURL)
	// GetCGP is unauthenticated for annotated games, same as the existing
	// GetGCG call used by volunteer mode, so no API key is needed here.
	client := worker.NewWooglesClient(wooglesURL, "", sc.config, sc.macondoVersion)

	gameID := sc.annoGameID
	ticker := time.NewTicker(AnnowatchPollInterval)
	defer ticker.Stop()

	log.Info().Str("game-id", gameID).Msg("annowatch loop started")

	poll := func() {
		cgpStr, err := client.FetchCGP(sc.annoCtx, gameID)
		if err != nil {
			log.Warn().Err(err).Str("game-id", gameID).Msg("annowatch: failed to fetch position")
			sc.showError(fmt.Errorf("annowatch: failed to fetch position: %w", err))
			return
		}
		if cgpStr == "" {
			return
		}

		sc.annoMu.Lock()
		changed := cgpStr != sc.annoLastCGP && cgpStr != sc.annoPendingCGP
		if changed {
			sc.annoPendingCGP = cgpStr
		}
		shouldLaunch := changed && !sc.annoBusy
		if shouldLaunch {
			sc.annoBusy = true
		}
		sc.annoMu.Unlock()

		if shouldLaunch {
			go sc.runAnnoAnalyses()
		}
	}

	poll()

	for {
		select {
		case <-sc.annoCtx.Done():
			log.Info().Str("game-id", gameID).Msg("annowatch loop cancelled")
			sc.cleanupAnnowatch()
			return
		case <-ticker.C:
			poll()
		}
	}
}

// runAnnoAnalyses is the sequential analysis worker. Only one instance ever
// runs at a time (annoBusy gates launching a second one). It claims the
// newest pending CGP, analyzes it, and — if a newer position arrived while it
// was working — loops to analyze that one too, until nothing is left pending
// or a stop has been requested.
func (sc *ShellController) runAnnoAnalyses() {
	for {
		sc.annoMu.Lock()
		cgpStr := sc.annoPendingCGP
		sc.annoLastCGP = cgpStr
		sc.annoPendingCGP = ""
		sc.annoMu.Unlock()

		if err := sc.runOneAnnoAnalysis(cgpStr); err != nil {
			log.Warn().Err(err).Msg("annowatch: analysis failed")
			sc.showError(fmt.Errorf("annowatch: analysis failed: %w", err))
		}

		sc.annoMu.Lock()
		hasNewerPending := sc.annoPendingCGP != "" && sc.annoPendingCGP != sc.annoLastCGP
		stopRequested := sc.annoStop
		keepGoing := hasNewerPending && !stopRequested
		if !keepGoing {
			sc.annoBusy = false
		}
		sc.annoMu.Unlock()

		if !keepGoing {
			if stopRequested {
				sc.cleanupAnnowatch()
			}
			return
		}
	}
}

// runOneAnnoAnalysis loads the given CGP position (without touching the
// interactive sc.game) and runs a full position analysis on it, printing the
// result with the rack clearly labeled at the top.
func (sc *ShellController) runOneAnnoAnalysis(cgpStr string) error {
	// Ensure the KWG/WMP are present, same prelude as loadCGP.
	lex := lexiconFromCGP(cgpStr)
	if lex == "" {
		lex = sc.config.GetString(config.ConfigDefaultLexicon)
	}
	if err := turnplayer.EnsureKWG(lex, sc.config.WGLConfig()); err != nil {
		return fmt.Errorf("could not ensure lexicon %s: %w", lex, err)
	}
	if _, wmpErr := wmppkg.EnsureWMP(sc.config.WGLConfig(), lex); wmpErr != nil {
		log.Info().Err(wmpErr).Str("lexicon", lex).
			Msg("WMP not available for this lexicon; sim will use the KWG algorithm")
	}

	parsed, err := cgp.ParseCGP(sc.config, cgpStr)
	if err != nil {
		return fmt.Errorf("failed to parse position: %w", err)
	}
	g := parsed.Game
	// Match cmd/lambda's bestbot CGP-loading sequence exactly (that path
	// has served thousands of games): SetBackupMode + SetStateStackLength
	// before RecalculateBoard, and no rack normalization — bestbot uses
	// both CGP racks exactly as given, with no SetRackFor/redraw step.
	// ParseCGP (NewFromSnapshot) only places letters on the board grid —
	// it never computes cross-sets or anchors, unlike the incremental
	// game.NewFromHistory+PlayToTurn path every other analyzer entry point
	// uses (which maintains them move-by-move via crossSetGen.UpdateForMove).
	// Without RecalculateBoard, the board's cross-sets/anchors are left at
	// whatever an empty board defaults to, not this layout's real
	// constraints — the interactive `load cgp` command and bestbot's lambda
	// both call this immediately after ParseCGP for the same reason.
	g.SetBackupMode(game.InteractiveGameplayMode)
	g.SetStateStackLength(1)
	g.RecalculateBoard()

	playerIdx := g.PlayerOnTurn()
	rack := g.RackFor(playerIdx).String()

	analyzer := gameanalysis.New(sc.config, gameanalysis.DefaultAnalysisConfig(), sc.macondoVersion)
	analysis, err := analyzer.AnalyzePosition(sc.annoCtx, g, playerIdx, g.History().Players)
	if err != nil {
		return fmt.Errorf("failed to analyze position (rack %q): %w", rack, err)
	}

	sc.showMessage(formatLivePositionAnalysis(sc.annoGameID, rack, analysis))
	return nil
}

// cleanupAnnowatch resets annowatch state. It is safe to call from either the
// polling loop or the analysis worker (whichever notices the stop first) —
// the annoMode guard ensures the reset and stop message only happen once.
func (sc *ShellController) cleanupAnnowatch() {
	sc.annoMu.Lock()
	if !sc.annoMode {
		sc.annoMu.Unlock()
		return
	}
	sc.annoMode = false
	sc.annoStop = false
	sc.annoBusy = false
	cancel := sc.annoCancel
	sc.annoMu.Unlock()

	if cancel != nil {
		cancel()
	}
	sc.showMessage("Annowatch stopped.")
	log.Info().Msg("annowatch stopped")
}

// formatLivePositionAnalysis formats a live to-move position analysis, with
// the rack clearly labeled at the top so it's obvious which rack the
// analysis below corresponds to.
func formatLivePositionAnalysis(gameID, rack string, analysis *gameanalysis.TurnAnalysis) string {
	var sb strings.Builder

	sb.WriteString(strings.Repeat("=", 60))
	sb.WriteString(fmt.Sprintf("\nannowatch: %s\n", gameID))
	sb.WriteString(strings.Repeat("=", 60))
	sb.WriteString("\n\n")

	sb.WriteString(fmt.Sprintf("Rack: %s\n", rack))
	sb.WriteString(fmt.Sprintf("Player: %s\n", analysis.PlayerName))
	sb.WriteString(fmt.Sprintf("Phase: %s (%d tiles in bag)\n\n", analysis.Phase, analysis.TilesInBag))

	if analysis.OptimalMove != nil {
		sb.WriteString(fmt.Sprintf("Best Play: %s\n", analysis.OptimalMove.ShortDescription()))
		sb.WriteString(fmt.Sprintf("  Score: %d points\n", analysis.OptimalMove.Score()))
		if analysis.OptimalIsBingo {
			sb.WriteString("  BINGO!\n")
		}
		sb.WriteString("\n")
	}

	switch {
	case len(analysis.TopSimPlays) > 0:
		sb.WriteString("Top plays (simulation):\n")
		for i, p := range analysis.TopSimPlays {
			sb.WriteString(fmt.Sprintf("  %d. %-20s %3d pts  win%% %5.1f  eq %6.1f\n",
				i+1, p.MoveDescription, p.Score, p.WinProb*100, p.Equity))
		}
	case len(analysis.TopPEGPlays) > 0:
		sb.WriteString("Top plays (pre-endgame):\n")
		for i, p := range analysis.TopPEGPlays {
			sb.WriteString(fmt.Sprintf("  %d. %-20s %3d pts  win%% %5.1f\n",
				i+1, p.MoveDescription, p.Score, p.WinProb*100))
		}
	case analysis.PrincipalVariation != nil:
		sb.WriteString("Endgame principal variation:\n")
		for _, m := range analysis.PrincipalVariation.Moves {
			sb.WriteString(fmt.Sprintf("  %d. %s (%d pts)\n", m.MoveNumber, m.MoveDescription, m.Score))
		}
		sb.WriteString(fmt.Sprintf("  Final spread: %+d\n", analysis.PrincipalVariation.FinalSpread))
	}

	return sb.String()
}
