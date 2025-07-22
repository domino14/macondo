package shell

import (
	"fmt"
	"io"

	"github.com/rs/zerolog/log"
	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/move"
)

// TUI accessor methods for ShellController
// These methods provide a cleaner interface for TUI integration

func (sc *ShellController) GetGame() *bot.BotTurnPlayer {
	return sc.game
}

func (sc *ShellController) GetCurTurnNum() int {
	return sc.curTurnNum
}

func (sc *ShellController) GetCurPlayList() []*move.Move {
	return sc.curPlayList
}

func (sc *ShellController) GetGenDisplayMoveList() string {
	return sc.genDisplayMoveList()
}

// TUI command wrappers
func (sc *ShellController) NextTurn() error {
	cmd := &shellcmd{args: nil, options: CmdOptions{}}
	_, err := sc.next(cmd)
	return err
}

func (sc *ShellController) PrevTurn() error {
	cmd := &shellcmd{args: nil, options: CmdOptions{}}
	_, err := sc.prev(cmd)
	return err
}

func (sc *ShellController) GenerateMoves(numMoves int) error {
	args := []string{fmt.Sprintf("%d", numMoves)}
	cmd := &shellcmd{args: args, options: CmdOptions{}}
	_, err := sc.generate(cmd)
	return err
}

func (sc *ShellController) LoadGCGFile(filepath string) error {
	args := []string{filepath}
	cmd := &shellcmd{args: args, options: CmdOptions{}}
	_, err := sc.load(cmd)
	return err
}

func (sc *ShellController) LoadCGPString(cgpString string) error {
	args := []string{"cgp", cgpString}
	cmd := &shellcmd{args: args, options: CmdOptions{}}
	_, err := sc.load(cmd)
	return err
}

func (sc *ShellController) LoadXTGame(gameID string) error {
	args := []string{"xt", gameID}
	cmd := &shellcmd{args: args, options: CmdOptions{}}
	_, err := sc.load(cmd)
	return err
}

func (sc *ShellController) LoadWooglesGame(gameID string) error {
	// Add debug logging for TUI
	if sc.writer == io.Discard {
		log.Info().Str("gameID", gameID).Msg("TUI: Starting Woogles game load")
	}
	
	args := []string{"woogles", gameID}
	cmd := &shellcmd{args: args, options: CmdOptions{}}
	_, err := sc.load(cmd)
	
	if sc.writer == io.Discard {
		if err != nil {
			log.Error().Err(err).Msg("TUI: Woogles game load failed")
		} else {
			log.Info().Msg("TUI: Woogles game load completed successfully")
		}
	}
	
	return err
}

func (sc *ShellController) NewGame() error {
	cmd := &shellcmd{args: nil, options: CmdOptions{}}
	_, err := sc.newGame(cmd)
	return err
}