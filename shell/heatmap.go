package shell

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
)

type Heat struct {
	numHits       int
	fractionOfMax float64
}

type HeatMap struct {
	board    *board.GameBoard
	squares  [][]Heat
	alphabet *tilemapping.TileMapping
}

func (sc *ShellController) PlaceMove(g *bot.BotTurnPlayer, play string) error {
	normalizedPlay := normalize(play)
	m, err := g.ParseMove(
		g.PlayerOnTurn(), sc.options.lowercaseMoves, strings.Fields(normalizedPlay))

	if err != nil {
		return err
	}
	g.SetBackupMode(game.SimulationMode)
	log.Debug().Str("move", m.ShortDescription()).Msg("Playing move")
	err = g.PlayMove(m, false, 0)
	return err
}

func (sc *ShellController) UnplaceMove(g *bot.BotTurnPlayer) {
	log.Debug().Msg("Undoing last move")
	g.UnplayLastMove()
	g.SetBackupMode(game.NoBackup)
}

func (sc *ShellController) CalculateHeatmap(s *montecarlo.Simmer, g *bot.BotTurnPlayer, play string,
	ply int) (*HeatMap, error) {
	iters, err := s.ReadHeatmap()
	if err != nil {
		return nil, err
	}
	log.Debug().Msgf("Read %d log lines", len(iters))
	h := &HeatMap{
		squares:  make([][]Heat, g.Board().Dim()),
		board:    g.Board(),
		alphabet: g.Alphabet(),
	}
	for idx := range h.squares {
		h.squares[idx] = make([]Heat, h.board.Dim())
	}

	maxNumHits := 0
	log.Debug().Msg("parsing-iterations")
	normalizedPlay := normalize(play)

	for i := range iters {
		for j := range iters[i].Plays {
			if normalizedPlay != normalize(iters[i].Plays[j].Play) {
				continue
			}
			if len(iters[i].Plays[j].Plies) <= ply {
				continue
			}
			analyzedPlay := normalize(iters[i].Plays[j].Plies[ply].Play)

			if strings.HasPrefix(analyzedPlay, "exchange ") ||
				analyzedPlay == "pass" || analyzedPlay == "UNHANDLED" {
				continue
			}

			// this is a tile-play move.
			playFields := strings.Fields(analyzedPlay)
			if len(playFields) != 2 {
				return nil, errors.New("unexpected play " + analyzedPlay)
			}
			coords := strings.ToUpper(playFields[0])
			row, col, vertical := move.FromBoardGameCoords(coords)
			mw, err := tilemapping.ToMachineWord(playFields[1], g.Alphabet())
			if err != nil {
				return nil, err
			}
			ri, ci := 1, 0
			if !vertical {
				ri, ci = 0, 1
			}

			for idx := range mw {
				if mw[idx] == 0 {
					continue // playthrough doesn't create heat map
				}
				newRow := row + (ri * idx)
				newCol := col + (ci * idx)
				h.squares[newRow][newCol].numHits++
				if h.squares[newRow][newCol].numHits > maxNumHits {
					maxNumHits = h.squares[newRow][newCol].numHits
				}
			}

		}
	}

	for ri := range h.squares {
		for ci := range h.squares[ri] {
			h.squares[ri][ci].fractionOfMax = float64(h.squares[ri][ci].numHits) / float64(maxNumHits)
		}
	}

	return h, nil
}

var exchRe = regexp.MustCompile(`\((exch|exchange) ([^)]+)\)`)
var throughPlayRe = regexp.MustCompile(`\(([^)]+)\)`)

func normalize(p string) string {
	// Trim leading and trailing whitespace
	trimmed := strings.TrimSpace(p)

	if trimmed == "(Pass)" {
		return "pass"
	}

	// Check for "(exch FOO)" or "(exchange FOO)" and extract the content
	if strings.HasPrefix(trimmed, "(exch ") || strings.HasPrefix(trimmed, "(exchange ") {
		// Define a regex to extract "exchange FOO" from "(exch FOO)" or "(exchange FOO)"
		matches := exchRe.FindStringSubmatch(trimmed)
		if len(matches) == 3 {
			return "exchange " + matches[2]
		}
	}

	// Define a regular expression to match groups in parentheses

	// Replace each match with as many dots as there are characters inside the parentheses
	normalized := throughPlayRe.ReplaceAllStringFunc(trimmed, func(match string) string {
		// Extract the content inside parentheses using capturing group
		content := throughPlayRe.FindStringSubmatch(match)[1]
		return strings.Repeat(".", len(content))
	})

	return normalized
}

// getHeatColor returns an ANSI escape sequence for a given heat level.
func getHeatColor(fraction float64) string {
	// Map the fraction (0 to 1) to grayscale colors (232 to 255 in ANSI 256-color palette)
	// 232 is darkest (black), 255 is lightest (white)
	start := 232
	end := 255
	colorCode := int(float64(start) + fraction*float64(end-start))
	return fmt.Sprintf("\033[48;5;%dm", colorCode) // Background color
}

// display renders the heatmap to the terminal.
func (h HeatMap) display() {
	fmt.Println()
	reset := "\033[0m" // Reset color
	for ri, row := range h.squares {
		for ci, heat := range row {
			color := getHeatColor(heat.fractionOfMax)
			letter := h.board.SQDisplayStr(ri, ci, h.alphabet, false)
			fmt.Printf("%s%s%s", color, letter, reset) // Colored block
		}
		fmt.Println() // Newline after each row
	}
}
