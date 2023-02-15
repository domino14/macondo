package bot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog/log"
)

// This file is an interface for `wolges` and its relevant projects.
// See github.com/andy-k/wolges
//     github.com/andy-k/wolges-wasm
//     github.com/andy-k/wolges-awsm

// see analyzer.tsx in liwords repo to see where a lot of this conversion
// code comes from.

const WolgesTimeout = 5 * time.Second

// Wolges ordering:
var GermanTiles = []rune("AÄBCDEFGHIJKLMNOÖPQRSTUÜVWXYZ")
var GermanBlankTiles = []rune("aäbcdefghijklmnoöpqrstuüvwxyz")
var NorwegianTiles = []rune("ABCDEFGHIJKLMNOPQRSTUVWXYÜZÆÄØÖÅ")
var NorwegianBlankTiles = []rune("abcdefghijklmnopqrstuvwxyüzæäøöå")

type WolgesAnalyzePayload struct {
	Rack    []int   `json:"rack"`
	Board   [][]int `json:"board"`
	Lexicon string  `json:"lexicon"`
	Leave   string  `json:"leave"`
	Rules   string  `json:"rules"`
	Count   int     `json:"count"`
}

type WolgesAnalyzeResponse struct {
	Equity float32 `json:"equity"`
	Action string  `json:"action"`
	Tiles  []int   `json:"tiles"` // used for exchange. If empty it is a pass.
	Down   bool    `json:"down"`
	Lane   int     `json:"lane"`
	Idx    int     `json:"idx"`
	Word   []int   `json:"word"`
	Score  int     `json:"score"`
}

func englishLabelToNum(c rune) int {
	if c >= 'A' && c <= 'Z' {
		return int(c - 0x40)
	}
	if c >= 'a' && c <= 'z' {
		return -int(c - 0x60)
	}
	return 0
}

func germanLabelToNum(c rune) int {
	for i := 0; i < len(GermanTiles); i++ {
		if c == GermanTiles[i] {
			return i + 1
		}
	}
	for i := 0; i < len(GermanBlankTiles); i++ {
		if c == GermanBlankTiles[i] {
			return -(i + 1)
		}
	}
	return 0
}

func norwegianLabelToNum(c rune) int {
	for i := 0; i < len(NorwegianTiles); i++ {
		if c == NorwegianTiles[i] {
			return i + 1
		}
	}
	for i := 0; i < len(NorwegianBlankTiles); i++ {
		if c == NorwegianBlankTiles[i] {
			return -(i + 1)
		}
	}
	return 0
}

func englishNumToLabel(n int) rune {
	switch {
	case n > 0:
		return rune(0x40 + n)
	case n < 0:
		return rune(0x60 - n)
	case n == 0:
		return '?'
	}
	return '?'
}

func germanNumToLabel(n int) rune {
	switch {
	case n > 0:
		return GermanTiles[n-1]
	case n < 0:
		return GermanBlankTiles[-1-n]
	}
	return '?'
}

func norwegianNumToLabel(n int) rune {
	switch {
	case n > 0:
		return NorwegianTiles[n-1]
	case n < 0:
		return NorwegianBlankTiles[-1-n]
	}
	return '?'
}

func labelToNumFor(ld string) func(rune) int {
	switch ld {
	case "english":
		return englishLabelToNum
	case "german":
		return germanLabelToNum
	case "norwegian":
		return norwegianLabelToNum
	}
	return englishLabelToNum
}

func numToLabelFor(ld string) func(int) rune {
	switch ld {
	case "english":
		return englishNumToLabel
	case "german":
		return germanNumToLabel
	case "norwegian":
		return norwegianNumToLabel
	}
	return englishNumToLabel
}

func wolgesAnalyze(cfg *config.Config, g *aiturnplayer.BotTurnPlayer) ([]*move.Move, error) {
	// cfg.WolgesAwsmURL
	// convert game to the needed data structure
	dim := g.Board().Dim()

	// there's some boards in this house
	wap := WolgesAnalyzePayload{}
	// assume square board.
	wap.Board = make([][]int, dim)
	for i := 0; i < dim; i++ {
		wap.Board[i] = make([]int, dim)
	}

	ourRack := g.RackFor(g.PlayerOnTurn())

	wap.Lexicon = g.LexiconName()

	leave := ""
	lowercasedLexicon := strings.ToLower(wap.Lexicon)
	switch {
	case strings.HasPrefix(lowercasedLexicon, "rd"):
		leave = "german"
	case strings.HasPrefix(lowercasedLexicon, "nsf"):
		leave = "norwegian"
	case strings.HasPrefix(lowercasedLexicon, "fra"):
		leave = "french"
	default:
		leave = "english"
	}
	if lowercasedLexicon == "csw21" {
		wap.Leave = "CSW21"
	} else {
		wap.Leave = leave
	}

	switch g.Rules().Variant() {
	case "", game.VarClassic:
		wap.Rules = "CrosswordGame"
	case game.VarWordSmog:
		wap.Rules = "WordSmog"
		wap.Lexicon += ".WordSmog"
	case game.VarClassicSuper:
		wap.Rules = "CrosswordGameSuper"
	case game.VarWordSmogSuper:
		wap.Rules = "WordSmogSuper"
		wap.Lexicon += ".WordSmog"
	}
	if leave != "english" {
		wap.Rules += "/" + leave
	}

	// populate board
	labelToNum := labelToNumFor(leave)
	for i := 0; i < g.Board().Dim(); i++ {
		for j := 0; j < g.Board().Dim(); j++ {
			// since wolges doesn't use our same letter ordering, let's just do
			// the conversion this way
			letter := g.Board().GetLetter(i, j).UserVisible(g.Alphabet())
			wap.Board[i][j] = labelToNum(letter)
		}
	}

	for _, c := range ourRack.String() {
		wap.Rack = append(wap.Rack, labelToNum(c))
	}

	wap.Count = 1

	bts, err := json.Marshal(wap)
	if err != nil {
		return nil, err
	}
	log.Debug().Str("payload", string(bts)).Msg("sending-to-wolges")
	// Now let's send a request.
	ctx, cancel := context.WithTimeout(context.Background(), WolgesTimeout)
	defer cancel()
	req, err := http.NewRequest("POST", cfg.WolgesAwsmURL+"/analyze", bytes.NewReader(bts))
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	readbts, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var r []WolgesAnalyzeResponse
	err = json.Unmarshal(readbts, &r)
	if err != nil {
		return nil, err
	}

	log.Info().Interface("r", r).Msg("from-wolges")
	if len(r) < 1 {
		return nil, errors.New("unexpected-wolges-response-length")
	}

	numToLabel := numToLabelFor(leave)
	best := r[0]
	switch best.Action {
	case "exchange":

		if len(best.Tiles) == 0 {
			// actually a pass
			p, err := g.NewPassMove(g.PlayerOnTurn())
			if err != nil {
				return nil, err
			}
			return []*move.Move{p}, nil

		} else {
			runes := make([]rune, 0, len(best.Tiles))
			for _, t := range best.Tiles {
				runes = append(runes, numToLabel(t))
			}
			exch, err := g.NewExchangeMove(g.PlayerOnTurn(), string(runes))
			if err != nil {
				return nil, err
			}
			return []*move.Move{exch}, nil
		}

	case "play":
		vertical := best.Down
		var row, col int
		if vertical {
			row = best.Idx
			col = best.Lane
		} else {
			row = best.Lane
			col = best.Idx
		}
		coords := move.ToBoardGameCoords(row, col, vertical)
		rack := g.RackLettersFor(g.PlayerOnTurn())
		runes := make([]rune, 0, len(best.Word))
		for _, t := range best.Word {
			if t == 0 {
				runes = append(runes, alphabet.ASCIIPlayedThrough)
			} else {
				runes = append(runes, numToLabel(t))
			}
		}

		m, err := g.CreateAndScorePlacementMove(coords, string(runes), string(rack))
		if err != nil {
			return nil, err
		}
		return []*move.Move{m}, nil
	}

	return nil, errors.New("not handled: " + best.Action)
}
