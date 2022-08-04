package bot

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	airunner "github.com/domino14/macondo/ai/runner"
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

// Wolges ordering:
var GermanTiles = strings.Split("AÄBCDEFGHIJKLMNOÖPQRSTUÜVWXYZ", "")
var GermanBlankTiles = strings.Split("aäbcdefghijklmnoöpqrstuüvwxyz", "")
var NorwegianTiles = strings.Split("ABCDEFGHIJKLMNOPQRSTUVWXYÜZÆÄØÖÅ", "")
var NorwegianBlankTiles = strings.Split("abcdefghijklmnopqrstuvwxyüzæäøöå", "")

type WolgesAnalyzePayload struct {
	Rack    []int   `json:"rack"`
	Board   [][]int `json:"board"`
	Lexicon string  `json:"lexicon"`
	Leave   string  `json:"leave"`
	Rules   string  `json:"rules"`
	Count   int     `json:"count"`
}

type WolgesAnalyzeResponse []struct {
	Equity float64 `json:"equity"`
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
		if string(c) == GermanTiles[i] {
			return i + 1
		}
	}
	for i := 0; i < len(GermanBlankTiles); i++ {
		if string(c) == GermanBlankTiles[i] {
			return -(i + 1)
		}
	}
	return 0
}

func norwegianLabelToNum(c rune) int {
	for i := 0; i < len(NorwegianTiles); i++ {
		if string(c) == NorwegianTiles[i] {
			return i + 1
		}
	}
	for i := 0; i < len(NorwegianBlankTiles); i++ {
		if string(c) == NorwegianBlankTiles[i] {
			return -(i + 1)
		}
	}
	return 0
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

func wolgesAnalyze(cfg *config.Config, g *airunner.AIGameRunner) ([]*move.Move, error) {
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
	wap.Leave = leave

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
			letter := g.Board().GetSquare(i, j).Letter().UserVisible(g.Alphabet())
			wap.Board[i][j] = labelToNum(letter)
		}
	}

	for _, c := range ourRack.String() {
		wap.Rack = append(wap.Rack, labelToNum(c))
	}

	wap.Count = 30

	bts, err := json.Marshal(wap)
	if err != nil {
		return nil, err
	}
	log.Debug().Str("payload", string(bts)).Msg("sending-to-wolges")
	// Now let's send a request.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequest("POST", cfg.WolgesAwsmURL+"/analyze", bytes.NewReader(bts))
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	readbts, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	r := &WolgesAnalyzeResponse{}
	err = json.Unmarshal(readbts, r)
	if err != nil {
		return nil, err
	}

	log.Info().Interface("r", r).Msg("from-wolges")
	return nil, errors.New("not-implemented")
}
