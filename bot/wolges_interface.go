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

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/tilemapping"
	"github.com/rs/zerolog/log"
)

// This file is an interface for `wolges` and its relevant projects.
// See github.com/andy-k/wolges
//     github.com/andy-k/wolges-wasm
//     github.com/andy-k/wolges-awsm

// see analyzer.tsx in liwords repo to see where a lot of this conversion
// code comes from.

const WolgesTimeout = 5 * time.Second

// // Wolges ordering:
// var GermanTiles = []rune("AÄBCDEFGHIJKLMNOÖPQRSTUÜVWXYZ")
// var GermanBlankTiles = []rune("aäbcdefghijklmnoöpqrstuüvwxyz")
// var NorwegianTiles = []rune("ABCDEFGHIJKLMNOPQRSTUVWXYÜZÆÄØÖÅ")
// var NorwegianBlankTiles = []rune("abcdefghijklmnopqrstuvwxyüzæäøöå")

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

func wolgesAnalyze(cfg *config.Config, g *bot.BotTurnPlayer) ([]*move.Move, error) {
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

	letterDistribution := ""
	lowercasedLexicon := strings.ToLower(wap.Lexicon)
	switch {
	case strings.HasPrefix(lowercasedLexicon, "rd"):
		letterDistribution = "german"
	case strings.HasPrefix(lowercasedLexicon, "nsf"):
		letterDistribution = "norwegian"
	case strings.HasPrefix(lowercasedLexicon, "fra"):
		letterDistribution = "french"
	case strings.HasPrefix(lowercasedLexicon, "disc"):
		letterDistribution = "catalan"
	default:
		letterDistribution = "english"
	}

	// assume english and catalan have super
	hasSuper := letterDistribution == "english" || letterDistribution == "catalan"
	wap.Leave = wap.Lexicon
	switch g.Rules().Variant() {
	case "", game.VarClassic:
		wap.Rules = "CrosswordGame"
	case game.VarWordSmog:
		wap.Rules = "WordSmog"
		wap.Lexicon += ".WordSmog"
		// assume always same leaves as classic
	case game.VarClassicSuper:
		wap.Rules = "CrosswordGameSuper"
		// assume always same lexicon as classic
		if hasSuper {
			wap.Leave = "super-" + wap.Leave
		}
	case game.VarWordSmogSuper:
		wap.Rules = "WordSmogSuper"
		// assume always same lexicon as wordsmog
		wap.Lexicon += ".WordSmog"
		if hasSuper {
			// assume always same leaves as classic super
			wap.Leave = "super-" + wap.Leave
		}
	}
	if letterDistribution != "english" {
		wap.Rules += "/" + letterDistribution
	}

	tm := g.Bag().LetterDistribution().TileMapping()

	// populate board
	for i := 0; i < g.Board().Dim(); i++ {
		for j := 0; j < g.Board().Dim(); j++ {
			// wolges now uses our letter ordering, except blanks are encoded differently.
			letter := g.Board().GetLetter(i, j)
			code := int(letter)
			if letter.IsBlanked() {
				code = -int(letter.Unblank())
			}
			wap.Board[i][j] = code
		}
	}

	for _, c := range ourRack.TilesOn() {
		wap.Rack = append(wap.Rack, int(c))
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
	log.Debug().Msg("made HTTP post, getting response...")
	resp, err := http.DefaultClient.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	readbts, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	log.Debug().Str("body", string(readbts)).Msg("raw-from-wolges")
	var r []WolgesAnalyzeResponse
	err = json.Unmarshal(readbts, &r)
	if err != nil {
		return nil, err
	}

	log.Info().Interface("r", r).Msg("from-wolges")
	if len(r) < 1 {
		return nil, errors.New("unexpected-wolges-response-length")
	}

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
			var str strings.Builder
			for _, t := range best.Tiles {
				str.WriteString(tm.Letter(tilemapping.MachineLetter(t)))
			}
			exch, err := g.NewExchangeMove(g.PlayerOnTurn(), str.String())
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
		var str strings.Builder
		for _, t := range best.Word {
			if t == 0 {
				str.WriteString(string(tilemapping.ASCIIPlayedThrough))
			} else {
				if t < 0 {
					// re-encode blank.
					t = (-t) | tilemapping.BlankMask
				}
				letter := tm.Letter(tilemapping.MachineLetter(t))
				str.WriteString(letter)
			}
		}

		m, err := g.CreateAndScorePlacementMove(coords, str.String(), string(rack))
		if err != nil {
			return nil, err
		}
		return []*move.Move{m}, nil
	}

	return nil, errors.New("not handled: " + best.Action)
}
