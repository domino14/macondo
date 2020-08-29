package game

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/domino14/macondo/alphabet"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/rs/zerolog/log"
)

func splitSubN(s string, n int) []string {
	sub := ""
	subs := []string{}

	runes := bytes.Runes([]byte(s))
	l := len(runes)
	for i, r := range runes {
		sub = sub + string(r)
		if (i+1)%n == 0 {
			subs = append(subs, sub)
			sub = ""
		} else if (i + 1) == l {
			subs = append(subs, sub)
		}
	}

	return subs
}

func addText(lines []string, row int, hpad int, text string) {
	maxTextSize := 42
	sp := splitSubN(text, maxTextSize)

	for _, chunk := range sp {
		str := lines[row] + strings.Repeat(" ", hpad) + chunk
		lines[row] = str
		row++
	}
}

// ToDisplayText turns the current state of the game into a displayable
// string.
func (g *Game) ToDisplayText() string {
	bt := g.Board().ToDisplayText(g.alph)
	// We need to insert rack, player, bag strings into the above string.
	bts := strings.Split(bt, "\n")
	hpadding := 3
	vpadding := 1
	bagColCount := 20

	notfirst := otherPlayer(g.wentfirst)

	log.Debug().Int("onturn", g.onturn).
		Int("wentfirst", g.wentfirst).Msg("todisplaytext")

	addText(bts, vpadding, hpadding,
		g.players[g.wentfirst].stateString(g.playing == pb.PlayState_PLAYING && g.onturn == g.wentfirst))
	addText(bts, vpadding+1, hpadding,
		g.players[notfirst].stateString(g.playing == pb.PlayState_PLAYING && g.onturn == notfirst))

	// Peek into the bag, and append the opponent's tiles:
	inbag := g.bag.Peek()
	opprack := g.players[otherPlayer(g.onturn)].rack.TilesOn()
	bagAndUnseen := append(inbag, opprack...)
	log.Debug().Str("inbag", alphabet.MachineWord(inbag).UserVisible(g.alph)).Msg("")
	log.Debug().Str("opprack", alphabet.MachineWord(opprack).UserVisible(g.alph)).Msg("")

	addText(bts, vpadding+3, hpadding, fmt.Sprintf("Bag + unseen: (%d)", len(bagAndUnseen)))

	vpadding = 6
	sort.Slice(bagAndUnseen, func(i, j int) bool {
		return bagAndUnseen[i] < bagAndUnseen[j]
	})

	bagDisp := []string{}
	cCtr := 0
	bagStr := ""
	for i := 0; i < len(bagAndUnseen); i++ {
		bagStr += string(bagAndUnseen[i].UserVisible(g.alph)) + " "
		cCtr++
		if cCtr == bagColCount {
			bagDisp = append(bagDisp, bagStr)
			bagStr = ""
			cCtr = 0
		}
	}
	if bagStr != "" {
		bagDisp = append(bagDisp, bagStr)
	}

	for p := vpadding; p < vpadding+len(bagDisp); p++ {
		addText(bts, p, hpadding, bagDisp[p-vpadding])
	}

	addText(bts, 12, hpadding, fmt.Sprintf("Turn %d:", g.turnnum))

	vpadding = 13

	for i, evt := range g.history.Events {
		log.Debug().Msgf("Event %d: %v", i, evt)
	}

	if g.turnnum-1 >= 0 {
		addText(bts, vpadding, hpadding,
			summary(g.history.Events[g.turnnum-1]))
	}

	vpadding = 17

	if g.playing == pb.PlayState_GAME_OVER && g.turnnum == len(g.history.Events) {
		addText(bts, vpadding, hpadding, "Game is over.")
	}

	return strings.Join(bts, "\n")

}
