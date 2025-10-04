package game

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/domino14/word-golib/tilemapping"

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
		if row >= len(lines) {
			l := len(lines)
			for i := l; i <= row; i++ {
				lines = append(lines, "")
			}
		}
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

	log.Debug().Int("onturn", g.onturn).Msg("todisplaytext")
	for pi := 0; pi < 2; pi++ {
		addText(bts, vpadding+pi, hpadding,
			g.players[pi].stateString(g.playing == pb.PlayState_PLAYING && g.onturn == pi))
	}

	// Peek into the bag, and append the opponent's tiles:
	inbag := g.bag.Peek()
	opprack := g.players[otherPlayer(g.onturn)].rack.TilesOn()
	bagAndUnseen := append(inbag, opprack...)
	log.Debug().Str("inbag", tilemapping.MachineWord(inbag).UserVisible(g.alph)).Msg("")
	log.Debug().Str("opprack", tilemapping.MachineWord(opprack).UserVisible(g.alph)).Msg("")

	addText(bts, vpadding+3, hpadding, fmt.Sprintf("Bag + unseen: (%d)", len(bagAndUnseen)))

	vpadding = 6
	sort.Slice(bagAndUnseen, func(i, j int) bool {
		return bagAndUnseen[i] < bagAndUnseen[j]
	})

	bagDisp := []string{}
	cCtr := 0
	bagStr := ""
	for i := 0; i < len(bagAndUnseen); i++ {
		bagStr += string(bagAndUnseen[i].UserVisible(g.alph, false)) + " "
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
	for i, evt := range g.events {
		log.Debug().Msgf("Event %d: %v", i, evt)
	}

	if g.turnnum-1 >= 0 && len(g.events) > g.turnnum-1 {
		// Create player info array for summary
		playerInfos := make([]*pb.PlayerInfo, len(g.players))
		for i, p := range g.players {
			playerInfos[i] = &pb.PlayerInfo{
				Nickname: p.Nickname,
				RealName: p.RealName,
			}
		}
		addText(bts, vpadding, hpadding,
			summary(playerInfos, g.events[g.turnnum-1]))
	}

	vpadding = 17

	if g.playing == pb.PlayState_GAME_OVER && g.turnnum == len(g.events) {
		addText(bts, vpadding, hpadding, "Game is over.")
	}
	if g.playing == pb.PlayState_WAITING_FOR_FINAL_PASS {
		addText(bts, vpadding, hpadding, "Waiting for final pass/challenge...")
	}
	vpadding = 19
	if g.turnnum-1 < len(g.events) && g.turnnum-1 >= 0 &&
		g.events[g.turnnum-1].Note != "" {
		// add it all the way at the bottom
		bts = append(bts, g.events[g.turnnum-1].Note)
	}
	return strings.Join(bts, "\n")

}
