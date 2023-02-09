// Package player is an automatic player of Crossword Game, using various
// forms of AI.
package player

import (
	"sort"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/strategy"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

const (
	// Infinity is a shockingly large number.
	Infinity = 10000000.0
)

// AIPlayer describes an artificial player. It uses a Strategizer to decide
// on moves to make.
type AIPlayer interface {
	// AssignEquity will assign equities to the given moves.
	AssignEquity([]*move.Move, *board.GameBoard, *alphabet.Bag, *alphabet.Rack)
	// BestPlay picks the best play from the list of plays.
	BestPlay([]*move.Move) *move.Move
	// TopPlays picks the top N plays from the list of plays.
	TopPlays([]*move.Move, int) []*move.Move

	Strategizer() strategy.Strategizer

	GetBotType() pb.BotRequest_BotCode
}

// RawEquityPlayer plays by equity only and does no look-ahead / sim.
type RawEquityPlayer struct {
	strategy strategy.Strategizer
	botType  pb.BotRequest_BotCode
}

func NewRawEquityPlayer(s strategy.Strategizer, botType pb.BotRequest_BotCode) *RawEquityPlayer {
	return &RawEquityPlayer{
		strategy: s,
		botType:  botType,
	}
}

func (p *RawEquityPlayer) Strategizer() strategy.Strategizer {
	return p.strategy
}

// AssignEquity uses the strategizer to assign an equity to every move.
// This is the sole module dedicated to assigning equities. (Perhaps it
// should be named something else?)
func (p *RawEquityPlayer) AssignEquity(moves []*move.Move, board *board.GameBoard,
	bag *alphabet.Bag, oppRack *alphabet.Rack) {
	for _, m := range moves {
		m.SetEquity(p.strategy.Equity(m, board, bag, oppRack))
	}
}

// BestPlay picks the highest equity play. It is assumed that these plays
// have already been assigned an equity but are not necessarily sorted.
func (p *RawEquityPlayer) BestPlay(moves []*move.Move) *move.Move {
	topEquity := -Infinity
	var topMove *move.Move
	for i := 0; i < len(moves); i++ {
		if moves[i].Equity() > topEquity {
			topEquity = moves[i].Equity()
			topMove = moves[i]
		}
	}
	return topMove
}

// TopPlays sorts the plays by equity and returns the top N. It assumes
// that the equities have already been assigned.
func (p *RawEquityPlayer) TopPlays(moves []*move.Move, n int) []*move.Move {
	sort.Slice(moves, func(i, j int) bool {
		return moves[j].Equity() < moves[i].Equity()
	})
	if n > len(moves) {
		n = len(moves)
	}
	return moves[:n]
}

func (p *RawEquityPlayer) GetBotType() pb.BotRequest_BotCode {
	return p.botType
}
