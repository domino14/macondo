package mechanics

import (
	"fmt"

	"github.com/domino14/macondo/move"
)

// A Turn is literally just an array of events. Most turns will have just
// one event attached to them.
type Turn []Event

// Display a summary of this turn's events.
func (t Turn) summary() string {
	summary := ""
	firstEvtPlacement := false
	for idx := range t {
		switch e := t[idx].(type) {
		case *TilePlacementEvent:
			summary += fmt.Sprintf("%s played %s %s for %d pts from a rack of %s",
				e.Nickname, e.Position, e.Play, e.Score, e.Rack)
			firstEvtPlacement = true
		case *PassingEvent:
			if len(e.Exchanged) > 0 {
				summary += fmt.Sprintf("%s exchanged %s from a rack of %s",
					e.Nickname, e.Exchanged, e.Rack)
			} else {
				summary += fmt.Sprintf("%s passed, holding a rack of %s",
					e.Nickname, e.Rack)
			}
		case *ScoreAdditionEvent:
			if e.Bonus > 0 {
				summary += fmt.Sprintf(" (+%d)", e.Bonus)
			} else {
				summary += fmt.Sprintf(" (+%d from opponent rack)", e.EndRackPoints)
			}

		case *ScoreSubtractionEvent:
			if firstEvtPlacement {
				summary += " (challenged off)"
			}
		}
	}
	return summary
}

// An Event is an atom of play, such as a scoring move or the return of
// phoney tiles after an unsuccessful challenge.
type Event interface {
	AppendNote(string)
	GetRack() string
	GetNickname() string
	GetType() EventType
}

type EventType string

const (
	RegMove        EventType = "move"
	LostChallenge            = "lost_challenge"
	Pass                     = "pass"
	ChallengeBonus           = "challenge_bonus"
	Exchange                 = "exchange"
	EndRackPts               = "end_rack_points"
	TimePenalty              = "time_penalty"
	EndRackPenalty           = "end_rack_penalty"
)

type BaseEvent struct {
	Nickname   string    `json:"nick"`
	Note       string    `json:"note"`
	Rack       string    `json:"rack"`
	Type       EventType `json:"type"`
	Cumulative int       `json:"cumul"`
	move       *move.Move
}

func (be *BaseEvent) AppendNote(note string) {
	be.Note = be.Note + note
}

func (be *BaseEvent) GetRack() string {
	return be.Rack
}

func (be *BaseEvent) GetNickname() string {
	return be.Nickname
}

func (be *BaseEvent) GetType() EventType {
	return be.Type
}

type TilePlacementEvent struct {
	BaseEvent
	Row       uint8  `json:"row"`
	Column    uint8  `json:"col"`
	Direction string `json:"dir,omitempty"`
	Position  string `json:"pos,omitempty"`
	Play      string `json:"play,omitempty"`
	Score     int    `json:"score"`
}

func (tp *TilePlacementEvent) CalculateCoordsFromPosition() {
	row, col, vertical := move.FromBoardGameCoords(tp.Position)
	if vertical {
		tp.Direction = "v"
	} else {
		tp.Direction = "h"
	}
	tp.Row = uint8(row)
	tp.Column = uint8(col)
}

type PassingEvent struct {
	BaseEvent
	Exchanged string `json:"exchanged,omitempty"`
}

type ScoreAdditionEvent struct {
	BaseEvent
	Bonus         int `json:"bonus,omitempty"`
	EndRackPoints int `json:"score"`
}

type ScoreSubtractionEvent struct {
	BaseEvent
	LostScore int `json:"lost_score"`
}
