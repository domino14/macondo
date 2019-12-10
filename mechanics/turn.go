package mechanics

import "github.com/domino14/macondo/move"

// A Turn is literally just an array of events. Most turns will have just
// one event attached to them.
type Turn []Event

// An Event is an atom of play, such as a scoring move or the return of
// phoney tiles after an unsuccessful challenge.
type Event interface {
	AppendNote(string)
	GetRack() string
	GetNickname() string
}

type BaseEvent struct {
	Nickname   string `json:"nick"`
	Note       string `json:"note"`
	Rack       string `json:"rack"`
	Type       string `json:"type"`
	Cumulative int    `json:"cumul"`
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

type TilePlacementEvent struct {
	BaseEvent
	Row       uint8  `json:"row"`
	Column    uint8  `json:"col"`
	Direction string `json:"dir,omitempty"`
	Position  string `json:"pos,omitempty"`
	Play      string `json:"play,omitempty"`
	Score     int    `json:"score"`
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
