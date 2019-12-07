package mechanics

import "github.com/domino14/macondo/move"

type Turn interface {
	AppendNote(string)
}

type BaseTurn struct {
	Nickname   string `json:"nick"`
	Note       string `json:"note"`
	Rack       string `json:"rack"`
	Type       string `json:"type"`
	Cumulative int    `json:"cumul"`
	move       *move.Move
}

func (bt *BaseTurn) AppendNote(note string) {
	bt.Note = bt.Note + note
}

type TilePlacementTurn struct {
	BaseTurn
	Row       uint8  `json:"row"`
	Column    uint8  `json:"col"`
	Direction string `json:"dir,omitempty"`
	Position  string `json:"pos,omitempty"`
	Play      string `json:"play,omitempty"`
	Score     int    `json:"score"`
}

type PassingTurn struct {
	BaseTurn
	Exchanged string `json:"exchanged,omitempty"`
}

type ScoreAdditionTurn struct {
	BaseTurn
	Bonus         int `json:"bonus,omitempty"`
	EndRackPoints int `json:"score"`
}

type ScoreSubtractionTurn struct {
	BaseTurn
	LostScore int `json:"lost_score"`
}

// func newPlacementTurn(m *move.Move, player *Player) TilePlacementTurn{
// 	tpt := &TilePlacementTurn{}
// 	tpt.Nickname = player.Nickname
// 	tpt.Rack = m.
// }
