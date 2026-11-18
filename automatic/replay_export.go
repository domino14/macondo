package automatic

import "github.com/domino14/macondo/turnplayer"

// SplitTurnHalves splits one game ID's per-turn log rows into its seatings
// (two in a -gamepairs run), for tools that replay logs themselves.
func SplitTurnHalves(rows [][]string) [][][]string { return splitTurnHalves(rows) }

// PlayLoggedRow applies one per-turn log row to a game in progress; setRack
// false leaves the racks the caller has already seated.
func PlayLoggedRow(g *turnplayer.BaseTurnPlayer, row []string, setRack bool) error {
	return playLoggedRow(g, row, setRack)
}
