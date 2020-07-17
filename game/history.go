package game

import (
	"strings"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// HistoryToVariant takes in a game history and returns the board configuration
// and letter distribution name.
func HistoryToVariant(h *pb.GameHistory) (boardLayout []string, letterDistributionName string) {

	if h.Variant == "CrosswordGame" {
		boardLayout = board.CrosswordGameBoard
	}
	letterDistributionName = "english"
	switch {
	case strings.HasPrefix(h.Lexicon, "OSPS"):
		letterDistributionName = "polish"
	case strings.HasPrefix(h.Lexicon, "FISE"):
		letterDistributionName = "spanish"
	}
	return
}
