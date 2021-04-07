package game

import (
	"strings"

	"github.com/domino14/macondo/board"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// HistoryToVariant takes in a game history and returns the board configuration
// and letter distribution name.
func HistoryToVariant(h *pb.GameHistory) (boardLayout []string, letterDistributionName string) {

	switch h.Variant {
	case "CrosswordGame":
		boardLayout = board.CrosswordGameBoard
	default:
		boardLayout = board.CrosswordGameBoard
	}
	letterDistributionName = "english"
	switch {
	case strings.HasPrefix(h.Lexicon, "OSPS"):
		letterDistributionName = "polish"
	case strings.HasPrefix(h.Lexicon, "FISE"):
		letterDistributionName = "spanish"
	case strings.HasPrefix(h.Lexicon, "Deutsch"):
		letterDistributionName = "german"
	}
	return boardLayout, letterDistributionName
}
