package game

import (
	"strings"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// HistoryToVariant takes in a game history and returns the board configuration
// and letter distribution name.
func HistoryToVariant(h *pb.GameHistory) (boardLayoutName, letterDistributionName string, variant Variant) {

	boardLayoutName = h.BoardLayout
	// XXX: the letter distribution name should come from the history.
	letterDistributionName = "english"
	lexicon := strings.ToLower(h.Lexicon)
	switch {
	case strings.HasPrefix(lexicon, "osps"):
		letterDistributionName = "polish"
	case strings.HasPrefix(lexicon, "fise"):
		letterDistributionName = "spanish"
	case strings.HasPrefix(lexicon, "rd"):
		letterDistributionName = "german"
	case strings.HasPrefix(lexicon, "nsf"):
		letterDistributionName = "norwegian"
	case strings.HasPrefix(lexicon, "fra"):
		letterDistributionName = "french"
	}
	variant = Variant(h.Variant)
	return
}
