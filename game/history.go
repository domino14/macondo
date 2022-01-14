package game

import (
	"strings"

	"github.com/domino14/macondo/board"
	"github.com/rs/zerolog/log"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// HistoryToVariant takes in a game history and returns the board configuration
// and letter distribution name.
func HistoryToVariant(h *pb.GameHistory) (boardLayoutName, letterDistributionName string, variant Variant) {
	log.Debug().Interface("h", h).Msg("HistoryToVariant")
	boardLayoutName = h.BoardLayout
	letterDistributionName = h.LetterDistribution
	if h.LetterDistribution == "" {
		// If the letter distribution is not explicitly specified, we
		// make some assumptions.
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
		// Technically, SuperCrosswordGame is not a variant, at least as far as Macondo is concerned.
		// It is the same classic rules as CrosswordGame, but with a different board layout and
		// letter distribution.
		if boardLayoutName == board.SuperCrosswordGameLayout {
			letterDistributionName += "_super"
		}
	}
	variant = Variant(h.Variant)
	return
}
