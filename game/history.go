package game

import (
	"github.com/domino14/macondo/board"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// HistoryToVariant takes in a game history and returns the board configuration
// and letter distribution name.
func HistoryToVariant(h *pb.GameHistory) (boardLayoutName, letterDistributionName string, variant Variant) {
	log.Debug().Interface("h", h).Msg("HistoryToVariant")
	boardLayoutName = h.BoardLayout
	letterDistributionName = h.LetterDistribution
	var err error
	if h.LetterDistribution == "" {
		// If the letter distribution is not explicitly specified, we
		// make some assumptions.
		letterDistributionName, err = tilemapping.ProbableLetterDistributionName(h.Lexicon)
		if err != nil {
			log.Err(err).Str("lexicon", h.Lexicon).Msg("Could not determine letter distribution name. Defaulting to english.")
			letterDistributionName = "english"
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
