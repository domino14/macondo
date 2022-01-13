package board

import (
	"fmt"
	"os"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/alphabet"
)

var (
	ColorSupport = os.Getenv("MACONDO_DISABLE_COLOR") != "on"
)

// A BonusSquare is a bonus square (duh)
type BonusSquare rune

const (
	Bonus4WS BonusSquare = '~'
	Bonus4LS BonusSquare = '^'
	// Bonus3WS is a triple word score
	Bonus3WS BonusSquare = '='
	// Bonus3LS is a triple letter score
	Bonus3LS BonusSquare = '"'
	// Bonus2LS is a double letter score
	Bonus2LS BonusSquare = '\''
	// Bonus2WS is a double word score
	Bonus2WS BonusSquare = '-'
)

// A Square is a single square in a game board. It contains the bonus markings,
// if any, a letter, if any (' ' if empty), and any cross-sets and cross-scores
type Square struct {
	letter alphabet.MachineLetter
	bonus  BonusSquare

	hcrossSet CrossSet
	vcrossSet CrossSet
	// the scores of the tiles on either side of this square.
	hcrossScore int
	vcrossScore int
	hAnchor     bool
	vAnchor     bool
}

func init() {
	if ColorSupport {
		log.Info().Msg("Terminal color support is on.")
	}
}

func (s Square) String() string {
	bonus := string(s.bonus)
	return fmt.Sprintf("<(%v) (%s)>", s.letter, bonus)
}

func (s Square) Info() string {
	return fmt.Sprintf("[let %v bonus %v hc %v vc %v hcs %v vcs %v ha %v va %v]",
		s.letter, s.bonus, s.hcrossSet, s.vcrossSet, s.hcrossScore, s.vcrossScore,
		s.hAnchor, s.vAnchor,
	)
}

func (s *Square) copyFrom(s2 *Square) {
	s.letter = s2.letter
	s.bonus = s2.bonus
	s.hcrossSet = s2.hcrossSet
	s.vcrossSet = s2.vcrossSet
	s.hcrossScore = s2.hcrossScore
	s.vcrossScore = s2.vcrossScore
	s.hAnchor = s2.hAnchor
	s.vAnchor = s2.vAnchor
}

func (s *Square) equals(s2 *Square) bool {
	if s.bonus != s2.bonus {
		log.Debug().Msg("Bonuses not equal")
		return false
	}
	if s.letter != s2.letter {
		log.Debug().Msg("Letters not equal")
		return false
	}
	if s.hcrossSet != s2.hcrossSet {
		log.Debug().Msgf("horiz cross-sets not equal: %v %v", s.hcrossSet, s2.hcrossSet)
		return false
	}
	if s.vcrossSet != s2.vcrossSet {
		log.Debug().Msgf("vert cross-sets not equal: %v %v", s.vcrossSet, s2.vcrossSet)
		return false
	}
	if s.hcrossScore != s2.hcrossScore {
		log.Debug().Msgf("horiz cross-scores not equal: %v %v", s.hcrossScore, s2.hcrossScore)
		return false
	}
	if s.vcrossScore != s2.vcrossScore {
		log.Debug().Msgf("vert cross-scores not equal: %v %v", s.vcrossScore, s2.vcrossScore)
		return false
	}
	if s.hAnchor != s2.hAnchor {
		log.Debug().Msgf("horiz anchors not equal: %v %v", s.hAnchor, s2.hAnchor)
		return false
	}
	if s.vAnchor != s2.vAnchor {
		log.Debug().Msgf("vert anchors not equal: %v %v", s.vAnchor, s2.vAnchor)
		return false
	}
	return true
}

func (s *Square) Letter() alphabet.MachineLetter {
	return s.letter
}

func (s *Square) SetLetter(letter alphabet.MachineLetter) {
	s.letter = letter
}

func (b BonusSquare) displayString() string {
	if !ColorSupport {
		return string(b)
	}
	switch b {

	case Bonus4WS:
		return fmt.Sprintf("\033[33m%s\033[0m", string(b))
	case Bonus3WS:
		return fmt.Sprintf("\033[31m%s\033[0m", string(b))
	case Bonus2WS:
		return fmt.Sprintf("\033[35m%s\033[0m", string(b))
	case Bonus4LS:
		return fmt.Sprintf("\033[95m%s\033[0m", string(b))
	case Bonus3LS:
		return fmt.Sprintf("\033[34m%s\033[0m", string(b))
	case Bonus2LS:
		return fmt.Sprintf("\033[36m%s\033[0m", string(b))
	default:
		return "?"
	}
}

func (s Square) DisplayString(alph *alphabet.Alphabet) string {
	var bonusdisp string
	if s.bonus != ' ' {
		bonusdisp = s.bonus.displayString()
	} else {
		bonusdisp = " "
	}
	if s.letter == alphabet.EmptySquareMarker {
		return bonusdisp
	}
	return string(s.letter.UserVisible(alph))

}

func (s Square) BadDisplayString(alph *alphabet.Alphabet) string {
	var hadisp, vadisp, bonusdisp string
	if s.hAnchor {
		hadisp = "→"
	} else {
		hadisp = " "
	}
	if s.vAnchor {
		vadisp = "↓"
	} else {
		vadisp = " "
	}
	if s.bonus != 0 {
		bonusdisp = string(s.bonus)
	} else {
		bonusdisp = " "
	}
	if s.letter == alphabet.EmptySquareMarker {
		return fmt.Sprintf("[%v%v%v]", bonusdisp, hadisp, vadisp)
	}
	return fmt.Sprintf("[%v%v%v]", s.letter.UserVisible(alph), hadisp, vadisp)

}

func (s *Square) SetCrossSet(cs CrossSet, dir BoardDirection) {
	if dir == HorizontalDirection {
		s.hcrossSet = cs
	} else if dir == VerticalDirection {
		s.vcrossSet = cs
	}
}

func (s *Square) SetCrossScore(score int, dir BoardDirection) {
	if dir == HorizontalDirection {
		s.hcrossScore = score
	} else if dir == VerticalDirection {
		s.vcrossScore = score
	}
}

func (s *Square) GetCrossSet(dir BoardDirection) *CrossSet {
	if dir == HorizontalDirection {
		return &s.hcrossSet
	} else if dir == VerticalDirection {
		return &s.vcrossSet
	}
	return nil
}

func (s *Square) GetCrossScore(dir BoardDirection) int {
	if dir == HorizontalDirection {
		return s.hcrossScore
	} else if dir == VerticalDirection {
		return s.vcrossScore
	}
	return 0
}

func (s *Square) setAnchor(dir BoardDirection) {
	if dir == HorizontalDirection {
		s.hAnchor = true
	} else if dir == VerticalDirection {
		s.vAnchor = true
	}
}

func (s *Square) resetAnchors() {
	s.hAnchor = false
	s.vAnchor = false
}

func (s *Square) IsEmpty() bool {
	return s.letter == alphabet.EmptySquareMarker
}

func (s *Square) anchor(dir BoardDirection) bool {
	if dir == HorizontalDirection {
		return s.hAnchor
	}
	return s.vAnchor
}
