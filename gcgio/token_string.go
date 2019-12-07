package gcgio

// A Token is an event in a GCG file.
type Token uint8

const (
	PlayerToken Token = iota
	MoveToken
	NoteToken
	LexiconToken
	LostChallengeToken
	PassToken
	ChallengeBonusToken
	ExchangeToken
	EndRackPointsToken
)

func (i Token) String() string {
	switch i {
	case PlayerToken:
		return ""
	case MoveToken:
		return "move"
	case NoteToken:
		return "note"
	case LexiconToken:
		return ""
	case LostChallengeToken:
		return "lost_challenge"
	case PassToken:
		return "pass"
	case ChallengeBonusToken:
		return "challenge_bonus"
	case ExchangeToken:
		return "exchange"
	case EndRackPointsToken:
		return "end_rack_points"
	}
	return ""
}
