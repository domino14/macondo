package runner

import (
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func ShowChallengeRule(rule pb.ChallengeRule) string {
	var ret string
	switch rule {
	case pb.ChallengeRule_VOID:
		ret = "void"
	case pb.ChallengeRule_SINGLE:
		ret = "single"
	case pb.ChallengeRule_DOUBLE:
		ret = "double"
	case pb.ChallengeRule_TRIPLE:
		ret = "triple"
	case pb.ChallengeRule_FIVE_POINT:
		ret = "5pt"
	case pb.ChallengeRule_TEN_POINT:
		ret = "10pt"
	}
	return ret
}
