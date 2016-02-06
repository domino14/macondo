package anagrammer

import (
	"fmt"
	"net/http"
)

type AnagramServiceArgs struct {
	Lexicon string `json:"lexicon"`
	Letters string `json:"letters"`
	Mode    string `json:"mode"`
}

type AnagramServiceReply struct {
	Words    []string `json:"words"`
	NumWords int      `json:"numWords"`
}

type AnagramService struct{}

func (a *AnagramService) Anagram(r *http.Request, args *AnagramServiceArgs,
	reply *AnagramServiceReply) error {
	// We cast to a SimpleGaddag here because only that one has
	// GetAlphabet defined.
	dawg, ok := Dawgs[args.Lexicon]
	if !ok {
		return fmt.Errorf("Lexicon %v not found", args.Lexicon)
	}
	mode := ModeBuild
	if args.Mode == "exact" {
		mode = ModeExact
	}
	sols := Anagram(args.Letters, dawg, mode)
	reply.Words = sols
	reply.NumWords = len(sols)
	return nil
}

type BlankChallengeArgs struct {
	WordLength   int    `json:"wordLength"`
	NumQuestions int    `json:"numQuestions"`
	Lexicon      string `json:"lexicon"`
	MaxSolutions int    `json:"maxSolutions"`
	// How many to generate with 2 blanks (default is 1)
	Num2Blanks int `json:"num2Blanks"`
}

type BlankChallengeReply struct {
	Questions    []*Question `json:"questions"`
	NumQuestions int         `json:"numQuestions"`
	NumAnswers   int         `json:"numAnswers"`
}

type BuildChallengeArgs struct {
	WordLength   int    `json:"wordLength"`
	Lexicon      string `json:"lexicon"`
	MinSolutions int    `json:"minSolutions"`
	MaxSolutions int    `json:"maxSolutions"`
}

type BuildChallengeReply struct {
	Question   *Question `json:"question"`
	NumAnswers int       `json:"numAnswers"`
}

func (a *AnagramService) BlankChallenge(r *http.Request, args *BlankChallengeArgs,
	reply *BlankChallengeReply) error {

	dawg, ok := Dawgs[args.Lexicon]
	if !ok {
		return fmt.Errorf("Lexicon %v not found", args.Lexicon)
	}
	blanks, numAnswers := GenerateBlanks(args, dawg)
	reply.Questions = blanks
	reply.NumQuestions = len(blanks)
	reply.NumAnswers = numAnswers
	return nil
}

func (a *AnagramService) BuildChallenge(r *http.Request, args *BuildChallengeArgs,
	reply *BuildChallengeReply) error {

	dawg, ok := Dawgs[args.Lexicon]
	if !ok {
		return fmt.Errorf("Lexicon %v not found", args.Lexicon)
	}
	question, numAnswers := GenerateBuildChallenge(args, dawg)
	reply.Question = question
	reply.NumAnswers = numAnswers
	return nil
}

/**
 * curl -H "Content-Type: application/json" http://localhost:8088/rpc -X POST -d \
 * '{"jsonrpc":"2.0","method":"AnagramService.BlankChallenge","id":1, \
 * "params":{"wordLength": 7, "numQuestions": 5, "lexicon": "America", \
 * "maxSolutions": 7, "num2Blanks": 1}}'
 */
