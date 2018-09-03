package anagrammer

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// AuthorizationKey is used for non-user exposed methods
var AuthorizationKey = os.Getenv("AUTH_KEY")

func init() {
	if AuthorizationKey == "" {
		panic("No auth key defined")
	}
}

type AnagramServiceArgs struct {
	Lexicon   string `json:"lexicon"`
	Letters   string `json:"letters"`
	Mode      string `json:"mode"`
	AuthToken string `json:"authToken"`
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
	start := time.Now()

	dawg, ok := Dawgs[args.Lexicon]
	if !ok {
		return fmt.Errorf("Lexicon %v not found", args.Lexicon)
	}
	mode := ModeBuild
	if args.Mode == "exact" {
		mode = ModeExact
	}

	if strings.Count(args.Letters, "?") > 2 {
		if args.AuthToken != AuthorizationKey {
			return errors.New("query too complex")
		}
	}

	sols := Anagram(args.Letters, dawg, mode)
	reply.Words = sols
	reply.NumWords = len(sols)

	elapsed := time.Since(start)
	log.Printf("Anagram took %s", elapsed)

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
	WordLength            int    `json:"wordLength"`
	MinWordLength         int    `json:"minWordLength"`
	RequireLengthSolution bool   `json:"requireLengthSolution"`
	Lexicon               string `json:"lexicon"`
	MinSolutions          int    `json:"minSolutions"`
	MaxSolutions          int    `json:"maxSolutions"`
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

	blanks, numAnswers, err := GenerateBlanks(r.Context(), args, dawg)
	if err != nil {
		return err
	}
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
	question, numAnswers, err := GenerateBuildChallenge(r.Context(), args, dawg)
	if err != nil {
		return err
	}
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
