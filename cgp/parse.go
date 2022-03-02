package cgp

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/rs/zerolog/log"
)

var nre *regexp.Regexp

func init() {
	nre = regexp.MustCompile("[0-9]+")
}

// ParseCGP returns an instantiated Game instance from the given CGP string.
func ParseCGP(cfg *config.Config, cgpstr string) (*game.Game, error) {

	var err error

	fields := strings.SplitN(cgpstr, " ", 5)
	if len(fields) < 4 {
		return nil, errors.New("must have at least 4 space-separated fields")
	}
	// The board:
	rows := strings.Split(fields[0], "/")

	playerRacks := strings.Split(fields[1], "/")

	playerScores := strings.Split(fields[2], "/")

	if len(playerRacks) != len(playerScores) {
		return nil, errors.New("player racks and scores do not match")
	}
	if len(playerRacks) != 2 {
		return nil, errors.New("only 2-player games are supported at the moment")
	}
	scores := make([]int, len(playerScores))
	for i, s := range playerScores {
		score, err := strconv.Atoi(s)
		if err != nil {
			return nil, err
		}
		scores[i] = score
	}

	nzero, err := strconv.Atoi(fields[3])
	if err != nil {
		return nil, err
	}

	var ops []string
	if len(fields) == 5 {
		ops = strings.Split(fields[4], ";")
	}

	// These are our defaults, but they can be overridden by operations.
	boardLayoutName := "CrosswordGame"
	letterDistributionName := "english"
	lexiconName := "NWL20"
	maxScorelessTurns := game.DefaultMaxScorelessTurns
	variant := game.VarClassic
	gid := ""

	for _, op := range ops {
		op := strings.TrimSpace(op)
		if len(op) == 0 {
			continue
		}
		opWithParams := strings.SplitN(op, " ", 2)
		switch opWithParams[0] {
		case "bdn":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for bdn operation")
			}
			boardLayoutName = opWithParams[1]
		case "gid":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for gid operation")
			}
			gid = opWithParams[1]
		case "ld":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for ld operation")
			}
			letterDistributionName = opWithParams[1]

		case "lex":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for lex operation")
			}
			lexiconName = opWithParams[1]

		case "mcnz":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for mcnz operation")
			}
			s := opWithParams[1]
			maxScorelessTurns, err = strconv.Atoi(s)
			if err != nil {
				return nil, err
			}
		}
	}

	rules, err := game.NewBasicGameRules(cfg, lexiconName, boardLayoutName, letterDistributionName,
		game.CrossScoreAndSet, variant)
	if err != nil {
		return nil, err
	}

	// "Decompress" the gameboard letters.
	fullRows := make([]string, len(rows))

	for i, row := range rows {
		fullRows[i], err = rowToLetters(row)
		if err != nil {
			return nil, err
		}
	}
	players := []*pb.PlayerInfo{}
	lastKnownRacks := []string{}
	for i, rack := range playerRacks {
		players = append(players, &pb.PlayerInfo{
			Nickname: fmt.Sprintf("player%d", i+1),
		})
		lastKnownRacks = append(lastKnownRacks, rack)
	}

	g, err := game.NewFromSnapshot(rules, players, lastKnownRacks, scores, fullRows)
	if err != nil {
		return nil, err
	}
	g.SetMaxScorelessTurns(maxScorelessTurns)
	g.SetScorelessTurns(nzero)
	g.History().StartingCgp = cgpstr
	g.History().Uid = gid
	g.History().IdAuth = "" //  maybe provide this later, id

	log.Debug().Msgf("got gid %v", gid)
	return g, nil
}

func rowToLetters(row string) (string, error) {
	// turn row into letters
	var letters strings.Builder

	writeSpaces := func(letters *strings.Builder, nspaces string) error {
		if nspaces != "" {
			n, err := strconv.Atoi(nspaces)
			if err != nil {
				return err
			}
			for i := 0; i < n; i++ {
				letters.WriteRune(' ')
			}
		}
		return nil
	}

	lastN := ""
	for _, rn := range row {
		if rn >= '0' && rn <= '9' {
			lastN += string(rn)
		} else {
			// parse the number then clear it out.
			err := writeSpaces(&letters, lastN)
			if err != nil {
				return "", err
			}
			lastN = ""
			letters.WriteRune(rn)
		}
	}
	err := writeSpaces(&letters, lastN)
	if err != nil {
		return "", err
	}
	return letters.String(), nil
}
