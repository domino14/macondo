package cgp

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/word-golib/tilemapping"
)

type ParsedCGP struct {
	*game.Game
	Opcodes map[string]string
}

// ParseCGP returns an instantiated Game instance from the given CGP string.
func ParseCGP(cfg *config.Config, cgpstr string) (*ParsedCGP, error) {

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
	lexiconName := "NWL23"
	maxScorelessTurns := game.DefaultMaxScorelessTurns
	variant := game.VarClassic
	gid := ""
	opcodes := map[string]string{}

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
			opcodes["bdn"] = opWithParams[1]
		case "gid":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for gid operation")
			}
			gid = opWithParams[1]
			opcodes["gid"] = opWithParams[1]
		case "ld":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for ld operation")
			}
			letterDistributionName = opWithParams[1]
			opcodes["ld"] = opWithParams[1]

		case "lex":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for lex operation")
			}
			lexiconName = opWithParams[1]
			opcodes["lex"] = opWithParams[1]

		case "mcnz":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for mcnz operation")
			}
			s := opWithParams[1]
			maxScorelessTurns, err = strconv.Atoi(s)
			if err != nil {
				return nil, err
			}
			opcodes["mcnz"] = opWithParams[1]

		case "var":
			if len(opWithParams) != 2 {
				return nil, errors.New("wrong number of arguments for var operation")
			}
			variant = game.Variant(opWithParams[1])
			opcodes["var"] = opWithParams[1]

		case "tmr":
			opcodes["tmr"] = opWithParams[1]

		}

	}

	rules, err := game.NewBasicGameRules(cfg, lexiconName, boardLayoutName, letterDistributionName,
		game.CrossScoreAndSet, variant)
	if err != nil {
		return nil, err
	}

	// "Decompress" the gameboard letters.
	fullRows := make([][]tilemapping.MachineLetter, len(rows))

	for i, row := range rows {
		fullRows[i], err = rowToLetters(row, rules.LetterDistribution().TileMapping())
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
	g.SetStartingCGP(cgpstr)
	g.SetUid(gid)
	g.SetIdAuth("") //  maybe provide this later, id

	log.Debug().Msgf("got gid %v", gid)
	return &ParsedCGP{Game: g, Opcodes: opcodes}, nil
}

func rowToLetters(row string, tm *tilemapping.TileMapping) ([]tilemapping.MachineLetter, error) {
	mls := []tilemapping.MachineLetter{}
	var lettersTemp string
	var multiTile string
	beganMulti := false
	lastN := ""
	for _, rn := range row {

		if rn == '[' {
			if lettersTemp != "" {
				mout, err := tilemapping.ToMachineLetters(lettersTemp, tm)
				if err != nil {
					return nil, err
				}
				mls = append(mls, mout...)
				lettersTemp = ""
			}
			beganMulti = true
			multiTile = ""
		} else if rn == ']' {
			mout, err := tilemapping.ToMachineLetters(multiTile, tm)
			if err != nil {
				return nil, err
			}
			if len(mout) > 1 {
				// If the multitile mapped to more than 1 "machine letter" this
				// is unexpected. As a last resort, try to parse the tile
				// _including_ the [].
				mout, err = tilemapping.ToMachineLetters("["+multiTile+"]", tm)
				if err != nil {
					return nil, err
				}
			}
			mls = append(mls, mout...)
			multiTile = ""
			beganMulti = false
		} else if rn >= '0' && rn <= '9' {
			if lettersTemp != "" {
				mout, err := tilemapping.ToMachineLetters(lettersTemp, tm)
				if err != nil {
					return nil, err
				}
				mls = append(mls, mout...)
				lettersTemp = ""
			}
			lastN += string(rn)
		} else {
			// parse the number then clear it out.
			if lastN != "" {
				n, err := strconv.Atoi(lastN)
				if err != nil {
					return nil, err
				}
				for idx := 0; idx < n; idx++ {
					mls = append(mls, 0)
				}
				lastN = ""
			}
			if beganMulti {
				multiTile += string(rn)
			} else {
				lettersTemp += string(rn)
			}
		}
	}
	if len(lettersTemp) > 0 {
		mout, err := tilemapping.ToMachineLetters(lettersTemp, tm)
		if err != nil {
			return nil, err
		}
		mls = append(mls, mout...)
		lettersTemp = ""
	}

	if lastN != "" {
		n, err := strconv.Atoi(lastN)
		if err != nil {
			return nil, err
		}
		for idx := 0; idx < n; idx++ {
			mls = append(mls, 0)
		}
	}

	return mls, nil
}
