package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	"github.com/domino14/macondo/move"
	"github.com/rs/zerolog"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var filepath = "allanno.csv"
var url = "http://cross-tables.com/rest/allanno.php"
var DefaultConfig = config.DefaultConfig()

type EqlossData struct {
	Items []*EqlossItem
}

type EqlossItem struct {
	YourPlay           string
	BestPlay           string
	Eqloss             float64
	PhonyTilesReturned bool
}

func main() {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	playerIdPtr := flag.Int("playerid", -1, "the cross-tables id of the player")
	minGameIdPtr := flag.Int("mingameid", -1, "the minimum cross-tables game id")
	maxGameIdPtr := flag.Int("maxgameid", 100000000, "the maximum cross-tables game id")
	tourneyIdPtr := flag.Int("tournamentid", -1, "the cross-tables id of the tournament")
	playerNamePtr := flag.String("playername", "", "the name of the player")
	playLimitPtr := flag.Int("playlimit", 10, "the number of top equity loss moves listed")

	flag.Parse()

	if *playerIdPtr == -1 {
		panic("must specify a playerid")
	}
	if *playerNamePtr == "" {
		panic("must specify a playername")
	}

	downloadFile(filepath, url)

	file, err := os.Open(filepath)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	lexicons := []string{}
	gameids := []int{}
	scanner := bufio.NewScanner(file)
	headerSkipped := false
	linenum := 0
	for scanner.Scan() {
		linenum++
		if !headerSkipped {
			headerSkipped = true
			continue
		}
		line := scanner.Text()
		data := strings.Split(line, ",")
		if len(data) != 9 {
			continue
		}
		gameid, err := strconv.Atoi(data[0])
		if err != nil {
			panic(err)
		}
		playerOneId, err := strconv.Atoi(data[1])
		if err != nil {
			panic(err)
		}
		playerTwoId, err := strconv.Atoi(data[2])
		if err != nil {
			panic(err)
		}
		tourneyId, err := strconv.Atoi(data[5])
		if err != nil {
			panic(err)
		}
		lexicon := data[7]
		if (playerOneId == *playerIdPtr || playerTwoId == *playerIdPtr) &&
			(*tourneyIdPtr == -1 || tourneyId == *tourneyIdPtr) &&
			gameid >= *minGameIdPtr && gameid <= *maxGameIdPtr {
			gameids = append(gameids, gameid)
			lexicons = append(lexicons, lexicon)
		}
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	eqlossData := &EqlossData{Items: []*EqlossItem{}}
	totalEqloss := 0.0
	for idx, gameid := range gameids {
		lexicon := lexicons[idx]
		gcgFilepath := "gcg"
		downloadFile(gcgFilepath, fmt.Sprintf("https://www.cross-tables.com/annotated/selfgcg/%d/anno%d.gcg", gameid/100, gameid))
		eqloss, err := getEquityLoss(gcgFilepath, lexicon, *playerNamePtr, gameid, eqlossData)
		if err != nil {
			panic(err)
		}
		totalEqloss += eqloss
	}
	fmt.Printf("Average Equity Loss: %.2f\nTotal Games: %d\n", totalEqloss/float64(len(gameids)), len(gameids))
	fmt.Printf("Total Plays: %d\n", len(eqlossData.Items))

	sort.Slice(eqlossData.Items, func(i, j int) bool {
		return eqlossData.Items[i].Eqloss > eqlossData.Items[j].Eqloss
	})

	maxPlayLen := -1
	for i := 0; i < 10 && i < len(eqlossData.Items); i++ {
		if len(eqlossData.Items[i].YourPlay) > maxPlayLen {
			maxPlayLen = len(eqlossData.Items[i].YourPlay)
		}
	}

	fmt.Printf("\n\nTop %d Saddest Anime Moments:\n\n", *playLimitPtr)
	fmt.Printf("%-"+strconv.Itoa(maxPlayLen)+"s | %s  | %s | %s\n", "Your Play", "Loss", "Walid?", "Best Play")
	for i := 0; i < *playLimitPtr && i < len(eqlossData.Items); i++ {
		wetoed := "Strong"
		if eqlossData.Items[i].PhonyTilesReturned {
			wetoed = "NO    "
		}
		fmt.Printf("%-"+strconv.Itoa(maxPlayLen)+"s | %.2f | %s | %s\n", eqlossData.Items[i].YourPlay, eqlossData.Items[i].Eqloss, wetoed, eqlossData.Items[i].BestPlay)
	}
}

func getEquityLoss(filepath string, lexicon string, playerName string, gameid int, eqlossData *EqlossData) (float64, error) {
	rules, err := game.NewBasicGameRules(DefaultConfig, lexicon, board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		panic(err)
	}

	gameHistory, err := gcgio.ParseGCG(DefaultConfig, filepath)
	if err != nil {
		panic(err)
	}
	p1Nickname := gameHistory.Players[0].Nickname
	p2Nickname := gameHistory.Players[1].Nickname
	if playerName != p1Nickname && playerName != p2Nickname {
		panic(fmt.Sprintf("player %s not found in (%s, %s) for game %d", playerName, p1Nickname, p2Nickname, gameid))
	}

	gameHistory.ChallengeRule = pb.ChallengeRule_FIVE_POINT

	g, err := game.NewFromHistory(gameHistory, rules, 0)
	if err != nil {
		panic(err)
	}

	totalEqloss := 0.0
	numMoves := 0
	history := g.History()
	players := history.Players
	botConfig := &bot.BotConfig{Config: *DefaultConfig}
	for evtIdx, evt := range history.Events {
		if players[evt.PlayerIndex].Nickname == playerName &&
			(evt.Type == pb.GameEvent_TILE_PLACEMENT_MOVE ||
				evt.Type == pb.GameEvent_EXCHANGE ||
				evt.Type == pb.GameEvent_PASS) {
			err := g.PlayToTurn(evtIdx)
			if err != nil {
				panic(err)
			}
			runner, err := bot.NewBotTurnPlayerFromGame(g, botConfig, pb.BotRequest_HASTY_BOT)
			if err != nil {
				panic(err)
			}
			thisMove, err := game.MoveFromEvent(evt, g.Alphabet(), g.Board())
			if err != nil {
				panic(err)
			}

			runner.AssignEquity([]*move.Move{thisMove}, g.Board(), g.Bag(), nil)
			topMove := runner.GenerateMoves(1)[0]

			thisMoveEquity := thisMove.Equity()
			phonyTilesReturned := len(history.Events) > evtIdx+1 && history.Events[evtIdx+1].Type == pb.GameEvent_PHONY_TILES_RETURNED
			if phonyTilesReturned {
				thisMoveEquity = 0.0
			}

			eqloss := topMove.Equity() - thisMoveEquity
			eqlossData.Items = append(eqlossData.Items, &EqlossItem{BestPlay: topMove.ShortDescription(), YourPlay: thisMove.ShortDescription(), Eqloss: eqloss, PhonyTilesReturned: phonyTilesReturned})
			totalEqloss += eqloss
			numMoves++
		}
	}
	return totalEqloss, nil
}

func downloadFile(filepath string, url string) {

	// Create the file
	out, err := os.Create(filepath)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		panic(fmt.Sprintf("bad status: %s", resp.Status))
	}

	// Writer the body to file
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		panic(err)
	}
}
