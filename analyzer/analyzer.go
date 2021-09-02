package analyzer

import (
	"encoding/json"
	"fmt"
	"strings"

	airunner "github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
)

var SampleJson = []byte(`{
"size": 15,
"rack": "EINRSTZ",
"lexicon": "CSW19",
"board": [
  "...............",
  "...............",
  "...............",
  "...............",
  "...............",
  "...............",
  "...............",
  "...HELLO.......",
  "...............",
  "...............",
  "...............",
  "...............",
  "...............",
  "...............",
  "..............."
]}`)

type JsonBoard struct {
	Size    int
	Lexicon string
	Board   []string
	Rack    string
}

type JsonMove struct {
	Action             string
	Row                int
	Column             int
	Vertical           bool
	DisplayCoordinates string
	Tiles              string
	Leave              string
	Equity             float64
	Score              int
}

type Analyzer struct {
	config  *config.Config
	options *runner.GameOptions
	game    *airunner.AIGameRunner
}

func MakeJsonMove(m *move.Move) JsonMove {
	j := JsonMove{}
	j.Action = m.MoveTypeString()
	j.Row, j.Column, j.Vertical = m.CoordsAndVertical()
	j.DisplayCoordinates = m.BoardCoords()
	j.Tiles = m.TilesString()
	j.Leave = m.LeaveString()
	j.Equity = m.Equity()
	j.Score = m.Score()
	return j
}

func NewAnalyzer(config *config.Config) *Analyzer {
	options := &runner.GameOptions{}
	an := &Analyzer{}
	an.config = config
	an.options = options
	an.game = nil
	return an
}

// Create an analyzer with an empty config. This will not have any relative
// resource paths resolved to actual paths; the caller is responsible for
// precaching everything so that we never actually hit the file system.
func NewDefaultAnalyzer() *Analyzer {
	cfg := &config.Config{}
	cfg.Load([]string{})
	cfg.Debug = false
	return NewAnalyzer(cfg)
}

func (an *Analyzer) newGame() error {
	players := []*pb.PlayerInfo{
		{Nickname: "self", RealName: "Macondo Bot"},
		{Nickname: "opponent", RealName: "Arthur Dent"},
	}

	game, err := airunner.NewAIGameRunner(an.config, an.options, players, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return err
	}
	an.game = game
	return nil
}

func (an *Analyzer) loadJson(j []byte) error {
	// Load a game position from a json blob
	var b = JsonBoard{}
	json.Unmarshal(j, &b)
	an.options.SetLexicon([]string{b.Lexicon})
	err := an.newGame()
	if err != nil {
		fmt.Println("Creating game failed!")
		return err
	}
	var g = an.game
	bd := g.Board()
	letters := []alphabet.MachineLetter{}
	for row, str := range b.Board {
		str = strings.Replace(str, ".", " ", -1)
		letters = append(letters, bd.SetRow(row, str, g.Alphabet())...)
	}
	// Reset the state of the bag; empty player racks (they are set to
	// random racks) and refill the bag from scratch
	g.ThrowRacksIn()
	g.Bag().Refill()
	// Then remove the visible tiles on the board
	err = g.Bag().RemoveTiles(letters)
	if err != nil {
		fmt.Println("Removing board tiles failed!")
		return err
	}
	// Set the current rack. This will also give opponent a random rack
	// from what remains, and edit the bag accordingly.
	err = g.SetCurrentRack(b.Rack)
	if err != nil {
		fmt.Println("Setting rack to " + b.Rack + " failed!")
		return err
	}
	g.RecalculateBoard()

	return nil
}

func (an *Analyzer) Analyze(jsonBoard []byte) ([]byte, error) {
	err := an.loadJson(jsonBoard)
	if err != nil {
		fmt.Println("Loading game failed!")
		return nil, err
	}
	moves := an.game.GenerateMoves(15)
	out := make([]JsonMove, len(moves))
	for i, m := range moves {
		out[i] = MakeJsonMove(m)
	}
	return json.Marshal(out)
}

func (an *Analyzer) RunTest() error {
	// Analyse the SampleJson test board
	moves, err := an.Analyze(SampleJson)
	if err != nil {
		return err
	}
	// Display the board
	g := an.game
	fmt.Println(g.Board().ToDisplayText(g.Alphabet()))
	// Display the moves
	var ms []JsonMove
	json.Unmarshal(moves, &ms)
	for _, m := range ms {
		fmt.Printf("%s %-15s %-7s %.1f\n", m.DisplayCoordinates, m.Tiles, m.Leave, m.Equity)
	}
	return nil
}

func AnalyzeBoard(jsonBoard []byte) ([]byte, error) {
	an := NewDefaultAnalyzer()
	return an.Analyze(jsonBoard)
}
