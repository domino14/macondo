package analyzer

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
)

var SampleJson = []byte(`{
"size": 15,
"rack": "EINRSTZ",
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
	Size  int
	Board []string
	Rack  string
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
}

type Analyzer struct {
	config  *config.Config
	options *runner.GameOptions
	game    *runner.AIGameRunner
}

func MakeJsonMove(m *move.Move) JsonMove {
	j := JsonMove{}
	j.Action = m.MoveTypeString()
	j.Row, j.Column, j.Vertical = m.CoordsAndVertical()
	j.DisplayCoordinates = m.BoardCoords()
	j.Tiles = m.TilesString()
	j.Leave = m.LeaveString()
	j.Equity = m.Equity()
	return j
}

func NewAnalyzer(config *config.Config, options *runner.GameOptions) *Analyzer {
	an := &Analyzer{}
	an.config = config
	an.options = options
	an.game = nil
	return an
}

func (an *Analyzer) newGame() error {
	players := []*pb.PlayerInfo{
		{Nickname: "self", RealName: "Macondo Bot"},
		{Nickname: "opponent", RealName: "Arthur Dent"},
	}

	game, err := runner.NewAIGameRunner(an.config, an.options, players)
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
	var g = an.game
	bd := g.Board()
	for row, str := range b.Board {
		str = strings.Replace(str, ".", " ", -1)
		bd.SetRow(row, str, g.Alphabet())
	}
	g.SetCurrentRack(b.Rack)
	g.RecalculateBoard()
	return nil
}

func (an *Analyzer) Analyze(jsonBoard []byte) ([]byte, error) {
	err := an.newGame()
	if err != nil {
		fmt.Println("Creating game failed!")
		return nil, err
	}
	err = an.loadJson(SampleJson)
	if err != nil {
		fmt.Println("Loading game failed!")
		return nil, err
	}
	moves := an.game.GenerateMoves(15)
	out := make([]JsonMove, 15)
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
