package analyzer

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/runner"
)

var SampleJson = []byte(`{
"scores": [0, 0],
"onturn": 0,
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
	Scores  []int
	Onturn  int
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
	game    *runner.AIGameRunner
	moves   []*move.Move
	simmer  *montecarlo.Simmer
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
	err := json.Unmarshal(j, &b)
	if err != nil {
		return fmt.Errorf("parse json failed: %w", err)
	}
	if len(b.Scores) != 2 {
		return errors.New("invalid scores")
	}
	if b.Onturn < 0 || b.Onturn > 1 {
		return errors.New("invalid onturn")
	}
	an.options.SetLexicon([]string{b.Lexicon})
	err = an.newGame()
	if err != nil {
		return fmt.Errorf("creating game failed: %w", err)
	}
	var g = an.game
	g.SetPointsFor(0, b.Scores[0])
	g.SetPointsFor(1, b.Scores[1])
	g.SetPlayerOnTurn(b.Onturn)
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
		return fmt.Errorf("removing board tiles failed: %w", err)
	}
	// Set the current rack. This will also give opponent a random rack
	// from what remains, and edit the bag accordingly.
	err = g.SetCurrentRack(b.Rack)
	if err != nil {
		return fmt.Errorf("setting rack to %v failed: %w", b.Rack, err)
	}
	g.RecalculateBoard()

	return nil
}

func (an *Analyzer) LoadGame(jsonBoard []byte) error {
	err := an.loadJson(jsonBoard)
	if err != nil {
		return fmt.Errorf("loading game failed: %w", err)
	}
	return nil
}

func (an *Analyzer) GenerateMoves(numPlays int) {
	an.moves = an.game.GenerateMoves(numPlays)
}

func (an *Analyzer) ToJsonMoves() ([]byte, error) {
	out := make([]JsonMove, len(an.moves))
	for i, m := range an.moves {
		out[i] = MakeJsonMove(m)
	}
	return json.Marshal(out)
}

func (an *Analyzer) Analyze(jsonBoard []byte) ([]byte, error) {
	err := an.LoadGame(jsonBoard)
	if err != nil {
		return nil, err
	}
	an.GenerateMoves(15)
	return an.ToJsonMoves()
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
	err = json.Unmarshal(moves, &ms)
	if err != nil {
		return err
	}
	for _, m := range ms {
		fmt.Printf("%s %-15s %-7s %.1f\n", m.DisplayCoordinates, m.Tiles, m.Leave, m.Equity)
	}
	return nil
}

func AnalyzeBoard(jsonBoard []byte) ([]byte, error) {
	an := NewDefaultAnalyzer()
	return an.Analyze(jsonBoard)
}

func (an *Analyzer) SimInit() error {
	simmer := &montecarlo.Simmer{}
	simmer.Init(&an.game.Game, an.game.AIPlayer())
	simmer.Reset()
	err := simmer.PrepareSim(2, an.moves)
	if err != nil {
		return fmt.Errorf("init sim failed: %w", err)
	}
	an.simmer = simmer
	return nil
}

func (an *Analyzer) SimSingleThread(iters int) error {
	simmer := an.simmer
	if simmer == nil {
		return errors.New("sim not initialized")
	}
	simmer.SimSingleThread(iters)
	return nil
}

// temp
func (an *Analyzer) SimState() ([]byte, error) {
	simmer := an.simmer
	if simmer == nil {
		return nil, errors.New("sim not initialized")
	}
	return json.Marshal(struct {
		EquityStats  string `json:"equity_stats"`
		ScoreDetails string `json:"score_details"`
	}{
		EquityStats:  simmer.EquityStats(),
		ScoreDetails: simmer.ScoreDetails(),
	})
}
