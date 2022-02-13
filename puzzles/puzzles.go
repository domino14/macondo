package puzzles

import (
	"github.com/domino14/macondo/ai/runner"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var PuzzleFunctions = []func(moves []*move.Move) (bool, pb.PuzzleTag){
	EquityPuzzle,
	OnlyBingoPuzzle,
}

func CreatePuzzlesFromGame(conf *config.Config, g *game.Game) ([]*pb.PuzzleCreationResponse, error) {
	evts := g.History().Events
	puzzles := []*pb.PuzzleCreationResponse{}
	for evtIdx := range evts {
		g.PlayToTurn(evtIdx)
		runner, err := runner.NewAIGameRunnerFromGame(g, conf, pb.BotRequest_HASTY_BOT)
		if err != nil {
			return nil, err
		}
		moves := runner.GenerateMoves(1000000)
		turnIsPuzzle := false
		tags := []pb.PuzzleTag{}
		for _, puzzleFunc := range PuzzleFunctions {
			turnIsPuzzleType, tag := puzzleFunc(moves)
			turnIsPuzzle = turnIsPuzzle || turnIsPuzzleType
			if turnIsPuzzleType {
				tags = append(tags, tag)
			}
		}
		if turnIsPuzzle {
			puzzles = append(puzzles, &pb.PuzzleCreationResponse{
				GameId:     g.Uid(),
				TurnNumber: int32(evtIdx),
				Answer:     g.EventFromMove(moves[0]),
				Tags:       tags})
		}
	}
	return puzzles, nil
}

func EquityPuzzle(moves []*move.Move) (bool, pb.PuzzleTag) {
	return len(moves) >= 2 && moves[0].Equity() > moves[1].Equity()+10, pb.PuzzleTag_EQUITY
}

func OnlyBingoPuzzle(moves []*move.Move) (bool, pb.PuzzleTag) {
	tag := pb.PuzzleTag_ONLY_BINGO
	if len(moves) == 0 || moves[0].TilesPlayed() != 7 {
		return false, tag
	}
	for _, mv := range moves[1:] {
		if mv.TilesPlayed() == 7 {
			return false, tag
		}
	}
	return true, tag
}
