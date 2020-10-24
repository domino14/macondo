package analyzer

import (
	"fmt"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/runner"
)

type Analyzer struct {
	config *config.Config
	game   *runner.AIGameRunner
}

func (an *Analyzer) loadGCG(filepath string) error {
	var err error
	var history *pb.GameHistory
	history, err = gcgio.ParseGCG(an.config, filepath)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Loaded game repr; players: %v", history.Players)
	lexicon := history.Lexicon
	if lexicon == "" {
		lexicon = an.config.DefaultLexicon
		log.Info().Msgf("gcg file had no lexicon, so using default lexicon %v",
			lexicon)
	}
	boardLayout, ldName := game.HistoryToVariant(history)
	rules, err := runner.NewAIGameRules(an.config, boardLayout, lexicon, ldName)
	if err != nil {
		return err
	}
	g, err := game.NewFromHistory(history, rules, 0)
	if err != nil {
		return err
	}
	an.game, err = runner.NewAIGameRunnerFromGame(g, an.config)
	if err != nil {
		return err
	}
	an.game.SetChallengeRule(pb.ChallengeRule_DOUBLE)
	return nil
}

func AnalyzeMove(g *runner.AIGameRunner, evt *pb.GameEvent) {
	moves := g.GenerateMoves(1)
	m := moves[0]
	fmt.Println("Move:", evt)
	fmt.Println("Generated move:", m.ShortDescription())
}

func AnalyzeGameTurn(g *runner.AIGameRunner, turn int) {
	hist := g.History()
	evt := hist.Events[turn]
	err := g.PlayToTurn(turn)
	if err != nil {
		panic(err)
	}
	AnalyzeMove(g, evt)
}

func AnalyzeFullGame(g *runner.AIGameRunner) {
	hist := g.History()
	nturns := len(hist.Events)
	var p string
	for i := 0; i < nturns; i++ {
		evt := hist.Events[i]
		if p != evt.Nickname {
			p = evt.Nickname
			AnalyzeGameTurn(g, i)
		}
	}
}

func AnalyzeGCG(conf *config.Config, filepath string) {
	a := Analyzer{config: conf, game: nil}
	a.loadGCG(filepath)
	AnalyzeFullGame(a.game)
}
