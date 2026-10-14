package turnplayer

import (
	"context"
	"sort"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/samber/lo"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/turnplayer"
)

// StaticPassPenalty is a penalty for passing without making a move.
const StaticPassPenalty = -50.0

type AIStaticTurnPlayer struct {
	turnplayer.BaseTurnPlayer

	calculators []equity.EquityCalculator
	eval        staticEval

	gen movegen.MoveGenerator
	cfg *config.Config
}

func NewAIStaticTurnPlayer(conf *config.Config, opts *turnplayer.GameOptions,
	players []*pb.PlayerInfo, calculators []equity.EquityCalculator) (*AIStaticTurnPlayer, error) {

	opts.SetDefaults(conf)
	rules, err := game.NewBasicGameRules(conf, opts.Lexicon.Name, opts.BoardLayoutName,
		opts.Lexicon.Distribution, game.CrossScoreAndSet, opts.Variant)

	if err != nil {
		return nil, err
	}
	p, err := turnplayer.BaseTurnPlayerFromRules(opts, players, rules)
	if err != nil {
		return nil, err
	}
	return AddAIFields(p, conf, calculators)
}

func NewAIStaticTurnPlayerFromGame(g *game.Game, conf *config.Config, calculators []equity.EquityCalculator) (*AIStaticTurnPlayer, error) {
	gr := &turnplayer.BaseTurnPlayer{Game: g}
	return AddAIFields(gr, conf, calculators)
}

func AddAIFields(p *turnplayer.BaseTurnPlayer, conf *config.Config, calculators []equity.EquityCalculator) (*AIStaticTurnPlayer, error) {
	gd, err := kwg.GetKWG(conf.WGLConfig(), p.LexiconName(), kwg.WithDistribution(p.Rules().LetterDistributionName()))
	if err != nil {
		return nil, err
	}
	gen := movegen.NewGordonGenerator(gd, p.Board(), p.Bag().LetterDistribution())
	if game.IsWordSmog(p.Rules().Variant()) {
		gen.SetWordSmog(p.Rules().AlphaDawg())
	}
	gen.SetEquityCalculators(calculators)
	// The top-N recorder computes equity inline and needs the game for bag and
	// opponent-rack state. Callers that swap in their own game copy (the
	// simmer, the autoplay runner) still override this afterwards.
	gen.SetGame(p.Game)
	ret := &AIStaticTurnPlayer{*p, calculators, staticEvalFor(calculators), gen, conf}
	return ret, nil
}

func (p *AIStaticTurnPlayer) AssignEquity(plays []*move.Move, board *board.GameBoard, bag *tilemapping.Bag, oppRack *tilemapping.Rack) {
	for _, m := range plays {
		if m.Action() == move.MoveTypePass {
			m.SetEquity(StaticPassPenalty)
		} else {
			m.SetEquity(lo.SumBy(p.calculators, func(c equity.EquityCalculator) float64 {
				return c.Equity(m, board, bag, oppRack)
			}))
		}
	}
}

func (p *AIStaticTurnPlayer) TopPlays(plays []*move.Move, ct int) []*move.Move {
	sort.Slice(plays, func(i, j int) bool {
		ei, ej := plays[i].Equity(), plays[j].Equity()
		if ei != ej {
			return ei > ej
		}
		return plays[i].TiebreaksBetter(plays[j])
	})
	if ct > len(plays) {
		ct = len(plays)
	}
	return plays[:ct]
}

func (p *AIStaticTurnPlayer) BestPlay(ctx context.Context) (*move.Move, error) {
	return p.GenerateMoves(1)[0], nil
}

// SetLastMoves does nothing for now.
func (p *AIStaticTurnPlayer) SetLastMoves(m []*move.Move) { panic("I am not used currently") }

func (p *AIStaticTurnPlayer) AddLastMove(m *move.Move) { panic("I am not used currently") }

// GenerateMoves generates the best numPlays moves for the player on turn,
// best first.
func (p *AIStaticTurnPlayer) GenerateMoves(numPlays int) []*move.Move {
	return p.GenerateMovesFor(p.Game, numPlays)
}

// GenerateMovesFor generates the best numPlays moves for the player on turn
// in g -- this player's game or a copy of it -- ranked by this player's
// static evaluation.
func (p *AIStaticTurnPlayer) GenerateMovesFor(g *game.Game, numPlays int) []*move.Move {
	plays, err := magpieGenerate(g, p.eval, numPlays)
	if err != nil {
		panic(err)
	}
	return plays
}

func (p *AIStaticTurnPlayer) MoveGenerator() movegen.MoveGenerator {
	return p.gen
}

func (p *AIStaticTurnPlayer) Calculators() []equity.EquityCalculator {
	return p.calculators
}

func (p *AIStaticTurnPlayer) SetCalculators(c []equity.EquityCalculator) {
	p.calculators = c
	p.eval = staticEvalFor(c)
}

func (p *AIStaticTurnPlayer) GetBotType() pb.BotRequest_BotCode {
	// No bot associated with just a plain AIStaticTurnPlayer
	return pb.BotRequest_UNKNOWN
}
