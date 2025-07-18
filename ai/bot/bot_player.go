package bot

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/magpie"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/preendgame"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/stats"
	"github.com/domino14/macondo/turnplayer"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"lukechampine.com/frand"
)

type BotConfig struct {
	config.Config
	PEGAdjustmentFile    string
	LeavesFile           string
	MinSimPlies          int
	StochasticStaticEval bool
	BotSpec              *pb.BotSpec
	// If UseOppRacksInAnalysis is true, will use opponent rack info for simulation/pre-endgames/etc
	UseOppRacksInAnalysis bool
}

type BotTurnPlayer struct {
	aiturnplayer.AIStaticTurnPlayer
	botType               pb.BotRequest_BotCode
	endgamer              *negamax.Solver
	preendgamer           *preendgame.Solver
	simmer                *montecarlo.Simmer
	simmerCalcs           []equity.EquityCalculator
	simThreads            int
	minSimPlies           int
	cfg                   *BotConfig
	lastMoves             []*move.Move
	inferencer            *rangefinder.RangeFinder
	lastCalculatedDetails string
	pertinentLogs         []string
	magpie                *magpie.Magpie
}

func NewBotTurnPlayer(conf *BotConfig, opts *turnplayer.GameOptions,
	players []*pb.PlayerInfo, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {

	opts.SetDefaults(&conf.Config)
	rules, err := game.NewBasicGameRules(&conf.Config, opts.Lexicon.Name, opts.BoardLayoutName,
		opts.Lexicon.Distribution, game.CrossScoreAndSet, opts.Variant)

	if err != nil {
		return nil, err
	}
	p, err := turnplayer.BaseTurnPlayerFromRules(opts, players, rules)
	if err != nil {
		return nil, err
	}
	return addBotFields(p, conf, botType)
}

func NewBotTurnPlayerFromGame(g *game.Game, conf *BotConfig, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {
	gr := &turnplayer.BaseTurnPlayer{Game: g}
	return addBotFields(gr, conf, botType)
}

func addBotFields(p *turnplayer.BaseTurnPlayer, conf *BotConfig, botType pb.BotRequest_BotCode) (*BotTurnPlayer, error) {
	var calculators []equity.EquityCalculator

	if botType == pb.BotRequest_NO_LEAVE_BOT {
		calculators = []equity.EquityCalculator{equity.NewNoLeaveCalculator()}
	} else {
		c1, err := equity.NewExhaustiveLeaveCalculator(
			p.LexiconName(), &conf.Config, conf.LeavesFile)
		if err != nil {
			return nil, err
		}
		c2 := &equity.OpeningAdjustmentCalculator{}
		c3, err := equity.NewPreEndgameAdjustmentCalculator(
			&conf.Config, p.LexiconName(), conf.PEGAdjustmentFile)
		if err != nil {
			return nil, err
		}
		c4 := &equity.EndgameAdjustmentCalculator{}
		calculators = []equity.EquityCalculator{c1, c2, c3, c4}
	}
	aip, err := aiturnplayer.AddAIFields(p, &conf.Config, calculators)
	if err != nil {
		return nil, err
	}
	btp := &BotTurnPlayer{
		AIStaticTurnPlayer: *aip,
		botType:            botType,
		cfg:                conf,
	}

	// If it is a simming bot, add more fields.
	if hasSimming(botType, conf.BotSpec) {
		log.Info().Msg("adding fields for simmer")
		leaveFile := "" // use default
		if p.Rules().BoardName() == board.SuperCrosswordGameLayout {
			leaveFile = "super-leaves.klv2"
		}
		c, err := equity.NewCombinedStaticCalculator(
			p.LexiconName(), p.Config(), leaveFile, equity.PEGAdjustmentFilename)
		if err != nil {
			return nil, err
		}
		btp.simmer = &montecarlo.Simmer{}
		btp.simmerCalcs = []equity.EquityCalculator{c}
		if conf.MinSimPlies > 0 {
			btp.SetMinSimPlies(conf.MinSimPlies)
		}
		// botspec overrides the default conf's minsimplies, if set.
		if conf.BotSpec != nil && conf.BotSpec.Params.MinSimPlies > 0 {
			btp.SetMinSimPlies(int(conf.BotSpec.Params.MinSimPlies))
		}
	}
	if HasEndgame(botType, conf.BotSpec) {
		log.Info().Msg("adding fields for endgame")
		btp.endgamer = &negamax.Solver{}
	}
	if HasPreendgame(botType, conf.BotSpec) {
		log.Info().Msg("adding fields for pre-endgame")
		btp.preendgamer = &preendgame.Solver{}
	}
	if HasInfer(botType, conf.BotSpec) {
		log.Info().Msg("adding fields for rangefinder")
		btp.inferencer = &rangefinder.RangeFinder{}
	}

	return btp, nil
}

func (p *BotTurnPlayer) GenerateMoves(numPlays int) []*move.Move {
	curRack := p.RackFor(p.PlayerOnTurn())
	oppRack := p.RackFor(p.NextPlayer())
	gen := p.MoveGenerator()
	// in case we don't have full rack info:
	unseen := int(oppRack.NumTiles()) + p.Bag().TilesRemaining()
	exchAllowed := unseen-game.RackTileLimit >= p.ExchangeLimit()
	gen.SetMaxCanExchange(game.MaxCanExchange(unseen-game.RackTileLimit, p.ExchangeLimit()))
	gen.GenAll(curRack, exchAllowed)
	plays := gen.(*movegen.GordonGenerator).Plays()

	p.AssignEquity(plays, p.Board(), p.Bag(), oppRack)
	if numPlays == 1 {
		// Plays aren't sorted yet
		sort.Slice(plays, func(i, j int) bool {
			return plays[i].Equity() > plays[j].Equity()
		})
		return []*move.Move{filter(p.Config(), p.Game, curRack, plays, p.botType, p.LexiconName())}
	}

	return p.TopPlays(plays, numPlays)
}

type moveEval struct {
	move *move.Move
	eval float32
	idx  int
}

// ChooseMoveWithExploration selects a move using a temperature-controlled
// softmax distribution. Assume that moves are already sorted from
// best to worst.
func ChooseMoveWithExploration(moves []*move.Move, temperature float64) (*move.Move, error) {
	if len(moves) == 0 {
		return nil, fmt.Errorf("moves slice cannot be empty")
	}

	// For pure exploitation (temperature = 0 or close to it), just find the best move.
	if temperature < 1e-6 {
		return moves[0], nil
	}

	// --- Softmax Implementation ---

	// 1. Find the max score for numerical stability.
	// Subtracting the max score before exponentiating prevents large scores
	// from causing float64 overflow, without changing the final probabilities.
	maxScore := moves[0].Equity()

	// 2. Calculate the exponentiated scores divided by temperature and sum them up.
	sum := 0.0
	probabilities := make([]float64, len(moves))
	for i, m := range moves {
		// Apply temperature and subtract maxScore for stability
		prob := math.Exp((m.Equity() - maxScore) / temperature)
		probabilities[i] = prob
		sum += prob
	}

	// 3. Normalize to get the final probabilities
	for i := range probabilities {
		probabilities[i] /= sum
	}

	// --- Weighted Random Choice Implementation ---

	// 4. Create a cumulative distribution for sampling.
	// Example: [0.1, 0.8, 0.1] becomes [0.1, 0.9, 1.0]
	cdf := make([]float64, len(probabilities))
	cdf[0] = probabilities[0]
	for i := 1; i < len(probabilities); i++ {
		cdf[i] = cdf[i-1] + probabilities[i]
	}

	// 5. Pick a random number and find which "bucket" it falls into.
	r := frand.Float64()
	for i, c := range cdf {
		if r < c {
			return moves[i], nil
		}
	}

	// Fallback to the last move in case of floating point inaccuracies
	return moves[len(moves)-1], nil
}

func (p *BotTurnPlayer) Reset() {
	p.lastMoves = nil
	p.pertinentLogs = nil
}

func (p *BotTurnPlayer) GetPertinentLogs() []string {
	return p.pertinentLogs
}

func (p *BotTurnPlayer) AddLastMove(m *move.Move) {
	p.lastMoves = append(p.lastMoves, m)
}

func (p *BotTurnPlayer) bestMLStaticTurn() (*move.Move, error) {

	if p.Bag().TilesRemaining() == 0 {
		// The bag is empty. Let's use the HastyBot endgame algorithm.
		log.Debug().Msg("Using HastyBot endgame algorithm for fast ML bot")
		return p.GenerateMoves(1)[0], nil
	}
	// Fast ML bot uses a different method
	moves := p.GenerateMoves(50)

	if len(moves) == 1 {
		return moves[0], nil
	}
	var lc *equity.ExhaustiveLeaveCalculator
	for _, c := range p.Calculators() {
		if e, ok := c.(*equity.ExhaustiveLeaveCalculator); ok {
			lc = e
			break
		}
	}
	if lc == nil {
		return nil, errors.New("no ExhaustiveLeaveCalculator found for fast ML bot")
	}
	resp, err := p.MLEvaluateMoves(moves, lc, p.lastMoves)
	if err != nil {
		log.Error().Err(err).Msg("Failed to evaluate moves for fast ML bot")
		return nil, err
	}
	pairs := make([]moveEval, len(moves))
	for i, m := range moves {
		pairs[i] = moveEval{
			move: m,
			eval: resp.Value[i],
			idx:  i + 1, // Store original index for reference
		}
	}
	// Sort by evaluation in descending order
	sort.Slice(pairs, func(i, j int) bool {
		// If the evaluations are equal, prefer the move with more tiles played
		// This helps in the endgame.
		if stats.FuzzyEqual(float64(pairs[i].eval), float64(pairs[j].eval)) {
			return pairs[i].move.TilesPlayed() > pairs[j].move.TilesPlayed()
		}
		return pairs[i].eval > pairs[j].eval
	})
	return pairs[0].move, nil
}

func (p *BotTurnPlayer) BestPlay(ctx context.Context) (*move.Move, error) {
	switch p.botType {
	// HastyBot is handled in the GameRunner.
	case pb.BotRequest_FAST_ML_BOT:
		return p.bestMLStaticTurn()
	case pb.BotRequest_RANDOM_BOT_WITH_TEMPERATURE:
		// Random bot just picks a random move among top N (not currenetly configurable)
		moves := p.GenerateMoves(50)
		if len(moves) == 0 {
			return nil, errors.New("no moves available for random bot")
		}
		temperature := 0.0
		// // Choose more exploratory moves early in the game.
		if p.Bag().TilesRemaining() > 60 {
			temperature = 1.0
		}
		move, err := ChooseMoveWithExploration(moves, temperature)
		if err != nil {
			log.Error().Err(err).Msg("Failed to choose move with exploration for random bot")
			return nil, err
		}
		return move, nil
	case pb.BotRequest_CUSTOM_BOT:
		if p.cfg.BotSpec == nil {
			return nil, errors.New("no bot spec provided for custom bot")
		}
		fallthrough
	default:
		return p.bestPlayByBotCapability(ctx)
	}

}

func (p *BotTurnPlayer) bestPlayByBotCapability(ctx context.Context) (*move.Move, error) {
	logger := zerolog.Ctx(ctx)
	var moves []*move.Move
	// First determine what stage of the game we are in.
	tr := p.Game.Bag().TilesRemaining()
	// We don't necessarily know the number of tiles on our opponent's rack.
	opp := p.Game.RackFor(p.Game.NextPlayer()).NumTiles()
	// If this is an annotated game, we may not have full rack info.
	unseen := int(opp) + tr
	// Assume our own rack is fully known, however. So if unseen == 7, the bag
	// is empty and we should assign the oppRack accordingly.
	useEndgame := false
	usePreendgame := false
	useMontecarlo := false
	useStaticAlgo := false
	endgamePlies := 0
	simPlies := 0

	if unseen <= 7 {
		if HasEndgame(p.botType, p.cfg.BotSpec) {
			useEndgame = true
		} else {
			// Use the static algorithm for the endgame if no endgame solver is available.
			useStaticAlgo = true
		}
		if tr > 0 {
			logger.Debug().Msg("assigning all unseen to opp")
			// bag is actually empty. Assign all of unseen to the opponent.
			mls := make([]tilemapping.MachineLetter, tr)
			err := p.Game.Bag().Draw(tr, mls)
			if err != nil {
				return nil, err
			}
			for _, t := range mls {
				p.Game.RackFor(p.Game.NextPlayer()).Add(t)
			}
		}
		// Just some sort of estimate
		endgamePlies = (int(float32(unseen)*1.5) + int(p.Game.RackFor(p.Game.PlayerOnTurn()).NumTiles()))
	} else if unseen > 7 && unseen <= 8 && HasPreendgame(p.botType, p.cfg.BotSpec) {
		usePreendgame = true
	} else if unseen > 8 && unseen <= 14 && hasSimming(p.botType, p.cfg.BotSpec) {
		moves = p.GenerateMoves(80)
		simPlies = unseen
		useMontecarlo = true
	} else if hasSimming(p.botType, p.cfg.BotSpec) {
		moves = p.GenerateMoves(40)
		if p.minSimPlies > 2 {
			simPlies = p.minSimPlies
		} else {
			simPlies = 2
		}
		useMontecarlo = true
	}
	simThreads := p.simThreads
	if p.simThreads == 0 {
		simThreads = p.simmer.Threads()
	}

	logger.Info().
		Str("onTurn", p.Game.NickOnTurn()).
		Int("playerID", p.Game.PlayerOnTurn()).
		Str("botType", p.botType.String()).
		Int("simPlies", simPlies).
		Int("simThreads", simThreads).
		Int("endgamePlies", endgamePlies).
		Bool("useEndgame", useEndgame).
		Bool("usePreendgame", usePreendgame).
		Bool("useMontecarlo", useMontecarlo).
		Bool("useStaticAlgo", useStaticAlgo).
		Int("unseen", unseen).
		Bool("useKnownOppRack", p.cfg.UseOppRacksInAnalysis).
		Bool("stochasticStaticEval", p.cfg.StochasticStaticEval).
		Int("consideredMoves", len(moves)).Msg("best-play-by-bot-capability")

	if useEndgame {
		return endGameBest(ctx, p, endgamePlies)
	} else if usePreendgame {
		return preendgameBest(ctx, p)
	} else if useMontecarlo {
		if p.cfg.BotSpec != nil && p.cfg.BotSpec.Params.SimUseMagpie {
			m, rawOutput, err := montecarloBestWithMagpie(ctx, p, simPlies, moves)
			if err != nil {
				return nil, err
			}
			p.pertinentLogs = append(p.pertinentLogs, rawOutput)
			return m, nil
		}
		return monteCarloBest(ctx, p, simPlies, moves)
	}
	// Otherwise, just use the static best play?
	return p.GenerateMoves(1)[0], nil
}

// Returns a string summary of details from a previous call to BestPlay.
func (p *BotTurnPlayer) BestPlayDetails(ctx context.Context) string {
	if hasSimming(p.botType, p.cfg.BotSpec) ||
		HasEndgame(p.botType, p.cfg.BotSpec) ||
		HasInfer(p.botType, p.cfg.BotSpec) ||
		HasPreendgame(p.botType, p.cfg.BotSpec) {

		return p.lastCalculatedDetails
	} else {
		return "(No summary)"
	}
}

func (p *BotTurnPlayer) SetEquityCalculators(calcs []equity.EquityCalculator) {
	p.SetCalculators(calcs)
}

func (p *BotTurnPlayer) GetBotType() pb.BotRequest_BotCode {
	return p.botType
}

func (p *BotTurnPlayer) SetBotType(b pb.BotRequest_BotCode) {
	p.botType = b
}

func (p *BotTurnPlayer) SetSimThreads(t int) {
	p.simThreads = t
}

func (p *BotTurnPlayer) SetMinSimPlies(t int) {
	p.minSimPlies = t
}
