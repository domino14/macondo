package automatic

import (
	"fmt"
	"math"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/wmp"
)

// WMPAgreementResult holds the results of a WMP agreement test.
type WMPAgreementResult struct {
	GamesPlayed   int
	TurnsPlayed   int
	Disagreements int
	Details       []string
}

// RunWMPAgreementTest plays numGames deterministic games, comparing the
// top play produced by the non-WMP movegen against the top play produced
// by the WMP movegen on every turn. Both must match by ShortDescription,
// score, and equity (within 1e-6).
//
// This exercises exactly the path Simmer's rollout loop drives:
// SetPlayRecorderTopPlay (shadow enabled), addExchange=true, with an
// equity calculator wired in. It broadens the assertions of
// montecarlo/wmp_equivalence_test.go across diverse positions drawn
// from real deterministic play rather than a fixed CGP catalogue.
//
// Games advance along the noWMP generator's top move (the canonical
// path), so a disagreement is reported at the earliest turn where the
// two generators diverge. The game then continues along the noWMP
// choice, which keeps the seed->position mapping reproducible across
// branches and across WMP-on/off runs.
func RunWMPAgreementTest(cfg *config.Config, w *wmp.WMP, numGames int) (*WMPAgreementResult, error) {
	if w == nil {
		return nil, fmt.Errorf("WMP is required for the agreement test; got nil")
	}

	lexicon := cfg.GetString(config.ConfigDefaultLexicon)
	letterDist := cfg.GetString(config.ConfigDefaultLetterDistribution)

	rules, err := game.NewBasicGameRules(cfg, lexicon, board.CrosswordGameLayout,
		letterDist, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		return nil, fmt.Errorf("creating rules: %w", err)
	}

	gd, err := kwg.GetKWG(cfg.WGLConfig(), lexicon)
	if err != nil {
		return nil, fmt.Errorf("loading KWG: %w", err)
	}

	calc, err := equity.NewCombinedStaticCalculator(
		lexicon, cfg, "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, fmt.Errorf("creating equity calculator: %w", err)
	}
	calcs := []equity.EquityCalculator{calc}

	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "NoWMP"},
		{Nickname: "p2", RealName: "WMP"},
	}

	result := &WMPAgreementResult{}

	// Reusable move buffers. TopPlayOnlyRecorder writes into the
	// generator's single `winner` field, so Plays()[0] is a pointer
	// that gets overwritten on the next GenAll. Snapshot into stable
	// buffers before comparing or playing the move forward.
	topNoBuf := &move.Move{}
	topWMPBuf := &move.Move{}

	for gidx := 0; gidx < numGames; gidx++ {
		g, err := game.NewGame(rules, playerInfos)
		if err != nil {
			return nil, fmt.Errorf("creating game: %w", err)
		}

		// Deterministic seed from game index.
		seed := [32]byte{}
		seed[0] = byte(gidx)
		seed[1] = byte(gidx >> 8)
		seed[2] = byte(gidx >> 16)
		seed[3] = byte(gidx >> 24)
		g.SeedBag(seed)
		g.StartGame()

		ld := g.Bag().LetterDistribution()
		alph := g.Alphabet()

		turnNum := 0
		for g.Playing() == pb.PlayState_PLAYING {
			playerIdx := g.PlayerOnTurn()
			rackLetters := g.RackLettersFor(playerIdx)
			rackNo := tilemapping.RackFromString(rackLetters, alph)
			rackWMP := tilemapping.RackFromString(rackLetters, alph)

			// Gate exchanges on bag size to match the game's exchange
			// rule. Without this the generator happily emits exchange
			// candidates that PlayMove can't execute, and the
			// non-validating PlayMove(addToHistory=false) path below
			// ends up calling bag.Exchange on a near-empty bag and
			// failing deep inside the tile draw.
			addExchange := g.Bag().TilesRemaining() >= game.DefaultExchangeLimit

			// Recreate the generators each turn. The WMP and non-WMP
			// paths carry internal state (shadow, wmpMoveGen,
			// leavemap) that we want to start clean for every
			// comparison so any divergence surfaces at its earliest
			// cause rather than at a downstream symptom.
			genNo := movegen.NewGordonGenerator(gd, g.Board(), ld)
			genNo.SetEquityCalculators(calcs)
			genNo.SetGame(g)
			genNo.SetPlayRecorderTopPlay()

			genWMP := movegen.NewGordonGenerator(gd, g.Board(), ld)
			genWMP.SetEquityCalculators(calcs)
			genWMP.SetGame(g)
			genWMP.SetPlayRecorderTopPlay()
			genWMP.SetWMP(w)

			// Each generator gets its own rack to eliminate any
			// shared-rack mutation effects.
			genNo.GenAll(rackNo, addExchange)
			playsNo := genNo.Plays()
			if len(playsNo) == 0 {
				return nil, fmt.Errorf("game %d turn %d: noWMP produced no plays", gidx, turnNum)
			}
			topNoBuf.CopyFrom(playsNo[0])

			genWMP.GenAll(rackWMP, addExchange)
			playsWMP := genWMP.Plays()
			if len(playsWMP) == 0 {
				return nil, fmt.Errorf("game %d turn %d: WMP produced no plays", gidx, turnNum)
			}
			topWMPBuf.CopyFrom(playsWMP[0])

			result.TurnsPlayed++

			mismatch := topNoBuf.ShortDescription() != topWMPBuf.ShortDescription() ||
				topNoBuf.Score() != topWMPBuf.Score() ||
				math.Abs(topNoBuf.Equity()-topWMPBuf.Equity()) > 1e-6
			if mismatch {
				detail := fmt.Sprintf("game %d turn %d:\n  noWMP: %s score=%d eq=%.4f\n  WMP:   %s score=%d eq=%.4f",
					gidx, turnNum,
					topNoBuf.ShortDescription(), topNoBuf.Score(), topNoBuf.Equity(),
					topWMPBuf.ShortDescription(), topWMPBuf.Score(), topWMPBuf.Equity())
				result.Details = append(result.Details, detail)
				result.Disagreements++
				log.Warn().Str("detail", detail).Msg("WMP disagreement")
			}

			// Advance by the noWMP top move (canonical path).
			if err := g.PlayMove(topNoBuf, false, 0); err != nil {
				return nil, fmt.Errorf("game %d turn %d: %w", gidx, turnNum, err)
			}
			turnNum++
		}

		result.GamesPlayed++
		if gidx > 0 && gidx%1000 == 0 {
			log.Info().
				Int("games", gidx).
				Int("turns", result.TurnsPlayed).
				Int("disagreements", result.Disagreements).
				Msg("progress")
		}
	}

	return result, nil
}
