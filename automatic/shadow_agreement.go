package automatic

import (
	"fmt"

	"github.com/domino14/word-golib/kwg"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// ShadowAgreementResult holds the results of a shadow agreement test.
type ShadowAgreementResult struct {
	GamesPlayed    int
	TurnsPlayed    int
	Disagreements  int
	Details        []string
}

// RunShadowAgreementTest plays numGames game pairs, comparing moves generated
// with and without shadow. Both algorithms must produce the same top move by
// score on every turn. Games use deterministic seeds for reproducibility.
func RunShadowAgreementTest(cfg *config.Config, numGames int) (*ShadowAgreementResult, error) {
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

	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "NoShadow"},
		{Nickname: "p2", RealName: "Shadow"},
	}

	result := &ShadowAgreementResult{}

	for gidx := 0; gidx < numGames; gidx++ {
		g, err := game.NewGame(rules, playerInfos)
		if err != nil {
			return nil, fmt.Errorf("creating game: %w", err)
		}

		// Deterministic seed from game index
		seed := [32]byte{}
		seed[0] = byte(gidx)
		seed[1] = byte(gidx >> 8)
		seed[2] = byte(gidx >> 16)
		seed[3] = byte(gidx >> 24)
		g.SeedBag(seed)
		g.StartGame()

		ld := g.Bag().LetterDistribution()
		genNS := movegen.NewGordonGenerator(gd, g.Board(), ld)
		genS := movegen.NewGordonGenerator(gd, g.Board(), ld)
		genS.SetShadowEnabled(true)

		turnNum := 0
		for g.Playing() == pb.PlayState_PLAYING {
			playerIdx := g.PlayerOnTurn()
			rack := g.RackFor(playerIdx)

			// Generate with no shadow
			genNS.SetPlayRecorder(movegen.AllPlaysRecorder)
			genNS.SetSortingParameter(movegen.SortByScore)
			playsNS := genNS.GenAll(rack, false)

			// Generate with shadow
			genS.SetPlayRecorder(movegen.AllPlaysRecorder)
			genS.SetSortingParameter(movegen.SortByScore)
			playsS := genS.GenAll(rack, false)

			result.TurnsPlayed++

			// Compare top move by score
			if len(playsNS) > 0 && len(playsS) > 0 {
				topNS := bestByScore(playsNS)
				topS := bestByScore(playsS)
				if topNS.Score() != topS.Score() {
					detail := fmt.Sprintf("game %d turn %d: score mismatch noshadow=%d(%s) shadow=%d(%s)",
						gidx, turnNum, topNS.Score(), topNS.ShortDescription(),
						topS.Score(), topS.ShortDescription())
					result.Details = append(result.Details, detail)
					result.Disagreements++
					log.Warn().Str("detail", detail).Msg("shadow disagreement")
				}
			}

			// Play the best move from the no-shadow generator
			bestPlay := playsNS[0]
			err := g.PlayMove(bestPlay, false, 0)
			if err != nil {
				return nil, fmt.Errorf("game %d turn %d: %w", gidx, turnNum, err)
			}
			turnNum++
		}

		result.GamesPlayed++
		if gidx > 0 && gidx%100 == 0 {
			log.Info().Int("games", gidx).Int("disagreements", result.Disagreements).Msg("progress")
		}
	}

	return result, nil
}

func bestByScore(plays []*move.Move) *move.Move {
	best := plays[0]
	for _, p := range plays[1:] {
		if p.Score() > best.Score() {
			best = p
		}
	}
	return best
}
