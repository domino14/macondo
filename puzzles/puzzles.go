package puzzles

import (
	"errors"
	"fmt"
	"regexp"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

const defaultMargin = 10.0

// tagFn is the type for all puzzle tag functions.
type tagFn func(ctx *tagCtx) (bool, pb.PuzzleTag)

// tagCtx holds per-turn state shared across tag functions and computeStats.
type tagCtx struct {
	g            *game.Game
	moves        []*move.Move // equity-sorted
	equityMargin float64
	scoreMargin  float64

	// lazy cached fields
	wordsFormedCache []tilemapping.MachineWord
	wordsFormedErr   error
	wordsDone        bool

	bestByScore  *move.Move
	secondScore  int
	scoreDone    bool
}

func newTagCtx(g *game.Game, moves []*move.Move, req *pb.PuzzleGenerationRequest) *tagCtx {
	em := req.GetEquityMargin()
	if em == 0 {
		em = defaultMargin
	}
	sm := req.GetScoreMargin()
	if sm == 0 {
		sm = defaultMargin
	}
	return &tagCtx{g: g, moves: moves, equityMargin: em, scoreMargin: sm}
}

func (ctx *tagCtx) getWords() ([]tilemapping.MachineWord, error) {
	if !ctx.wordsDone {
		ctx.wordsFormedCache, ctx.wordsFormedErr = ctx.g.ValidateMove(ctx.moves[0])
		ctx.wordsDone = true
	}
	return ctx.wordsFormedCache, ctx.wordsFormedErr
}

func (ctx *tagCtx) getBestByScore() (*move.Move, int) {
	if !ctx.scoreDone {
		ctx.bestByScore, ctx.secondScore = topTwoByScore(ctx.moves)
		ctx.scoreDone = true
	}
	return ctx.bestByScore, ctx.secondScore
}

var PuzzleFunctions = []tagFn{
	EquityPuzzle,
	BingoPuzzle,
	OnlyBingoPuzzle,
	BlankBingoPuzzle,
	NonBingoPuzzle,
	PowerTilePuzzle,
	BingoNineOrAbovePuzzle,
	CELOnlyPuzzle,
	PointsPuzzle,
}

func CreatePuzzlesFromGame(conf *config.Config, eqLossLimit int, g *game.Game, req *pb.PuzzleGenerationRequest) ([]*pb.PuzzleCreationResponse, error) {
	evts := g.History().Events
	puzzles := []*pb.PuzzleCreationResponse{}
	err := validatePuzzleGenerationRequest(req)
	if err != nil {
		return nil, err
	}
	totalEquityLoss := 0.0
	puzzleCalc, err := equity.NewCombinedStaticCalculator(g.LexiconName(), conf, "", "")
	if err != nil {
		return nil, err
	}

	eqCalcs := []equity.EquityCalculator{puzzleCalc}
	for evtIdx, evt := range evts {
		if evt.Type != pb.GameEvent_TILE_PLACEMENT_MOVE &&
			evt.Type != pb.GameEvent_EXCHANGE &&
			evt.Type != pb.GameEvent_PASS {
			continue
		}
		err := g.PlayToTurn(evtIdx)
		if err != nil {
			return nil, err
		}
		// Don't create endgame puzzles for now
		if g.Bag().TilesRemaining() < 7 {
			continue
		}

		player, err := turnplayer.NewAIStaticTurnPlayerFromGame(g, conf, eqCalcs)
		if err != nil {
			return nil, err
		}

		moves := player.GenerateMoves(1000000)

		// Let's keep a running tally of equity loss for this game.
		topEquity := moves[0].Equity()
		madeMove, err := game.MoveFromEvent(evt, g.Alphabet(), g.Board())
		if err != nil {
			return nil, err
		}
		player.AssignEquity([]*move.Move{madeMove}, g.Board(), g.Bag(), nil)
		totalEquityLoss += (topEquity - madeMove.Equity())

		if totalEquityLoss > float64(eqLossLimit) {
			log.Info().Str("gid", g.Uid()).Float64("eqloss", totalEquityLoss).Msg("too much equity loss")
			return nil, nil
		}

		ctx := newTagCtx(g, moves, req)

		tags := []pb.PuzzleTag{}
		for _, fn := range PuzzleFunctions {
			turnIsPuzzleType, tag := fn(ctx)
			if turnIsPuzzleType {
				tags = append(tags, tag)
			}
		}

		stats := computeStats(ctx)

		for _, bucket := range req.Buckets {
			if tagsFitInBucket(tags, bucket) {
				puzzles = append(puzzles, &pb.PuzzleCreationResponse{
					GameId:      g.Uid(),
					TurnNumber:  int32(evtIdx),
					Answer:      g.EventFromMove(moves[0]),
					BucketIndex: int32(bucket.Index),
					Tags:        tags,
					Stats:       stats,
				})
				break
			}
		}
	}
	return puzzles, nil
}

func InitializePuzzleGenerationRequest(req *pb.PuzzleGenerationRequest) error {
	err := validatePuzzleGenerationRequest(req)
	if err != nil {
		return err
	}
	for idx, b := range req.Buckets {
		b.Index = int32(idx)
	}
	return nil
}

func EquityPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	moves := ctx.moves
	return len(moves) >= 2 && moves[0].Equity() > moves[1].Equity()+ctx.equityMargin, pb.PuzzleTag_EQUITY
}

func BingoPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	return moveIsBingo(ctx.moves[0]), pb.PuzzleTag_BINGO
}

func OnlyBingoPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	tag := pb.PuzzleTag_ONLY_BINGO
	if len(ctx.moves) == 0 || !moveIsBingo(ctx.moves[0]) {
		return false, tag
	}
	for _, m := range ctx.moves[1:] {
		if moveIsBingo(m) && m.Action() == move.MoveTypePlay {
			return false, tag
		}
	}
	return true, tag
}

func BlankBingoPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	m := ctx.moves[0]
	return moveIsBingo(m) && moveContainsBlank(m), pb.PuzzleTag_BLANK_BINGO
}

func NonBingoPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	return !moveIsBingo(ctx.moves[0]), pb.PuzzleTag_NON_BINGO
}

// XXX: Must be expanded to other languages
func PowerTilePuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	ld := ctx.g.Bag().LetterDistribution()
	for _, tile := range ctx.moves[0].Tiles() {
		if ld.Score(tile) > 6 {
			return true, pb.PuzzleTag_POWER_TILE
		}
	}
	return false, pb.PuzzleTag_POWER_TILE
}

func BingoNineOrAbovePuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	m := ctx.moves[0]
	return moveIsBingo(m) && moveLength(m) >= 9, pb.PuzzleTag_BINGO_NINE_OR_ABOVE
}

func CELOnlyPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	m := ctx.moves[0]
	evt := ctx.g.EventFromMove(m)
	wordsFormed, err := ctx.getWords()
	if err != nil {
		log.Err(err).Msg("cel-only-validation-error")
		return false, pb.PuzzleTag_CEL_ONLY
	}
	evt.WordsFormed = convertToVisible(wordsFormed, ctx.g.Alphabet())
	isCEL, err := isCELEvent(evt, ctx.g.History(), ctx.g.Config())
	if err != nil {
		log.Err(err).Msg("cel-only-phony-error")
		return false, pb.PuzzleTag_CEL_ONLY
	}
	return isCEL, pb.PuzzleTag_CEL_ONLY
}

// PointsPuzzle fires when the answer is the top-scoring play by at least score_margin.
func PointsPuzzle(ctx *tagCtx) (bool, pb.PuzzleTag) {
	best, secondScore := ctx.getBestByScore()
	if best != ctx.moves[0] {
		return false, pb.PuzzleTag_POINTS
	}
	return float64(ctx.moves[0].Score()-secondScore) > ctx.scoreMargin, pb.PuzzleTag_POINTS
}

// topTwoByScore returns the highest-scoring move and the second-highest score.
// Does not modify the original slice order.
func topTwoByScore(moves []*move.Move) (*move.Move, int) {
	if len(moves) == 0 {
		return nil, 0
	}
	best := moves[0]
	secondScore := 0
	for _, m := range moves[1:] {
		if m.Score() > best.Score() {
			secondScore = best.Score()
			best = m
		} else if m.Score() > secondScore {
			secondScore = m.Score()
		}
	}
	return best, secondScore
}

func tagsFitInBucket(tags []pb.PuzzleTag, bucket *pb.PuzzleBucket) bool {
	tagMap := map[pb.PuzzleTag]bool{}
	for _, tag := range tags {
		tagMap[tag] = true
	}

	excludesMap := map[pb.PuzzleTag]bool{}
	for _, tag := range bucket.Excludes {
		excludesMap[tag] = true
	}

	for _, tag := range tags {
		if excludesMap[tag] {
			return false
		}
	}

	for _, includeTag := range bucket.Includes {
		if !tagMap[includeTag] {
			return false
		}
	}
	return true
}

func validatePuzzleGenerationRequest(req *pb.PuzzleGenerationRequest) error {
	if req == nil {
		return errors.New("puzzle generation request is nil")
	}
	if req.Buckets == nil {
		return errors.New("buckets are nil in puzzle generation request")
	}
	bucketEncryptions := map[string]bool{}
	numberOfTags := len(pb.PuzzleTag_name)
	for idx, bucket := range req.Buckets {
		be, err := validatePuzzleBucket(bucket, numberOfTags)
		if err != nil {
			return fmt.Errorf("error for bucket %d: %s", idx, err.Error())
		}
		if bucketEncryptions[be] {
			return fmt.Errorf("bucket %d is not unique", idx)
		}
		bucketEncryptions[be] = true
	}
	return nil
}

func validatePuzzleBucket(pzlBucket *pb.PuzzleBucket, numberOfTags int) (string, error) {
	// This function checks that tags appear at most once
	// across the Includes and Excludes fields. If the bucket
	// is valid, it returns the bucket encryption.

	// The puzzle bucket encryption can be viewed as a base-3
	// number where the digit in the int(tag) place is
	// 2 if the tag must be included, 1 if the tag must
	// be excluded, and 0 if the tag has no conditions.

	be := make([]byte, numberOfTags)
	tagValidationMap := map[pb.PuzzleTag]bool{}
	for _, tag := range pzlBucket.Includes {
		if tagValidationMap[tag] {
			return "", fmt.Errorf("invalid puzzle bucket, tag %s appears more than once", tag.String())
		}
		tagValidationMap[tag] = true
		be[int(tag)] = 2
	}
	for _, tag := range pzlBucket.Excludes {
		if tagValidationMap[tag] {
			return "", fmt.Errorf("invalid puzzle bucket, tag %s appears more than once", tag.String())
		}
		tagValidationMap[tag] = true
		be[int(tag)] = 1
	}
	return string(be), nil
}

func moveLength(m *move.Move) int {
	return len(m.Tiles())
}

func moveIsBingo(m *move.Move) bool {
	return m.TilesPlayed() == game.RackTileLimit
}

func moveContainsBlank(m *move.Move) bool {
	for _, ml := range m.Tiles() {
		if ml.IsBlanked() {
			return true
		}
	}
	return false
}

func isCELEvent(event *pb.GameEvent, history *pb.GameHistory, cfg *config.Config) (bool, error) {
	kwg, err := kwg.GetKWG(cfg.WGLConfig(), "ECWL")
	if err != nil {
		return false, err
	}
	for _, word := range event.WordsFormed {
		phony, err := isPhony(kwg, word, history.Variant)
		if err != nil || phony {
			return false, err
		}
	}
	return true, nil
}

func isPhony(k *kwg.KWG, word, variant string) (bool, error) {
	lex := kwg.Lexicon{KWG: *k}
	machineWord, err := tilemapping.ToMachineWord(word, lex.GetAlphabet())
	if err != nil {
		return false, err
	}
	var valid bool
	switch string(variant) {
	case string(game.VarWordSmog):
		valid = lex.HasAnagram(machineWord)
	default:
		valid = lex.HasWord(machineWord)
	}
	return !valid, nil
}

func convertToVisible(words []tilemapping.MachineWord, alph *tilemapping.TileMapping) []string {
	uvstrs := make([]string, len(words))
	for idx, w := range words {
		uvstrs[idx] = w.UserVisible(alph)
	}
	return uvstrs
}

var lexre = regexp.MustCompile(`(.* lex )([A-Za-z\d]+)(;.*)`)

// IsEquityPuzzleStillValid returns a boolean indicating whether an equity puzzle is stll
// valid given a new lexicon. The old answer must still be the clear winner.
func IsEquityPuzzleStillValid(conf *config.Config, g *game.Game, turnNumber int,
	answer *pb.GameEvent, updatedLexiconName string) (bool, error) {
	// Recalculate equity using the updated lexicon
	puzzleCalc, err := equity.NewCombinedStaticCalculator(updatedLexiconName, conf, "", "")
	if err != nil {
		return false, err
	}
	// add the rack to the game event. It is saved without a rack.

	err = g.PlayToTurn(turnNumber)
	if err != nil {
		return false, err
	}
	// In the future puzzles should just use cgp only.
	cgpRepr := g.ToCGP(false)
	cgpRepr = lexre.ReplaceAllString(cgpRepr, "${1}"+updatedLexiconName+"${3}")
	newGame, err := cgp.ParseCGP(conf, cgpRepr)
	if err != nil {
		return false, err
	}

	player, err := turnplayer.NewAIStaticTurnPlayerFromGame(newGame.Game, conf, []equity.EquityCalculator{puzzleCalc})
	if err != nil {
		return false, err
	}
	// Calculate anchors and cross-sets:
	player.RecalculateBoard()

	moves := player.GenerateMoves(1000000)
	ctx := newTagCtx(newGame.Game, moves, &pb.PuzzleGenerationRequest{})
	ok, _ := EquityPuzzle(ctx)

	newAnsEvt := g.EventFromMove(moves[0])
	return ok &&
		// for legacy reasons the new answer event would contain a rack but
		// the one that comes in (answer) might or might not, depending on whether
		// it was loaded from the database (which strips the rack :( )
		newAnsEvt.Position == answer.Position &&
		newAnsEvt.PlayedTiles == answer.PlayedTiles &&
		newAnsEvt.Exchanged == answer.Exchanged &&
		newAnsEvt.Score == answer.Score &&
		newAnsEvt.IsBingo == answer.IsBingo &&
		newAnsEvt.Type == answer.Type, nil
}
