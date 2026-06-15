package shell

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/domino14/macondo/automatic"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/puzzles"
)

// puzzlegen generates puzzles from game sources and prints matching ones.
//
// Usage:
//
//	puzzlegen woogles <gameID> [<gameID>...] [-filter EXPR] [options]
//	puzzlegen gcg <path> [<path>...] [-filter EXPR] [options]
//	puzzlegen xt <gameID> [<gameID>...] [-filter EXPR] [options]
//	puzzlegen selfplay [-numgames N] [-lexicon NAME] [-filter EXPR] [options]
func (sc *ShellController) puzzlegen(cmd *shellcmd) (*Response, error) {
	if len(cmd.args) == 0 {
		return nil, fmt.Errorf("usage: puzzlegen <woogles|gcg|xt|selfplay> [args...] [options]")
	}

	source := strings.ToLower(cmd.args[0])
	sourceArgs := cmd.args[1:]

	filterExpr := cmd.options.String("filter")
	lexiconOverride := cmd.options.String("lexicon")

	equityMargin := 10.0
	if s := cmd.options.String("equity-margin"); s != "" {
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid -equity-margin %q: %w", s, err)
		}
		equityMargin = f
	}
	scoreMargin := 10.0
	if s := cmd.options.String("score-margin"); s != "" {
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid -score-margin %q: %w", s, err)
		}
		scoreMargin = f
	}
	eqLossLimit, err := cmd.options.IntDefault("eqloss-limit", 1000)
	if err != nil {
		return nil, err
	}
	numGames, err := cmd.options.IntDefault("numgames", 1)
	if err != nil {
		return nil, err
	}

	var filter predicate
	if filterExpr != "" {
		filter, err = compileFilter(filterExpr)
		if err != nil {
			return nil, fmt.Errorf("invalid -filter: %w", err)
		}
	}

	req := &pb.PuzzleGenerationRequest{
		EquityMargin: equityMargin,
		ScoreMargin:  scoreMargin,
		Buckets:      []*pb.PuzzleBucket{{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}}},
	}
	if err := puzzles.InitializePuzzleGenerationRequest(req); err != nil {
		return nil, err
	}

	totalAll, totalMatched := 0, 0
	tagCounts := map[string]int{}

	switch source {
	case "woogles":
		if len(sourceArgs) == 0 {
			return nil, fmt.Errorf("puzzlegen woogles requires at least one game ID")
		}
		for _, gameID := range sourceArgs {
			sc.showMessage(fmt.Sprintf("Fetching %s...", gameID))
			gh, err := sc.loadGameHistoryFromWoogles(gameID)
			if err != nil {
				sc.showMessage(fmt.Sprintf("  [%s] fetch error: %v", gameID, err))
				continue
			}
			all, matched := sc.processPuzzleHistory(gh, lexiconOverride, eqLossLimit, req, filter, tagCounts)
			sc.showMessage(fmt.Sprintf("  [%s] %d/%d puzzles matched", gameID, matched, all))
			totalAll += all
			totalMatched += matched
		}

	case "gcg":
		if len(sourceArgs) == 0 {
			return nil, fmt.Errorf("puzzlegen gcg requires at least one file path")
		}
		for _, path := range sourceArgs {
			gh, err := sc.loadGameHistoryFromFile(path)
			if err != nil {
				sc.showMessage(fmt.Sprintf("  [%s] error: %v", path, err))
				continue
			}
			all, matched := sc.processPuzzleHistory(gh, lexiconOverride, eqLossLimit, req, filter, tagCounts)
			sc.showMessage(fmt.Sprintf("  [%s] %d/%d puzzles matched", path, matched, all))
			totalAll += all
			totalMatched += matched
		}

	case "xt":
		if len(sourceArgs) == 0 {
			return nil, fmt.Errorf("puzzlegen xt requires at least one cross-tables game ID")
		}
		for _, gameID := range sourceArgs {
			sc.showMessage(fmt.Sprintf("Fetching cross-tables game %s...", gameID))
			gh, err := sc.loadGameHistoryFromCrossTables(gameID)
			if err != nil {
				sc.showMessage(fmt.Sprintf("  [%s] fetch error: %v", gameID, err))
				continue
			}
			all, matched := sc.processPuzzleHistory(gh, lexiconOverride, eqLossLimit, req, filter, tagCounts)
			sc.showMessage(fmt.Sprintf("  [%s] %d/%d puzzles matched", gameID, matched, all))
			totalAll += all
			totalMatched += matched
		}

	case "selfplay":
		lexicon := lexiconOverride
		if lexicon == "" {
			lexicon = sc.config.GetString(config.ConfigDefaultLexicon)
		}
		// Temporarily set the lexicon on config so NewGameRunner picks it up.
		origLexicon := sc.config.GetString(config.ConfigDefaultLexicon)
		if lexiconOverride != "" {
			sc.config.Set(config.ConfigDefaultLexicon, lexicon)
			defer sc.config.Set(config.ConfigDefaultLexicon, origLexicon)
		}
		for i := 0; i < numGames; i++ {
			sc.showMessage(fmt.Sprintf("Self-play game %d/%d...", i+1, numGames))
			r := automatic.NewGameRunner(nil, sc.config)
			if err := r.CompVsCompStatic(true); err != nil {
				sc.showMessage(fmt.Sprintf("  [game %d] error: %v", i+1, err))
				continue
			}
			g := r.Game()
			pzls, err := puzzles.CreatePuzzlesFromGame(sc.config, eqLossLimit, g, req)
			if err != nil {
				sc.showMessage(fmt.Sprintf("  [game %d] puzzle error: %v", i+1, err))
				continue
			}
			matched := 0
			for _, pz := range pzls {
				totalAll++
				if filter == nil || filter(pz) {
					totalMatched++
					matched++
					for _, t := range pz.GetTags() {
						tagCounts[t.String()]++
					}
					sc.showMessage(pgFormatPuzzle(pz))
				}
			}
			sc.showMessage(fmt.Sprintf("  [game %d] %d/%d puzzles matched", i+1, matched, len(pzls)))
		}

	default:
		return nil, fmt.Errorf("unknown source %q; use woogles, gcg, xt, or selfplay", source)
	}

	sc.showMessage(fmt.Sprintf("\n━━━ SUMMARY: %d matched / %d total ━━━", totalMatched, totalAll))
	if len(tagCounts) > 0 {
		sc.showMessage("Tag distribution (matched puzzles):")
		type kv struct {
			k string
			v int
		}
		sorted := make([]kv, 0, len(tagCounts))
		for k, v := range tagCounts {
			sorted = append(sorted, kv{k, v})
		}
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].k < sorted[j].k })
		for _, pair := range sorted {
			sc.showMessage(fmt.Sprintf("  %-42s %d", pair.k, pair.v))
		}
	}
	return nil, nil
}

// processPuzzleHistory converts one GameHistory into puzzles and streams matches.
func (sc *ShellController) processPuzzleHistory(
	gh *pb.GameHistory, lexiconOverride string, eqLossLimit int,
	req *pb.PuzzleGenerationRequest, filter predicate, tagCounts map[string]int,
) (all, matched int) {
	lexicon := gh.GetLexicon()
	if lexiconOverride != "" {
		lexicon = lexiconOverride
	}
	if lexicon == "" {
		lexicon = sc.config.GetString(config.ConfigDefaultLexicon)
	}
	letterDist := strings.ToLower(sc.config.GetString(config.ConfigDefaultLetterDistribution))
	if letterDist == "" {
		letterDist = "english"
	}

	// VOID triggers lexicon word-checking during replay; override so phonies don't block puzzle gen.
	if gh.GetChallengeRule() == pb.ChallengeRule_VOID {
		gh.ChallengeRule = pb.ChallengeRule_DOUBLE
	}

	rules, err := game.NewBasicGameRules(sc.config, lexicon, board.CrosswordGameLayout, letterDist, game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		sc.showMessage(fmt.Sprintf("  rules error: %v", err))
		return
	}
	g, err := game.NewFromHistory(gh, rules, 0)
	if err != nil {
		sc.showMessage(fmt.Sprintf("  history error: %v", err))
		return
	}
	pzls, err := puzzles.CreatePuzzlesFromGame(sc.config, eqLossLimit, g, req)
	if err != nil {
		sc.showMessage(fmt.Sprintf("  puzzle error: %v", err))
		return
	}
	all = len(pzls)
	for _, pz := range pzls {
		if filter == nil || filter(pz) {
			matched++
			for _, t := range pz.GetTags() {
				tagCounts[t.String()]++
			}
			sc.showMessage(pgFormatPuzzle(pz))
		}
	}
	return
}

// pgFormatPuzzle formats a single puzzle response for display.
func pgFormatPuzzle(pz *pb.PuzzleCreationResponse) string {
	s := pz.GetStats()
	ans := pz.GetAnswer()
	line := fmt.Sprintf("  turn%3d  %-5s %-18s  score=%3d  eq_adv=%+6.2f  score_adv=%+4d",
		pz.GetTurnNumber(),
		ans.GetPosition(),
		ans.GetPlayedTiles(),
		s.GetScore(),
		s.GetEquityAdvantage(),
		s.GetScoreAdvantage(),
	)
	line += fmt.Sprintf("  words=%d  maxCross=%d", s.GetWordsFormed(), s.GetMaxCrossWordLength())
	if s.GetBonusSquaresCovered() > 0 {
		line += fmt.Sprintf("  bonus=%s", pgBonusSummary(s))
	}
	if s.GetLongestHookedWordLength() > 0 {
		line += fmt.Sprintf("  hook=%d", s.GetLongestHookedWordLength())
	}
	if s.GetLongestExtendedWordLength() > 0 {
		line += fmt.Sprintf("  ext=%d", s.GetLongestExtendedWordLength())
	}
	if s.GetMaxPlayedThroughTileScore() > 0 {
		line += fmt.Sprintf("  thruScore=%d", s.GetMaxPlayedThroughTileScore())
	}
	if s.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword() > 0 {
		line += fmt.Sprintf("  DLS/TLS×cross=%d", s.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword())
	}
	line += fmt.Sprintf("  rack_max=%d", s.GetMaxRackTileScore())
	line += fmt.Sprintf("\n           tags: %s", strings.Join(pgTagNames(pz.GetTags()), " "))
	return line
}

func pgBonusSummary(s *pb.PuzzleStats) string {
	var parts []string
	if s.GetTwsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("TWS×%d", s.GetTwsCovered()))
	}
	if s.GetDwsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("DWS×%d", s.GetDwsCovered()))
	}
	if s.GetTlsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("TLS×%d", s.GetTlsCovered()))
	}
	if s.GetDlsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("DLS×%d", s.GetDlsCovered()))
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, " ")
}

func pgTagNames(tags []pb.PuzzleTag) []string {
	out := make([]string, len(tags))
	for i, t := range tags {
		out[i] = t.String()
	}
	return out
}

// ─── filter expression compiler ──────────────────────────────────────────────

// predicate is a compiled filter expression.
type predicate func(*pb.PuzzleCreationResponse) bool

type pgTokKind int

const (
	pgTokIdent  pgTokKind = iota // field name or value
	pgTokOp                      // >=, <=, !=, =, >, <
	pgTokAnd                     // AND / and
	pgTokOr                      // OR  / or
	pgTokNot                     // NOT / not
	pgTokLParen                  // (
	pgTokRParen                  // )
	pgTokEOF
)

type pgTok struct {
	kind pgTokKind
	val  string
}

func pgTokenize(s string) ([]pgTok, error) {
	var toks []pgTok
	for i := 0; i < len(s); {
		c := s[i]
		if c == ' ' || c == '\t' {
			i++
			continue
		}
		switch {
		case c == '(':
			toks = append(toks, pgTok{pgTokLParen, "("})
			i++
		case c == ')':
			toks = append(toks, pgTok{pgTokRParen, ")"})
			i++
		case i+1 < len(s) && (c == '>' || c == '<' || c == '!') && s[i+1] == '=':
			toks = append(toks, pgTok{pgTokOp, s[i : i+2]})
			i += 2
		case c == '>' || c == '<' || c == '=' || c == '!':
			if c == '!' {
				return nil, fmt.Errorf("bare '!' at position %d; did you mean '!='?", i)
			}
			toks = append(toks, pgTok{pgTokOp, string(c)})
			i++
		case pgIsWordChar(c):
			j := i
			for j < len(s) && pgIsWordChar(s[j]) {
				j++
			}
			word := s[i:j]
			switch strings.ToLower(word) {
			case "and":
				toks = append(toks, pgTok{pgTokAnd, word})
			case "or":
				toks = append(toks, pgTok{pgTokOr, word})
			case "not":
				toks = append(toks, pgTok{pgTokNot, word})
			default:
				toks = append(toks, pgTok{pgTokIdent, word})
			}
			i = j
		default:
			return nil, fmt.Errorf("unexpected character %q at position %d in filter", c, i)
		}
	}
	toks = append(toks, pgTok{pgTokEOF, ""})
	return toks, nil
}

func pgIsWordChar(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		(c >= '0' && c <= '9') || c == '_' || c == '.'
}

// compileFilter parses expr and returns a predicate, or nil for an empty expression.
func compileFilter(expr string) (predicate, error) {
	if strings.TrimSpace(expr) == "" {
		return nil, nil
	}
	toks, err := pgTokenize(expr)
	if err != nil {
		return nil, err
	}
	p, pos, err := pgParseExpr(toks, 0)
	if err != nil {
		return nil, err
	}
	if toks[pos].kind != pgTokEOF {
		return nil, fmt.Errorf("unexpected token %q in filter", toks[pos].val)
	}
	return p, nil
}

// expr := term ( 'or' term )*
func pgParseExpr(toks []pgTok, pos int) (predicate, int, error) {
	left, pos, err := pgParseTerm(toks, pos)
	if err != nil {
		return nil, pos, err
	}
	for pos < len(toks) && toks[pos].kind == pgTokOr {
		pos++
		right, newPos, err := pgParseTerm(toks, pos)
		if err != nil {
			return nil, newPos, err
		}
		pos = newPos
		l, r := left, right
		left = func(pz *pb.PuzzleCreationResponse) bool { return l(pz) || r(pz) }
	}
	return left, pos, nil
}

// term := factor ( 'and' factor )*
func pgParseTerm(toks []pgTok, pos int) (predicate, int, error) {
	left, pos, err := pgParseFactor(toks, pos)
	if err != nil {
		return nil, pos, err
	}
	for pos < len(toks) && toks[pos].kind == pgTokAnd {
		pos++
		right, newPos, err := pgParseFactor(toks, pos)
		if err != nil {
			return nil, newPos, err
		}
		pos = newPos
		l, r := left, right
		left = func(pz *pb.PuzzleCreationResponse) bool { return l(pz) && r(pz) }
	}
	return left, pos, nil
}

// factor := 'not' factor | '(' expr ')' | atom
func pgParseFactor(toks []pgTok, pos int) (predicate, int, error) {
	if pos >= len(toks) || toks[pos].kind == pgTokEOF {
		return nil, pos, fmt.Errorf("unexpected end of filter expression")
	}
	switch toks[pos].kind {
	case pgTokNot:
		inner, newPos, err := pgParseFactor(toks, pos+1)
		if err != nil {
			return nil, newPos, err
		}
		return func(pz *pb.PuzzleCreationResponse) bool { return !inner(pz) }, newPos, nil
	case pgTokLParen:
		inner, newPos, err := pgParseExpr(toks, pos+1)
		if err != nil {
			return nil, newPos, err
		}
		if newPos >= len(toks) || toks[newPos].kind != pgTokRParen {
			return nil, newPos, fmt.Errorf("missing closing ')' in filter expression")
		}
		return inner, newPos + 1, nil
	case pgTokRParen:
		return nil, pos, fmt.Errorf("unexpected ')' in filter expression")
	default:
		return pgParseAtom(toks, pos)
	}
}

// atom := IDENT OP VALUE
func pgParseAtom(toks []pgTok, pos int) (predicate, int, error) {
	if toks[pos].kind != pgTokIdent {
		return nil, pos, fmt.Errorf("expected field name, got %q", toks[pos].val)
	}
	if pos+1 >= len(toks) || toks[pos+1].kind != pgTokOp {
		return nil, pos + 1, fmt.Errorf("expected operator after %q", toks[pos].val)
	}
	if pos+2 >= len(toks) || toks[pos+2].kind == pgTokEOF || toks[pos+2].kind != pgTokIdent {
		return nil, pos + 2, fmt.Errorf("expected value after %q %q", toks[pos].val, toks[pos+1].val)
	}

	field := strings.ToLower(toks[pos].val)
	op := toks[pos+1].val
	valueStr := toks[pos+2].val

	if field == "tag" {
		if op != "=" && op != "!=" {
			return nil, pos + 1, fmt.Errorf("operator %q not allowed for 'tag' (use = or !=)", op)
		}
		tagVal, ok := pgResolveTag(strings.ToUpper(valueStr))
		if !ok {
			return nil, pos + 2, fmt.Errorf("unknown tag %q (valid: BINGO, EQUITY, ONLY_BINGO, BLANK_BINGO, NON_BINGO, POWER_TILE, BINGO_NINE_OR_ABOVE, CEL_ONLY, POINTS)", valueStr)
		}
		if op == "=" {
			return func(pz *pb.PuzzleCreationResponse) bool {
				for _, t := range pz.GetTags() {
					if t == tagVal {
						return true
					}
				}
				return false
			}, pos + 3, nil
		}
		return func(pz *pb.PuzzleCreationResponse) bool {
			for _, t := range pz.GetTags() {
				if t == tagVal {
					return false
				}
			}
			return true
		}, pos + 3, nil
	}

	getter, ok := pgResolveStatField(field)
	if !ok {
		return nil, pos, fmt.Errorf("unknown field %q (valid stat fields: score, words_formed, main_word_length, tiles_played, equity_advantage, score_advantage, top_score_play_tiles_played, tws_covered, dws_covered, dls_covered, tls_covered, bonus_squares_covered, max_cross_word_length, min_cross_word_length, longest_hooked_word_length, longest_extended_word_length, max_played_through_tile_score, max_fresh_tile_face_value_on_letter_bonus_with_crossword, max_rack_tile_score)", toks[pos].val)
	}

	threshold, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return nil, pos + 2, fmt.Errorf("invalid numeric value %q for field %q", valueStr, field)
	}

	var cmp func(a, b float64) bool
	switch op {
	case ">=":
		cmp = func(a, b float64) bool { return a >= b }
	case "<=":
		cmp = func(a, b float64) bool { return a <= b }
	case ">":
		cmp = func(a, b float64) bool { return a > b }
	case "<":
		cmp = func(a, b float64) bool { return a < b }
	case "=":
		cmp = func(a, b float64) bool { return a == b }
	case "!=":
		cmp = func(a, b float64) bool { return a != b }
	default:
		return nil, pos + 1, fmt.Errorf("unknown operator %q", op)
	}

	th := threshold
	return func(pz *pb.PuzzleCreationResponse) bool {
		return cmp(getter(pz.GetStats()), th)
	}, pos + 3, nil
}

func pgResolveTag(name string) (pb.PuzzleTag, bool) {
	v, ok := pb.PuzzleTag_value[name]
	return pb.PuzzleTag(v), ok
}

func pgResolveStatField(name string) (func(*pb.PuzzleStats) float64, bool) {
	switch name {
	case "score":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetScore()) }, true
	case "words_formed":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetWordsFormed()) }, true
	case "main_word_length":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetMainWordLength()) }, true
	case "tiles_played":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetTilesPlayed()) }, true
	case "equity_advantage":
		return func(s *pb.PuzzleStats) float64 { return s.GetEquityAdvantage() }, true
	case "score_advantage":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetScoreAdvantage()) }, true
	case "top_score_play_tiles_played":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetTopScorePlayTilesPlayed()) }, true
	case "tws_covered":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetTwsCovered()) }, true
	case "dws_covered":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetDwsCovered()) }, true
	case "dls_covered":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetDlsCovered()) }, true
	case "tls_covered":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetTlsCovered()) }, true
	case "bonus_squares_covered":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetBonusSquaresCovered()) }, true
	case "max_cross_word_length":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetMaxCrossWordLength()) }, true
	case "min_cross_word_length":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetMinCrossWordLength()) }, true
	case "longest_hooked_word_length":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetLongestHookedWordLength()) }, true
	case "longest_extended_word_length":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetLongestExtendedWordLength()) }, true
	case "max_played_through_tile_score":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetMaxPlayedThroughTileScore()) }, true
	case "max_fresh_tile_face_value_on_letter_bonus_with_crossword":
		return func(s *pb.PuzzleStats) float64 {
			return float64(s.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword())
		}, true
	case "max_rack_tile_score":
		return func(s *pb.PuzzleStats) float64 { return float64(s.GetMaxRackTileScore()) }, true
	}
	return nil, false
}
