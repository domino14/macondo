// taufit — maximum-likelihood estimation of the rangefinder softmax
// temperature (tau) from real BestBot-vs-human games on Woogles.
//
// The rangefinder weighs candidate opponent racks with
// P(observed play | rack) = softmax over logit(win prob) / tau, where win
// probs come from a shallow 2-ply mini-sim. BestBot itself chooses moves with
// a 5-ply sim, so tau calibrates the approximation noise between the 2-ply
// shallow sim and BestBot's actual decision process. In a finished game
// BestBot's rack is known at every turn, so at each of its turns we can
// replay the position, run the same shallow sim, and record the log-odds
// vector of the candidates plus which candidate was actually played. The
// log-odds are tau-independent, so the MLE grid search over tau runs
// instantly from the extracted ingredients.
//
// Three modes:
//
//	taufit -mode fetch    # download game metadata + GCGs from woogles
//	taufit -mode extract  # replay games, run shallow sims, emit ingredients JSONL
//	taufit -mode fit      # grid-search tau by maximum likelihood
//
// Example:
//
//	taufit -mode fetch -max-games 2000
//	taufit -mode extract -workers 16
//	taufit -mode fit -curve-csv curve.csv
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/ai/simplesimmer"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/rangefinder"
	"github.com/domino14/macondo/turnplayer"
)

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

// GameMeta is one line of games.jsonl, written by fetch mode.
type GameMeta struct {
	GameID             string `json:"game_id"`
	Lexicon            string `json:"lexicon"`
	LetterDistribution string `json:"letter_distribution"`
	BoardLayout        string `json:"board_layout"`
	Variant            string `json:"variant"`
	ChallengeRule      string `json:"challenge_rule"`
	GameMode           string `json:"game_mode"`
	RatingMode         string `json:"rating_mode"`
	TimeControlName    string `json:"time_control_name"`
	GameEndReason      string `json:"game_end_reason"`
	CreatedAt          string `json:"created_at"`
	BotNickname        string `json:"bot_nickname"`
	BotRating          string `json:"bot_rating"`
	HumanNickname      string `json:"human_nickname"`
	HumanRating        string `json:"human_rating"`
}

// PositionRecord is one line of ingredients.jsonl, written by extract mode.
// Logits (log-odds of the shallow-sim win probs) are tau-independent: the fit
// step batch-evaluates the softmax likelihood across the whole tau grid from
// them without re-simming.
type PositionRecord struct {
	GameID        string    `json:"game_id"`
	EventIdx      int       `json:"event_idx"`
	Player        string    `json:"player"`
	OppRating     string    `json:"opp_rating"`
	Lexicon       string    `json:"lexicon"`
	ChallengeRule string    `json:"challenge_rule"`
	MoveType      string    `json:"move_type"`
	TilesPlayed   int       `json:"tiles_played"`
	TilesInBag    int       `json:"tiles_in_bag"`
	Rack          string    `json:"rack"`
	PlayedMove    string    `json:"played_move"`
	TargetIdx     int       `json:"target_idx"`
	Moves         []string  `json:"moves"`
	WinProbs      []float64 `json:"win_probs"`
	Logits        []float64 `json:"logits"`
	ElapsedMs     int64     `json:"elapsed_ms"`
}

func main() {
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	mode := flag.String("mode", "", "fetch | extract | fit")
	dataDir := flag.String("data-dir", "taufit-data", "directory for games.jsonl and gcgs/")
	verbose := flag.Bool("verbose", false, "enable verbose logging")

	// fetch flags
	baseURL := flag.String("base-url", "https://woogles.io", "woogles API base URL")
	username := flag.String("username", "BestBot", "bot username to fetch games for")
	lexicon := flag.String("lexicon", "NWL23", "only fetch games with this lexicon")
	maxGames := flag.Int("max-games", 1000, "max games to fetch (fetch) or process (extract)")
	pageSize := flag.Int("page-size", 20, "GetRecentGames page size")
	sleepMs := flag.Int("sleep-ms", 200, "sleep between API requests in milliseconds")

	// extract flags
	output := flag.String("output", "", "ingredients JSONL path (default <data-dir>/ingredients.jsonl)")
	workers := flag.Int("workers", 8, "parallel games during extract")
	simIters := flag.Int("sim-iters", 200, "mini-sim iterations (matches rangefinder default)")
	candidates := flag.Int("candidates", 10, "top-N static candidates for the mini-sim (matches inferSingle)")
	player := flag.String("player", "bot", "whose moves to evaluate: bot | human")

	// fit flags
	input := flag.String("input", "", "ingredients JSONL path (default <data-dir>/ingredients.jsonl)")
	tauMin := flag.Float64("tau-min", 0.005, "grid search lower bound")
	tauMax := flag.Float64("tau-max", 2.0, "grid search upper bound")
	gridPoints := flag.Int("grid-points", 200, "log-spaced grid points")
	includeExchanges := flag.Bool("include-exchanges", false, "include exchange moves in the fit")
	includeBingos := flag.Bool("include-bingos", false, "include 7-tile plays (deployment inference skips them)")
	minBag := flag.Int("min-bag", 2, "min tiles in bag pre-move (BestBot uses MC only with 2+)")
	curveCSV := flag.String("curve-csv", "", "optional path to write the LL(tau) curve as CSV")

	flag.Parse()

	if *verbose {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	ctx, cancel := context.WithCancel(context.Background())
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Warn().Str("signal", sig.String()).Msg("shutting down")
		cancel()
	}()

	var err error
	switch *mode {
	case "fetch":
		err = runFetch(ctx, fetchCfg{
			baseURL:  *baseURL,
			username: *username,
			lexicon:  *lexicon,
			maxGames: *maxGames,
			pageSize: *pageSize,
			sleep:    time.Duration(*sleepMs) * time.Millisecond,
			dataDir:  *dataDir,
		})
	case "extract":
		outPath := *output
		if outPath == "" {
			outPath = filepath.Join(*dataDir, "ingredients.jsonl")
		}
		err = runExtract(ctx, extractCfg{
			dataDir:    *dataDir,
			output:     outPath,
			workers:    *workers,
			simIters:   *simIters,
			candidates: *candidates,
			player:     *player,
			maxGames:   *maxGames,
		})
	case "fit":
		inPath := *input
		if inPath == "" {
			inPath = filepath.Join(*dataDir, "ingredients.jsonl")
		}
		err = runFit(fitCfg{
			input:            inPath,
			tauMin:           *tauMin,
			tauMax:           *tauMax,
			gridPoints:       *gridPoints,
			includeExchanges: *includeExchanges,
			includeBingos:    *includeBingos,
			minBag:           *minBag,
			curveCSV:         *curveCSV,
		})
	default:
		fmt.Fprintln(os.Stderr, "usage: taufit -mode fetch|extract|fit [flags]")
		flag.PrintDefaults()
		os.Exit(2)
	}
	if err != nil && ctx.Err() == nil {
		log.Fatal().Err(err).Str("mode", *mode).Msg("taufit failed")
	}
}

// ---------------------------------------------------------------------------
// fetch
// ---------------------------------------------------------------------------

type fetchCfg struct {
	baseURL  string
	username string
	lexicon  string
	maxGames int
	pageSize int
	sleep    time.Duration
	dataDir  string
}

// recentGamesResponse mirrors the fields we need from
// game_service.GameMetadataService/GetRecentGames (Connect JSON, snake_case).
type recentGamesResponse struct {
	GameInfo []struct {
		Players []struct {
			Nickname string `json:"nickname"`
			Rating   string `json:"rating"`
			IsBot    bool   `json:"is_bot"`
		} `json:"players"`
		TimeControlName string `json:"time_control_name"`
		GameEndReason   string `json:"game_end_reason"`
		GameID          string `json:"game_id"`
		CreatedAt       string `json:"created_at"`
		Type            string `json:"type"`
		GameRequest     struct {
			Lexicon string `json:"lexicon"`
			Rules   struct {
				BoardLayoutName        string `json:"board_layout_name"`
				LetterDistributionName string `json:"letter_distribution_name"`
				VariantName            string `json:"variant_name"`
			} `json:"rules"`
			ChallengeRule string `json:"challenge_rule"`
			GameMode      string `json:"game_mode"`
			RatingMode    string `json:"rating_mode"`
			BotType       string `json:"bot_type"`
		} `json:"game_request"`
	} `json:"game_info"`
}

func postJSON(ctx context.Context, client *http.Client, url string, reqBody any, respBody any) error {
	buf, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(buf))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: status %d: %s", url, resp.StatusCode, string(body))
	}
	return json.Unmarshal(body, respBody)
}

func runFetch(ctx context.Context, fc fetchCfg) error {
	gcgDir := filepath.Join(fc.dataDir, "gcgs")
	if err := os.MkdirAll(gcgDir, 0o755); err != nil {
		return err
	}

	// Load already-fetched game IDs so re-runs are resumable and append-only.
	gamesPath := filepath.Join(fc.dataDir, "games.jsonl")
	seen := map[string]bool{}
	if f, err := os.Open(gamesPath); err == nil {
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 1<<20), 1<<20)
		for scanner.Scan() {
			var gm GameMeta
			if json.Unmarshal(scanner.Bytes(), &gm) == nil {
				seen[gm.GameID] = true
			}
		}
		f.Close()
	}
	log.Info().Int("already-fetched", len(seen)).Msg("resuming fetch")

	gamesFile, err := os.OpenFile(gamesPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer gamesFile.Close()
	enc := json.NewEncoder(gamesFile)

	client := &http.Client{Timeout: 30 * time.Second}
	recentURL := fc.baseURL + "/api/game_service.GameMetadataService/GetRecentGames"
	gcgURL := fc.baseURL + "/api/game_service.GameMetadataService/GetGCG"

	fetched := 0
	scanned := 0
	offset := 0
	for fetched < fc.maxGames {
		if ctx.Err() != nil {
			break
		}
		var page recentGamesResponse
		err := postJSON(ctx, client, recentURL, map[string]any{
			"username": fc.username,
			"numGames": fc.pageSize,
			"offset":   offset,
		}, &page)
		if err != nil {
			return fmt.Errorf("GetRecentGames offset %d: %w", offset, err)
		}
		if len(page.GameInfo) == 0 {
			log.Info().Int("offset", offset).Msg("no more games")
			break
		}
		offset += len(page.GameInfo)
		scanned += len(page.GameInfo)

		for _, gi := range page.GameInfo {
			if fetched >= fc.maxGames || ctx.Err() != nil {
				break
			}
			if seen[gi.GameID] {
				continue
			}
			if gi.Type != "NATIVE" ||
				gi.GameRequest.BotType != "SIMMING_BOT" ||
				gi.GameRequest.Lexicon != fc.lexicon ||
				gi.GameRequest.Rules.VariantName != "classic" {
				continue
			}
			switch gi.GameEndReason {
			case "NONE", "ABORTED", "CANCELLED":
				continue
			}
			if len(gi.Players) != 2 {
				continue
			}
			botIdx := -1
			for i, p := range gi.Players {
				if p.IsBot {
					if botIdx != -1 {
						botIdx = -2 // two bots
						break
					}
					botIdx = i
				}
			}
			if botIdx < 0 {
				continue // zero or two bots
			}

			// Download the GCG before recording the metadata line, so
			// games.jsonl only ever references GCGs we actually have.
			var gcgResp struct {
				GCG string `json:"gcg"`
			}
			if err := postJSON(ctx, client, gcgURL, map[string]any{"gameId": gi.GameID}, &gcgResp); err != nil {
				log.Warn().Err(err).Str("game-id", gi.GameID).Msg("GetGCG failed; skipping game")
				continue
			}
			if err := os.WriteFile(filepath.Join(gcgDir, gi.GameID+".gcg"), []byte(gcgResp.GCG), 0o644); err != nil {
				return err
			}

			gm := GameMeta{
				GameID:             gi.GameID,
				Lexicon:            gi.GameRequest.Lexicon,
				LetterDistribution: gi.GameRequest.Rules.LetterDistributionName,
				BoardLayout:        gi.GameRequest.Rules.BoardLayoutName,
				Variant:            gi.GameRequest.Rules.VariantName,
				ChallengeRule:      gi.GameRequest.ChallengeRule,
				GameMode:           gi.GameRequest.GameMode,
				RatingMode:         gi.GameRequest.RatingMode,
				TimeControlName:    gi.TimeControlName,
				GameEndReason:      gi.GameEndReason,
				CreatedAt:          gi.CreatedAt,
				BotNickname:        gi.Players[botIdx].Nickname,
				BotRating:          gi.Players[botIdx].Rating,
				HumanNickname:      gi.Players[1-botIdx].Nickname,
				HumanRating:        gi.Players[1-botIdx].Rating,
			}
			if err := enc.Encode(gm); err != nil {
				return err
			}
			seen[gi.GameID] = true
			fetched++
			time.Sleep(fc.sleep)
		}

		if scanned%200 < fc.pageSize {
			fmt.Fprintf(os.Stderr, "fetch: scanned %d games, kept %d\n", scanned, fetched)
		}
		time.Sleep(fc.sleep)
	}
	fmt.Fprintf(os.Stderr, "fetch done: scanned %d, kept %d new games (dir %s)\n", scanned, fetched, fc.dataDir)
	return nil
}

// ---------------------------------------------------------------------------
// extract
// ---------------------------------------------------------------------------

type extractCfg struct {
	dataDir    string
	output     string
	workers    int
	simIters   int
	candidates int
	player     string
	maxGames   int
}

func loadGamesIndex(dataDir string) ([]GameMeta, error) {
	f, err := os.Open(filepath.Join(dataDir, "games.jsonl"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var games []GameMeta
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		var gm GameMeta
		if err := json.Unmarshal(scanner.Bytes(), &gm); err != nil {
			return nil, err
		}
		games = append(games, gm)
	}
	return games, scanner.Err()
}

func runExtract(ctx context.Context, ec extractCfg) error {
	games, err := loadGamesIndex(ec.dataDir)
	if err != nil {
		return fmt.Errorf("loading games index (run -mode fetch first?): %w", err)
	}
	if ec.maxGames > 0 && len(games) > ec.maxGames {
		games = games[:ec.maxGames]
	}

	cfg := config.DefaultConfig()

	// Ensure lexica are present before spawning workers.
	lexica := map[string]bool{}
	for _, gm := range games {
		lexica[gm.Lexicon] = true
	}
	for lex := range lexica {
		if err := turnplayer.EnsureKWG(lex, cfg.WGLConfig()); err != nil {
			return fmt.Errorf("ensuring lexicon %s: %w", lex, err)
		}
	}

	outFile, err := os.Create(ec.output)
	if err != nil {
		return err
	}
	defer outFile.Close()
	writer := bufio.NewWriterSize(outFile, 1<<20)
	defer writer.Flush()

	gameChan := make(chan GameMeta, ec.workers*2)
	recordChan := make(chan PositionRecord, 256)

	var gamesDone, positionsDone, targetMissing atomic.Int64

	var writerWg sync.WaitGroup
	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		enc := json.NewEncoder(writer)
		for rec := range recordChan {
			if err := enc.Encode(rec); err != nil {
				log.Error().Err(err).Msg("failed to write record")
			}
			positionsDone.Add(1)
		}
	}()

	stopProgress := make(chan struct{})
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				fmt.Fprintf(os.Stderr, "extract: %d/%d games, %d positions (%d target-missing)\n",
					gamesDone.Load(), len(games), positionsDone.Load(), targetMissing.Load())
			case <-stopProgress:
				return
			}
		}
	}()

	var workerWg sync.WaitGroup
	for w := 0; w < ec.workers; w++ {
		workerWg.Add(1)
		go func() {
			defer workerWg.Done()
			for gm := range gameChan {
				if ctx.Err() != nil {
					return
				}
				// A panic on one malformed game must not kill an hours-long
				// extract across thousands of games.
				func() {
					defer func() {
						if r := recover(); r != nil {
							log.Error().Interface("panic", r).Str("game-id", gm.GameID).Msg("game panicked")
						}
					}()
					if err := extractGame(ctx, cfg, ec, gm, recordChan, &targetMissing); err != nil {
						log.Warn().Err(err).Str("game-id", gm.GameID).Msg("game skipped")
					}
				}()
				gamesDone.Add(1)
			}
		}()
	}

	for _, gm := range games {
		if ctx.Err() != nil {
			break
		}
		gameChan <- gm
	}
	close(gameChan)
	workerWg.Wait()
	close(recordChan)
	writerWg.Wait()
	close(stopProgress)

	fmt.Fprintf(os.Stderr, "extract done: %d games, %d positions written to %s (%d target-missing)\n",
		gamesDone.Load(), positionsDone.Load(), ec.output, targetMissing.Load())
	return nil
}

// extractGame replays one game and emits a PositionRecord for each qualifying
// move by the tracked player (BestBot by default).
func extractGame(ctx context.Context, cfg *config.Config, ec extractCfg,
	gm GameMeta, recordChan chan<- PositionRecord, targetMissing *atomic.Int64) error {

	gcgBytes, err := os.ReadFile(filepath.Join(ec.dataDir, "gcgs", gm.GameID+".gcg"))
	if err != nil {
		return err
	}
	history, err := gcgio.ParseGCGFromReader(cfg, bytes.NewReader(gcgBytes))
	if err != nil {
		return fmt.Errorf("parsing GCG: %w", err)
	}
	// The GCG doesn't record the challenge rule; the parsed default (VOID)
	// makes replay reject phony words that legitimately stayed on the board
	// in non-VOID games. Same workaround as gameanalysis: replay as DOUBLE.
	if history.ChallengeRule == pb.ChallengeRule_VOID {
		history.ChallengeRule = pb.ChallengeRule_DOUBLE
	}

	trackedNick := gm.BotNickname
	oppRating := gm.HumanRating
	if ec.player == "human" {
		trackedNick = gm.HumanNickname
		oppRating = gm.BotRating
	}
	trackedIdx := -1
	for i, p := range history.Players {
		if p.Nickname == trackedNick {
			trackedIdx = i
		}
	}
	if trackedIdx == -1 {
		return fmt.Errorf("player %q not found in GCG", trackedNick)
	}

	boardLayout, ldName, variant := game.HistoryToVariant(history)
	rules, err := game.NewBasicGameRules(cfg, history.Lexicon, boardLayout, ldName,
		game.CrossScoreAndSet, variant)
	if err != nil {
		return fmt.Errorf("creating rules: %w", err)
	}

	for i, evt := range history.Events {
		if ctx.Err() != nil {
			return nil
		}
		if int(evt.PlayerIndex) != trackedIdx {
			continue
		}
		var moveType string
		switch evt.Type {
		case pb.GameEvent_TILE_PLACEMENT_MOVE:
			moveType = "placement"
		case pb.GameEvent_EXCHANGE:
			moveType = "exchange"
		default:
			continue
		}
		// Skip plays that were challenged off the board: the recorded move
		// effectively became a pass, so it doesn't reflect a chosen play.
		if i+1 < len(history.Events) &&
			history.Events[i+1].Type == pb.GameEvent_PHONY_TILES_RETURNED {
			continue
		}

		g, err := game.NewFromHistory(history, rules, i)
		if err != nil {
			return fmt.Errorf("replay to event %d: %w", i, err)
		}
		if g.Playing() != pb.PlayState_PLAYING {
			continue
		}
		// BestBot only uses Monte Carlo with 2+ tiles in the bag; with 1 it
		// switches to the pre-endgame solver and with 0 to the endgame
		// solver, so those decisions aren't softmax-of-sim shaped at all.
		tilesInBag := g.Bag().TilesRemaining()
		if tilesInBag < 2 {
			continue
		}

		playedMove, err := game.MoveFromEvent(evt, g.Alphabet(), g.Board())
		if err != nil {
			return fmt.Errorf("move from event %d: %w", i, err)
		}

		simmer, err := simplesimmer.NewSimpleSimmerFromGame(g)
		if err != nil {
			return fmt.Errorf("creating simmer at event %d: %w", i, err)
		}
		simmer.SetMaxIters(ec.simIters)

		start := time.Now()
		// Passing playedMove guarantees the candidate set contains the move
		// actually made — GenAndSim appends it if the top-N static
		// generation didn't already produce it.
		if _, err := simmer.GenAndSim(ctx, ec.candidates, playedMove); err != nil {
			return fmt.Errorf("sim at event %d: %w", i, err)
		}
		elapsed := time.Since(start)

		bestPlays := simmer.BestPlays().PlaysNoLock()
		moves := make([]string, len(bestPlays))
		winProbs := make([]float64, len(bestPlays))
		logits := make([]float64, len(bestPlays))
		targetIdx := -1
		for j, sp := range bestPlays {
			moves[j] = sp.Move().ShortDescription()
			winProbs[j] = sp.WinProb()
			logits[j] = rangefinder.Logit(sp.WinProb())
			if rangefinder.MovesAreTheSame(sp.Move(), playedMove, g.Board()) {
				targetIdx = j
			}
		}
		if targetIdx == -1 {
			// Should only happen on sim anomalies; the added move guarantees
			// membership. Count it so the fit can report data loss.
			targetMissing.Add(1)
			log.Warn().Str("game-id", gm.GameID).Int("event", i).
				Str("played", playedMove.ShortDescription()).Msg("played move not found in simmed plays")
		}

		recordChan <- PositionRecord{
			GameID:        gm.GameID,
			EventIdx:      i,
			Player:        trackedNick,
			OppRating:     oppRating,
			Lexicon:       gm.Lexicon,
			ChallengeRule: gm.ChallengeRule,
			MoveType:      moveType,
			TilesPlayed:   playedMove.TilesPlayed(),
			TilesInBag:    tilesInBag,
			Rack:          evt.Rack,
			PlayedMove:    playedMove.ShortDescription(),
			TargetIdx:     targetIdx,
			Moves:         moves,
			WinProbs:      winProbs,
			Logits:        logits,
			ElapsedMs:     elapsed.Milliseconds(),
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------

type fitCfg struct {
	input            string
	tauMin           float64
	tauMax           float64
	gridPoints       int
	includeExchanges bool
	includeBingos    bool
	minBag           int
	curveCSV         string
}

func loadRecords(path string) ([]PositionRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var recs []PositionRecord
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		var r PositionRecord
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			return nil, err
		}
		recs = append(recs, r)
	}
	return recs, scanner.Err()
}

// tauGrid returns gridPoints log-spaced values in [tauMin, tauMax].
func tauGrid(tauMin, tauMax float64, gridPoints int) []float64 {
	grid := make([]float64, gridPoints)
	logMin, logMax := math.Log(tauMin), math.Log(tauMax)
	for i := range grid {
		frac := float64(i) / float64(gridPoints-1)
		grid[i] = math.Exp(logMin + frac*(logMax-logMin))
	}
	return grid
}

// logLikelihoods returns the summed log-likelihood at each tau in grid.
func logLikelihoods(recs []PositionRecord, grid []float64) []float64 {
	ll := make([]float64, len(grid))
	for _, r := range recs {
		for gi, tau := range grid {
			p := rangefinder.SoftmaxOverLogOdds(r.Logits, r.TargetIdx, tau)
			ll[gi] += math.Log(p)
		}
	}
	return ll
}

// refineMax fits a parabola in log(tau) through the grid max and its
// neighbors and returns the vertex (clamped to the neighbor interval). Falls
// back to the grid point at the boundary.
func refineMax(grid, ll []float64) (tauStar, llStar float64, bestIdx int) {
	bestIdx = 0
	for i := range ll {
		if ll[i] > ll[bestIdx] {
			bestIdx = i
		}
	}
	if bestIdx == 0 || bestIdx == len(grid)-1 {
		return grid[bestIdx], ll[bestIdx], bestIdx
	}
	x0, x1, x2 := math.Log(grid[bestIdx-1]), math.Log(grid[bestIdx]), math.Log(grid[bestIdx+1])
	y0, y1, y2 := ll[bestIdx-1], ll[bestIdx], ll[bestIdx+1]
	denom := (x0-x1)*(x0-x2)*(x1-x2)
	if denom == 0 {
		return grid[bestIdx], ll[bestIdx], bestIdx
	}
	a := (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
	b := (x2*x2*(y0-y1) + x1*x1*(y2-y0) + x0*x0*(y1-y2)) / denom
	if a >= 0 {
		return grid[bestIdx], ll[bestIdx], bestIdx
	}
	xv := -b / (2 * a)
	xv = math.Max(x0, math.Min(x2, xv))
	c := y1 - a*x1*x1 - b*x1
	return math.Exp(xv), a*xv*xv + b*xv + c, bestIdx
}

// profileCI returns the tau range where LL >= llStar - 1.92 (approx 95% CI
// via Wilks), linearly interpolated between grid points on the log scale.
func profileCI(grid, ll []float64, llStar float64) (lo, hi float64) {
	threshold := llStar - 1.92
	lo, hi = grid[0], grid[len(grid)-1]
	for i := 0; i < len(grid)-1; i++ {
		if ll[i] < threshold && ll[i+1] >= threshold {
			frac := (threshold - ll[i]) / (ll[i+1] - ll[i])
			lo = math.Exp(math.Log(grid[i]) + frac*(math.Log(grid[i+1])-math.Log(grid[i])))
			break
		}
	}
	for i := len(grid) - 1; i > 0; i-- {
		if ll[i] < threshold && ll[i-1] >= threshold {
			frac := (threshold - ll[i]) / (ll[i-1] - ll[i])
			hi = math.Exp(math.Log(grid[i]) - frac*(math.Log(grid[i])-math.Log(grid[i-1])))
			break
		}
	}
	return lo, hi
}

// bagBuckets partitions positions by tiles in the bag pre-move; shared by the
// stdout breakdown and the curve CSV facets.
var bagBuckets = []struct {
	name    string
	min, mx int
}{{"bag 2-10", 2, 10}, {"bag 11-30", 11, 30}, {"bag 31-45", 31, 45}, {"bag 46+", 46, 1000}}

func filterBag(recs []PositionRecord, b struct {
	name    string
	min, mx int
}) []PositionRecord {
	var sub []PositionRecord
	for _, r := range recs {
		if r.TilesInBag >= b.min && r.TilesInBag <= b.mx {
			sub = append(sub, r)
		}
	}
	return sub
}

func fitSubset(name string, recs []PositionRecord, grid []float64, w io.Writer) {
	if len(recs) == 0 {
		fmt.Fprintf(w, "%-22s n=0 (skipped)\n", name)
		return
	}
	ll := logLikelihoods(recs, grid)
	tauStar, llStar, _ := refineMax(grid, ll)
	lo, hi := profileCI(grid, ll, llStar)

	top1 := 0
	for _, r := range recs {
		argmax := 0
		for j := range r.Logits {
			if r.Logits[j] > r.Logits[argmax] {
				argmax = j
			}
		}
		if argmax == r.TargetIdx {
			top1++
		}
	}
	fmt.Fprintf(w, "%-22s n=%-6d tau*=%.4f  CI95=[%.4f, %.4f]  meanLL=%.4f  top1=%.1f%%\n",
		name, len(recs), tauStar, lo, hi, llStar/float64(len(recs)),
		100*float64(top1)/float64(len(recs)))
}

func runFit(fc fitCfg) error {
	all, err := loadRecords(fc.input)
	if err != nil {
		return fmt.Errorf("loading ingredients (run -mode extract first?): %w", err)
	}

	var recs []PositionRecord
	var nTargetMissing, nTooFew, nFiltered int
	for _, r := range all {
		if r.TargetIdx < 0 {
			nTargetMissing++
			continue
		}
		if len(r.Logits) < 2 {
			nTooFew++
			continue
		}
		if !fc.includeExchanges && r.MoveType != "placement" {
			nFiltered++
			continue
		}
		if !fc.includeBingos && r.TilesPlayed >= 7 {
			nFiltered++
			continue
		}
		if r.TilesInBag < fc.minBag {
			nFiltered++
			continue
		}
		recs = append(recs, r)
	}

	fmt.Printf("Loaded %d positions; fitting on %d (excluded: %d target-missing, %d <2 candidates, %d filtered)\n\n",
		len(all), len(recs), nTargetMissing, nTooFew, nFiltered)
	if len(recs) == 0 {
		return fmt.Errorf("no positions left after filtering")
	}

	grid := tauGrid(fc.tauMin, fc.tauMax, fc.gridPoints)
	ll := logLikelihoods(recs, grid)
	tauStar, llStar, bestIdx := refineMax(grid, ll)
	lo, hi := profileCI(grid, ll, llStar)
	if bestIdx == 0 || bestIdx == len(grid)-1 {
		fmt.Printf("WARNING: max is at the grid boundary (tau=%.4f); widen -tau-min/-tau-max\n\n", grid[bestIdx])
	}

	// Reference points: current default and a uniform-choice baseline.
	llDefault := 0.0
	llUniform := 0.0
	for _, r := range recs {
		llDefault += math.Log(rangefinder.SoftmaxOverLogOdds(r.Logits, r.TargetIdx, rangefinder.SoftmaxTemperature))
		llUniform += math.Log(1.0 / float64(len(r.Logits)))
	}
	n := float64(len(recs))

	fmt.Printf("MLE tau = %.4f   (95%% CI [%.4f, %.4f])\n", tauStar, lo, hi)
	fmt.Printf("mean log-likelihood per position:\n")
	fmt.Printf("  at tau* = %.4f:       %.4f\n", tauStar, llStar/n)
	fmt.Printf("  at default tau %.3f:  %.4f\n", rangefinder.SoftmaxTemperature, llDefault/n)
	fmt.Printf("  uniform baseline:      %.4f\n\n", llUniform/n)

	w := os.Stdout
	fmt.Fprintln(w, "By tiles in bag:")
	for _, b := range bagBuckets {
		fitSubset("  "+b.name, filterBag(recs, b), grid, w)
	}

	fmt.Fprintln(w, "\nBy tiles played:")
	for tp := 0; tp <= 7; tp++ {
		var sub []PositionRecord
		for _, r := range recs {
			if r.TilesPlayed == tp {
				sub = append(sub, r)
			}
		}
		if len(sub) > 0 {
			fitSubset(fmt.Sprintf("  played %d", tp), sub, grid, w)
		}
	}

	fmt.Fprintln(w, "\nBy opponent rating:")
	ratingBuckets := []struct {
		name    string
		min, mx int
	}{{"  opp <1400", 0, 1399}, {"  opp 1400-1799", 1400, 1799}, {"  opp 1800+", 1800, 10000}}
	for _, b := range ratingBuckets {
		var sub []PositionRecord
		for _, r := range recs {
			rating, err := strconv.Atoi(strings.TrimSuffix(r.OppRating, "?"))
			if err != nil {
				continue
			}
			if rating >= b.min && rating <= b.mx {
				sub = append(sub, r)
			}
		}
		fitSubset(b.name, sub, grid, w)
	}

	// Target-rank histogram: where did the played move rank by win prob?
	rankCounts := map[int]int{}
	for _, r := range recs {
		rank := 0
		for j := range r.Logits {
			if r.Logits[j] > r.Logits[r.TargetIdx] {
				rank++
			}
		}
		rankCounts[rank]++
	}
	ranks := make([]int, 0, len(rankCounts))
	for k := range rankCounts {
		ranks = append(ranks, k)
	}
	sort.Ints(ranks)
	fmt.Fprintln(w, "\nPlayed-move rank in shallow sim (0 = sim's top choice):")
	for _, k := range ranks {
		fmt.Fprintf(w, "  rank %d: %d (%.1f%%)\n", k, rankCounts[k], 100*float64(rankCounts[k])/n)
	}

	if fc.curveCSV != "" {
		f, err := os.Create(fc.curveCSV)
		if err != nil {
			return err
		}
		// Long format: one row per (bucket, tau), "all" plus the bag facets.
		// uniform_mean_ll is that bucket's uniform-choice baseline (constant
		// across tau, repeated for convenience).
		fmt.Fprintln(f, "bucket,n,tau,log_likelihood,mean_ll,uniform_mean_ll")
		writeCurve := func(name string, sub []PositionRecord) {
			if len(sub) == 0 {
				return
			}
			subLL := logLikelihoods(sub, grid)
			uniform := 0.0
			for _, r := range sub {
				uniform += math.Log(1.0 / float64(len(r.Logits)))
			}
			uniform /= float64(len(sub))
			for i, tau := range grid {
				fmt.Fprintf(f, "%s,%d,%g,%g,%g,%g\n", name, len(sub), tau,
					subLL[i], subLL[i]/float64(len(sub)), uniform)
			}
		}
		writeCurve("all", recs)
		for _, b := range bagBuckets {
			writeCurve(b.name, filterBag(recs, b))
		}
		f.Close()
		fmt.Fprintf(w, "\nLL curves (all + bag facets) written to %s\n", fc.curveCSV)
	}
	return nil
}
