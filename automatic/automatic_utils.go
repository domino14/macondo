package automatic

// Data collection for automatic game. Allow computer vs computer games, etc.

import (
	"context"
	"errors"
	"expvar"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/rs/zerolog/log"
	"github.com/samber/lo"
	"golang.org/x/sync/errgroup"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

var (
	CVCCounter *expvar.Int
	IsPlaying  *expvar.Int
)

func init() {
	CVCCounter = expvar.NewInt("cvcCounter")
	IsPlaying = expvar.NewInt("isPlaying")
}

// DeterministicConfig holds configuration for deterministic autoplay runs
type DeterministicConfig struct {
	Seeds    [][32]byte // Pre-generated seeds (nil = generate new)
	SeedFile string     // File to read/write seeds
	NumGames int        // Number of games (used when generating seeds)
	// MasterSeed derives a seed per unit of work when there is no seed file.
	MasterSeed uint64
	// GamePairs plays each seed twice with the bots swapping seats.
	GamePairs bool
}

// CompVsCompStatic plays out a game to the end using best static turns.
func (r *GameRunner) CompVsCompStatic(addToHistory bool) error {
	err := r.Init(
		[]AutomaticRunnerPlayer{
			{BotCode: pb.BotRequest_HASTY_BOT},
			{BotCode: pb.BotRequest_HASTY_BOT},
		})

	if err != nil {
		return err
	}
	err = r.playFull(addToHistory, 0, [32]byte{})
	if err != nil {
		return err
	}
	log.Debug().Msgf("Game over. Score: %v - %v", r.game.PointsFor(0),
		r.game.PointsFor(1))
	return nil
}

func (r *GameRunner) playFull(addToHistory bool, gidx int, seed [32]byte) error {
	if err := r.playGame(addToHistory, gidx, seed); err != nil {
		return err
	}
	if r.gamechan != nil {
		r.gamechan <- r.gameSummary() + "\n"
	}
	return nil
}

// playGame plays one game to the end. gidx decides which bot moves first: even
// indices put player 1 in the first seat, odd indices put player 2 there.
func (r *GameRunner) playGame(addToHistory bool, gidx int, seed [32]byte) error {
	r.StartGameWithSeed(gidx, seed)
	log.Trace().Msgf("playing full, game %v", r.game.History().Uid)

	for r.game.Playing() == pb.PlayState_PLAYING {
		err := r.PlayBestTurn(r.game.PlayerOnTurn(), addToHistory)
		if err != nil {
			return err
		}
	}
	return nil
}

// playPair plays the same seed twice with the bots swapping seats. Both games
// deal the same tiles to the same seats, so the only thing that can differ
// between them is what the bots chose to do with those tiles.
func (r *GameRunner) playPair(addToHistory bool, pairIdx int, seed [32]byte) error {
	r.recordMoves = true
	defer func() {
		r.recordMoves = false
		r.movesPlayed = nil
	}()

	summaries := make([]string, 0, 2)
	plays := make([][]*move.Move, 0, 2)
	for half := 0; half < 2; half++ {
		r.movesPlayed = nil
		if err := r.playGame(addToHistory, 2*pairIdx+half, seed); err != nil {
			return err
		}
		// The row needs the divergence verdict, which we only have once both
		// halves are played, so hold it back until then.
		summaries = append(summaries, r.gameSummary())
		plays = append(plays, r.movesPlayed)
	}

	divergent := movesDiverge(plays[0], plays[1])
	if r.gamechan != nil {
		for _, summary := range summaries {
			r.gamechan <- fmt.Sprintf("%s,%d,%v\n", summary, pairIdx, divergent)
		}
	}
	return nil
}

// gameSummary renders the finished game as a row of the per-game log, without
// the trailing newline. Nicknames track the bots rather than the seats, so "p1"
// is always the first bot no matter which side of the board it played from.
func (r *GameRunner) gameSummary() string {
	return fmt.Sprintf("%v,%d,%d,%d,%d,%d,%d,%s",
		r.game.Uid(),
		r.game.PointsForNick("p1"),
		r.game.PointsForNick("p2"),
		r.game.BingosForNick("p1"),
		r.game.BingosForNick("p2"),
		r.game.TurnsForNick("p1"),
		r.game.TurnsForNick("p2"),
		r.game.FirstPlayer().RealName,
	)
}

// movesDiverge reports whether the two halves of a pair ever chose differently.
// The halves see identical positions in identical order, so a pair that never
// diverges contributes nothing but a tie to the comparison -- the fraction that
// does diverge is the useful signal about how often a change actually bites.
func movesDiverge(first, second []*move.Move) bool {
	if len(first) != len(second) {
		return true
	}
	for i := range first {
		if !first[i].Equals(second[i], false, false) {
			return true
		}
	}
	return false
}

func prettyName(b pb.BotRequest_BotCode) string {
	protoName := b.String()

	components := strings.Split(protoName, "_")
	return strings.Join(lo.Map(components, func(i string, idx int) string {
		return strings.Title(strings.ToLower(i))
	}), "")
}

func playerNames(players []AutomaticRunnerPlayer) []string {
	botct := map[string]int{}
	botctorig := map[string]int{}
	for _, p := range players {
		s := p.BotCode.String()
		botct[s]++
		botctorig[s]++
	}
	names := []string{}
	for _, p := range players {
		s := p.BotCode.String()

		if botct[s] == botctorig[s] {
			names = append(names, prettyName(p.BotCode))
		} else {
			names = append(names, prettyName(p.BotCode)+strconv.Itoa(botctorig[s]-botct[s]))
		}
		botct[s]--
	}
	return names
}

type Job struct {
	// gidx indexes the unit of work: a game normally, a pair of games when
	// game pairs are on.
	gidx int
	seed [32]byte
}

// seedForUnit picks the seed for one unit of work. A seed file wins if there is
// one; otherwise the run's master seed derives it. A zero seed means the game
// runs off the global RNG, which is the old non-reproducible behavior.
func seedForUnit(detConfig *DeterministicConfig, seeds [][32]byte, unitIdx int) [32]byte {
	if unitIdx < len(seeds) {
		return seeds[unitIdx]
	}
	if detConfig != nil && detConfig.MasterSeed != 0 {
		return DeriveSeed(detConfig.MasterSeed, unitIdx)
	}
	return [32]byte{}
}

// StartAutoplayFromConfig runs an autoplay experiment defined by a protojson
// config file. It returns the resolved experiment ID. Output files are written
// to cfg.OutputDir (or the current directory if empty):
//
//   - {experimentId}.txt       — per-turn log
//   - games-{experimentId}.txt — per-game summary
//   - {experimentId}.config.json — copy of the config for reproducibility
func StartAutoplayFromConfig(ctx context.Context, appCfg *config.Config, expCfg *pb.AutoplayConfig) (string, error) {
	experimentID := ResolveExperimentID(expCfg)

	outputDir := expCfg.OutputDir
	if outputDir == "" {
		outputDir = "."
	}
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return "", fmt.Errorf("creating output dir: %w", err)
	}

	// Write a copy of the config for reproducibility (with the resolved ID).
	expCfg.ExperimentId = experimentID
	cfgJSON, err := protojson.MarshalOptions{Multiline: true}.Marshal(expCfg)
	if err != nil {
		return "", fmt.Errorf("marshalling config: %w", err)
	}
	cfgPath := filepath.Join(outputDir, experimentID+".config.json")
	if err := os.WriteFile(cfgPath, cfgJSON, 0644); err != nil {
		return "", fmt.Errorf("writing config file: %w", err)
	}

	logfile := filepath.Join(outputDir, experimentID+".txt")
	players := PlayersFromConfig(expCfg)

	numGames := int(expCfg.NumGames)
	if numGames == 0 {
		numGames = 1_000_000_000
	}
	threads := int(expCfg.Threads)
	if threads == 0 {
		threads = runtime.NumCPU()
	}

	lexicon := expCfg.Lexicon
	if lexicon == "" {
		lexicon = appCfg.GetString(config.ConfigDefaultLexicon)
	}
	letterDist := expCfg.LetterDistribution
	if letterDist == "" {
		// Infer the letter distribution from the resolved lexicon name rather
		// than falling back to the app config's default. The config default
		// reflects whatever lexicon the shell last used, which may differ from
		// the lexicon requested here (e.g., "-lexicon NWL23" with a shell that
		// has FILE2017/spanish as its default). Using the wrong distribution
		// gives the game Spanish tiles with an English word graph, causing
		// immediate KWG dead-paths and a crash in the leave-map populator.
		inferred, err := tilemapping.ProbableLetterDistributionName(lexicon)
		if err == nil {
			letterDist = inferred
		} else {
			letterDist = appCfg.GetString(config.ConfigDefaultLetterDistribution)
		}
	}

	if expCfg.Description != "" {
		log.Info().Str("experiment", experimentID).Str("description", expCfg.Description).Msg("starting-autoplay")
	}

	// Handle seeding.
	var detConfig *DeterministicConfig
	if expCfg.GenerateSeeds || expCfg.Deterministic || expCfg.SeedFile != "" ||
		expCfg.GamePairs || expCfg.Seed != 0 {

		detConfig = &DeterministicConfig{
			SeedFile:   expCfg.SeedFile,
			NumGames:   numGames,
			MasterSeed: expCfg.Seed,
			GamePairs:  expCfg.GamePairs,
		}
		if expCfg.GenerateSeeds {
			if expCfg.SeedFile == "" {
				return experimentID, fmt.Errorf("generate_seeds requires seed_file to be set")
			}
			seeds, err := GenerateSeeds(numGames)
			if err != nil {
				return experimentID, fmt.Errorf("generating seeds: %w", err)
			}
			if err := SaveSeeds(seeds, expCfg.SeedFile); err != nil {
				return experimentID, fmt.Errorf("saving seeds: %w", err)
			}
			log.Info().Int("n", numGames).Str("file", expCfg.SeedFile).Msg("generated-seeds")
			return experimentID, nil
		}
		if expCfg.Deterministic {
			if expCfg.SeedFile == "" {
				return experimentID, fmt.Errorf("deterministic requires seed_file to be set")
			}
			seeds, err := LoadSeeds(expCfg.SeedFile)
			if err != nil {
				return experimentID, fmt.Errorf("loading seeds: %w", err)
			}
			detConfig.Seeds = seeds
			log.Info().Int("n", len(seeds)).Str("file", expCfg.SeedFile).Msg("loaded-seeds")
		}
		// Whatever asked for determinism needs a seed to work from. If the run
		// has no seed file to take them from and named no master seed, pick one
		// and say so, so the run can still be replayed later.
		if detConfig.MasterSeed == 0 && len(detConfig.Seeds) == 0 {
			masterSeed, err := RandomMasterSeed()
			if err != nil {
				return experimentID, err
			}
			detConfig.MasterSeed = masterSeed
			log.Info().Uint64("seed", masterSeed).
				Msg("no seed given; generated one -- pass it back with -seed to replay this run")
		}
	}

	err = StartCompVCompStaticGames(ctx, appCfg, numGames, expCfg.Block, threads,
		logfile, lexicon, letterDist, players, detConfig)
	if err != nil {
		return experimentID, err
	}
	return experimentID, nil
}

func StartCompVCompStaticGames(ctx context.Context, cfg *config.Config,
	numGames int, block bool, threads int,
	outputFilename, lexicon, letterDistribution string,
	players []AutomaticRunnerPlayer, detConfig *DeterministicConfig) error {

	if len(players) != 2 {
		return errors.New("must have two players")
	}

	gamePairs := detConfig != nil && detConfig.GamePairs
	if gamePairs {
		// A pair is only worth playing if both halves make the same decisions
		// in the same position. A sim spread over several threads finishes its
		// iterations in whatever order the scheduler picks, so it does not, and
		// the pairing buys nothing.
		for idx, p := range players {
			if bot.HasSimming(p.BotCode) && p.SimThreads != 1 {
				return fmt.Errorf(
					"player %d sims and would need simthreads=1 for game pairs to be reproducible (got %d)",
					idx+1, p.SimThreads)
			}
		}
	}

	// Handle deterministic mode
	var seeds [][32]byte
	if detConfig != nil {
		if len(detConfig.Seeds) > 0 {
			// Use provided seeds
			seeds = detConfig.Seeds
			if len(seeds) < numGames {
				return fmt.Errorf("not enough seeds: have %d, need %d", len(seeds), numGames)
			}
		} else if detConfig.SeedFile != "" {
			// Generate new seeds and save
			var err error
			seeds, err = GenerateSeeds(numGames)
			if err != nil {
				return fmt.Errorf("failed to generate seeds: %w", err)
			}
			err = SaveSeeds(seeds, detConfig.SeedFile)
			if err != nil {
				return fmt.Errorf("failed to save seeds: %w", err)
			}
			log.Info().Msgf("Generated and saved %d seeds to %s", numGames, detConfig.SeedFile)
		}
	}

	if threads > 1 && lo.SomeBy(players, func(p AutomaticRunnerPlayer) bool {
		return bot.HasEndgame(p.BotCode) || bot.HasPreendgame(p.BotCode)
	}) {
		return errors.New("cannot run multiple games in parallel if either player uses endgame or pre-endgame")
	}

	if IsPlaying.Value() > 0 {
		return errors.New("games are already being played, please wait till complete")
	}

	logfile, err := os.Create(outputFilename)
	if err != nil {
		return err
	}

	glfilename := filepath.Join(
		path.Dir(outputFilename),
		"games-"+path.Base(outputFilename))
	gamelogfile, err := os.Create(glfilename)
	if err != nil {
		return err
	}

	if gamePairs {
		log.Info().Msgf("Starting %v game pairs (%v games), %v threads", numGames, numGames*2, threads)
	} else {
		log.Info().Msgf("Starting %v games, %v threads", numGames, threads)
	}

	CVCCounter.Set(0)
	jobs := make(chan Job, threads*5)
	logChan := make(chan string, 100)
	gameChan := make(chan string, 10)
	var wg sync.WaitGroup
	// var fwg sync.WaitGroup

	g, ctx := errgroup.WithContext(ctx)
	addToHistory := lo.SomeBy(players, func(p AutomaticRunnerPlayer) bool {
		return bot.HasInfer(p.BotCode) || p.OracleInference
	})

	for i := 1; i <= threads; i++ {
		wg.Add(1)
		i := i
		g.Go(func() error {
			defer wg.Done()
			r := GameRunner{logchan: logChan, gamechan: gameChan,
				config: cfg, lexicon: lexicon, letterDistribution: letterDistribution,
				gamePairs: gamePairs}
			err := r.Init(players)
			if err != nil {
				log.Err(err).Msg("error initializing runner")
				return err
			}

			IsPlaying.Add(1)
			defer IsPlaying.Add(-1)
			for j := range jobs {
				if gamePairs {
					err = r.playPair(addToHistory, j.gidx, j.seed)
				} else {
					err = r.playFull(addToHistory, j.gidx, j.seed)
				}
				if err != nil {
					log.Err(err).Int("job", j.gidx).Msg("error-playFull")
					return err
				}
				if gamePairs {
					CVCCounter.Add(2)
				} else {
					CVCCounter.Add(1)
				}
			}
			log.Err(err).Msgf("exiting-gameplay-thread-%d", i)
			return nil
		})
	}

	g.Go(func() error {
		queuingJobs := true
		i := 0
	gameLoop:
		for queuingJobs {
			select {
			case jobs <- Job{gidx: i, seed: seedForUnit(detConfig, seeds, i)}:
				if i%1000 == 0 {
					log.Info().Msgf("Queued %v jobs", i)
				}
				i++
			case <-ctx.Done():
				// exit early
				log.Err(ctx.Err()).Msg("Context done")
				log.Info().Msg("Got stop signal, exiting soon...")
				break gameLoop
			default:
				// do nothing

			}
			if i == numGames {
				queuingJobs = false
			}
		}

		close(jobs)
		log.Info().Int("jobsQueued", i).Msg("Finished queueing all jobs.")
		wg.Wait()
		log.Info().Msg("All games finished.")
		close(logChan)
		close(gameChan)
		log.Info().Msg("Exiting feeder subroutine!")
		return ctx.Err()
	})

	g.Go(func() error {
		logfile.WriteString("playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n")
		for msg := range logChan {
			logfile.WriteString(msg)
		}
		logfile.Close()
		log.Info().Msg("Exiting turn logger goroutine!")
		return nil
	})

	g.Go(func() error {
		pnames := playerNames(players)
		header := fmt.Sprintf("gameID,%s_score,%s_score,%s_bingos,%s_bingos,%s_turns,%s_turns,first",
			pnames[0], pnames[1], pnames[0], pnames[1], pnames[0], pnames[1])
		if gamePairs {
			header += ",pair,divergent"
		}
		header += "\n"

		gamelogfile.WriteString(header)
		for msg := range gameChan {
			gamelogfile.WriteString(msg)
		}
		gamelogfile.Close()
		log.Info().Msg("Exiting game logger goroutine!")
		return nil
	})

	if block {
		err = g.Wait()
		return err
	}
	return nil

}
