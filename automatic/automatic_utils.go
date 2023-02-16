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
	"strconv"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"
	"github.com/samber/lo"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

var (
	CVCCounter *expvar.Int
	IsPlaying  *expvar.Int
)

func init() {
	CVCCounter = expvar.NewInt("cvcCounter")
	IsPlaying = expvar.NewInt("isPlaying")
}

// CompVsCompStatic plays out a game to the end using best static turns.
func (r *GameRunner) CompVsCompStatic(addToHistory bool) error {
	err := r.Init(
		[]AutomaticRunnerPlayer{
			{"", "", pb.BotRequest_HASTY_BOT},
			{"", "", pb.BotRequest_HASTY_BOT},
		})

	if err != nil {
		return err
	}
	r.playFullStatic(addToHistory)
	log.Debug().Msgf("Game over. Score: %v - %v", r.game.PointsFor(0),
		r.game.PointsFor(1))
	return nil
}

func (r *GameRunner) playFullStatic(addToHistory bool) {
	r.StartGame()
	log.Trace().Msgf("playing full static, game %v", r.game.History().Uid)

	for r.game.Playing() == pb.PlayState_PLAYING {
		r.PlayBestStaticTurn(r.game.PlayerOnTurn(), addToHistory)
	}

	if r.gamechan != nil {
		r.gamechan <- fmt.Sprintf("%v,%d,%d,%d,%d,%d,%d,%s\n",
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

type Job struct{}

func StartCompVCompStaticGames(ctx context.Context, cfg *config.Config,
	numGames int, block bool, threads int,
	outputFilename, lexicon, letterDistribution string,
	players []AutomaticRunnerPlayer) error {

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

	log.Info().Msgf("Starting %v games, %v threads", numGames, threads)

	CVCCounter.Set(0)
	jobs := make(chan Job, 100)
	logChan := make(chan string, 100)
	gameChan := make(chan string, 10)
	var wg sync.WaitGroup
	var fwg sync.WaitGroup
	fwg.Add(3)

	for i := 1; i <= threads; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := GameRunner{logchan: logChan, gamechan: gameChan,
				config: cfg, lexicon: lexicon, letterDistribution: letterDistribution}
			err := r.Init(players)
			if err != nil {
				log.Err(err).Msg("error initializing runner")
				return
			}

			IsPlaying.Add(1)
			for range jobs {
				r.playFullStatic(false)
				CVCCounter.Add(1)
			}
			IsPlaying.Add(-1)
		}(i)
	}

	go func() {
		defer fwg.Done()
	gameLoop:
		for i := 1; i < numGames+1; i++ {
			jobs <- Job{}
			if i%1000 == 0 {
				log.Info().Msgf("Queued %v jobs", i)
			}
			select {
			case <-ctx.Done():
				// exit early
				log.Info().Msg("Got stop signal, exiting soon...")
				break gameLoop
			default:
				// do nothing

			}
		}

		close(jobs)
		log.Info().Msg("Finished queueing all jobs.")
		wg.Wait()
		log.Info().Msg("All games finished.")
		close(logChan)
		close(gameChan)
		log.Info().Msg("Exiting feeder subroutine!")
	}()

	go func() {
		defer fwg.Done()
		logfile.WriteString("playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n")
		for msg := range logChan {
			logfile.WriteString(msg)
		}
		logfile.Close()
		log.Info().Msg("Exiting turn logger goroutine!")
	}()

	go func() {
		defer fwg.Done()
		pnames := playerNames(players)
		header := fmt.Sprintf("gameID,%s_score,%s_score,%s_bingos,%s_bingos,%s_turns,%s_turns,first\n",
			pnames[0], pnames[1], pnames[0], pnames[1], pnames[0], pnames[1])

		gamelogfile.WriteString(header)
		for msg := range gameChan {
			gamelogfile.WriteString(msg)
		}
		gamelogfile.Close()
		log.Info().Msg("Exiting game logger goroutine!")
	}()

	if block {
		fwg.Wait()
	}
	return nil

}
