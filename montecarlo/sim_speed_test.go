package montecarlo

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"

	aiturnplayer "github.com/domino14/macondo/ai/turnplayer"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// A speed harness for whole sims, for comparing with another engine on the
// same positions. It does nothing unless SIM_SPEED is set.
//
//	SIM_SPEED=gen      write positions (CGP, one a line) to SIM_SPEED_FILE
//	SIM_SPEED=run      sim the positions in SIM_SPEED_FILE
//	SIM_SPEED_LEXICON  lexicon for gen (default CSW21)
//	SIM_SPEED_THREADS  comma-separated thread counts (default 1,8,16)
//	SIM_SPEED_PLAYS    candidate plays, the top N by equity (default 15)
//	SIM_SPEED_ITERS    iterations; each rolls every candidate out once (default 2000)
//	SIM_SPEED_PLIES    plies (default 5)
//	SIM_SPEED_WMP      "0" to sim without the word map
//
// One line is printed per position and thread count, with the rollouts
// (iterations x candidates) and rollouts per second.
func TestSimSpeed(t *testing.T) {
	mode := os.Getenv("SIM_SPEED")
	if mode == "" {
		t.Skip("SIM_SPEED not set")
	}
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	file := os.Getenv("SIM_SPEED_FILE")
	if file == "" {
		t.Fatal("SIM_SPEED_FILE not set")
	}
	switch mode {
	case "gen":
		simSpeedGen(t, file)
	case "run":
		simSpeedRun(t, file)
	default:
		t.Fatalf("SIM_SPEED=%s: want gen or run", mode)
	}
}

func simSpeedEnvInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			panic(err)
		}
		return n
	}
	return def
}

// simSpeedGen plays seeded games with the top static move and stops each at
// a different point: two early, two in the middle, two late.
func simSpeedGen(t *testing.T, file string) {
	lexicon := os.Getenv("SIM_SPEED_LEXICON")
	if lexicon == "" {
		lexicon = "CSW21"
	}
	rules, err := game.NewBasicGameRules(DefaultConfig, lexicon,
		board.CrosswordGameLayout, "English", game.CrossScoreAndSet, game.VarClassic)
	if err != nil {
		t.Fatal(err)
	}
	calcs, _ := defaultSimCalculators(lexicon)
	players := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player1"},
		{Nickname: "p2", RealName: "Player2"},
	}
	// Tiles left in the bag when the game stops.
	stops := []int{84, 80, 55, 45, 20, 9}
	out, err := os.Create(file)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	for i, stop := range stops {
		for seedIdx := i * 100; ; seedIdx++ {
			g, err := game.NewGame(rules, players)
			if err != nil {
				t.Fatal(err)
			}
			var seed [32]byte
			seed[0], seed[1] = byte(seedIdx), byte(seedIdx>>8)
			g.SeedBag(seed)
			g.StartGame()
			p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
			if err != nil {
				t.Fatal(err)
			}
			for g.Playing() == pb.PlayState_PLAYING && g.Bag().TilesRemaining() > stop {
				m := aiturnplayer.GenBestStaticTurn(g, p, g.PlayerOnTurn())
				if err := g.PlayMove(m, false, 0); err != nil {
					t.Fatal(err)
				}
			}
			if g.Playing() != pb.PlayState_PLAYING || g.Bag().TilesRemaining() < 7 {
				continue
			}
			fmt.Fprintln(out, simSpeedCGP(g))
			break
		}
	}
}

// simSpeedCGP is the position with the opponent's rack left out.
func simSpeedCGP(g *game.Game) string {
	full := g.ToCGP(false)
	f := strings.Fields(full)
	// board, racks, scores, zero-point turns, then the operations.
	racks := strings.SplitN(f[1], "/", 2)
	f[1] = racks[0] + "/"
	return strings.Join(f, " ")
}

func simSpeedRun(t *testing.T, file string) {
	in, err := os.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	var positions []string
	sc := bufio.NewScanner(in)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			positions = append(positions, line)
		}
	}
	threadCounts := []int{1, 8, 16}
	if v := os.Getenv("SIM_SPEED_THREADS"); v != "" {
		threadCounts = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			threadCounts = append(threadCounts, n)
		}
	}
	nplays := simSpeedEnvInt("SIM_SPEED_PLAYS", 15)
	iters := simSpeedEnvInt("SIM_SPEED_ITERS", 2000)
	plies := simSpeedEnvInt("SIM_SPEED_PLIES", 5)
	useWMP := os.Getenv("SIM_SPEED_WMP") != "0"

	for pi, pos := range positions {
		for _, threads := range threadCounts {
			parsed, err := cgp.ParseCGP(DefaultConfig, pos)
			if err != nil {
				t.Fatal(err)
			}
			parsed.RecalculateBoard()
			g := parsed.Game
			lexicon := g.LexiconName()
			calcs, leaves := defaultSimCalculators(lexicon)
			p, err := aiturnplayer.NewAIStaticTurnPlayerFromGame(g, DefaultConfig, calcs)
			if err != nil {
				t.Fatal(err)
			}
			plays := p.GenerateMoves(nplays)
			s := &Simmer{}
			s.Init(g, calcs, leaves.(*equity.CombinedStaticCalculator), DefaultConfig)
			s.SetThreads(threads)
			if useWMP {
				s.TryLoadWMP(DefaultConfig.WGLConfig(), lexicon)
			}
			if err := s.PrepareSim(plies, plays); err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			if err := s.simulate(context.Background(), simRun{maxIters: iters, quiet: true}); err != nil {
				t.Fatal(err)
			}
			el := time.Since(start).Seconds()
			rollouts := s.Iterations() * len(plays)
			best := s.WinningPlay()
			fmt.Printf("SIMSPEED pos=%d bag=%d threads=%d plays=%d rollouts=%d secs=%.3f rollouts_per_sec=%.0f wmp=%v best=%q win=%.3f\n",
				pi, g.Bag().TilesRemaining(), threads, len(plays), rollouts, el,
				float64(rollouts)/el, s.WMP() != nil,
				g.Board().MoveDescriptionWithPlaythrough(best.Move()), best.WinProb())
		}
	}
}
