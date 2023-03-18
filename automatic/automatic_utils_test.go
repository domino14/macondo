package automatic

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog"
)

var DefaultConfig = config.DefaultConfig()

func TestCompVsCompStatic(t *testing.T) {
	logchan := make(chan string)
	runner := NewGameRunner(logchan, &DefaultConfig)
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		err := runner.CompVsCompStatic(false)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(runner.game.Board().ToDisplayText(tilemapping.EnglishAlphabet()))
		close(logchan)
	}()

	go func() {
		for msg := range logchan {
			log.Printf("From logchan: %v", msg)
		}
	}()

	wg.Wait()

	if runner.game.Turn() < 6 {
		t.Errorf("Expected game.turnnum < 6, got %v", runner.game.Turn())
	}
}

func TestPlayerNames(t *testing.T) {
	is := is.New(t)
	is.Equal(playerNames([]AutomaticRunnerPlayer{
		{"", "", macondo.BotRequest_HASTY_BOT},
		{"", "", macondo.BotRequest_HASTY_BOT},
	}), []string{"HastyBot", "HastyBot1"})
	is.Equal(playerNames([]AutomaticRunnerPlayer{
		{"", "", macondo.BotRequest_HASTY_BOT},
		{"", "", macondo.BotRequest_HASTY_BOT},
		{"", "", macondo.BotRequest_HASTY_BOT},
	}), []string{"HastyBot", "HastyBot1", "HastyBot2"})
	is.Equal(playerNames([]AutomaticRunnerPlayer{
		{"", "", macondo.BotRequest_HASTY_BOT},
		{"", "", macondo.BotRequest_NO_LEAVE_BOT},
		{"", "", macondo.BotRequest_HASTY_BOT},
	}), []string{"HastyBot", "NoLeaveBot", "HastyBot1"})
	is.Equal(playerNames([]AutomaticRunnerPlayer{
		{"", "", macondo.BotRequest_NO_LEAVE_BOT},
		{"", "", macondo.BotRequest_HASTY_BOT},
		{"", "", macondo.BotRequest_HASTY_BOT},
	}), []string{"NoLeaveBot", "HastyBot", "HastyBot1"})
	is.Equal(playerNames([]AutomaticRunnerPlayer{
		{"", "", macondo.BotRequest_LEVEL1_CEL_BOT},
		{"", "", macondo.BotRequest_LEVEL3_CEL_BOT},
	}), []string{"Level1CelBot", "Level3CelBot"})
}

func BenchmarkCompVsCompStatic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner := NewGameRunner(nil, &DefaultConfig)
		runner.CompVsCompStatic(false)
	}
}

func BenchmarkPlayFull(b *testing.B) {
	// themonolith - 12th gen linux computer
	// 87	  12813797 ns/op	    4971 B/op	     140 allocs/op
	runner := NewGameRunner(nil, &DefaultConfig)
	for i := 0; i < b.N; i++ {
		runner.playFull(false, i)
	}
}

func TestCompVCompSeries(t *testing.T) {
	is := is.New(t)
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	nGames := 400
	nThreads := 4
	err := StartCompVCompStaticGames(
		context.Background(), &DefaultConfig, nGames, true, nThreads,
		"/tmp/testcompvcomp.txt", "NWL20", "English",
		[]AutomaticRunnerPlayer{
			{"", "", macondo.BotRequest_HASTY_BOT, 0},
			{"", "", macondo.BotRequest_NO_LEAVE_BOT, 0},
		})

	is.NoErr(err)

	// Ensure every logged game is not corrupt
	fin, err := os.Open("/tmp/games-testcompvcomp.txt")
	is.NoErr(err)
	cr := csv.NewReader(fin)
	gameIDs := []string{}
	for {
		record, err := cr.Read()
		if err == io.EOF {
			break
		}
		is.NoErr(err)
		if record[0] == "gameID" {
			continue
		}
		gameIDs = append(gameIDs, record[0])
	}
	is.Equal(len(gameIDs), nGames)

	for _, gid := range gameIDs {
		f := new(strings.Builder)
		err := ExportGCG(&DefaultConfig, "/tmp/testcompvcomp.txt", "english", "NWL20", board.CrosswordGameLayout,
			gid, f)
		is.NoErr(err)
	}

	s, err := AnalyzeLogFile("/tmp/games-testcompvcomp.txt")
	is.NoErr(err)

	// HastyBot should win more than 64% of its games roughly. Let's make
	// it 55% for this test, or 220. Test fails 0.013% of the time which
	// seems acceptable.
	r, err := regexp.Compile(`HastyBot wins: ?(\d+\.?\d+)\s*`)
	is.NoErr(err)
	subMatches := r.FindSubmatch([]byte(s))
	is.Equal(len(subMatches), 2)
	f, err := strconv.ParseFloat(string(subMatches[1]), 64)
	is.NoErr(err)
	is.True(f > 220)
}
