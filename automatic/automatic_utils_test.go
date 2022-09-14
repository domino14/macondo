package automatic

import (
	"fmt"
	"log"
	"os"
	"sync"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/testcommon"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	testcommon.CreateGaddags(DefaultConfig, []string{"NWL20"})
	os.Exit(m.Run())
}

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
		fmt.Println(runner.game.Board().ToDisplayText(alphabet.EnglishAlphabet()))
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

func BenchmarkCompVsCompStatic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner := NewGameRunner(nil, &DefaultConfig)
		runner.CompVsCompStatic(false)
	}
}

func BenchmarkPlayFullStatic(b *testing.B) {
	// themonolith - 12th gen linux computer
	// 87	  12813797 ns/op	    4971 B/op	     140 allocs/op
	runner := NewGameRunner(nil, &DefaultConfig)
	for i := 0; i < b.N; i++ {
		runner.playFullStatic(false)
	}
}
