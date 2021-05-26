package automatic

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/config"

	"github.com/domino14/macondo/gaddagmaker"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"NWL20"} {
		gdgPath := filepath.Join(DefaultConfig.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(DefaultConfig.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
	os.Exit(m.Run())
}

func TestCompVsCompStatic(t *testing.T) {
	logchan := make(chan string)
	runner := NewGameRunner(logchan, &DefaultConfig)
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		err := runner.CompVsCompStatic()
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
		runner.CompVsCompStatic()
	}
}

func BenchmarkPlayFullStatic(b *testing.B) {
	runner := NewGameRunner(nil, &DefaultConfig)
	for i := 0; i < b.N; i++ {
		runner.playFullStatic()
	}
}
