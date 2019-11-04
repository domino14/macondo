package automatic

import (
	"log"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
)

var LexiconDir = os.Getenv("LEXICON_PATH")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/nwl18.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "NWL18.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/nwl18.gaddag")
	}
	os.Exit(m.Run())
}
func TestCompVsCompStatic(t *testing.T) {
	gd, err := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	if err != nil {
		t.Errorf("expected err to be nil, got %v", err)
	}
	logchan := make(chan string)
	runner := &GameRunner{logchan: logchan}
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		runner.CompVsCompStatic(gd)
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
	gd, _ := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner := &GameRunner{}
		runner.CompVsCompStatic(gd)
	}
}

func BenchmarkPlayFullStatic(b *testing.B) {
	gd, _ := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	runner := &GameRunner{}
	runner.Init(gd)
	for i := 0; i < b.N; i++ {
		runner.playFullStatic()
	}
}
