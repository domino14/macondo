package xwordgame

import (
	"log"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/nwl18.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "NWL18.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/nwl18.gaddag")
	}
	os.Exit(m.Run())
}
func TestCompVsCompStatic(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	logchan := make(chan string)
	game := &XWordGame{logchan: logchan}
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		game.CompVsCompStatic(gd)
		close(logchan)
	}()

	go func() {
		for msg := range logchan {
			log.Printf("From logchan: %v", msg)
		}
	}()

	wg.Wait()

	if game.turnnum < 6 {
		t.Errorf("Expected game.turnnum < 6, got %v", game.turnnum)
	}
}

func BenchmarkCompVsCompStatic(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		game := &XWordGame{}
		game.CompVsCompStatic(gd)
	}
}

func BenchmarkPlayFullStatic(b *testing.B) {
	gd := gaddag.LoadGaddag("/tmp/nwl18.gaddag")
	game := &XWordGame{}
	game.Init(gd)
	for i := 0; i < b.N; i++ {
		game.playFullStatic()
	}
}
