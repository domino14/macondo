package preendgame

import (
	"context"
	"fmt"
	"testing"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/tilemapping"
	"github.com/matryer/is"
)

var DefaultConfig = config.DefaultConfig()

func TestMoveTilesToBeginning(t *testing.T) {
	is := is.New(t)
	ld, err := tilemapping.EnglishLetterDistribution(&DefaultConfig)
	is.NoErr(err)
	for i := 0; i < 1000; i++ {

		bag := tilemapping.NewBag(ld, tilemapping.EnglishAlphabet())
		bag.Shuffle()
		// Q, A, B, ?
		lastTiles := []tilemapping.MachineLetter{17, 1, 2, 0}
		moveTilesToBeginning(lastTiles, bag)

		bagTiles := bag.Peek()
		is.Equal(len(bagTiles), 100)
		is.Equal(bagTiles[3], tilemapping.MachineLetter(17))
		is.Equal(bagTiles[2], tilemapping.MachineLetter(1))
		is.Equal(bagTiles[1], tilemapping.MachineLetter(2))
		is.Equal(bagTiles[0], tilemapping.MachineLetter(0))
	}
}

func Test1PEGPass(t *testing.T) {
	is := is.New(t)
	// Pass wins, and has a W-L-D of 5-2-1 in terms of guaranteed wins/ties and possible losses
	cgpStr := "11ONZE/10J2O1/8A1E1DO/7QUETEE1H/10E1F1U/8ECUMERA/8C1R1TIR/7WOKS2ET/6DUR6/5G2N1M4/4HALLALiS3/1G1P1P1OM1XI3/VIVONS1BETEL3/IF1N3AS1RYAL1/ETUDIAIS7 AEINRST/ 301/300 0 lex FRA20; ld french;"
	g, err := cgp.ParseCGP(&DefaultConfig, cgpStr)
	is.NoErr(err)

	g.RecalculateBoard()

	gd, err := kwg.Get(&DefaultConfig, "FRA20")
	is.NoErr(err)
	peg := new(Solver)
	err = peg.Init(g, gd)
	is.NoErr(err)

	ctx := context.Background()
	err = peg.Solve(ctx)
	is.NoErr(err)
	fmt.Println(peg.plays)
}

func TestStraightforward1PEG(t *testing.T) {
	is := is.New(t)
	cgpStr := "15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;"
	g, err := cgp.ParseCGP(&DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	peg := new(Solver)
	err = peg.Init(g, gd)
	is.NoErr(err)
	ctx := context.Background()
	err = peg.Solve(ctx)
	is.NoErr(err)
	// 13L ONYX wins 7.5/8 endgames, tying only with the Y. it is counter-intuitive.
	is.Equal(peg.plays[0].play.ShortDescription(), "13L ONYX")
	is.Equal(peg.plays[0].wins, float32(7.5))
	is.Equal(peg.plays[0].drawsWith, [][]tilemapping.MachineLetter{{25}})
}

// Test a complex pre-endgame with 1 in the bag.
// the best move here is to pass to avoid the Q.
/*
func TestComplicated1PEG(t *testing.T) {
	is := is.New(t)
	// https://www.cross-tables.com/annotated.php?u=42794#26#
	// note: the game above has the wrong rack for Matt. EEILOSS gives the 100% win pass.
	cgpStr := "13AW/11F1LI/10JURAT/9LINER1/8O1T4/5C1WAsTiNG1/4DAMAR1E4/3PARED2ROUEN/2YA1K9/1BERG1OATH4V/3COUP1I5E/3H1TESTILY2N/4FAN1I2OXID/9MIB2U/7ZEES3E EEILOSS/ 297/300 lex NWL20;"
	g, err := cgp.ParseCGP(&DefaultConfig, cgpStr)
	is.NoErr(err)
	gd, err := kwg.Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	peg := new(Solver)
	err = peg.Init(g, gd)
	is.NoErr(err)

}*/
