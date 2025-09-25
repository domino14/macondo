package preendgame

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"
	"github.com/rs/zerolog"

	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/testhelpers"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	// endgame/pre-endgame debug logs are very noisy.
	level := zerolog.GlobalLevel()
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	exitVal := m.Run()
	zerolog.SetGlobalLevel(level)

	os.Exit(exitVal)
}

func TestMoveTilesToBeginning(t *testing.T) {
	is := is.New(t)
	ld, err := tilemapping.EnglishLetterDistribution(DefaultConfig.WGLConfig())
	is.NoErr(err)

	ct := func(bag *tilemapping.Bag, letter tilemapping.MachineLetter, from int) int {
		n := 0
		for i := from; i < len(bag.Tiles()); i++ {
			if bag.Tiles()[i] == letter {
				n++
			}
		}
		return n
	}

	for i := 0; i < 1000; i++ {

		bag := tilemapping.NewBag(ld, testhelpers.EnglishAlphabet())
		bag.Shuffle()
		// Q, A, B, ?
		lastTiles := []tilemapping.MachineLetter{17, 1, 2, 0}
		moveTilesToBeginning(lastTiles, bag)

		bagTiles := bag.Peek()
		is.Equal(len(bagTiles), 100)
		is.Equal(bagTiles[0], tilemapping.MachineLetter(17))
		is.Equal(bagTiles[1], tilemapping.MachineLetter(1))
		is.Equal(bagTiles[2], tilemapping.MachineLetter(2))
		is.Equal(bagTiles[3], tilemapping.MachineLetter(0))
		is.Equal(ct(bag, 17, 4), 0)
		is.Equal(ct(bag, 1, 4), 8)
		is.Equal(ct(bag, 2, 4), 1)
		is.Equal(ct(bag, 0, 4), 1)
	}

	for i := 0; i < 1000; i++ {

		bag := tilemapping.NewBag(ld, testhelpers.EnglishAlphabet())
		bag.Shuffle()
		// E, E, E, E
		lastTiles := []tilemapping.MachineLetter{5, 5, 5, 5}
		moveTilesToBeginning(lastTiles, bag)

		bagTiles := bag.Peek()
		is.Equal(len(bagTiles), 100)
		is.Equal(bagTiles[3], tilemapping.MachineLetter(5))
		is.Equal(bagTiles[2], tilemapping.MachineLetter(5))
		is.Equal(bagTiles[1], tilemapping.MachineLetter(5))
		is.Equal(bagTiles[0], tilemapping.MachineLetter(5))
		is.Equal(ct(bag, 5, 4), 8)
	}
}

func Test1PEGPass(t *testing.T) {
	is := is.New(t)
	// Pass wins, and has a W-L-D of 5-2-1 in terms of guaranteed wins/ties and possible losses
	cgpStr := "11ONZE/10J2O1/8A1E1DO/7QUETEE1H/10E1F1U/8ECUMERA/8C1R1TIR/7WOKS2ET/6DUR6/5G2N1M4/4HALLALiS3/1G1P1P1OM1XI3/VIVONS1BETEL3/IF1N3AS1RYAL1/ETUDIAIS7 AEINRST/ 301/300 0 lex FRA20; ld french;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)

	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "FRA20")
	is.NoErr(err)

	peg := new(Solver)
	err = peg.Init(g.Game, gd)
	is.NoErr(err)

	ctx := context.Background()
	plays, err := peg.Solve(ctx)
	is.NoErr(err)
	is.Equal(plays[0].Play.ShortDescription(), "(Pass)")
	is.Equal(plays[0].Points, float32(5.5))
	// Wins for ?, A, I, N, U, draw for B, Loss for both Es
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{0}), PEGWin)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{1}), PEGWin)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{9}), PEGWin)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{14}), PEGWin)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{21}), PEGWin)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{2}), PEGDraw)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{5}), PEGLoss)
	is.Equal(len(plays[0].outcomesArray), 7)
}

func TestStraightforward1PEG(t *testing.T) {
	is := is.New(t)
	cgpStr := "15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)
	ctx := context.Background()
	plays, err := peg.Solve(ctx)
	is.NoErr(err)
	// 13L ONYX wins 7.5/8 endgames, tying only with the Y. it is counter-intuitive.
	is.Equal(plays[0].Play.ShortDescription(), "13L ONYX")
	is.Equal(plays[0].Points, float32(7.5))
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{25}), PEGDraw)
}

func TestKnownTilesPEG(t *testing.T) {
	is := is.New(t)
	cgpStr := "15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)
	ctx := context.Background()
	// Set known opp rack to AACEISU. This means the Y is surely in the bag.
	// The previous best move of ONYX wins 7.5/8 endgames tying only with the Y.
	// STEY, STYE, and OXY win if the Y is in the bag, so those are the new winners.
	peg.SetKnownOppRack([]tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21})
	plays, err := peg.Solve(ctx)
	is.NoErr(err)
	is.Equal(plays[0].Play.ShortDescription(), " N3 STEY")
	is.Equal(plays[0].Points, float32(1.0))
	is.Equal(plays[0].Spread, 10)
	is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{25}), PEGWin)

	is.Equal(plays[1].Play.ShortDescription(), " N3 STYE")
	is.Equal(plays[1].Points, float32(1.0))
	is.Equal(plays[1].Spread, 9)
	is.Equal(plays[1].OutcomeFor([]tilemapping.MachineLetter{25}), PEGWin)

	is.Equal(plays[2].Play.ShortDescription(), "13L OXY")
	is.Equal(plays[2].Points, float32(1.0))
	is.Equal(plays[2].Spread, 1)
	is.Equal(plays[2].OutcomeFor([]tilemapping.MachineLetter{25}), PEGWin)
}

// This benchmark is really only here to measure allocations etc.
func BenchmarkStraightforward1PEG(b *testing.B) {
	is := is.New(b)
	cgpStr := "15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)
	ctx := context.Background()
	peg.SetIterativeDeepening(false)
	b.ResetTimer()

	// 10/24/23 - 1		16283647648 ns/op	43704360960 B/op	831437490 allocs/op
	// 11/5/23 -  1		15224121119 ns/op	19675753312 B/op	143766012 allocs/op
	// 11/9/23 -  1		14481023059 ns/op	18559717408 B/op	53698650 allocs/op
	// 11/11/23 - 1		13938763155 ns/op	18363324144 B/op	52887568 allocs/op
	// 11/11/23 - 1		13747110199 ns/op	11981979688 B/op	37294808 allocs/op
	// 11/18/23 - 1		8170869576 ns/op	10322733488 B/op	21333124 allocs/op
	// 11/26/23 - 1		5073027621 ns/op	9482291240 B/op		12543465 allocs/op
	// 11/29/23 - 1		3247604595 ns/op	9274420416 B/op	 	9328476 allocs/op
	// 3/3/24   - 1		3157266203 ns/op	9241820392 B/op	 	8688517 allocs/op
	// With striped-table mutexes: (Think there was a TT-related very rare bug otherwise)
	// 3/11/24  - 1		3267509980 ns/op	9348210584 B/op		10011243 allocs/op
	for i := 0; i < b.N; i++ {
		plays, err := peg.Solve(ctx)
		is.NoErr(err)

		// 13L ONYX wins 7.5/8 endgames, tying only with the Y. it is counter-intuitive.
		is.Equal(plays[0].Play.ShortDescription(), "13L ONYX")
		is.Equal(plays[0].Points, float32(7.5))
		is.Equal(plays[0].OutcomeFor([]tilemapping.MachineLetter{25}), PEGDraw)
	}
}

func BenchmarkSlowPEG(b *testing.B) {
	is := is.New(b)

	cgpStr := "AnORETIC7/7R7/2C1EMEU7/1JANNY1X7/2P12/1SIDELING6/2ZAG7Q2/3MOVED3AA2/6FEAL1IT2/2NEGATES2D3/3DOH2KEEFS2/WITH7UN2/I10LO2/LABOUR6B2/Y6POTTOS2 ?AENORW/EIIIRUV 332/384 0 lex NWL20;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)
	ctx := context.Background()
	peg.SetEndgamePlies(5)
	peg.SetIterativeDeepening(true)
	b.ResetTimer()
	// ~135.7 seconds on themonolith
	for i := 0; i < b.N; i++ {
		plays, err := peg.Solve(ctx)
		is.NoErr(err)

		is.Equal(plays[0].Play.ShortDescription(), "14M .O")
		is.Equal(plays[0].Points, float32(7))
	}
}

// Test a complex pre-endgame with 1 in the bag.
// There are several winning moves, one of them being a pass. Note that we
// need to look forward a bit more (increase endgame plies to at least 7) since there is a Q
// stick situation that isn't handled properly otherwise.

func TestComplicated1PEG(t *testing.T) {
	t.Skip()
	is := is.New(t)
	// https://www.cross-tables.com/annotated.php?u=42794#26#
	// note: the game above has the wrong rack for Matt. EEILOSS gives the 100% win pass.
	cgpStr := "13AW/11F1LI/10JURAT/9LINER1/8O1T4/5C1WAsTiNG1/4DAMAR1E4/3PARED2ROUEN/2YA1K9/1BERG1OATH4V/3COUP1I5E/3H1TESTILY2N/4FAN1I2OXID/9MIB2U/7ZEES3E EEILOSS/ 297/300 0 lex NWL20;"
	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	peg.maxEndgamePlies = 7
	is.NoErr(err)

	ctx := context.Background()
	winners, err := peg.Solve(ctx)
	is.NoErr(err)
	is.Equal(winners[0].Points, float32(8))
	is.Equal(winners[6].Points, float32(8))
	is.Equal(winners[7].Points, float32(7))
}

func TestPolishPEGEarlyCutoff(t *testing.T) {
	is := is.New(t)
	cgpStr := "13P1/12SYĆ/11HET1/11IMA1/3C6T1EN1/2PiLŻE2WOŹNI1/1K1C5SI1Y2/DASHI2ZATNĘ3/1R1A4BAO4/1M1C1Z1ROŃ5/3ZŁU1A7/3Y1PÓKI1WERS1/GOJ2Y1IWIE4/1BOA3JĄŁ5/KANADZIE7 GLLMNOŚ/?DEFRUZ 307/288 0 lex OSPS49; ld polish;"

	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "OSPS49")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	peg.maxEndgamePlies = 5
	peg.earlyCutoffOptim = true
	peg.iterativeDeepening = false
	is.NoErr(err)

	ctx := context.Background()
	winners, err := peg.Solve(ctx)
	is.NoErr(err)
	is.Equal(winners[0].Points, float32(3))
	is.Equal(winners[0].Play.ShortDescription(), " G4 ŚL.")
}

func TestPossibleTilesInBag(t *testing.T) {
	is := is.New(t)

	mTiles := []tilemapping.MachineLetter{3, 1, 21, 19, 5, 25}
	unseenTiles := []tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21, 25}
	pt := possibleTilesInBag(unseenTiles, mTiles, nil)
	is.Equal(pt, []tilemapping.MachineLetter{1, 9})

	unseenTiles = []tilemapping.MachineLetter{1, 1, 1, 3, 5, 19, 21, 25}
	pt = possibleTilesInBag(unseenTiles, mTiles, nil)
	is.Equal(pt, []tilemapping.MachineLetter{1})

	mTiles = []tilemapping.MachineLetter{1, 1}
	unseenTiles = []tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21, 25}
	pt = possibleTilesInBag(unseenTiles, mTiles, nil)
	is.Equal(pt, []tilemapping.MachineLetter{3, 5, 9, 19, 21, 25})

	// player plays AA, unseen is AACEISUY, known rack is ACEIY
	// possible tiles in the bag should be SU
	unseenTiles = []tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21, 25}
	knownRack := []tilemapping.MachineLetter{1, 3, 5, 9, 25}
	pt = possibleTilesInBag(unseenTiles, mTiles, knownRack)
	is.Equal(pt, []tilemapping.MachineLetter{19, 21})

	unseenTiles = []tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21, 25}
	knownRack = []tilemapping.MachineLetter{1, 1, 3, 5, 9, 25}
	pt = possibleTilesInBag(unseenTiles, mTiles, knownRack)
	is.Equal(pt, []tilemapping.MachineLetter{19, 21})

	unseenTiles = []tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21, 25}
	knownRack = []tilemapping.MachineLetter{3, 5, 9, 19, 21, 1}
	pt = possibleTilesInBag(unseenTiles, mTiles, knownRack)
	is.Equal(pt, []tilemapping.MachineLetter{25})

	// player plays SAUCY. we know the unseen tiles are AACEISUY.
	// we know they had ACEISU.
	// so the only tile that can be in the bag is an A.
	mTiles = []tilemapping.MachineLetter{19, 1, 21, 3, 25}
	unseenTiles = []tilemapping.MachineLetter{1, 1, 3, 5, 9, 19, 21, 25}
	knownRack = []tilemapping.MachineLetter{1, 3, 5, 9, 19, 21}
	pt = possibleTilesInBag(unseenTiles, mTiles, knownRack)
	is.Equal(pt, []tilemapping.MachineLetter{1})
}

func TestMoveIsPossible(t *testing.T) {
	is := is.New(t)
	m := []tilemapping.MachineLetter{3, 15, 15, 11, 9, 5}
	partial := []tilemapping.MachineLetter{11, 12, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{11, 12}
	is.True(moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{3, 15, 15, 11, 9, 12}
	is.True(moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{3, 15, 15, 11, 9, 12, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{15, 15, 11, 9, 12, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{15, 15, 11, 9, 12}
	is.True(moveIsPossible(m, partial))

	// COoKIE
	m = []tilemapping.MachineLetter{3, 15, 15 | 0x80, 11, 9, 5}
	partial = []tilemapping.MachineLetter{11, 12, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{11, 12}
	is.True(moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{3, 15, 0, 11, 9, 12}
	is.True(moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{3, 15, 0, 11, 9, 12, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{15, 0, 11, 9, 12, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{15, 0, 11, 9, 12}
	is.True(moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{15, 15, 11, 9, 12}
	is.True(!moveIsPossible(m, partial))

	partial = []tilemapping.MachineLetter{15, 15, 11, 9, 0}
	is.True(moveIsPossible(m, partial))

	// SAUCY
	partial = []tilemapping.MachineLetter{1, 3, 5, 9, 19, 21}
	m = []tilemapping.MachineLetter{19, 1, 21, 3, 25}
	is.True(moveIsPossible(m, partial))

}

func TestTwoInBag(t *testing.T) {
	t.Skip()
	is := is.New(t)
	// https://www.cross-tables.com/annotated.php?u=34161#17
	cgpStr := "1T13/1W3Q9/VERB1U9/1E1OPIUM5C1/1LAWIN1I5O1/1Y3A1E5R1/7V4NO1/NOTArIZE1C2UN1/6ODAH2LA1/3TAHA2I2LED/2JUT4R2A1O/3G5P4D/3R3BrIEFING/3I5L4E/3K2DESYNES1M AEFGSTX/EEIOOST 370/341 0 lex CSW19;"

	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	peg.maxEndgamePlies = 2
	peg.skipTiebreaker = true
	is.NoErr(err)

	ctx := context.Background()
	winners, err := peg.Solve(ctx)

	is.NoErr(err)
	fmt.Println(winners)
}

func TestTwoInBagSingleMove(t *testing.T) {
	is := is.New(t)
	// https://www.cross-tables.com/annotated.php?u=34161#17
	cgpStr := "1T13/1W3Q9/VERB1U9/1E1OPIUM5C1/1LAWIN1I5O1/1Y3A1E5R1/7V4NO1/NOTArIZE1C2UN1/6ODAH2LA1/3TAHA2I2LED/2JUT4R2A1O/3G5P4D/3R3BrIEFING/3I5L4E/3K2DESYNES1M AEFGSTX/EEIOOST 370/341 0 lex CSW19;"

	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW19")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)
	peg.maxEndgamePlies = 3
	peg.iterativeDeepening = false
	m := move.NewScoringMoveSimple(10, "6F", ".X.", "AEFGST", g.Alphabet())
	peg.solveOnlyMoves = []*move.Move{m}
	ctx := context.Background()
	winners, err := peg.Solve(ctx)

	is.NoErr(err)
	is.Equal(winners[0].Points, float32(70)) // wins 70/72 games
	// It only loses games where IE are in the bag, in that order (i.e. "we" draw the I
	// after the X and opp bingoes with TOREROS)
	// Note that internally we represent bag right to left:
	is.Equal(winners[0].OutcomeFor([]tilemapping.MachineLetter{5, 9}), PEGLoss)
	is.Equal(winners[0].OutcomeFor([]tilemapping.MachineLetter{9, 5}), PEGWin)
}

// The problem with this test is as follows:
// By the time it gets to the bag option AH (which is backwards, HA is in the
// bag, in that order), it does this:
// - Plays HIED for US, draws the H
// - Plays every other possible play for opponent with a rack of ACEEINO. They
// all lose in the endgame.
// - Finally, play Pass for opponent
// - Plays Pass for us (it tries pass first as an optimization)
// - This causes the game to end and we lose the game since we lose more
// points than the opponent does on our tiles.
// - This sets an UnfinalizedWinPctStat of PEGLoss for this move (HIED) because
// we lost a game. It incorrectly thinks that because we lost after the opponent
// passed, and we passed back, that we're going to lose this eventuality. We have
// a perfectly valid SCATHING play that wins the game easily.
// - So,
// - We need to see the position from our perspective after the opponent has
// passed. We have a 1-in-the bag pre-endgame:
/*
//	  A B C D E F G H I J K L M N O     ->              player1  ACHINST  353
//	  ------------------------------                    player2           357
//	1|=     '       =       ' B E N |
//	2|  -       "       J U D O S   |   Bag + unseen: (8)
//	3|    -       '   '     R - T E |
//	4|'     -       B R A V I     X |   A C E E E I N O
//	5|        -   Q I     - L     U |
//	6|  "       P   O K A   L   " L |
//	7|    '     O ' T O M E   '   T |
//	8|=     '   U   A A   N ' O R e |
//	9|    '     R '   '   G   F I D |
// 10|  "       I       " R     Z   |   Turn 2:
// 11|      H I E D       A   V A U |   player2 passed, holding a rack of ACEEINO
// 12|'     I D S   P   F I N O   ' |
// 13|    T E E   ' O Y   L   W     |
// 14|  - Y   M a R T E N S     -   |
// 15|G A G E       E W     '     = |
//	  ------------------------------
//
//	If we have _any_ play that wins ALL endgames, then we should mark the top-level play
// (HIED) as a win. If SCATHING won 7 out of 8 endgames, and that was the biggest number,
// that means that the opponent might win after passing. In that case, we should
// check for which endgames SCATHING wins. If it loses for the endgame that has the
// A in the bag, then we mark HIED as a loss for the (H, A) in the bag scenario,
// since the opponent can win by passing, then us playing our best play of SCATHING,
// and then losing because we drew the A.
// If we have three plays that are tied for winning 7 out of 8 endgames, let's say,
// and two of them lose if we draw the A, and the third one wins if we draw the A,
// we still have to mark HIED as a loss for the (H, A) in the bag scenario,
// because we don't know which of the three plays we might pick. We have to make
// sure that our best pre-endgame play(s) all win if we draw the A (in this example).
*/
func TestAnotherTwoInBag(t *testing.T) {
	is := is.New(t)
	cgpStr := "12BEN/9JUDOS1/11R1TE/7BRAVI2X/6QI3L2U/5P1OKA1L2L/5O1TOME3T/5U1AA1N1ORe/5R4G1FID/5I4R2Z1/3HIE4A1VAU/3IDS1P1FINO2/2TEE2OY1L1W2/2Y1MaRTENS4/GAGE3EW6 ACDINST/ACEEHIO 345/357 0 lex CSW21;"

	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "CSW21")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)

	m := move.NewScoringMoveSimple(8, "11D", "...D", "ACINST", g.Alphabet())

	peg.SetEndgamePlies(4)
	peg.SetEarlyCutoffOptim(false)
	peg.SetSkipNonEmptyingOptim(false)
	peg.SetSkipTiebreaker(false)
	peg.SetSkipLossOptim(false)
	peg.SetIterativeDeepening(false)
	peg.SetSolveOnly([]*move.Move{m})

	ctx := context.Background()
	winners, err := peg.Solve(ctx)
	fmt.Println(peg.SolutionStats(1))
	is.NoErr(err)
	is.Equal(winners[0].Points, float32(28.5)) // wins 27/72 and ties 3/72 games
	/**
		W-L-T: 27-42-3
	Guaranteed Wins: 'H+A H+C H+E H+I H+N H+O O+A O+C O+E O+H O+I O+N E+C N+A N+C N+E N+H N+I N+O'
	Guaranteed Ties: 'E+H'
	Possible Losses: 'A+E A+O A+I A+N A+C A+H E+A E+E E+I E+N E+O I+C I+E I+H I+A I+N I+O C+E C+H C+I C+N C+O C+A'
	*/
}

func TestFourInBag(t *testing.T) {
	// This test is not expected to finish in any reasonable amount of time yet.
	// It is only here aspirationally.
	t.Skip()
	is := is.New(t)

	cgpStr := "7LITERARY/6QI7/1YET3NEBULA2/2FAX2G7/4INVOKED4/9T5/9E5/5AVOWs5/9I5/1CLIME1R1A5/4ENWOUND4/PATEN1HO5J1/L5OF4BIG/U5AI1HUE1G1/M6EDITRESS ACEOOSZ/ANOPRRT 331/336 0 lex NWL20;"

	g, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)
	g.RecalculateBoard()

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL20")
	is.NoErr(err)
	peg := new(Solver)

	err = peg.Init(g.Game, gd)
	is.NoErr(err)
	peg.maxEndgamePlies = 4
	peg.iterativeDeepening = true

	ctx := context.Background()
	_, err = peg.Solve(ctx)

	is.NoErr(err)

}
