// shadow_test.go contains tests ported from magpie's test/shadow_test.c.
// These test the shadow algorithm's upper bound scoring on various board
// positions. Only KWG-only tests are included (no WMP tests).
package movegen_test

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/cgp"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

var DefaultConfig = config.DefaultConfig()

// loadAndShadow loads a CGP, sets the rack, runs shadow, and returns sorted anchors.
func loadAndShadow(t *testing.T, cgpStr, rackStr string) []movegen.Anchor {
	t.Helper()
	is := is.New(t)

	gh, err := cgp.ParseCGP(DefaultConfig, cgpStr)
	is.NoErr(err)

	g := gh.Game
	g.RecalculateBoard()

	lexName := gh.Game.LexiconName()
	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), lexName)
	is.NoErr(err)

	ld := g.Bag().LetterDistribution()
	alph := gd.GetAlphabet()
	rack := tilemapping.RackFromString(rackStr, alph)

	gen := movegen.NewGordonGenerator(gd, g.Board(), ld)
	anchors := gen.RunShadowOnly(rack)

	// Verify anchors are in descending order of score
	for i := 1; i < len(anchors); i++ {
		if anchors[i].HighestPossibleScore > anchors[i-1].HighestPossibleScore {
			t.Errorf("anchors not sorted: [%d].score=%d > [%d].score=%d",
				i, anchors[i].HighestPossibleScore, i-1, anchors[i-1].HighestPossibleScore)
		}
	}
	return anchors
}

// assertAnchorScore checks that the shadow's upper bound is >= the expected
// score. The shadow's bound must be at least as high as the actual achievable
// score (a too-low bound is a correctness bug). A bound higher than expected
// is safe (just less efficient pruning).
func assertAnchorScore(t *testing.T, anchors []movegen.Anchor, idx, expected int) {
	t.Helper()
	if idx >= len(anchors) {
		t.Errorf("anchor index %d out of range (have %d anchors)", idx, len(anchors))
		return
	}
	actual := anchors[idx].HighestPossibleScore
	if actual < expected {
		t.Errorf("anchor[%d].score = %d, want >= %d (upper bound too low!)", idx, actual, expected)
	}
}

// CGP strings ported from magpie test_constants.h.
// Macondo uses "lex X;" not "-lex X;".
const (
	emptyCGP  = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex CSW21;"
	kaOpening = "15/15/15/15/15/15/15/6KA7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/4 0 lex CSW21;"
	aaOpening = "15/15/15/15/15/15/15/6AA7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/4 0 lex CSW21;"

	dougVEmely       = "15/15/15/15/15/15/15/3WINDY7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;"
	tripleLetters    = "15/15/15/15/15/15/1PROTEAN7/3WINDY7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;"
	tripleDouble     = "15/15/15/15/15/15/15/15/15/2PAV10/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;"
	bottomLeftRE     = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/RE13 ADEEGIL/AEILOUY 0/32 0 lex CSW21;"
	laterBetweenDWS  = "15/15/15/15/5LATER5/15/15/15/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;"
	vsOxy            = "1PACIFYING5/1IS12/YE13/1REQUALIFIED3/H1L12/EDS12/NO3T9/1RAINWASHING3/UM3O9/T2E1O9/1WAKEnERS6/1OnETIME7/OOT2E1B7/N6U7/1JACULATING4 / 0/0 0 lex NWL20;"

	qiQisCGP = "15/15/15/15/15/15/15/6QI7/6I8/6S8/15/15/15/15/15 FRUITED/EGGCUPS 22/12 0 lex CSW21;"

	shuttledRavioli = "14Z/5V7NA/5R4P2AN/5O1N1ARGUFY/5TOO2I2F1/4T1COWKS2E1/2REE1UN2A2R1/1RAVIoLI2G3Q/1EXON1IN2E1P1A/1C1L3GEM2AHI/BEMUD2SHUTTlED/1D1E8AI1/YET9IS1/ODA9ST1/W1JOLLER7 BII/EO 477/388 0 lex CSW21;"
	toelessCGP      = "15/15/15/15/15/15/5Q2J6/5UVAE6/5I2U6/5Z9/15/15/15/15/15 TOELESS/EEGIPRW 42/38 0 lex CSW21;"
	addleCGP        = "5E1p7/1A1C1G1E5S1/1JIZ1G1R3V1P1/HO1APE1A3U1I1/OW1RAD1EXFIL1T1/MA2W2OI2G1T1/IN2K2N2MOBE1/E2FYRDS2o1ANS/3O6V1N1H/3U6ADD1E/3S6BOO1E/3T6LOR1L/INCITANT1AYRE2/3E5U5/3REQUITE5 L/I 467/473 0 lex CSW21;"
	magpiesCGP      = "15/15/15/15/15/15/15/MAGPIE2LUMBAGO/15/15/15/15/15/15/15 SS/Q 0/0 0 lex CSW21;"
	ueyCGP          = "T2F3C7/O2O1BEHOWLING1/A1PI2ME4IO1/s1OD1NUR2AIDA1/T1L1NARTJIES3/I1E2NE1I6/E1A2N2B6/SAXONY1UEY5/2E5D6/15/15/15/15/15/15 ACEOOQV/DEGPRS? 271/283 0 lex CSW21;"
)

// TestShadowScore tests shadow upper bound scoring, ported from magpie's
// test_shadow_score. These test the highest possible score at each anchor.
func TestShadowScore(t *testing.T) {
	// Empty board tests
	anchors := loadAndShadow(t, emptyCGP, "OU")
	if len(anchors) < 1 {
		t.Fatalf("expected >= 1 anchor, got %d", len(anchors))
	}
	assertAnchorScore(t, anchors, 0, 4)

	anchors = loadAndShadow(t, emptyCGP, "ID")
	assertAnchorScore(t, anchors, 0, 6)

	anchors = loadAndShadow(t, emptyCGP, "AX")
	assertAnchorScore(t, anchors, 0, 18)

	anchors = loadAndShadow(t, emptyCGP, "BD")
	assertAnchorScore(t, anchors, 0, 10)

	anchors = loadAndShadow(t, emptyCGP, "QK")
	assertAnchorScore(t, anchors, 0, 30)

	anchors = loadAndShadow(t, emptyCGP, "AESR")
	assertAnchorScore(t, anchors, 0, 8)

	anchors = loadAndShadow(t, emptyCGP, "TNCL")
	assertAnchorScore(t, anchors, 0, 12)

	anchors = loadAndShadow(t, emptyCGP, "AAAAA")
	assertAnchorScore(t, anchors, 0, 12)

	anchors = loadAndShadow(t, emptyCGP, "CAAAA")
	assertAnchorScore(t, anchors, 0, 20)

	anchors = loadAndShadow(t, emptyCGP, "CAKAA")
	assertAnchorScore(t, anchors, 0, 32)

	anchors = loadAndShadow(t, emptyCGP, "AIERZ")
	assertAnchorScore(t, anchors, 0, 48)

	anchors = loadAndShadow(t, emptyCGP, "AIERZN")
	assertAnchorScore(t, anchors, 0, 50)

	anchors = loadAndShadow(t, emptyCGP, "AIERZNL")
	assertAnchorScore(t, anchors, 0, 102)

	anchors = loadAndShadow(t, emptyCGP, "?")
	assertAnchorScore(t, anchors, 0, 0)

	anchors = loadAndShadow(t, emptyCGP, "??")
	assertAnchorScore(t, anchors, 0, 0)

	anchors = loadAndShadow(t, emptyCGP, "??OU")
	assertAnchorScore(t, anchors, 0, 4)

	anchors = loadAndShadow(t, emptyCGP, "??OUA")
	assertAnchorScore(t, anchors, 0, 8)

	// KA opening
	anchors = loadAndShadow(t, kaOpening, "EE")
	if len(anchors) < 6 {
		t.Errorf("KA+EE: expected >= 6 anchors, got %d", len(anchors))
	}
	assertAnchorScore(t, anchors, 0, 10)
	assertAnchorScore(t, anchors, 1, 9)
	assertAnchorScore(t, anchors, 2, 8)
	assertAnchorScore(t, anchors, 3, 5)
	assertAnchorScore(t, anchors, 4, 5)
	assertAnchorScore(t, anchors, 5, 3)

	anchors = loadAndShadow(t, kaOpening, "E?")
	assertAnchorScore(t, anchors, 0, 8)
	assertAnchorScore(t, anchors, 1, 8)
	assertAnchorScore(t, anchors, 2, 8)
	assertAnchorScore(t, anchors, 3, 7)
	assertAnchorScore(t, anchors, 4, 7)
	assertAnchorScore(t, anchors, 5, 7)
	assertAnchorScore(t, anchors, 6, 3)
	assertAnchorScore(t, anchors, 7, 3)
	assertAnchorScore(t, anchors, 8, 2)

	anchors = loadAndShadow(t, kaOpening, "J")
	if len(anchors) < 4 {
		t.Errorf("KA+J: expected >= 4 anchors, got %d", len(anchors))
	}
	assertAnchorScore(t, anchors, 0, 21)
	assertAnchorScore(t, anchors, 1, 14)
	assertAnchorScore(t, anchors, 2, 9)
	assertAnchorScore(t, anchors, 3, 9)

	// AA opening
	anchors = loadAndShadow(t, aaOpening, "JF")
	if len(anchors) < 6 {
		t.Errorf("AA+JF: expected >= 6 anchors, got %d", len(anchors))
	}
	assertAnchorScore(t, anchors, 0, 42)
	assertAnchorScore(t, anchors, 1, 25)
	assertAnchorScore(t, anchors, 2, 25)
	assertAnchorScore(t, anchors, 3, 18)
	assertAnchorScore(t, anchors, 4, 14)
	assertAnchorScore(t, anchors, 5, 13)

	anchors = loadAndShadow(t, aaOpening, "JFU")
	assertAnchorScore(t, anchors, 0, 44)

	anchors = loadAndShadow(t, kaOpening, "JFU")
	assertAnchorScore(t, anchors, 0, 32)

	anchors = loadAndShadow(t, aaOpening, "JFUG")
	assertAnchorScore(t, anchors, 0, 47)

	anchors = loadAndShadow(t, aaOpening, "JFUGX")
	assertAnchorScore(t, anchors, 0, 61)

	anchors = loadAndShadow(t, aaOpening, "JFUGXL")
	assertAnchorScore(t, anchors, 0, 102)

	// DOUG_V_EMELY
	anchors = loadAndShadow(t, dougVEmely, "Q")
	assertAnchorScore(t, anchors, 0, 14)
	assertAnchorScore(t, anchors, 1, 11)

	anchors = loadAndShadow(t, dougVEmely, "BD")
	assertAnchorScore(t, anchors, 0, 14)

	anchors = loadAndShadow(t, dougVEmely, "BOH")
	assertAnchorScore(t, anchors, 0, 24)

	anchors = loadAndShadow(t, dougVEmely, "BOHGX")
	assertAnchorScore(t, anchors, 0, 44)

	anchors = loadAndShadow(t, dougVEmely, "BOHGXZ")
	assertAnchorScore(t, anchors, 0, 116)

	anchors = loadAndShadow(t, dougVEmely, "BOHGXZQ")
	assertAnchorScore(t, anchors, 0, 206)

	// TRIPLE_LETTERS
	anchors = loadAndShadow(t, tripleLetters, "A")
	assertAnchorScore(t, anchors, 0, 6)

	anchors = loadAndShadow(t, tripleLetters, "Z")
	assertAnchorScore(t, anchors, 0, 33)

	anchors = loadAndShadow(t, tripleLetters, "ZLW")
	assertAnchorScore(t, anchors, 0, 73)

	anchors = loadAndShadow(t, tripleLetters, "ZLW?")
	assertAnchorScore(t, anchors, 0, 79)

	anchors = loadAndShadow(t, tripleLetters, "QZLW")
	assertAnchorScore(t, anchors, 0, 85)

	// TRIPLE_DOUBLE
	anchors = loadAndShadow(t, tripleDouble, "K")
	assertAnchorScore(t, anchors, 0, 13)

	anchors = loadAndShadow(t, tripleDouble, "KT")
	assertAnchorScore(t, anchors, 0, 20)

	anchors = loadAndShadow(t, tripleDouble, "KT?")
	assertAnchorScore(t, anchors, 0, 24)

	// BOTTOM_LEFT_RE
	anchors = loadAndShadow(t, bottomLeftRE, "M")
	assertAnchorScore(t, anchors, 0, 8)

	anchors = loadAndShadow(t, bottomLeftRE, "MN")
	assertAnchorScore(t, anchors, 0, 16)

	anchors = loadAndShadow(t, bottomLeftRE, "MNA")
	assertAnchorScore(t, anchors, 0, 20)

	anchors = loadAndShadow(t, bottomLeftRE, "MNAU")
	assertAnchorScore(t, anchors, 0, 22)

	anchors = loadAndShadow(t, bottomLeftRE, "MNAUT")
	assertAnchorScore(t, anchors, 0, 30)

	anchors = loadAndShadow(t, bottomLeftRE, "MNAUTE")
	assertAnchorScore(t, anchors, 0, 39)

	// LATER_BETWEEN_DOUBLE_WORDS
	anchors = loadAndShadow(t, laterBetweenDWS, "Z")
	assertAnchorScore(t, anchors, 0, 21)

	anchors = loadAndShadow(t, laterBetweenDWS, "ZL")
	assertAnchorScore(t, anchors, 0, 64)

	anchors = loadAndShadow(t, laterBetweenDWS, "ZLI")
	assertAnchorScore(t, anchors, 0, 68)

	anchors = loadAndShadow(t, laterBetweenDWS, "ZLIE")
	assertAnchorScore(t, anchors, 0, 72)

	anchors = loadAndShadow(t, laterBetweenDWS, "ZLIER")
	assertAnchorScore(t, anchors, 0, 76)

	anchors = loadAndShadow(t, laterBetweenDWS, "ZLIERA")
	assertAnchorScore(t, anchors, 0, 80)

	anchors = loadAndShadow(t, laterBetweenDWS, "ZLIERAI")
	assertAnchorScore(t, anchors, 0, 212)

	// VS_OXY
	anchors = loadAndShadow(t, vsOxy, "A")
	assertAnchorScore(t, anchors, 0, 18)

	anchors = loadAndShadow(t, vsOxy, "O")
	assertAnchorScore(t, anchors, 0, 63)

	anchors = loadAndShadow(t, vsOxy, "E")
	assertAnchorScore(t, anchors, 0, 72)

	anchors = loadAndShadow(t, vsOxy, "PB")
	assertAnchorScore(t, anchors, 0, 156)

	anchors = loadAndShadow(t, vsOxy, "PA")
	assertAnchorScore(t, anchors, 0, 50)

	anchors = loadAndShadow(t, vsOxy, "PBA")
	assertAnchorScore(t, anchors, 0, 174)

	anchors = loadAndShadow(t, vsOxy, "Z")
	assertAnchorScore(t, anchors, 0, 42)

	anchors = loadAndShadow(t, vsOxy, "ZE")
	assertAnchorScore(t, anchors, 0, 160)

	anchors = loadAndShadow(t, vsOxy, "AZE")
	assertAnchorScore(t, anchors, 0, 184)

	anchors = loadAndShadow(t, vsOxy, "AZEB")
	assertAnchorScore(t, anchors, 0, 484)

	anchors = loadAndShadow(t, vsOxy, "AZEBP")
	assertAnchorScore(t, anchors, 0, 604)

	anchors = loadAndShadow(t, vsOxy, "AZEBPX")
	assertAnchorScore(t, anchors, 0, 686)

	anchors = loadAndShadow(t, vsOxy, "AZEBPXO")
	assertAnchorScore(t, anchors, 0, 1780)

	anchors = loadAndShadow(t, vsOxy, "AZEBPQO")
	assertAnchorScore(t, anchors, 0, 1836)

	// qi_qis
	anchors = loadAndShadow(t, qiQisCGP, "FRUITED")
	if len(anchors) < 8 {
		t.Errorf("qi_qis+FRUITED: expected >= 8 anchors, got %d", len(anchors))
	}
	assertAnchorScore(t, anchors, 0, 128)
	assertAnchorScore(t, anchors, 1, 103)
	assertAnchorScore(t, anchors, 2, 88)

	// shuttled_ravioli
	anchors = loadAndShadow(t, shuttledRavioli, "BII")
	assertAnchorScore(t, anchors, 0, 25)
	assertAnchorScore(t, anchors, 1, 16)
	assertAnchorScore(t, anchors, 2, 15)

	// toeless
	anchors = loadAndShadow(t, toelessCGP, "TOELESS")
	assertAnchorScore(t, anchors, 0, 86)

	// addle
	anchors = loadAndShadow(t, addleCGP, "L")
	assertAnchorScore(t, anchors, 0, 17)
	assertAnchorScore(t, anchors, 1, 17)
	assertAnchorScore(t, anchors, 2, 16)
	assertAnchorScore(t, anchors, 3, 14)
	assertAnchorScore(t, anchors, 4, 10)
	assertAnchorScore(t, anchors, 5, 9)

	// magpies
	anchors = loadAndShadow(t, magpiesCGP, "SS")
	assertAnchorScore(t, anchors, 0, 15)
}

// TestShadowTopPlayAgreement generates all plays (no shadow) and verifies
// that the shadow top play matches the best play from the full move list.
func TestShadowTopPlayAgreement(t *testing.T) {
	is := is.New(t)

	rules, err := game.NewBasicGameRules(DefaultConfig, "NWL23",
		board.CrosswordGameLayout, "English",
		game.CrossScoreAndSet, game.VarClassic)
	is.NoErr(err)

	playerInfos := []*pb.PlayerInfo{
		{Nickname: "p1", RealName: "Player1"},
		{Nickname: "p2", RealName: "Player2"},
	}

	numGames := 1000
	var numDisagreements atomic.Int64
	var totalTurns atomic.Int64

	gd, err := kwg.GetKWG(DefaultConfig.WGLConfig(), "NWL23")
	is.NoErr(err)

	threads := runtime.NumCPU()
	jobs := make(chan int, threads*2)

	var wg sync.WaitGroup
	for th := 0; th < threads; th++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for gidx := range jobs {
				g, err := game.NewGame(rules, playerInfos)
				if err != nil {
					t.Errorf("game create: %v", err)
					return
				}

				seed := [32]byte{}
				seed[0] = byte(gidx)
				seed[1] = byte(gidx >> 8)
				seed[2] = byte(gidx >> 16)
				g.SeedBag(seed)
				g.StartGame()

				genAll := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
				genS := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())

				turnNum := 0
				for g.Playing() == pb.PlayState_PLAYING {
					rack := g.RackFor(g.PlayerOnTurn())

					// Generate all plays (no shadow)
					genAll.SetPlayRecorder(movegen.AllPlaysRecorder)
					allPlays := genAll.GenAll(rack, false)
					topAll := bestByScore(allPlays)

					// Generate with shadow
					genS.SetPlayRecorderTopPlay()
					shadowPlays := genS.GenAll(rack, false)

					totalTurns.Add(1)

					if len(allPlays) > 0 && len(shadowPlays) > 0 {
						if topAll.Score() != shadowPlays[0].Score() {
							t.Errorf("Game %d turn %d: score mismatch all=%d(%s) shadow=%d(%s)",
								gidx, turnNum, topAll.Score(), topAll.ShortDescription(),
								shadowPlays[0].Score(), shadowPlays[0].ShortDescription())
							numDisagreements.Add(1)
						}
					}

					g.PlayMove(allPlays[0], false, 0)
					turnNum++
				}
			}
		}()
	}

	for gidx := 0; gidx < numGames; gidx++ {
		if numDisagreements.Load() >= 10 {
			break
		}
		jobs <- gidx
	}
	close(jobs)
	wg.Wait()

	t.Logf("Played %d games, %d turns, %d disagreements (%d threads)\n",
		numGames, totalTurns.Load(), numDisagreements.Load(), threads)
	is.Equal(int(numDisagreements.Load()), 0)
}

func bestByScore(plays []*move.Move) *move.Move {
	best := plays[0]
	for _, p := range plays[1:] {
		if p.Score() > best.Score() {
			best = p
		}
	}
	return best
}

func sortPlays(plays []*move.Move) {
	sort.Slice(plays, func(i, j int) bool {
		if plays[i].Score() != plays[j].Score() {
			return plays[i].Score() > plays[j].Score()
		}
		return plays[i].ShortDescription() < plays[j].ShortDescription()
	})
}
