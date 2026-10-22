package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/domino14/word-golib/kwg"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// A second logged game, with a five-position endgame (bag empty before
// turns 19-23), so the vector checks cover empty-bag positions too.
var endgameGameTurns = []string{
	"p1,685417a6010f67545800000f,1,AADLUVW, 8G VAW,18,18,3,ADLU,14.113,86,0",
	"p2,685417a6010f67545800000f,2,NNOSTVY, H7 N.VY,10,10,3,NOST,20.645,83,18",
	"p1,685417a6010f67545800000f,3,?ADLOPU, J2 UPLOADs,72,90,7,,72.000,80,10",
	"p2,685417a6010f67545800000f,4,AEJNOST,10F JO.,29,39,2,AENST,52.473,73,90",
	"p1,685417a6010f67545800000f,5,AILOSUU, 3J .OILU,14,104,4,ASU,16.279,71,39",
	"p2,685417a6010f67545800000f,6,AAENRST, O3 SANTERA,82,121,7,,82.000,67,104",
	"p1,685417a6010f67545800000f,7,AAEIMSU, N9 AMUSIA,30,134,6,E,30.584,60,121",
	"p2,685417a6010f67545800000f,8,BEEEEGN, L1 BE.GNE,24,145,5,EE,19.820,54,134",
	"p1,685417a6010f67545800000f,9,?AAEELS,12H ELAStA.E,68,202,7,,68.000,49,145",
	"p2,685417a6010f67545800000f,10,ACEEKMR, 4C MACKERE.,84,229,7,,84.000,42,202",
	"p1,685417a6010f67545800000f,11,CDIOOTY, C2 CO.ITY,34,236,5,DO,33.049,35,229",
	"p2,685417a6010f67545800000f,12,DIINOSX,15H DIOXINS,107,336,7,,107.000,30,236",
	"p1,685417a6010f67545800000f,13,DEEFOOU,11E OOF,26,262,3,DEEU,18.918,23,336",
	"p2,685417a6010f67545800000f,14,EGHIIOT,12A HOGTIE,40,376,6,I,37.708,20,262",
	"p1,685417a6010f67545800000f,15,DEEENPU,14J PE,29,291,2,DEENU,26.272,14,376",
	"p2,685417a6010f67545800000f,16,BHIIRRT, A8 BIRT.,30,406,4,HIR,30.478,12,291",
	"p1,685417a6010f67545800000f,17,DEEFNQU, H1 QUE.N,42,333,4,DEF,44.052,8,406",
	"p2,685417a6010f67545800000f,18,GHILRTW, C9 WRI.HT,34,440,5,GL,29.985,4,333",
	"p1,685417a6010f67545800000f,19,DEEFIRR, 1L .EEF,27,360,3,DIRR,7.000,0,440",
	"p2,685417a6010f67545800000f,20,DGLNTZ, 6I G.Z.D,16,456,3,LNT,0.000,0,360",
	"p1,685417a6010f67545800000f,21,DIRR,13L IR.D,18,378,3,R,6.000,0,456",
	"p2,685417a6010f67545800000f,22,LNT, L1 ......T,10,466,1,LN,-4.000,0,378",
	"p1,685417a6010f67545800000f,23,R, 3G R.,6,388,1,,10.000,0,466",
}

// The producer builds training vectors by replaying a log; the bot builds
// inference vectors by playing each candidate on a scratch copy
// (game.MLVectorsForMoves). Both are meant to describe the same state: the
// position right after the move, the mover holding only their leave, the
// opponent's rack in the unseen pool, nothing drawn. For every turn of a
// logged game, the inference vector for the move actually played must be
// identical to the producer's training vector for that turn.
func TestInferenceVectorsMatchTraining(t *testing.T) {
	for name, rows := range map[string][]string{
		"sample game":  sampleGameTurns,
		"endgame game": endgameGameTurns,
		"exchange":     {"p1,exchgame,1,AAEIOUV,(exch AAEIOU),0,0,6,V,-20.000,86,0", "p2,exchgame,2,DEHLORW, 8D WORLD,26,26,5,EH,30.000,81,0"},
	} {
		t.Run(name, func(t *testing.T) {
			checkVectors(t, rows)
		})
	}
}

func checkVectors(t *testing.T, rows []string) {
	t.Helper()
	hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
	var turns []Turn
	sc := NewTurnScanner(strings.NewReader(hdr + strings.Join(rows, "\n") + "\n"))
	for sc.Scan() {
		turns = append(turns, sc.Turn())
	}
	gid := turns[0].GameID

	// Training vectors, by turn.
	table := NewGameAssembler(NPlies)
	want := map[int]outputVector{}
	for _, tn := range turns {
		for _, v := range table.FeedTurn(tn) {
			want[v.turn] = v
		}
	}
	// An unfinished game (the exchange case) never releases; take the
	// pending vectors directly.
	if gw := table.games[gid]; gw != nil {
		for _, v := range gw.pending {
			want[v.turn] = v
		}
	}

	for i, tn := range turns {
		ref, ok := want[tn.TurnNumber]
		if !ok {
			continue // last turn of a finished game: never a training position
		}
		// Replay to just before this turn, then do what the game runner does.
		ga := NewGameAssembler(NPlies)
		var history []*move.Move
		gw := ga.newGameWindow(turns[0])
		ga.games[gid] = gw
		for _, prev := range turns[:i] {
			ga.FeedTurn(prev)
			history = append(history, gw.plies[len(gw.plies)-1].move)
		}
		g := gw.game.Game
		onturn := g.PlayerOnTurn()
		if err := g.SetRackFor(onturn, tilemapping.RackFromString(tn.Rack, g.Alphabet())); err != nil {
			t.Fatalf("turn %d: set rack: %v", tn.TurnNumber, err)
		}
		m, err := gw.game.ParseMove(onturn, false, strings.Fields(stats.Normalize(tn.Play)), shouldTranspose(gid))
		if err != nil {
			t.Fatalf("turn %d: parse move: %v", tn.TurnNumber, err)
		}
		planes, scalars, err := g.MLVectorsForMoves([]*move.Move{m}, ga.eqCalc, history)
		if err != nil {
			t.Fatalf("turn %d: inference vectors: %v", tn.TurnNumber, err)
		}
		got := append(planes, scalars...)
		if len(got) != len(*ref.features) {
			t.Fatalf("turn %d: vector lengths %d vs %d", tn.TurnNumber, len(got), len(*ref.features))
		}
		var diffs []string
		for j := range got {
			if got[j] != (*ref.features)[j] {
				diffs = append(diffs, fmt.Sprintf("%s: train %v infer %v", featureName(j), (*ref.features)[j], got[j]))
			}
		}
		if len(diffs) > 0 {
			t.Errorf("turn %d (%s %s): %d features differ, e.g.\n  %s", tn.TurnNumber, tn.PlayerID, tn.Play, len(diffs),
				strings.Join(diffs[:min(len(diffs), 6)], "\n  "))
		}
	}
}

// The English layout: 85 board planes, 72 scalars (game.TestEnglishMLLayout
// pins it).
const (
	enPlaneLen = 85 * game.NN_H * game.NN_W
	enScalars  = 72
)

// featureName describes a vector index for error messages.
func featureName(j int) string {
	if j < enPlaneLen {
		plane, sq := j/(15*15), j%(15*15)
		return fmt.Sprintf("plane %d (%c%d)", plane, 'A'+sq%15, sq/15+1)
	}
	s := j - enPlaneLen
	switch {
	case s < 27:
		return fmt.Sprintf("rack[%d]", s)
	case s < 54:
		return fmt.Sprintf("unseen[%d]", s-27)
	case s < 57:
		return fmt.Sprintf("history[%d]", s-54)
	case s < 63:
		return fmt.Sprintf("power[%d]", s-57)
	case s < 67:
		return fmt.Sprintf("vcratio[%d]", s-63)
	default:
		return []string{"turnsSinceOppBingo", "moveScore", "leaveValue", "bagUnseen", "spread"}[s-67]
	}
}

// The bot builds the board planes as bytes for uint8 models
// (game.MLByteVectorsForMoves): a copy of the position's planes patched
// for each candidate. They must equal the float planes, and the scalars
// must be the same, for every legal move at every turn.
func TestByteVectorsMatchFloat(t *testing.T) {
	for name, rows := range map[string][]string{
		"sample game":  sampleGameTurns,
		"endgame game": endgameGameTurns,
	} {
		t.Run(name, func(t *testing.T) {
			hdr := "playerID,gameID,turn,rack,play,score,totalscore,tilesplayed,leave,equity,tilesremaining,oppscore\n"
			var turns []Turn
			sc := NewTurnScanner(strings.NewReader(hdr + strings.Join(rows, "\n") + "\n"))
			for sc.Scan() {
				turns = append(turns, sc.Turn())
			}
			ga := NewGameAssembler(NPlies)
			gw := ga.newGameWindow(turns[0])
			ga.games[turns[0].GameID] = gw
			var history []*move.Move
			checked := 0
			for _, tn := range turns {
				g := gw.game.Game
				onturn := g.PlayerOnTurn()
				if err := g.SetRackFor(onturn, tilemapping.RackFromString(tn.Rack, g.Alphabet())); err != nil {
					t.Fatalf("turn %d: set rack: %v", tn.TurnNumber, err)
				}
				gd, err := kwg.GetKWG(g.Config().WGLConfig(), g.LexiconName())
				if err != nil {
					t.Fatal(err)
				}
				gen := movegen.NewGordonGenerator(gd, g.Board(), g.Bag().LetterDistribution())
				gen.SetGenPass(true)
				moves := gen.GenAll(g.RackFor(onturn), g.Bag().TilesRemaining() >= game.RackTileLimit)
				planes, scalars, err := g.MLVectorsForMoves(moves, ga.eqCalc, history)
				if err != nil {
					t.Fatalf("turn %d: float vectors: %v", tn.TurnNumber, err)
				}
				boards, bscalars, err := g.MLByteVectorsForMoves(moves, ga.eqCalc, history)
				if err != nil {
					t.Fatalf("turn %d: byte vectors: %v", tn.TurnNumber, err)
				}
				if len(boards) != len(planes) || len(bscalars) != len(scalars) {
					t.Fatalf("turn %d: lengths %d/%d vs %d/%d", tn.TurnNumber, len(boards), len(bscalars), len(planes), len(scalars))
				}
				for j := range planes {
					if float32(boards[j]) != planes[j] {
						mi := j / enPlaneLen
						t.Fatalf("turn %d move %s: %s: float %v byte %v", tn.TurnNumber,
							moves[mi].ShortDescription(), featureName(j%enPlaneLen), planes[j], boards[j])
					}
				}
				for j := range scalars {
					if bscalars[j] != scalars[j] {
						t.Fatalf("turn %d: scalar %s: float %v byte path %v", tn.TurnNumber,
							featureName(enPlaneLen+j%enScalars), scalars[j], bscalars[j])
					}
				}
				checked += len(moves)
				ga.FeedTurn(tn)
				if len(gw.plies) == 0 || gw.game.Game.Playing() != pb.PlayState_PLAYING {
					break
				}
				history = append(history, gw.plies[len(gw.plies)-1].move)
			}
			t.Logf("%d turns, %d candidate moves checked", len(history), checked)
			if checked < 100 {
				t.Fatalf("only %d moves checked", checked)
			}
		})
	}
}
