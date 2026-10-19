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
	table := NewGameAssembler(NPlies, nil, 0, 0, 0)
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
		ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
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

// featureName describes a vector index for error messages.
func featureName(j int) string {
	if j < game.NN_N_PLANES {
		plane, sq := j/(15*15), j%(15*15)
		return fmt.Sprintf("plane %d (%c%d)", plane, 'A'+sq%15, sq/15+1)
	}
	s := j - game.NN_N_PLANES
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
			ga := NewGameAssembler(NPlies, nil, 0, 0, 0)
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
						mi := j / game.NN_N_PLANES
						t.Fatalf("turn %d move %s: %s: float %v byte %v", tn.TurnNumber,
							moves[mi].ShortDescription(), featureName(j%game.NN_N_PLANES), planes[j], boards[j])
					}
				}
				for j := range scalars {
					if bscalars[j] != scalars[j] {
						t.Fatalf("turn %d: scalar %s: float %v byte path %v", tn.TurnNumber,
							featureName(game.NN_N_PLANES+j%game.NN_N_SCAL), scalars[j], bscalars[j])
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
