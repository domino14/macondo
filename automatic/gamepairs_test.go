package automatic

import (
	"testing"

	"github.com/domino14/word-golib/tilemapping"
	"github.com/matryer/is"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

func exchange(tiles ...tilemapping.MachineLetter) *move.Move {
	return move.NewExchangeMove(tilemapping.MachineWord(tiles), nil, nil)
}

func TestMovesDiverge(t *testing.T) {
	is := is.New(t)

	same := []*move.Move{exchange(1, 2), exchange(3), exchange(4, 5)}
	identical := []*move.Move{exchange(1, 2), exchange(3), exchange(4, 5)}
	is.True(!movesDiverge(same, identical))

	differentMove := []*move.Move{exchange(1, 2), exchange(9), exchange(4, 5)}
	is.True(movesDiverge(same, differentMove))

	// One game running longer than the other means they parted ways somewhere.
	shorter := []*move.Move{exchange(1, 2), exchange(3)}
	is.True(movesDiverge(same, shorter))

	is.True(!movesDiverge(nil, nil))
}

func TestDeriveSeed(t *testing.T) {
	is := is.New(t)

	// The same master seed and unit index always give the same game, so a run
	// can be replayed, and a worker can start at any unit without replaying the
	// ones before it.
	is.Equal(DeriveSeed(42, 7), DeriveSeed(42, 7))
	is.True(DeriveSeed(42, 7) != DeriveSeed(42, 8))
	is.True(DeriveSeed(42, 7) != DeriveSeed(43, 7))
	is.True(DeriveSeed(42, 7) != [32]byte{})
}

func TestSeedForUnit(t *testing.T) {
	is := is.New(t)

	// No deterministic config at all: games run off the global RNG as before.
	is.Equal(seedForUnit(nil, nil, 3), [32]byte{})

	// A seed file wins over the master seed.
	fileSeeds := [][32]byte{DeriveSeed(1, 1), DeriveSeed(1, 2)}
	detConfig := &DeterministicConfig{MasterSeed: 99}
	is.Equal(seedForUnit(detConfig, fileSeeds, 1), fileSeeds[1])

	// Past the end of the seed file, the master seed takes over rather than
	// silently handing out a zero seed.
	is.Equal(seedForUnit(detConfig, fileSeeds, 5), DeriveSeed(99, 5))
	is.Equal(seedForUnit(detConfig, nil, 5), DeriveSeed(99, 5))
}

func TestRandomMasterSeedIsNeverZero(t *testing.T) {
	is := is.New(t)
	for i := 0; i < 100; i++ {
		seed, err := RandomMasterSeed()
		is.NoErr(err)
		// Callers read a zero master seed as "not set", so it must never come
		// back from here.
		is.True(seed != 0)
	}
}

// A bot playing itself must produce two identical halves: same seed, same bag
// order, and both seats played by something that makes the same choices. That
// makes it the sharpest test of the pairing -- any difference at all is a bug,
// not strategy. It also guards the recording itself: move generators hand back
// one reusable move object per turn, so recording pointers rather than copies
// used to leave every game holding two moves and flagging every pair divergent.
func TestSelfPlayPairDoesNotDiverge(t *testing.T) {
	is := is.New(t)

	players := []AutomaticRunnerPlayer{
		{BotCode: pb.BotRequest_HASTY_BOT},
		{BotCode: pb.BotRequest_HASTY_BOT},
	}
	r := &GameRunner{config: config.DefaultConfig(), lexicon: "NWL18",
		letterDistribution: "English", gamePairs: true}
	is.NoErr(r.Init(players))
	r.recordMoves = true

	seed := DeriveSeed(1234, 0)
	halves := make([][]*move.Move, 2)
	scores := make([][2]int, 2)
	for half := 0; half < 2; half++ {
		r.movesPlayed = nil
		is.NoErr(r.playGame(false, half, seed))
		halves[half] = r.movesPlayed
		scores[half] = [2]int{r.game.PointsForNick("p1"), r.game.PointsForNick("p2")}
	}

	is.True(len(halves[0]) > 1)
	// Every move is its own object, not the generator's reused one.
	distinct := map[*move.Move]bool{}
	for _, m := range halves[0] {
		distinct[m] = true
	}
	is.Equal(len(distinct), len(halves[0]))

	is.True(!movesDiverge(halves[0], halves[1]))
	// The seats swap between halves, so the same two scores come back mirrored
	// and the pair is an exact tie.
	is.Equal(scores[0][0], scores[1][1])
	is.Equal(scores[0][1], scores[1][0])
}
