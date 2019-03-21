package endgame

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"

	"github.com/domino14/macondo/move"

	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/stretchr/testify/assert"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_csw15.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "CSW15.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_csw15.gaddag")
	}
	os.Exit(m.Run())
}

func TestAddEntryToTable(t *testing.T) {

	pt := make(playTable)
	pt.addEntry("foo", 0, -15, nil)

	assert.Equal(t, 1, len(pt))
	assert.Equal(t, 1, len(pt["foo"]))
	assert.Nil(t, pt["foo"][0].move)
	assert.Equal(t, -15, pt["foo"][0].value)
}

func TestAddEntryToTableAlreadyExisting(t *testing.T) {
	pt := make(playTable)
	pt.addEntry("foo", 3, 15, nil)
	pt.addEntry("foo", 3, 40, nil)

	assert.Equal(t, 1, len(pt))
	assert.Equal(t, 1, len(pt["foo"]))
	assert.Nil(t, pt["foo"][3].move)
	assert.Equal(t, 40, pt["foo"][3].value)
}

func TestGetEntryFromTable(t *testing.T) {

	pt := make(playTable)
	pt.addEntry("foo", 0, -15, nil)

	entry, ok := pt.get("bar", 0)
	assert.False(t, ok)
	assert.Nil(t, entry)

	entry2, ok := pt.get("foo", 1)
	assert.False(t, ok)
	assert.Nil(t, entry2)

	entry3, ok := pt.get("foo", 0)
	assert.True(t, ok)
	assert.Nil(t, entry3.move)
	assert.Equal(t, -15, entry3.value)
}

func TestCanBePlayedWith(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_csw15.gaddag")
	alph := gd.GetAlphabet()

	type pwtest struct {
		play     string
		rack     string
		playable bool
		leave    string
	}
	playedWithTests := []pwtest{
		{".....ULARY", "ABLRUY", true, "B"},
		{".....ULARY", "ALRUY", true, ""},
		{".....ULARY", "ALRU", false, ""},
		{".....ULARY", "AALLRUY", true, "AL"},
		{".....ULARY", "VOCAB", false, ""},
		{".....ULARY", "ALRU", false, ""},
		{"OX.P...B..AZ..E", "ABEOPXZ", true, ""},
		{"OX.P...B..AZ..E", "ABEOPXY", false, ""},
		// blanks
		{".....ULArY", "ALRUY", false, ""},
		{".....ULArY", "ALUY?", true, ""},
		{".....ULArY", "ALUY??", true, "?"},
		{".....ULarY", "LUY??", true, ""},
		{".....ULarY", "ALRUY??", true, "AR"},
		{"COoKIES", "CEIKS??", false, ""},
		{"COoKIES", "CEIKOS?", true, ""},
	}

	for _, tt := range playedWithTests {
		mv := move.NewScoringMoveSimple(30, "B3", tt.play, "", alph)
		rack, err := alphabet.ToMachineWord(tt.rack, alph)

		assert.Nil(t, err)

		playable, leave := canBePlayedWith(mv.Tiles(), rack)
		assert.Equal(t, tt.playable, playable,
			"for %v expected %v got %v", tt, tt.playable, playable)
		assert.Equal(t, tt.leave, leave.UserVisible(alph))
	}

}

func TestGeneratePlayTables(t *testing.T) {
	gd := gaddag.LoadGaddag("/tmp/gen_csw15.gaddag")
	b := board.MakeBoard(board.CrosswordGameBoard)
	alph := gd.GetAlphabet()
	dist := alphabet.EnglishLetterDistribution()
	bag := dist.MakeBag(alph)

	generator := movegen.NewGordonGenerator(gd, bag, b, &strategy.NoLeaveStrategy{})
	generator.SetBoardToGame(alph, board.JDvsNB)
	s := new(DynamicProgSolver)
	s.Init(b, generator, bag, gd)

	ourRack := alphabet.RackFromString("RR", alph)
	theirRack := alphabet.RackFromString("LN", alph)

	s.Solve(ourRack, theirRack)

	assert.Equal(t, 2, len(s.playTableOurs))

	singleR := alphabet.RackFromString("R", alph)

	assert.Equal(t, -4, s.playTableOurs[ourRack.Hashable()][0].value)
	assert.Equal(t, 5, s.playTableOurs[ourRack.Hashable()][1].value)
	assert.Equal(t, 8, s.playTableOurs[ourRack.Hashable()][2].value)
	assert.Equal(t, -2, s.playTableOurs[singleR.Hashable()][0].value)
	assert.Equal(t, 4, s.playTableOurs[singleR.Hashable()][1].value)

	singleL := alphabet.RackFromString("L", alph)
	singleN := alphabet.RackFromString("N", alph)
	assert.Equal(t, 3, len(s.playTableOpp))

	assert.Equal(t, -4, s.playTableOpp[theirRack.Hashable()][0].value)
	assert.Equal(t, 3, s.playTableOpp[theirRack.Hashable()][1].value)
	assert.Equal(t, 9, s.playTableOpp[theirRack.Hashable()][2].value)
	assert.Equal(t, -2, s.playTableOpp[singleL.Hashable()][0].value)
	assert.Equal(t, 4, s.playTableOpp[singleL.Hashable()][1].value)
	assert.Equal(t, -2, s.playTableOpp[singleN.Hashable()][0].value)
	assert.Equal(t, 5, s.playTableOpp[singleN.Hashable()][1].value)
}

func TestSolve(t *testing.T) {

}
