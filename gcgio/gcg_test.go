package gcgio

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/stretchr/testify/assert"
)

var DefaultConfig = config.DefaultConfig()

func TestMain(m *testing.M) {
	for _, lex := range []string{"NWL18"} {
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

func slurp(filename string) string {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	contents, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatal(err)
	}
	return string(contents)
}

func TestParseGCG(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/vs_andy.gcg")
	expected := slurp("./testdata/vs_andy.json")

	assert.Nil(t, err)
	assert.NotNil(t, history)

	repr, err := json.Marshal(history)
	assert.Nil(t, err)

	assert.JSONEq(t, expected, string(repr))
}

func TestParseOtherGCG(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/doug_v_emely.gcg")
	expected := slurp("./testdata/doug_v_emely.json")

	assert.Nil(t, err)
	assert.NotNil(t, history)

	repr, err := json.Marshal(history)
	assert.Nil(t, err)

	assert.JSONEq(t, expected, string(repr))
}

func TestParseGCGWithWithdrawnPhonyBingo(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/josh2.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)

	history.Lexicon = "CSW19"
	expected := slurp("./testdata/josh2.json")

	repr, err := json.Marshal(history)
	assert.Nil(t, err)
	assert.JSONEq(t, expected, string(repr))
}

func TestParseGCGWithChallengeBonus(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/vs_frentz.gcg")
	history.Lexicon = "CSW12"
	expected := slurp("./testdata/vs_frentz.json")

	assert.Nil(t, err)
	assert.NotNil(t, history)

	repr, err := json.Marshal(history)
	fmt.Println(string(repr))

	assert.Nil(t, err)
	assert.JSONEq(t, expected, string(repr))
}

func TestParseSpecialChar(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/name_iso8859-1.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "césar", history.Players[0].Nickname)
	assert.Equal(t, "hércules", history.Players[1].Nickname)
}

func TestParseSpecialUTF8NoHeader(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/name_utf8_noheader.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	// Since there was no encoding header, the name gets all messed up:
	assert.Equal(t, "cÃ©sar", history.Players[0].Nickname)
}

func TestParseSpecialUTF8WithHeader(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/name_utf8_with_header.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "césar", history.Players[0].Nickname)
}

func TestParseUnsupportedEncoding(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/name_weird_encoding_with_header.gcg")
	assert.NotNil(t, err)
	assert.Nil(t, history)
}

func TestParseDOSMode(t *testing.T) {
	// file has CRLF carriage returns. we should handle it.
	history, err := ParseGCG(&DefaultConfig, "./testdata/utf8_dos.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, history)
	assert.Equal(t, "angwantibo", history.Players[0].Nickname)
	assert.Equal(t, "Michal_Josko", history.Players[1].Nickname)
}

func TestToGCG(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/doug_v_emely.gcg")

	assert.Nil(t, err)
	assert.NotNil(t, history)

	gcgstr, err := GameHistoryToGCG(history, false)
	assert.Nil(t, err)

	// ignore encoding line:
	linesNew := strings.Split(gcgstr, "\n")[1:]
	linesOld := strings.Split(slurp("./testdata/doug_v_emely.gcg"), "\n")

	assert.Equal(t, len(linesNew), len(linesOld))
	for idx, ln := range linesNew {
		assert.Equal(t, strings.Fields(ln), strings.Fields(linesOld[idx]))
	}
}

func TestToGCGExcludePenultimatePass(t *testing.T) {
	history, err := ParseGCG(&DefaultConfig, "./testdata/doug_v_emely_double_challenge.gcg")

	assert.Nil(t, err)
	assert.NotNil(t, history)

	gcgstr, err := GameHistoryToGCG(history, false)
	assert.Nil(t, err)

	// ignore encoding line:
	linesNew := strings.Split(gcgstr, "\n")[1:]
	linesOld := strings.Split(slurp("./testdata/doug_v_emely.gcg"), "\n")

	assert.Equal(t, len(linesNew), len(linesOld))
	for idx, ln := range linesNew {
		assert.Equal(t, strings.Fields(ln), strings.Fields(linesOld[idx]))
	}
}

func TestDuplicateNicknames(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#player1 dougie Doungy B
#player2 dougie Cesar D
>dougie: FOO 8D FOO +12 12`)
	history, err := ParseGCGFromReader(&DefaultConfig, reader)
	assert.Nil(t, history)
	assert.Equal(t, errDuplicateNames, err)
}

func TestPragmaWrongPlace(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#player1 dougie Doungy B
#player2 cesar Cesar D
>dougie: FOO 8H FOO +12 12
#lexicon OSPD4`)
	history, err := ParseGCGFromReader(&DefaultConfig, reader)
	assert.Nil(t, history)
	assert.Equal(t, errPragmaPrecedeEvent, err)
}

func TestIsBingo(t *testing.T) {
	reader := strings.NewReader(`#character-encoding UTF-8
#lexicon CSW19
#player1 dougie Doungy B
#player2 cesar Cesar D
>dougie: FOODIES 8D FOODIES +80 80
>cesar: ABCDEFG D7 E. +5 5
`)
	history, err := ParseGCGFromReader(&DefaultConfig, reader)
	assert.Nil(t, err)
	assert.True(t, history.Events[0].IsBingo)
	assert.False(t, history.Events[1].IsBingo)
}
