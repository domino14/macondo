package gcgio

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

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
	gamerepr, err := ParseGCG("./testdata/vs_andy.gcg")
	expected := slurp("./testdata/vs_andy.json")

	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)

	repr, err := json.Marshal(gamerepr)
	assert.Nil(t, err)

	assert.JSONEq(t, expected, string(repr))
}

func TestParseOtherGCG(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/doug_v_emely.gcg")
	expected := slurp("./testdata/doug_v_emely.json")

	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)

	repr, err := json.Marshal(gamerepr)
	assert.Nil(t, err)

	assert.JSONEq(t, expected, string(repr))
}

func TestParseGCGWithChallengeBonus(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/vs_frentz.gcg")
	expected := slurp("./testdata/vs_frentz.json")

	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)

	repr, err := json.Marshal(gamerepr)
	assert.Nil(t, err)
	assert.JSONEq(t, expected, string(repr))
}

func TestParseSpecialChar(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/name_iso8859-1.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)
	assert.Equal(t, "césar", gamerepr.Players[0].Nickname)
	assert.Equal(t, "hércules", gamerepr.Players[1].Nickname)
}

func TestParseSpecialUTF8NoHeader(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/name_utf8_noheader.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)
	// Since there was no encoding header, the name gets all messed up:
	assert.Equal(t, "cÃ©sar", gamerepr.Players[0].Nickname)
}

func TestParseSpecialUTF8WithHeader(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/name_utf8_with_header.gcg")
	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)
	assert.Equal(t, "césar", gamerepr.Players[0].Nickname)
}

func TestParseUnsupportedEncoding(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/name_weird_encoding_with_header.gcg")
	assert.NotNil(t, err)
	assert.Nil(t, gamerepr)
}

func TestToGCG(t *testing.T) {
	gamerepr, err := ParseGCG("./testdata/doug_v_emely.gcg")

	assert.Nil(t, err)
	assert.NotNil(t, gamerepr)

	gcgstr := GameReprToGCG(gamerepr)

	assert.Equal(t, gcgstr, slurp("./testdata/doug_v_emely.gcg"))
}
