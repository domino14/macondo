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
