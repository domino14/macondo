package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMoveBlanksToEnd(t *testing.T) {
	assert.Equal(t, moveBlanksToEnd("?"), "?")
	assert.Equal(t, moveBlanksToEnd("I?"), "I?")
	assert.Equal(t, moveBlanksToEnd("?AB?C"), "ABC??")
	assert.Equal(t, moveBlanksToEnd("??"), "??")
	assert.Equal(t, moveBlanksToEnd("?FED"), "FED?")
	assert.Equal(t, moveBlanksToEnd("X?X"), "XX?")
}
