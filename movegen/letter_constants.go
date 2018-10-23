package movegen

import "github.com/domino14/macondo/alphabet"

// EmptySquareMarker is a MachineLetter representation of an empty square
const EmptySquareMarker = alphabet.MaxAlphabetSize + 1

// PlayedThroughMarker is a MachineLetter representation of a filled-in square
// that was played through.
const PlayedThroughMarker = alphabet.MaxAlphabetSize + 2
