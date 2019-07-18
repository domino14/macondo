package move

import "testing"

type coordTestStruct struct {
	row      int
	col      int
	vertical bool
	output   string
}

var coordTests = []coordTestStruct{
	{0, 0, false, "1A"},
	{0, 0, true, "A1"},
	{14, 14, false, "15O"},
	{14, 14, true, "O15"},
	{9, 8, false, "10I"},
	{9, 8, true, "I10"},
	{1, 7, false, "2H"},
	{1, 7, true, "H2"},
}

func TestToBoardGameCoords(t *testing.T) {
	for _, tc := range coordTests {
		calc := ToBoardGameCoords(tc.row, tc.col, tc.vertical)
		if calc != tc.output {
			t.Errorf("For row=%v col=%v vertical=%v got %v, expected %v",
				tc.row, tc.col, tc.vertical, calc, tc.output)
		}
	}
}

func TestFromBoardGameCoords(t *testing.T) {
	for _, tc := range coordTests {
		row, col, vertical := FromBoardGameCoords(tc.output)
		if row != tc.row || col != tc.col || vertical != tc.vertical {
			t.Errorf("For coord %v expected (%v, %v, %v) got (%v, %v, %v)",
				tc.output, tc.row, tc.col, tc.vertical, row, col, vertical)
		}
	}
}
