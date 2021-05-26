package board

var (
	// CrosswordGameBoard is a board for a fun Crossword Game, featuring lots
	// of wingos and blonks.
	CrosswordGameBoard []string
)

const (
	CrosswordGameLayout = "CrosswordGame"
)

func init() {
	CrosswordGameBoard = []string{
		`=  '   =   '  =`,
		` -   "   "   - `,
		`  -   ' '   -  `,
		`'  -   '   -  '`,
		`    -     -    `,
		` "   "   "   " `,
		`  '   ' '   '  `,
		`=  '   -   '  =`,
		`  '   ' '   '  `,
		` "   "   "   " `,
		`    -     -    `,
		`'  -   '   -  '`,
		`  -   ' '   -  `,
		` -   "   "   - `,
		`=  '   =   '  =`,
	}
}
