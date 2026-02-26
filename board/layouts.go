package board

var (
	// CrosswordGameBoard is a board for a fun Crossword Game, featuring lots
	// of wingos and blonks.
	CrosswordGameBoard []string
	// SuperCrosswordGameBoard is a board for a bigger Crossword game, featuring
	// even more wingos and blonks.
	SuperCrosswordGameBoard []string
	// CrossplayGameBoard is a board like Crossword but with alternate wingo blonk
	// arrangement.
	CrossplayGameBoard []string
)

const (
	CrosswordGameLayout      = "CrosswordGame"
	SuperCrosswordGameLayout = "SuperCrosswordGame"
	CrossplayGameLayout      = "CrossplayGame"
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
	SuperCrosswordGameBoard = []string{
		`~  '   =  '  =   '  ~`,
		` -  "   -   -   "  - `,
		`  -  ^   - -   ^  -  `,
		`'  =  '   =   '  =  '`,
		` "  -   "   "   -  " `,
		`  ^  -   ' '   -  ^  `,
		`   '  -   '   -  '   `,
		`=      -     -      =`,
		` -  "   "   "   "  - `,
		`  -  '   ' '   '  -  `,
		`'  =  '   -   '  =  '`,
		`  -  '   ' '   '  -  `,
		` -  "   "   "   "  - `,
		`=      -     -      =`,
		`   '  -   '   -  '   `,
		`  ^  -   ' '   -  ^  `,
		` "  -   "   "   -  " `,
		`'  =  '   =   '  =  '`,
		`  -  ^   - -   ^  -  `,
		` -  "   -   -   "  - `,
		`~  '   =  '  =   '  ~`,
	}
	CrossplayGameBoard = []string{
		`"  =   '   =  "`,
		` -    " "    - `,
		`    '     '    `,
		`=  '   -   '  =`,
		`  '  "   "  '  `,
		`    "  '  "    `,
		` "           " `,
		`'  - '   ' -  '`,
		` "           " `,
		`    "  '  "    `,
		`  '  "   "  '  `,
		`=  '   -   '  =`,
		`    '     '    `,
		` -    " "    - `,
		`"  =   '   =  "`,
	}
}
