package board

import "github.com/domino14/macondo/alphabet"

// This file contains some sample filled boards, used solely for testing.
type VsWho uint8

const (
	VsEd VsWho = iota
	VsMatt
	VsJeremy
	VsOxy
)

// SetBoardToGame sets the board to a specific game in progress. It is used to
// generate test cases.
func (b *GameBoard) SetBoardToGame(alph *alphabet.Alphabet, game VsWho) {
	// Set the board to a game
	if game == VsEd {

		// club games - 20150127vEd, beginning of turn 8
		// Quackle generates 219 total unique moves with a rack of AFGIIIS
		b.SetFromPlaintext(`
cesar: Turn 8
   A B C D E F G H I J K L M N O   -> cesar                    AFGIIIS   182
   ------------------------------     ed                       ADEILNV   226
 1|=     '       =       '     E| --Tracking-----------------------------------
 2|  -       "       "       - N| ?AAAAACCDDDEEIIIIKLNOOOQRRRRSTTTTUVVZ  37
 3|    -       '   '       -   d|
 4|'     -       '       -     U|
 5|        G L O W S   -       R|
 6|  "       "     P E T     " E|
 7|    '       ' F A X I N G   R|
 8|=     '     J A Y   T E E M S|
 9|    B     B O Y '       N    |
10|  " L   D O E     "     U "  |
11|    A N E W         - P I    |
12|'   M O   L E U       O N   '|
13|    E H     '   '     H E    |
14|  -       "       "       -  |
15|=     '       =       '     =|
   ------------------------------
`, alph)
	} else if game == VsMatt {
		// tourney, 2018 Lake George vs Matt G
		b.SetFromPlaintext(`
cesar: Turn 10
   A B C D E F G H I J K L M N O      matt g                   AEEHIIL   341
   ------------------------------  -> cesar                    AABDELT   318
 1|=     '       Z E P   F     =| --Tracking-----------------------------------
 2|  F L U K Y       R   R   -  | AEEEGHIIIILMRUUWWY  18
 3|    -     E X   ' A   U -    |
 4|'   S C A R I E S T   I     '|
 5|        -         T O T      |
 6|  "       " G O   L O     "  |
 7|    '       O R ' E T A '    | ↓
 8|=     '     J A B S   b     =|
 9|    '     Q I   '     A '    | ↓
10|  "       I   N   "   N   "  | ↓
11|      R e S P O N D - D      | ↓
12|' H O E       V       O     '| ↓
13|  E N C O M I A '     N -    | ↓
14|  -       "   T   "       -  |
15|=     V E N G E D     '     =|
   ------------------------------
`, alph)
	} else if game == VsJeremy {
		// tourney, 2018 nov Manhattan vs Jeremy H
		b.SetFromPlaintext(`
jeremy hall: Turn 13
   A B C D E F G H I J K L M N O   -> jeremy hall              DDESW??   299
   ------------------------------     cesar                    AHIILR    352
 1|=     '       N       '     M| --Tracking-----------------------------------
 2|  -       Z O O N "       A A| AHIILR  6
 3|    -       ' B '       - U N|
 4|'   S -       L       L A D Y|
 5|    T   -     E     Q I   I  |
 6|  " A     P O R N "     N O R|
 7|    B I C E '   A A   D A   E|
 8|=     '     G U V S   O P   F|
 9|    '       '   E T   L A   U|
10|  "       J       R   E   U T|
11|        V O T E   I - R   N E|
12|'     -   G   M I C K I E S '|
13|    -       F E ' T   T H E W|
14|  -       " O R   "   E   X I|
15|=     '     O Y       '     G|
   ------------------------------
`, alph)
	} else if game == VsOxy {
		// lol
		b.SetFromPlaintext(`
cesar: Turn 11
   A B C D E F G H I J K L M N O      rubin                    ADDELOR   345
   ------------------------------  -> cesar                    OXPBAZE   129
 1|= P A C I F Y I N G   '     =| --Tracking-----------------------------------
 2|  I S     "       "       -  | ADDELORRRTVV  12
 3|Y E -       '   '       -    |
 4|' R E Q U A L I F I E D     '|
 5|H   L   -           -        |
 6|E D S     "       "       "  |
 7|N O '     T '   '       '    |
 8|= R A I N W A S H I N G     =|
 9|U M '     O '   '       '    |
10|T "   E   O       "       "  |
11|  W A K E n E R S   -        |
12|' O n E T I M E       -     '|
13|O O T     E ' B '       -    |
14|N -       "   U   "       -  |
15|= J A C U L A T I N G '     =|
   ------------------------------
`, alph)
	}
}
