package board

// This file contains some sample filled boards, used solely for testing.

import "github.com/domino14/macondo/alphabet"

// VsWho is a string representation of a board.
type VsWho string

const (
	// VsEd was a game I played against Ed, under club games 20150127vEd
	// Quackle generates 219 total unique moves with a rack of AFGIIIS
	VsEd VsWho = `
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
`
	// VsMatt was a game I played against Matt Graham, 2018 Lake George tourney
	VsMatt = `
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
`
	// VsJeremy was a game I played against Jeremy Hall, 2018-11 Manhattan tourney
	VsJeremy = `
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
`
	// VsOxy is a constructed game that has a gigantic play available.
	VsOxy = `
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
`
	// VsMatt2 at the 2018-11 Manhattan tourney
	VsMatt2 = `
cesar: Turn 8
   A B C D E F G H I J K L M N O   -> cesar                    EEILNT?   237
   ------------------------------     matt graham              EIJPSTW   171
 1|=     '       =       '     R| --Tracking-----------------------------------
 2|  -       "       "     Q - E| AABCDDDEEEEEHIIIIJLLLMNOPRSSSTTTUUVWWY  38
 3|    T I G E R   '     H I   I|
 4|'     -     O F       U     N|
 5|        O C E A N   P R A N K|
 6|  "       "   B A Z A R   "  |
 7|    '       '   '     A '    |
 8|=     '       M O O N Y     =|
 9|    '       D I F       '    |
10|  "       V E G   "       "  |
11|        -     S A n T O O R  |
12|'     -       '     O X     '|
13|    -       ' A G U E   -    |
14|  -       "       "       -  |
15|=     '       =       '     =|
   ------------------------------
`
	// VsRoy at the 2011 California Open
	VsRoy = `
cesar: Turn 10
   A B C D E F G H I J K L M N O      roy                      WZ        427
   ------------------------------  -> cesar                    EFHIKOQ   331
 1|=     '       =     L U R I D| --Tracking-----------------------------------
 2|  - O     "       "       - I| WZ  2
 3|    U       '   P R I C E R S|
 4|O U T R A T E S       O     T|
 5|    V   -           - u     E|
 6|G " I   C O L O N I A L   " N|
 7|A   E S     '   '     T '   D|
 8|N     E       U P B Y E     E|
 9|J   ' R     M   ' O   R '   D|
10|A B   E N " A G A V E S   "  |
11|  L   N O   F   M I X        |
12|' I   A N   I '   D   -     '|
13|  G A T E W A Y s       -    |
14|  H   E   "       "       -  |
15|= T   '       =       '     =|
   ------------------------------
`
	// VsMacondo1 is poor_endgame_timing.gcg
	VsMacondo1 = `
teich: Turn 12
   A B C D E F G H I J K L M N O      cesar                    ENNR      379
   ------------------------------  -> teich                    APRS?     469
 1|J O Y E D     =       '     =| --Tracking-----------------------------------
 2|U - E L   V       "       -  | ENNR  4
 3|G   W O   I '   '       -    |
 4|A I   P   G   '       -     '|
 5|    F E T A         -        |
 6|  Y I R R S       C       "  |
 7|    L   I   O B I A     '    |
 8|U     H A I K A   L   '     =|
 9|N   Z   L   '   ' O F   '    |
10|I T E M I S E D   T O     "  |
11|T   B A N     E   T O        |
12|E   R U G     V   e D -     '|
13|  H A D     ' O '       -    |
14|W E       "   I N Q U E S T  |
15|E X E C       R       M O A N|
   ------------------------------
`
	// JDvsNB is a test for endgames
	JDvsNB = `
Nathan Benedict: Turn 14
   A B C D E F G H I J K L M N O   -> Nathan Benedict          RR        365
   ------------------------------     JD                       LN        510
 1|G R R L       d     Q '     =| --Tracking-----------------------------------
 2|  E       J   E   G I     -  | LN  2
 3|  M M   L A ' N Y E     -    |
 4|'   O B I     O O N   -     '|
 5|    K I P     T   I F        |
 6|  W E T A "   I   T O     "  |
 7|S O ' C     ' V ' O B   '    |
 8|U T   H E D G E   R   ' V   =|
 9|I   '     U '   '       E    |
10|  "       P       F A A N "  |
11|    C   - E N T A Y L E D    |
12|W   H O A S   A       - I   '|
13|E   I       ' T '       N    |
14|D A Z E D "   O   "     g -  |
15|S     X I     U     U R S A E|
   ------------------------------
`
	// VsAlec at the 2019 Nationals
	VsAlec = `
cesar: Turn 11
   A B C D E F G H I J K L M N O      alec                     EGNOQR    420
   ------------------------------  -> cesar                    DGILOPR   369
 1|=     '       L       '     =| --Tracking-----------------------------------
 2|  -       "   A   J       -  | EGNOQR  6
 3|    -       ' T W I N E R    |
 4|'     A       E   V O T E R '|
 5|  B U R A N   E   E -        |
 6|  "   c H U T N E Y S     "  |
 7|    ' A     '   '       '    |
 8|W O O D       S       '     =|
 9|E   ' I F   ' L '       '    |
10|I "   A A "   E   "       "  |
11|R   C   U   P A M   -        |
12|D   A L G U A Z I l   -     C|
13|O   N   H   V E X       F   Y|
14|E - T     K I   T O O N I E S|
15|S     ' M I D I       ' B   T|
   ------------------------------
`
	// VsAlec2 same game as above just a couple turns later.
	VsAlec2 = `
cesar: Turn 12
   A B C D E F G H I J K L M N O      alec                     ENQR      438
   ------------------------------  -> cesar                    DGILOR    383
 1|=     '       L       '     =| --Tracking-----------------------------------
 2|  -       "   A   J O G   -  | ENQR  4
 3|    -       ' T W I N E R    |
 4|'     A       E   V O T E R '|
 5|  B U R A N   E   E -        |
 6|  "   c H U T N E Y S     "  |
 7|    P A     '   '       '    |
 8|W O O D       S       '     =|
 9|E   ' I F   ' L '       '    |
10|I "   A A "   E   "       "  |
11|R   C   U   P A M   -        |
12|D   A L G U A Z I l   -     C|
13|O   N   H   V E X       F   Y|
14|E - T     K I   T O O N I E S|
15|S     ' M I D I       ' B   T|
   ------------------------------
`
	// VsJoey from Lake George 2019
	VsJoey = `
Joey: Turn 11
   A B C D E F G H I J K L M N O      Cesar                    DIV       412
   ------------------------------  -> Joey                     AEFILMR   371
 1|A I D E R     U       '     =| --Tracking-----------------------------------
 2|b - E   E "   N   Z       -  | DIV  3
 3|A W N   T   ' M ' A T T -    |
 4|L I   C O B L E     O W     '|
 5|O P     U     E     A A      |
 6|N E     C U S T A R D S   Q  |
 7|E R ' O H   '   '   I   ' U  |
 8|S     K     F O B   E R G O T|
 9|    '     H E X Y L S   ' I  |
10|  "     J I N     "       N  |
11|    G O O P     N A I V E s T|
12|' D I R E     '       -     '|
13|    G A Y   '   '       -    |
14|  -       "       "       -  |
15|=     '       =       '     =|
   ------------------------------
`
	// VsCanik from 2019 Nationals
	VsCanik = `
cesar: Turn 12
   A B C D E F G H I J K L M N O      canik                    DEHILOR   389
   ------------------------------  -> cesar                    BGIV      384
 1|=     '       =   A   P I X Y| --Tracking-----------------------------------
 2|  -       "       S   L   -  | DEHILOR  7
 3|    T o W N L E T S   O -    |
 4|'     -       '   U   D A   R|
 5|      G E R A N I A L   U   I|
 6|  "       "       g     T " C|
 7|    '       '   W E     O B I|
 8|=     '     E M U     '   O N|
 9|    '       A I D       G O  |
10|  "       H U N   "     E T  |
11|        Z A   T     -   M E  |
12|' Q   F A K E Y       J O E S|
13|F I V E   E '   '     I T   C|
14|  -       S P O R R A N   - A|
15|=     '     O R E     N     D|
   ------------------------------
`
	// JoeVsPaul, sample endgame given in the Maven paper
	JoeVsPaul = `
joe: Turn 12
   A B C D E F G H I J K L M N O      joe                      LZ        296
   ------------------------------  -> paul                     ?AEIR     296
 1|=     '   B E R G S   '     =| --Tracking-----------------------------------
 2|  -     P A       U       -  | ILMZ 4
 3|    Q A I D '   U R N   -    |
 4|'     B E E   '   F   T S K '|
 5|  P   E T     V I A T I C    |
 6|M A   T A W       c     H "  |
 7|E S '     I S   ' E     A    |
 8|A T   F O L I A       ' V I M|
 9|L I ' L   E X   E       '    |
10|  N   O   D     N "   Y   "  |
11|  G N U -   C   J E T E      |
12|'   E R     O H O     N     '|
13|    O       G O Y       -    |
14|  I N D O W   U   "       -  |
15|=     ' D O R R       '     =|
   ------------------------------
`
	// VsJoel from Manhattan November 2019
	VsJoel = `
cesar: Turn 11
   A B C D E F G H I J K L M N O      joel s                   EIQSS     393
   ------------------------------  -> cesar                    AAFIRTW   373
 1|= L E M N I S C I     L   E R| --Tracking-----------------------------------
 2|  -       "   O   P A I N T  | EIQSS  5
 3|    -   A   ' L ' R A V E    |
 4|W E D G E     Z   I   R     '|
 5|        R   J A U N T E d    |
 6|  "     O X O     K       "  |
 7|    Y O B   '   P       '    |
 8|=     F A U N A E     '     =|
 9|    '   T   '   G U Y   '    |
10|  "       " B E S T E a D "  |
11|        -     T     H I E    |
12|'     -       H       - V U G|
13|    C O R M O I D       -    |
14|  -       "   O   "       -  |
15|=     '       N O N I D E A L|
   ------------------------------
`
	// Endgame from 2019 Worlds
	EldarVsNigel = `
Nigel Richards: Turn 11
   A B C D E F G H I J K L M N O   -> Nigel Richards           AEEIRUW   410
   ------------------------------     David Eldar              V         409
 1|=     ' E X O D E     '     =| --Tracking-----------------------------------
 2|  D O F F " K E R A T I N - U| V  1
 3|  O H O     '   '       Y E N|
 4|' P O O J A   B       M E W S|
 5|        - S Q U I N T Y     A|
 6|  "     R H I N O " e     " V|
 7|    B       ' C '   R   '   E|
 8|G O A T   D   E     Z I N   d|
 9|  U R A C I L S '   E   '    |
10|  P I G   S       " T     "  |
11|    L   - R         T        |
12|'   L -   A   G E N I I     '|
13|    A     T ' L '       -    |
14|  -       E   A   "       -  |
15|=     '   D   M       '     =|
   ------------------------------
`
	TestDupe = `
New Player 1: Turn 2
   A B C D E F G H I J K L M N O      Quackle                  AHIJNRR   76
   ------------------------------  -> New Player 1             Z         4
 1|=     '       =       '     =| --Tracking-----------------------------------
 2|  -       "       "       -  | ??AAAAAAAAABBCDDDDEEEEEEEEEEEFFGGGHHIIIIIIJK
 3|    -       '   '       -    | LLLLMMNNNNNOOOOOOOOPPQRRRRRRSSTTTTUUUUVVWWXY
 4|'     -       '       -     '| Y  89
 5|        -           -        |
 6|  "       "       "       "  |
 7|    '       '   '       '    |
 8|= I N C I T E S       '     =|
 9|I S '       '   '       '    |
10|T "       "       "       "  |
11|        -           -        |
12|'     -       '       -     '|
13|    -       '   '       -    |
14|  -       "       "       -  |
15|=     '       =       '     =|
   ------------------------------
`

	NoahVsMishu = `
whatnoloan: Turn 15
   A B C D E F G H I J K L M N O   -> whatnoloan               AEIINTY   327
   ------------------------------     mishu7                   CLLPR     368
 1|=     '       =     W H E T S| --Tracking-----------------------------------
 2|  -       "       "   O   -  | CLLPR  5
 3|    -       '   ' G L U G    |
 4|'     -       '       S     '|
 5|        -         R - E      |
 6|  "       "       I   D A I S|
 7|    '       '   ' A G   B O A|
 8|f     '     V O X   A T O N Y|
 9|I   '       ' F I V E R '    |
10|R "       W E T   "   E   "  |
11|E       Z A     M O A N E D  |
12|L     B I D   Q I     C     '|
13|O   J U N   '   M U   H O E  |
14|c -   R E T U N E S     I F  |
15|K N A P       O R E A D   T =|
   ------------------------------
`

	NoahVsMishu2 = `
whatnoloan: Turn 15
   A B C D E F G H I J K L M N O   -> whatnoloan               AEIINY    334
   ------------------------------     mishu7                   LLPR      374
 1|=     '       =     W H E T S| --Tracking-----------------------------------
 2|  -       "       "   O   -  | LLPR  4
 3|    -       '   ' G L U G    |
 4|'     -       '       S     '|
 5|        -         R - E   C  |
 6|  "       "       I   D A I S|
 7|    '       ' T ' A G   B O A|
 8|f     '     V O X   A T O N Y|
 9|I   '       ' F I V E R '    |
10|R "       W E T   "   E   "  |
11|E       Z A     M O A N E D  |
12|L     B I D   Q I     C     '|
13|O   J U N   '   M U   H O E  |
14|c -   R E T U N E S     I F  |
15|K N A P       O R E A D   T =|
   ------------------------------
`
	NoahVsMishu3 = `
whatnoloan: Turn 15
   A B C D E F G H I J K L M N O   -> whatnoloan               AEIY      339
   ------------------------------     mishu7                   LLP       381
 1|=     '       =     W H E T S| --Tracking-----------------------------------
 2|  -       "       "   O   -  | LLP  4
 3|    -       '   ' G L U G    |
 4|'     -       '       S     '|
 5|        -         R - E   C  |
 6|  "       "       I   D A I S|
 7|    '       ' T ' A G   B O A|
 8|f     '     V O X   A T O N Y|
 9|I   '       ' F I V E R '    |
10|R I N     W E T   "   E   "  |
11|E       Z A     M O A N E D  |
12|L     B I D   Q I     C     '|
13|O   J U N   '   M U   H O E R|
14|c -   R E T U N E S     I F  |
15|K N A P       O R E A D   T =|
   ------------------------------
`

	MavenVsMacondo = `
maven: Turn 23
   A B C D E F G H I J K L M N O      player1               EO      448
   ------------------------------  -> player2               AEEORS? 278
 1|D   V ' B E Z I Q u E '     W| --Tracking-----------------------------------
 2|J U I C E R       " L O U I E| EO  2
 3|I   V   G '     F A   H M   A|
 4|N   A H       ' E L   -   U N|
 5|S     U -         O P     R E|
 6|  "   E D "       F A     I D|
 7|    '   R   M I L T Y   ' N  |
 8|=     B I G O T     O C T A D|
 9|    T   N A '   '   R   ' R  |
10|  " O   K I       "       Y  |
11|    W O S T         -        |
12|'   N -   S P A E I N G     '|
13|    L       ' A X       -    |
14|  - E     "       "       -  |
15|=   T '       =       '     =|
   ------------------------------
`

	APolishEndgame = `
   A B C D E F G H I J K L M N O     ->                    1  BGHUWZZ  304
   ------------------------------                          2           258
 1|=     '       =       '     E |
 2|  -       "       "       E T |   Bag + unseen: (6)
 3|    -       '   '     F i Ś   |
 4|'     -       '       L I   ' |   I K M Ó Ź Ż
 5|        -         C Ł A       |
 6|  "       "       Z   N   "   |
 7|    '       '   H O I   '     |
 8|=     '     S T Ę P I Ć     = |
 9|    '     A U R ' Y     R O K |
10|  P O   S A M Y   " G N A Ń   |   Turn 46:
11|    C L E     P   J A   K     |   2 played 9M .OK for 12 pts from a rack of
12|'   L A R W O ' S A M B I E ' |   IKKMOŹŻ
13|C N I   W E N T O   O   I W Ą |
14|E - S     "   E N D       Y   |
15|Z D Z I A Ł a J   Y A R D   = |
   ------------------------------
   `

	APolishEndgame2 = `
   A B C D E F G H I J K L M N O                           1   BHUWZ   316
   ------------------------------    ->                    2   IKMÓŹŻ  258
 1|=     '       =       '     E|
 2|  -       "       "       E T|    Bag + unseen: (5)
 3|    -       '   '     F i Ś  |
 4|'     -       '       L I   '|    B H U W Z
 5|        -         C Ł A      |
 6|  "       "       Z   N   "  |
 7|    '       '   H O I   ' Z  |
 8|=     '     S T Ę P I Ć   G =|
 9|    '     A U R ' Y     R O K|
10|  P O   S A M Y   " G N A Ń  |    Turn 47:
11|    C L E     P   J A   K    |    1 played N7 ZG.. for 12 pts from a rack of
12|'   L A R W O ' S A M B I E '|     BGHUWZZ
13|C N I   W E N T O   O   I W Ą|
14|E - S     "   E N D       Y  |
15|Z D Z I A Ł a J   Y A R D   =|
   ------------------------------
   `
)

// SetToGame sets the board to a specific game in progress. It is used to
// generate test cases.
func (b *GameBoard) SetToGame(alph *alphabet.Alphabet, game VsWho) *TilesInPlay {
	// Set the board to a game
	tip := b.setFromPlaintext(string(game), alph)
	b.UpdateAllAnchors()
	return tip
}
