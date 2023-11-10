package tinymove

// TinyMove is a 64-bit representation of a move. We can probably make it
// smaller at the cost of higher decoding. It is made to be as small as possible
// to fit it in a transposition table.
type TinyMove uint64

// Schema:
// 42 bits (7 groups of 6 bits) representing each tile value in the move.
// 7 bit flags, representing whether the associated tile is a blank or not (1 = blank, 0 = no blank)
// 5 bits for row
// 5 bits for column
// 1 bit for horiz/vert (horiz = 0, vert = 1)

// If move is a pass, the entire value is 0.
// 63   59   55   51   47   43   39   35   31   27   23   19   15   11    7    3
//  xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
//    77 7777 6666 6655 5555 4444 4433 3333 2222 2211 1111  BBB BBBB  RRR RRCC CCCV

const ColBitMask = 0b00111110
const RowBitMask = 0b00000111_11000000
const BlanksBitMask = 127 << 12
const InvalidTinyMove TinyMove = 1 << 63

var TBitMasks = [7]uint64{63 << 20, 63 << 26, 63 << 32, 63 << 38, 63 << 44, 63 << 50, 63 << 56}
