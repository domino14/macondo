// The BOARD_DIM 21 build of the shim; see shim_impl.h. MAGPIE's 21x21 library
// has every global symbol renamed mg21_*, so the same names are renamed here
// before any MAGPIE header is read.
#include "mg21_rename.h"
#define BOARD_DIM 21
#define MGS(name) mgs21_##name
#include "shim_impl.h"
