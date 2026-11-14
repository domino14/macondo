#include "textflag.h"

// func prefetch(addr unsafe.Pointer)
TEXT ·prefetch(SB), NOSPLIT, $0-8
	MOVD addr+0(FP), R0
	PRFM (R0), PLDL1KEEP
	RET
