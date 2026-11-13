#include "textflag.h"

// func prefetch(addr unsafe.Pointer)
TEXT ·prefetch(SB), NOSPLIT, $0-8
	MOVQ addr+0(FP), AX
	PREFETCHT0 (AX)
	RET
