// amd64 implementation of scanBucketAsm.
//
// Hand-written Plan 9 amd64 translation of the bucket-scan loop.

#include "textflag.h"

// func scanBucketAsm(bucketStartsBase *uint32, entriesBase unsafe.Pointer,
//                    bucketIdx uint32, brLow, brHigh uint64) *Entry
TEXT ·scanBucketAsm(SB), NOSPLIT, $0-48
	MOVQ	bucketStartsBase+0(FP), AX   // AX = &bucketStarts[0]
	MOVQ	entriesBase+8(FP), BX        // BX = entries base
	MOVL	bucketIdx+16(FP), CX         // CX = bucketIdx
	MOVQ	brLow+24(FP), R8             // R8 = brLow
	MOVQ	brHigh+32(FP), R9            // R9 = brHigh

	// start = bucketStarts[bucketIdx]; end = bucketStarts[bucketIdx+1]
	MOVL	(AX)(CX*4), DX               // DX = start
	MOVL	4(AX)(CX*4), DI              // DI = end

	CMPL	DX, DI
	JE	notfound

	// base = entries + start*32
	SHLQ	$5, DX                       // DX = start*32
	ADDQ	BX, DX                       // DX = base

	// limit = entries + end*32
	SHLQ	$5, DI
	ADDQ	BX, DI                       // DI = limit

scanloop:
	// Compare BitRack key (offset 16, 16 bytes) against (R8, R9).
	CMPQ	16(DX), R8
	JNE	advance
	CMPQ	24(DX), R9
	JNE	advance

	MOVQ	DX, ret+40(FP)
	RET

advance:
	ADDQ	$32, DX
	CMPQ	DX, DI
	JNE	scanloop

notfound:
	MOVQ	$0, ret+40(FP)
	RET
