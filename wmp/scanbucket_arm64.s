// arm64 implementation of scanBucketAsm.
//
// Hand-written Plan 9 ARM64 translation of the bucket-scan loop.
// Loads two uint32 bucket starts to compute the entry range, then
// walks entries by 32-byte stride, comparing the trailing 16 bytes
// of each entry against the BitRack key in registers.

#include "textflag.h"

// func scanBucketAsm(bucketStartsBase *uint32, entriesBase unsafe.Pointer,
//                    bucketIdx uint32, brLow, brHigh uint64) *Entry
//
// Frame: 0 stack, 48 bytes for args + return (8 + 8 + 8 (uint32 padded) + 8 + 8 + 8).
// Argument layout (ABI0):
//   bucketStartsBase: +0(FP)   8 bytes (pointer)
//   entriesBase:      +8(FP)   8 bytes (pointer)
//   bucketIdx:        +16(FP)  4 bytes (uint32, padded to 8)
//   brLow:            +24(FP)  8 bytes
//   brHigh:           +32(FP)  8 bytes
//   ret:              +40(FP)  8 bytes (*Entry)
TEXT ·scanBucketAsm(SB), NOSPLIT, $0-48
	MOVD	bucketStartsBase+0(FP), R0   // R0 = &bucketStarts[0]
	MOVD	entriesBase+8(FP), R1        // R1 = entries base pointer
	MOVWU	bucketIdx+16(FP), R2         // R2 = bucketIdx (uint32, zero-extended)
	MOVD	brLow+24(FP), R3             // R3 = brLow
	MOVD	brHigh+32(FP), R4            // R4 = brHigh

	// Load start = bucketStarts[bucketIdx], end = bucketStarts[bucketIdx+1]
	// Each bucket-start is a uint32 (4 bytes).
	ADD	R2<<2, R0, R5                // R5 = &bucketStarts[bucketIdx]
	MOVWU	0(R5), R6                    // R6 = start
	MOVWU	4(R5), R7                    // R7 = end

	CMP	R6, R7                       // if start == end → empty bucket
	BEQ	notfound

	// base = entries + start * 32
	LSL	$5, R6, R8                   // R8 = start * 32
	ADD	R8, R1, R9                   // R9 = base = entries + start*32

	// limit = entries + end * 32
	LSL	$5, R7, R10                  // R10 = end * 32
	ADD	R10, R1, R10                 // R10 = limit

scanloop:
	// Load the BitRack key (16 trailing bytes of each 32-byte entry)
	// from R9 + 16. ARM64 supports paired loads via LDP.
	LDP	16(R9), (R11, R12)           // R11 = entry.Low, R12 = entry.High

	CMP	R3, R11
	BNE	advance
	CMP	R4, R12
	BNE	advance

	// Match — return current entry pointer
	MOVD	R9, ret+40(FP)
	RET

advance:
	ADD	$32, R9, R9
	CMP	R9, R10
	BNE	scanloop

notfound:
	MOVD	ZR, ret+40(FP)
	RET
