//go:build amd64 || arm64

// scanbucket_asm.go: experimental ABI0 assembly stub for scanBucket.
//
// The Go inliner already inlines scanBucket (cost 72, under the
// budget 80) into each of getWordEntry/getBlankEntry/getDoubleBlankEntry.
// Replacing it with an out-of-line asm function adds a per-call ABI0
// bridge that's a fraction of a ns but typically swamps the savings
// for sub-microsecond loops. We measure both ways to verify.

package wmp

import "unsafe"

// scanBucketAsm is the assembly version of scanBucket. Same
// semantics — given a bucket-starts table, an entries pointer, a
// bucket index, and the BitRack key (low/high), it walks the
// entries from start..end-1 and returns a pointer to the matching
// entry, or nil.
//
// We pass entries via a raw unsafe.Pointer (the address of
// entries[0]) so the asm doesn't need to crack a slice header.
//
//go:noescape
func scanBucketAsm(bucketStartsBase *uint32, entriesBase unsafe.Pointer, bucketIdx uint32, brLow, brHigh uint64) *Entry
