//go:build amd64 || arm64

package negamax

import "unsafe"

// prefetch asks the processor to start loading the cache line at addr. It
// never faults and changes nothing a program can observe except timing.
//
//go:noescape
func prefetch(addr unsafe.Pointer)
