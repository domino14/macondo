//go:build !amd64 && !arm64

package negamax

import "unsafe"

// prefetch does nothing on processors without an implementation.
func prefetch(addr unsafe.Pointer) {}
