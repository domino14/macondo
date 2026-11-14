//go:build linux

package negamax

import (
	"syscall"
	"unsafe"
)

// adviseHugePages asks the kernel to back a new table with huge pages, and
// reports whether it agreed. It is advice: where the kernel declines, the
// table works as before. See TranspositionTable.touchHugePage.
func adviseHugePages(table []TableEntry) bool {
	if len(table) == 0 {
		return false
	}
	first := uintptr(unsafe.Pointer(&table[0]))
	size := uintptr(len(table)) * unsafe.Sizeof(table[0])
	// Only whole huge pages inside the table.
	start := (first + hugePageSize - 1) &^ (hugePageSize - 1)
	end := (first + size) &^ (hugePageSize - 1)
	if end <= start {
		return false
	}
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&table[0])), size)
	return syscall.Madvise(bytes[start-first:end-first], syscall.MADV_HUGEPAGE) == nil
}
