//go:build !linux

package negamax

// adviseHugePages does nothing where there is no such advice to give.
func adviseHugePages(table []TableEntry) bool { return false }
