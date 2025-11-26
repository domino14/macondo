package negamax

import (
	"sync"
	"sync/atomic"
)

// WorkDeque is a simple double-ended queue for work-stealing
// Supports fast pop from the bottom (owner thread) and steal from the top (other threads)
// Stores indices into a shared move array
type WorkDeque struct {
	mu      sync.Mutex
	indices []int
	bottom  atomic.Int32 // Owner pops from bottom (local)
	top     atomic.Int32 // Thieves steal from top (remote)
}

// NewWorkDeque creates a work deque with the given indices
func NewWorkDeque(indices []int) *WorkDeque {
	d := &WorkDeque{
		indices: indices,
	}
	d.bottom.Store(int32(len(indices)))
	d.top.Store(0)
	return d
}

// Pop removes and returns an index from the bottom (owner thread, lock-free in common case)
func (d *WorkDeque) Pop() (int, bool) {
	b := d.bottom.Add(-1)
	if b < 0 {
		d.bottom.Store(0)
		return -1, false
	}

	t := d.top.Load()
	if b < t {
		// Empty or being stolen
		d.bottom.Store(t)
		return -1, false
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Re-check after acquiring lock
	t = d.top.Load()
	if b < t {
		d.bottom.Store(t)
		return -1, false
	}

	idx := d.indices[b]
	if b == t {
		// Last element - race with stealers
		if !d.top.CompareAndSwap(t, t+1) {
			// Lost race - stealer got it
			d.bottom.Store(t + 1)
			return -1, false
		}
		d.bottom.Store(t + 1)
	}
	return idx, true
}

// Steal attempts to steal an index from the top (other threads)
func (d *WorkDeque) Steal() (int, bool) {
	t := d.top.Load()
	b := d.bottom.Load()

	if t >= b {
		// Empty
		return -1, false
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Re-check after acquiring lock
	t = d.top.Load()
	b = d.bottom.Load()
	if t >= b {
		return -1, false
	}

	idx := d.indices[t]
	if !d.top.CompareAndSwap(t, t+1) {
		// Lost race
		return -1, false
	}
	return idx, true
}

// Size returns the current size of the deque
func (d *WorkDeque) Size() int {
	b := d.bottom.Load()
	t := d.top.Load()
	size := int(b - t)
	if size < 0 {
		return 0
	}
	return size
}
