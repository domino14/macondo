//! The positions the endgame's threads are in, for ABDADA (Weill, "The
//! ABDADA Distributed Minimax Search Algorithm", 1996), kept as MAGPIE keeps
//! them (its transposition table's nproc counters): a thread counts itself
//! into each position it searches from min_depth up, and out when done
//! there. A node's younger moves are searched exclusively: one whose
//! position another thread is in is put off until the node's other moves
//! are done, by when that thread's result may be in the transposition
//! table. Lockless counters indexed by the position's key; two positions
//! that share a counter cost only a move put off.
const std = @import("std");

/// The depth from which threads count themselves into positions: below it a
/// subtree costs less than the bookkeeping (MAGPIE's ABDADA_MIN_DEPTH).
pub const min_depth = 2;

const size = 1 << 18;

pub const Table = struct {
    counts: [size]std.atomic.Value(u8) = @splat(.init(0)),

    /// Whether a thread is in the position `key`.
    pub fn busy(self: *const Table, key: u64) bool {
        return self.counts[key & (size - 1)].load(.monotonic) > 0;
    }

    /// Counts a thread into the position `key`.
    pub fn enter(self: *Table, key: u64) void {
        _ = self.counts[key & (size - 1)].fetchAdd(1, .monotonic);
    }

    /// Counts a thread out of the position `key`.
    pub fn leave(self: *Table, key: u64) void {
        _ = self.counts[key & (size - 1)].fetchSub(1, .monotonic);
    }
};

const testing = std.testing;

test "a position is busy while a thread is in it" {
    const t = try testing.allocator.create(Table);
    defer testing.allocator.destroy(t);
    t.* = .{};
    try testing.expect(!t.busy(42));
    t.enter(42);
    t.enter(42);
    try testing.expect(t.busy(42));
    try testing.expect(!t.busy(43));
    t.leave(42);
    try testing.expect(t.busy(42));
    t.leave(42);
    try testing.expect(!t.busy(42));
}
