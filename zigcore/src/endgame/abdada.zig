//! The moves the endgame's threads are searching, for ABDADA (Weill, "The
//! ABDADA Distributed Minimax Search Algorithm", 1996): port of
//! endgame/negamax/abdada.go. A thread about to search a move another is
//! already searching at the same depth or deeper puts it off, and searches
//! it after the node's other moves, by when the other thread's result is
//! in the transposition table. Lockless: each slot is a move's hash and
//! depth, read and written atomically, and a lost or stale slot costs only
//! a move searched twice or put off once.
const std = @import("std");

/// The depth from which moves are put off: below it a subtree is cheaper
/// than the bookkeeping (MAGPIE's, and Go's DEFER_DEPTH).
pub const min_depth = 2;

const sets = 1 << 15;
const ways = 4;

const Slot = struct {
    hash: std.atomic.Value(u64) = .init(0),
    depth: std.atomic.Value(u32) = .init(0),
};

pub const Table = struct {
    slots: [sets][ways]Slot = @splat(@splat(.{})),

    /// A move at a position, hashed (Go's moveHash); never 0 in practice,
    /// which marks an empty slot.
    inline fn moveHash(node_key: u64, tm: u64) u64 {
        return node_key *% 1103515245 +% tm *% 12345;
    }

    /// Whether another thread is searching move `tm` of the position
    /// `node_key` at depth `depth` or deeper.
    pub fn busy(self: *const Table, node_key: u64, tm: u64, depth: u32) bool {
        const h = moveHash(node_key, tm);
        for (&self.slots[h % sets]) |*s| {
            if (s.hash.load(.monotonic) == h and s.depth.load(.monotonic) >= depth) return true;
        }
        return false;
    }

    /// Notes that this thread is searching the move: in an empty slot of
    /// its set if there is one, else over the set's first.
    pub fn start(self: *Table, node_key: u64, tm: u64, depth: u32) void {
        const h = moveHash(node_key, tm);
        const set = &self.slots[h % sets];
        for (set) |*s| {
            if (s.hash.load(.monotonic) == 0 and s.hash.cmpxchgStrong(0, h, .monotonic, .monotonic) == null) {
                s.depth.store(depth, .monotonic);
                return;
            }
        }
        set[0].hash.store(h, .monotonic);
        set[0].depth.store(depth, .monotonic);
    }

    /// Notes that the move's search is over.
    pub fn finish(self: *Table, node_key: u64, tm: u64) void {
        const h = moveHash(node_key, tm);
        for (&self.slots[h % sets]) |*s| {
            if (s.hash.load(.monotonic) == h) {
                s.hash.store(0, .monotonic);
                s.depth.store(0, .monotonic);
            }
        }
    }
};

const testing = std.testing;

test "a move is busy from its start to its finish, at its depth and below" {
    const t = try testing.allocator.create(Table);
    defer testing.allocator.destroy(t);
    t.* = .{};
    try testing.expect(!t.busy(42, 7, 3));
    t.start(42, 7, 3);
    try testing.expect(t.busy(42, 7, 3));
    try testing.expect(t.busy(42, 7, 2));
    try testing.expect(!t.busy(42, 7, 4));
    try testing.expect(!t.busy(42, 8, 3));
    t.finish(42, 7);
    try testing.expect(!t.busy(42, 7, 2));
}
