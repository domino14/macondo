//! The endgame's transposition table: port of
//! endgame/negamax/transposition_table.go. Entries are 16 bytes, four to a
//! bucket (one cache line). A store replaces the entry of the same
//! position, else an empty one, else the shallowest (pure entries first).
//! Lookups and stores are lockless: a slot's two words are read and
//! written atomically, the first last, and a read that sees the first word
//! change is dropped.

const std = @import("std");

pub const exact: u8 = 1;
pub const lower: u8 = 2;
pub const upper: u8 = 3;

pub const depth_mask: u8 = (1 << 6) - 1;
/// Marks an entry whose search reached only finished games: its value
/// holds at its depth and every greater one. The top bit of the fifth hash
/// byte, which the hash check then leaves out.
pub const pure_bit: u8 = 0x80;
const bottom3_byte_mask: u64 = (1 << 24) - 1;
const bucket_size = 4;

/// A move as the table stores it (tinymove.TinyMove); `invalid` is none.
pub const TinyMove = u64;
pub const invalid_move: TinyMove = 1 << 63;

/// The least number of entries, as the Go engine's: fewer and the five
/// stored hash bytes would not tell positions apart.
pub const min_size_power = 24;

pub const Entry = struct {
    word1: u64 = 0,
    word2: u64 = 0,

    pub inline fn flag(e: Entry) u8 {
        return @as(u8, @truncate(e.word1)) >> 6;
    }
    pub inline fn depth(e: Entry) u8 {
        return @as(u8, @truncate(e.word1)) & depth_mask;
    }
    pub inline fn valid(e: Entry) bool {
        return e.flag() != 0;
    }
    pub inline fn score(e: Entry) i16 {
        return @bitCast(@as(u16, @truncate(e.word1 >> 16)));
    }
    pub inline fn pure(e: Entry) bool {
        return @as(u8, @truncate(e.word1 >> 8)) & pure_bit != 0;
    }
    pub inline fn play(e: Entry) TinyMove {
        return e.word2;
    }
};

const Slot = extern struct { word1: u64, word2: u64 };

pub const Table = struct {
    slots: []Slot = &.{},
    mask: u64 = 0,

    /// Sizes the table to the largest power of two of entries that fits in
    /// `bytes` (at least 2^24), and clears it.
    pub fn reset(self: *Table, gpa: std.mem.Allocator, bytes: u64) std.mem.Allocator.Error!void {
        const want = @max(bytes / @sizeOf(Slot), 1);
        const power = @max(std.math.log2_int(u64, want), min_size_power);
        const n = @as(usize, 1) << @intCast(power);
        if (self.slots.len != n) {
            gpa.free(self.slots);
            self.slots = &.{};
            self.slots = try gpa.alloc(Slot, n);
        }
        @memset(self.slots, .{ .word1 = 0, .word2 = 0 });
        self.mask = n - 1;
    }

    pub fn deinit(self: *Table, gpa: std.mem.Allocator) void {
        gpa.free(self.slots);
        self.* = undefined;
    }

    inline fn load(self: *const Table, idx: u64) ?Entry {
        const s = &self.slots[idx];
        const w1 = @atomicLoad(u64, &s.word1, .monotonic);
        const w2 = @atomicLoad(u64, &s.word2, .monotonic);
        if (@atomicLoad(u64, &s.word1, .monotonic) != w1) return null;
        return .{ .word1 = w1, .word2 = w2 };
    }

    /// What identifies a position within its bucket.
    inline fn bucketHash(zval: u64) u64 {
        return zval & ~(@as(u64, pure_bit) << 24) & ~@as(u64, bucket_size - 1);
    }

    inline fn fullHash(e: Entry, base: u64) u64 {
        const top4 = e.word1 >> 32;
        const fifth = @as(u8, @truncate(e.word1 >> 8)) & ~pure_bit;
        return (top4 << 32) + (@as(u64, fifth) << 24) + (base & bottom3_byte_mask);
    }

    pub fn prefetch(self: *const Table, zval: u64) void {
        if (self.slots.len == 0) return;
        @prefetch(&self.slots[(zval & self.mask) & ~@as(u64, bucket_size - 1)], .{});
    }

    /// The entry for `zval`, or an invalid one.
    pub fn lookup(self: *Table, zval: u64) Entry {
        const base = (zval & self.mask) & ~@as(u64, bucket_size - 1);
        const want = bucketHash(zval);
        for (0..bucket_size) |i| {
            const e = self.load(base + i) orelse continue;
            if (!e.valid()) continue;
            if (fullHash(e, base) == want) {
                return e;
            }
        }
        return .{};
    }

    /// Stores an entry; `pure` marks one whose search reached only
    /// finished games.
    pub fn store(self: *Table, zval: u64, score: i16, flag_and_depth: u8, play: TinyMove, pure: bool) void {
        const top4: u64 = zval >> 32;
        var fifth: u8 = @as(u8, @truncate(zval >> 24)) & ~pure_bit;
        if (pure) fifth |= pure_bit;
        const base = (zval & self.mask) & ~@as(u64, bucket_size - 1);
        const want = bucketHash(zval);
        var victim = base;
        var victim_rank: usize = std.math.maxInt(usize);
        for (0..bucket_size) |i| {
            var rank: usize = 0;
            if (self.load(base + i)) |e| {
                if (e.valid()) {
                    if (fullHash(e, base) == want) {
                        victim = base + i;
                        break;
                    }
                    rank = if (e.pure()) 1 else 1 + @as(usize, e.depth());
                }
            }
            if (rank < victim_rank) {
                victim_rank = rank;
                victim = base + i;
            }
        }
        const w1 = (top4 << 32) | (@as(u64, @as(u16, @bitCast(score))) << 16) | (@as(u64, fifth) << 8) | flag_and_depth;
        const s = &self.slots[victim];
        @atomicStore(u64, &s.word2, play, .monotonic);
        @atomicStore(u64, &s.word1, w1, .release);
    }
};

test "store, look up and replace" {
    const gpa = std.testing.allocator;
    var t: Table = .{};
    defer t.deinit(gpa);
    try t.reset(gpa, 0);
    try std.testing.expectEqual(@as(usize, 1) << min_size_power, t.slots.len);
    const key: u64 = 0x1234_5678_9abc_def0;
    try std.testing.expect(!t.lookup(key).valid());
    t.store(key, -42, (exact << 6) | 5, 77, false);
    const e = t.lookup(key);
    try std.testing.expect(e.valid());
    try std.testing.expectEqual(-42, e.score());
    try std.testing.expectEqual(exact, e.flag());
    try std.testing.expectEqual(5, e.depth());
    try std.testing.expectEqual(77, e.play());
    try std.testing.expect(!e.pure());
    // The pure bit is not part of the key.
    t.store(key ^ (@as(u64, pure_bit) << 24), 3, (lower << 6) | 2, 1, true);
    const p = t.lookup(key);
    try std.testing.expectEqual(3, p.score());
    try std.testing.expect(p.pure());
    // Four other positions in the bucket: the shallowest goes.
    for (1..5) |i| t.store(key + (@as(u64, i) << 40), 0, (exact << 6) | @as(u8, @intCast(i + 3)), 0, false);
    try std.testing.expect(!t.lookup(key).valid());
    try std.testing.expect(t.lookup(key + (2 << 40)).valid());
}
