//! Zobrist hashing of endgame positions: the board, both racks, whose turn
//! it is and the turns without a score. Port of zobrist/hash.go.
//!
//! The keys are random. Macondo draws them from an unseeded generator, so
//! its hashes differ from run to run; the cross-checks hand the Go engine's
//! keys to this one (`fromJson`, the format of Zobrist.Dump) to compare
//! searches node for node.

const std = @import("std");
const tiles = @import("../data/tiles.zig");
const chacha8 = @import("../gocompat/chacha8.zig");
const MachineLetter = tiles.MachineLetter;

/// Letters per position table half: a blank is its letter plus this.
pub const max_letters = 35;
/// A rack holds up to this many of a letter (EEEEEEE is legal).
pub const max_count = 7;

pub const Zobrist = struct {
    their_turn: u64,
    /// By square (row * dim + col), then letter (blanks at letter + 35).
    pos: []u64,
    our_rack: [max_letters][max_count + 1]u64,
    their_rack: [max_letters][max_count + 1]u64,
    scoreless_turns: [3]u64,
    dim: usize,

    /// Keys drawn from a ChaCha8 stream seeded with `seed`.
    pub fn init(gpa: std.mem.Allocator, dim: usize, seed: [32]u8) std.mem.Allocator.Error!Zobrist {
        var rng: chacha8.Rng = .init(seed);
        const pos = try gpa.alloc(u64, dim * dim * 2 * max_letters);
        // As Go draws them: Uint64n(1<<63 - 2) + 1, never 0.
        const draw = struct {
            fn f(r: *chacha8.Rng) u64 {
                return r.uint64n((1 << 63) - 2) + 1;
            }
        }.f;
        for (pos) |*k| k.* = draw(&rng);
        var z: Zobrist = .{ .their_turn = 0, .pos = pos, .our_rack = undefined, .their_rack = undefined, .scoreless_turns = undefined, .dim = dim };
        for (&z.our_rack) |*row| {
            for (row) |*k| k.* = draw(&rng);
        }
        for (&z.their_rack) |*row| {
            for (row) |*k| k.* = draw(&rng);
        }
        for (&z.scoreless_turns) |*k| k.* = draw(&rng);
        z.their_turn = draw(&rng);
        return z;
    }

    pub const JsonError = std.mem.Allocator.Error || error{BadZobrist};

    /// The keys Macondo's Zobrist.Dump writes (JSON with TheirTurn,
    /// PosTable, OurRackTable, TheirRackTable and ScorelessTurns).
    pub fn fromJson(gpa: std.mem.Allocator, text: []const u8) JsonError!Zobrist {
        const Dump = struct {
            TheirTurn: u64,
            PosTable: [][]u64,
            OurRackTable: [][]u64,
            TheirRackTable: [][]u64,
            ScorelessTurns: [3]u64,
        };
        const parsed = std.json.parseFromSlice(Dump, gpa, text, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.BadZobrist,
        };
        defer parsed.deinit();
        const d = parsed.value;
        const dim = std.math.sqrt(d.PosTable.len);
        if (dim * dim != d.PosTable.len or d.OurRackTable.len != max_letters or d.TheirRackTable.len != max_letters) return error.BadZobrist;
        const pos = try gpa.alloc(u64, dim * dim * 2 * max_letters);
        errdefer gpa.free(pos);
        for (d.PosTable, 0..) |row, i| {
            if (row.len != 2 * max_letters) return error.BadZobrist;
            @memcpy(pos[i * 2 * max_letters ..][0 .. 2 * max_letters], row);
        }
        var z: Zobrist = .{ .their_turn = d.TheirTurn, .pos = pos, .our_rack = undefined, .their_rack = undefined, .scoreless_turns = d.ScorelessTurns, .dim = dim };
        for (0..max_letters) |i| {
            if (d.OurRackTable[i].len != max_count + 1 or d.TheirRackTable[i].len != max_count + 1) return error.BadZobrist;
            @memcpy(&z.our_rack[i], d.OurRackTable[i]);
            @memcpy(&z.their_rack[i], d.TheirRackTable[i]);
        }
        return z;
    }

    pub fn deinit(self: *Zobrist, gpa: std.mem.Allocator) void {
        gpa.free(self.pos);
        self.* = undefined;
    }

    /// The key of a square holding `ml`.
    pub inline fn square(self: *const Zobrist, idx: usize, ml: MachineLetter) u64 {
        const letter: usize = if (tiles.isBlanked(ml)) @as(usize, tiles.unblank(ml)) + max_letters else ml;
        return self.pos[idx * 2 * max_letters + letter];
    }

    /// The board's part of a key: squares in row-major order, `dim` wide.
    pub fn boardHash(self: *const Zobrist, squares: []const MachineLetter) u64 {
        var key: u64 = 0;
        for (squares, 0..) |ml, i| {
            if (ml != 0) key ^= self.square(i, ml);
        }
        return key;
    }

    /// The key of a position: the board, the solving player's rack
    /// (`ours`) and the other's, `num_letters` letters each (every count is
    /// hashed, 0 included), whether the other player is on turn, and the
    /// turns without a score.
    pub fn hash(self: *const Zobrist, squares: []const MachineLetter, ours: *const tiles.Rack, theirs: *const tiles.Rack, num_letters: usize, their_turn: bool, scoreless_turns: u32) u64 {
        var key = self.boardHash(squares);
        for (0..num_letters) |i| key ^= self.our_rack[i][ours.counts[i]];
        for (0..num_letters) |i| key ^= self.their_rack[i][theirs.counts[i]];
        if (their_turn) key ^= self.their_turn;
        key ^= self.scoreless_turns[scoreless_turns];
        return key;
    }

    inline fn rackTable(self: *const Zobrist, ours: bool) *const [max_letters][max_count + 1]u64 {
        return if (ours) &self.our_rack else &self.their_rack;
    }

    /// The key after a move from the position with key `key` (AddMove).
    /// `placed` holds the tiles the move put down by position along it (0
    /// for a square it played through), at (row, col) going across or
    /// down; `leave` is the mover's rack after the move. `scoreless` and
    /// `last_scoreless` are the counts after it and before it.
    pub fn addMove(self: *const Zobrist, key_in: u64, is_pass: bool, row: usize, col: usize, vertical: bool, placed: []const MachineLetter, leave: *const tiles.Rack, num_letters: usize, ours: bool, scoreless: u32, last_scoreless: u32) u64 {
        var key = key_in;
        if (!is_pass) {
            const table = self.rackTable(ours);
            var counts: [max_letters]u8 = @splat(0);
            for (placed, 0..) |ml, i| {
                if (ml == 0) continue;
                const r = if (vertical) row + i else row;
                const c = if (vertical) col else col + i;
                key ^= self.square(r * self.dim + c, ml);
                counts[if (tiles.isBlanked(ml)) 0 else ml] += 1;
            }
            for (0..num_letters) |i| counts[i] += leave.counts[i];
            for (placed) |ml| {
                if (ml == 0) continue;
                const t: usize = if (tiles.isBlanked(ml)) 0 else ml;
                key ^= table[t][counts[t]];
                counts[t] -= 1;
                key ^= table[t][counts[t]];
            }
        }
        if (last_scoreless != scoreless) {
            key ^= self.scoreless_turns[last_scoreless];
            key ^= self.scoreless_turns[scoreless];
        }
        return key ^ self.their_turn;
    }

    /// The key a tile play would lead to, worked out without playing it
    /// (AddTilePlayUnplayed): `rack` is the mover's rack before it and
    /// `scoreless` the count before it.
    pub fn addTilePlayUnplayed(self: *const Zobrist, key_in: u64, row: usize, col: usize, vertical: bool, placed: []const MachineLetter, rack: *const tiles.Rack, ours: bool, scoreless: u32) u64 {
        var key = key_in;
        const table = self.rackTable(ours);
        var counts: [max_letters]u8 = @splat(0);
        for (placed, 0..) |ml, i| {
            if (ml == 0) continue;
            const r = if (vertical) row + i else row;
            const c = if (vertical) col else col + i;
            key ^= self.square(r * self.dim + c, ml);
            const t: usize = if (tiles.isBlanked(ml)) 0 else ml;
            if (counts[t] == 0) counts[t] = rack.counts[t];
            key ^= table[t][counts[t]];
            counts[t] -= 1;
            key ^= table[t][counts[t]];
        }
        if (scoreless != 0) {
            key ^= self.scoreless_turns[scoreless];
            key ^= self.scoreless_turns[0];
        }
        return key ^ self.their_turn;
    }
};

test "a move's key is the key of the position it leads to" {
    var z = try Zobrist.init(std.testing.allocator, 15, @splat(7));
    defer z.deinit(std.testing.allocator);
    var squares: [225]MachineLetter = @splat(0);
    var ours: tiles.Rack = .fromLetters(&.{ 1, 5, 5, 0, 19 });
    const theirs: tiles.Rack = .fromLetters(&.{ 2, 3 });
    const before = z.hash(&squares, &ours, &theirs, 27, false, 0);
    // E, blank as T and S across from 8H, through nothing.
    const placed = [_]MachineLetter{ 5, tiles.blank(20), 19 };
    const unplayed = z.addTilePlayUnplayed(before, 7, 7, false, &placed, &ours, true, 0);
    for (placed, 0..) |ml, i| squares[7 * 15 + 7 + i] = ml;
    ours.take(5);
    ours.take(0);
    ours.take(19);
    const after = z.hash(&squares, &ours, &theirs, 27, true, 0);
    try std.testing.expectEqual(after, z.addMove(before, false, 7, 7, false, &placed, &ours, 27, true, 0, 0));
    try std.testing.expectEqual(after, unplayed);
    // A pass: the turn and the scoreless count change.
    const passed = z.addMove(after, true, 0, 0, false, &.{}, &theirs, 27, false, 1, 0);
    try std.testing.expectEqual(z.hash(&squares, &ours, &theirs, 27, false, 1), passed);
}
