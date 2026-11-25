//! The leave map: the equity adjustment of every subset of the rack, worked
//! out once per rack so that move generation reads a play's in one step.
//! Port of Macondo's leavemap package and movegen's populateLeaveMap (both
//! from MAGPIE's leave_map).
//!
//! Each rack tile has a bit; a subset of the rack is the set of bits of the
//! tiles it keeps. The n copies of a letter use n consecutive bits, and a
//! subset holding k of them has the lowest k set. Taking a tile from the
//! rack clears its letter's highest set bit and putting one back sets the
//! next one, so the index follows the rack as move generation plays tiles
//! and takes them back.

const std = @import("std");
const tiles = @import("tiles.zig");
const kwg_mod = @import("kwg.zig");
const equity = @import("equity.zig");
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;

pub const max_rack_size = 7;
pub const max_values = 1 << max_rack_size;

pub const LeaveMap = struct {
    /// By subset index: what the leave adds to a play's score (the leave
    /// value plus the pre-endgame adjustment, or the endgame adjustment).
    values: [max_values]f64 = undefined,
    /// Each letter's first bit.
    base: [tiles.max_alphabet_size]u8 = @splat(0),
    current: u32 = 0,
    total_tiles: u32 = 0,
    /// False for an empty rack or one of more than 7 tiles.
    initialized: bool = false,

    /// Gives each tile of `rack` its bit, and sets the index to the whole
    /// rack. Port of LeaveMap.Init.
    pub fn init(self: *LeaveMap, rack: *const Rack) void {
        self.initialized = false;
        self.base = @splat(0);
        var next: u32 = 0;
        for (rack.counts, 0..) |n, ml| {
            if (n == 0) continue;
            self.base[ml] = @intCast(@min(next, 255));
            next += n;
        }
        self.total_tiles = next;
        if (next == 0 or next > max_rack_size) return;
        self.current = (@as(u32, 1) << @intCast(next)) - 1;
        self.initialized = true;
    }

    /// A tile of letter `ml` comes back; `number_on_rack` is how many the
    /// rack held before.
    pub inline fn add(self: *LeaveMap, ml: MachineLetter, number_on_rack: u32) void {
        self.current |= @as(u32, 1) << @intCast(self.base[ml] + number_on_rack);
    }

    /// A tile of letter `ml` is played; `number_on_rack` is how many the
    /// rack holds after.
    pub inline fn take(self: *LeaveMap, ml: MachineLetter, number_on_rack: u32) void {
        self.current &= ~(@as(u32, 1) << @intCast(self.base[ml] + number_on_rack));
    }

    pub inline fn currentValue(self: *const LeaveMap) f64 {
        return self.values[self.current];
    }

    /// The `i`th tile of letter `ml` (from 0) moves from the leave into the
    /// tiles played, clearing the letter's bits from the top down. For
    /// enumerating the rack's subsets the other way round, as the word
    /// map's move generator does (MAGPIE's leave_map_complement_add_letter).
    pub inline fn complementAdd(self: *LeaveMap, ml: MachineLetter, i: u32, count: u32) void {
        self.current &= ~(@as(u32, 1) << @intCast(self.base[ml] + count - 1 - i));
    }

    /// Undoes complementAdd.
    pub inline fn complementTake(self: *LeaveMap, ml: MachineLetter, i: u32, count: u32) void {
        self.current |= @as(u32, 1) << @intCast(self.base[ml] + count - 1 - i);
    }
};

/// The highest leave value (or endgame adjustment) over the subsets of
/// each size: an upper bound for the shadow pass.
pub const BestLeaves = [max_rack_size + 1]f64;

/// Fills `lm`'s values for `rack` (which `lm` was initialized with), and
/// `best` with the best raw leave value of each size. Walks the leave
/// graph once, keeping the word index of the leave so far, as movegen's
/// populateLeaveMapIncremental does. `klv` null values every leave at 0.
pub fn populate(lm: *LeaveMap, rack: *Rack, klv: ?*const kwg_mod.Klv, ctx: *const equity.Context, best: *BestLeaves) void {
    populateVisiting(lm, rack, klv, ctx, best, null);
}

/// Called at each subset, with the rack holding it and its value.
pub const Visitor = struct {
    ctx: *anyopaque,
    visit: *const fn (ctx: *anyopaque, leave: *const Rack, value: f64) void,
};

/// populate, calling `visitor` (if any) at each subset: one walk for what
/// needs every leave of the rack, such as recording the exchanges (MAGPIE's
/// generate_exchange_moves).
pub fn populateVisiting(lm: *LeaveMap, rack: *Rack, klv: ?*const kwg_mod.Klv, ctx: *const equity.Context, best: *BestLeaves, visitor: ?Visitor) void {
    best.* = @splat(-std.math.inf(f64));
    var p: Populator = .{ .lm = lm, .rack = rack, .klv = klv, .ctx = ctx, .best = best, .visitor = visitor };
    // The rack's distinct letters: the walks go through those only.
    for (rack.counts[0..ctx.ld.size], 0..) |n, ml| {
        if (n == 0) continue;
        p.letters[p.num_letters] = @intCast(ml);
        p.num_letters += 1;
    }
    if (klv) |k| {
        p.rec(0, k.kwg.dawgRoot(), 0);
    } else {
        p.zero(0);
    }
}

const Populator = struct {
    lm: *LeaveMap,
    rack: *Rack,
    klv: ?*const kwg_mod.Klv,
    ctx: *const equity.Context,
    best: *BestLeaves,
    visitor: ?Visitor = null,
    letters: [max_rack_size]MachineLetter = undefined,
    num_letters: usize = 0,

    /// Records the value of the current leave, whose raw leave value is
    /// `raw` (0 if not in the leave graph).
    fn leaf(self: *Populator, raw_leave: f64) void {
        const n = self.rack.num_tiles;
        const played = self.lm.total_tiles - n;
        var raw = raw_leave;
        // The leave's tile scores matter only for the endgame adjustment.
        const leave_score = if (self.ctx.tiles_in_bag > 0) 0 else self.rack.scoreOn(self.ctx.ld);
        const v = self.ctx.leaveMapValue(raw, leave_score, n, played);
        // With the bag empty the adjustment stands in for the leave value.
        if (self.ctx.tiles_in_bag <= 0) raw = v;
        self.lm.values[self.lm.current] = v;
        if (raw > self.best[n]) self.best[n] = raw;
        if (self.visitor) |vis| vis.visit(vis.ctx, self.rack, v);
    }

    fn rec(self: *Populator, idx: usize, node: u32, word: i32) void {
        if (idx == self.num_letters) {
            var raw: f64 = 0;
            if (self.ctx.tiles_in_bag > 0 and self.rack.num_tiles > 0 and word >= 0) {
                raw = leaveValueByIndex(self.klv.?, word - 1);
            }
            return self.leaf(raw);
        }
        const m = self.letters[idx];
        const num = self.rack.counts[m];
        for (0..num) |_| {
            self.rack.take(m);
            self.lm.take(m, self.rack.counts[m]);
        }
        // None of this letter in the leave: the graph state is unchanged.
        self.rec(idx + 1, node, word);

        // Then one copy more at a time, moving down the graph for each.
        const k = &self.klv.?.kwg;
        var cur_node = node;
        var cur_word = word;
        for (0..num) |i| {
            self.lm.add(m, self.rack.counts[m]);
            self.rack.add(m);
            const next = incrementNodeToMl(self.klv.?, cur_node, cur_word, m);
            if (next.node == 0) {
                // No leave in the graph has this many: every subset with
                // this many or more is valued as if absent.
                self.zero(idx + 1);
                for (i + 1..num) |_| {
                    self.lm.add(m, self.rack.counts[m]);
                    self.rack.add(m);
                    self.zero(idx + 1);
                }
                return;
            }
            cur_node = k.arcIndex(next.node);
            cur_word = next.word + 1;
            self.rec(idx + 1, cur_node, cur_word);
        }
    }

    /// Records every subset of the letters from `idx` on, with the leave so
    /// far absent from the leave graph.
    fn zero(self: *Populator, idx: usize) void {
        if (idx == self.num_letters) return self.leaf(0);
        const m = self.letters[idx];
        const num = self.rack.counts[m];
        for (0..num) |_| {
            self.rack.take(m);
            self.lm.take(m, self.rack.counts[m]);
        }
        self.zero(idx + 1);
        for (0..num) |_| {
            self.lm.add(m, self.rack.counts[m]);
            self.rack.add(m);
            self.zero(idx + 1);
        }
    }
};

fn leaveValueByIndex(klv: *const kwg_mod.Klv, idx: i32) f64 {
    if (idx < 0 or idx >= klv.values.len) return 0;
    return klv.values[@intCast(idx)];
}

/// Moves along the sibling list at `node` to letter `ml`, adding to `word`
/// the words below the siblings passed. Node 0 if `ml` isn't there. Port of
/// incrementNodeToML (MAGPIE's increment_node_to_ml).
fn incrementNodeToMl(klv: *const kwg_mod.Klv, node: u32, word: i32, ml: MachineLetter) struct { node: u32, word: i32 } {
    if (node == 0) return .{ .node = 0, .word = -1 };
    const k = &klv.kwg;
    const counts = klv.counts.counts;
    var i = node;
    var w = word;
    while (true) : (i += 1) {
        if (k.tile(i) == ml) return .{ .node = i, .word = w };
        if (k.isEnd(i)) return .{ .node = 0, .word = -1 };
        w += counts[i] - counts[i + 1];
    }
}

// Tests ----------------------------------------------------------------------
// Ported from movegen/leave_map_test.go (itself from MAGPIE's
// test/leave_map_test.c), plus the pre-endgame and endgame values.

const testing = std.testing;
const testdata = @import("testdata.zig");

fn rackOf(pairs: []const struct { MachineLetter, u8 }) Rack {
    var r: Rack = .{};
    for (pairs) |p| {
        for (0..p[1]) |_| r.add(p[0]);
    }
    return r;
}

test "leave map index" {
    // Seven different tiles, a bit each.
    var rack = rackOf(&.{ .{ 1, 1 }, .{ 2, 1 }, .{ 3, 1 }, .{ 4, 1 }, .{ 5, 1 }, .{ 6, 1 }, .{ 7, 1 } });
    var lm: LeaveMap = .{};
    lm.init(&rack);
    try testing.expect(lm.initialized);
    try testing.expectEqual(127, lm.current);

    const cases = [_]struct { MachineLetter, u32, f64 }{ .{ 1, 126, 7 }, .{ 2, 125, 8 }, .{ 5, 111, 9 }, .{ 7, 63, 10 } };
    for (cases) |c| {
        rack.take(c[0]);
        lm.take(c[0], rack.counts[c[0]]);
        try testing.expectEqual(c[1], lm.current);
        lm.values[lm.current] = c[2];
        try testing.expectEqual(c[2], lm.currentValue());
        lm.add(c[0], rack.counts[c[0]]);
        rack.add(c[0]);
        try testing.expectEqual(127, lm.current);
    }
    // The values stay where they were put.
    for (cases) |c| {
        rack.take(c[0]);
        lm.take(c[0], rack.counts[c[0]]);
        try testing.expectEqual(c[2], lm.currentValue());
        lm.add(c[0], rack.counts[c[0]]);
        rack.add(c[0]);
    }
}

test "leave map with duplicate tiles" {
    // DDDIIUU.
    var rack = rackOf(&.{ .{ 4, 3 }, .{ 9, 2 }, .{ 21, 2 } });
    var lm: LeaveMap = .{};
    lm.init(&rack);
    try testing.expect(lm.initialized);
    try testing.expectEqual(7, lm.total_tiles);
    try testing.expectEqual(127, lm.current);
    lm.values[127] = 100;

    var diu: u32 = 0;
    for ([_]MachineLetter{ 4, 21, 4, 9, 4, 9, 21 }, 11..) |ml, v| {
        rack.take(ml);
        lm.take(ml, rack.counts[ml]);
        lm.values[lm.current] = @floatFromInt(v);
        if (v == 14) diu = lm.current; // leave DIU
    }
    try testing.expectEqual(0, lm.current);

    // Back in another order: the same index for the same leave.
    for ([_]MachineLetter{ 4, 9, 21 }) |ml| {
        lm.add(ml, rack.counts[ml]);
        rack.add(ml);
    }
    try testing.expectEqual(diu, lm.current);
    try testing.expectEqual(14, lm.currentValue());
    for ([_]MachineLetter{ 4, 9, 21, 4 }) |ml| {
        lm.add(ml, rack.counts[ml]);
        rack.add(ml);
    }
    try testing.expectEqual(127, lm.current);
    try testing.expectEqual(100, lm.currentValue());
}

/// Checks every subset's value against the equity module's, exactly, and
/// the best values by size.
fn verifySubsets(lm: *LeaveMap, rack: *Rack, ctx: *const equity.Context, best: *const BestLeaves, ml_start: usize, seen_best: *BestLeaves) !void {
    var ml = ml_start;
    while (ml < ctx.ld.size and rack.counts[ml] == 0) ml += 1;
    if (ml == ctx.ld.size) {
        var buf: [max_rack_size]MachineLetter = undefined;
        const n = rack.tiles(&buf);
        const raw = ctx.leaveValue(buf[0..n]);
        const want = ctx.leaveMapValue(raw, rack.scoreOn(ctx.ld), rack.num_tiles, lm.total_tiles - rack.num_tiles);
        try testing.expectEqual(want, lm.currentValue());
        const r = if (ctx.tiles_in_bag > 0) raw else want;
        if (r > seen_best[n]) seen_best[n] = r;
        return;
    }
    const m: MachineLetter = @intCast(ml);
    try verifySubsets(lm, rack, ctx, best, ml + 1, seen_best);
    const num = rack.counts[ml];
    for (0..num) |_| {
        rack.take(m);
        lm.take(m, rack.counts[ml]);
        try verifySubsets(lm, rack, ctx, best, ml + 1, seen_best);
    }
    for (0..num) |_| {
        lm.add(m, rack.counts[ml]);
        rack.add(m);
    }
}

test "leave map values match the leave values" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    const bytes = try testdata.loadFile("lexica/gaddag/NWL20.klv2");
    defer testing.allocator.free(bytes);
    var klv = try kwg_mod.Klv.init(testing.allocator, bytes);
    defer klv.deinit(testing.allocator);
    const peg = [_]f64{ 0, -8, 0, -0.5, -2, -3.5, -2, 2, 10, 7, 4, -1, -2 };

    const racks = [_][]const u8{
        "AEINRST", "DDDIIUU", "?AEINRS", "??EINRS", "QZ",      "RETAINS",
        "AAAAAAA", "EEEIILZ", "AAAEOOS", "AAEFGIK", "EIMNORS", "Q",
    };
    // Mid-game, with a pre-endgame table near the end, and the endgame.
    const settings = [_]struct { i32, []const f64, ?*const kwg_mod.Klv }{
        .{ 80, &.{}, &klv }, .{ 3, &peg, &klv }, .{ 0, &.{}, &klv }, .{ 4, &peg, null },
    };
    for (settings) |st| {
        const ctx: equity.Context = .{ .ld = &f.ld, .klv = st[2], .peg = st[1], .board_dim = 15, .board_empty = false, .tiles_in_bag = st[0], .opp_rack_score = 20 };
        for (racks) |text| {
            var mls: [8]MachineLetter = undefined;
            var rack: Rack = .fromLetters(mls[0..try f.ld.toMachineLetters(text, &mls)]);
            var lm: LeaveMap = .{};
            lm.init(&rack);
            try testing.expect(lm.initialized);
            var best: BestLeaves = undefined;
            populate(&lm, &rack, ctx.klv, &ctx, &best);
            try testing.expectEqual((@as(u32, 1) << @intCast(rack.num_tiles)) - 1, lm.current);
            var seen: BestLeaves = @splat(-std.math.inf(f64));
            try verifySubsets(&lm, &rack, &ctx, &best, 0, &seen);
            try testing.expectEqualSlices(f64, seen[0 .. rack.num_tiles + 1], best[0 .. rack.num_tiles + 1]);
        }
    }
}
