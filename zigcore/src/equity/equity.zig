//! Static equity: a move's score plus the value of what it leaves, with
//! adjustments for the opening, the pre-endgame and the endgame. Port of
//! Macondo's equity.CombinedStaticCalculator, the evaluator the static
//! player and the simulations use.
//!
//! The additions are made in the same order as in Go so that equities are
//! the same to the last bit.

const std = @import("std");
const tiles = @import("../data/tiles.zig");
const kwg_mod = @import("../data/kwg.zig");
const move_mod = @import("../board/move.zig");
const Move = move_mod.Move;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;

/// The equity the static player gives a pass in a full move list (Go's
/// turnplayer.StaticPassPenalty). The best-play search values a pass like
/// any other move instead.
pub const static_pass_penalty: f64 = -50.0;

/// Endgame: a play that doesn't go out loses twice its leave's score and
/// this much more; going out gains twice the opponent's rack.
pub const endgame_non_outplay_leave_penalty_multiplier = 2;
pub const endgame_non_outplay_constant_penalty: f64 = 10.0;

/// What equity needs to know about the position, for the player to move.
pub const Context = struct {
    ld: *const tiles.LetterDistribution,
    /// Leave values, or null to value leaves at 0.
    klv: ?*const kwg_mod.Klv,
    /// Pre-endgame adjustments, by tiles in the bag after the play plus 7.
    peg: []const f64 = &.{},
    board_dim: usize,
    board_empty: bool,
    tiles_in_bag: i32,
    /// The sum of the opponent's rack's tile scores.
    opp_rack_score: i32,

    pub fn leaveValue(self: *const Context, sorted_leave: []const MachineLetter) f64 {
        const klv = self.klv orelse return 0;
        return klv.leaveValue(sorted_leave);
    }

    /// The equity of `m`, which leaves `leave` on the rack. Port of
    /// CombinedStaticCalculator.Equity.
    pub fn equity(self: *const Context, m: *const Move, leave: *const Rack) f64 {
        var leave_adjustment: f64 = 0.0;
        var other_adjustments: f64 = 0.0;
        if (self.board_empty) other_adjustments += placementAdjustment(m, self.board_dim, self.ld);
        if (self.tiles_in_bag > 0) {
            var buf: [tiles.max_alphabet_size]MachineLetter = undefined;
            leave_adjustment = self.leaveValue(buf[0..leave.tiles(&buf)]);
            const bag_plus_seven = self.tiles_in_bag - @as(i32, m.tiles_played) + 7;
            if (bag_plus_seven < self.peg.len) other_adjustments += self.peg[@intCast(bag_plus_seven)];
        } else {
            other_adjustments += endgameAdjustment(leave.scoreOn(self.ld), leave.num_tiles, self.opp_rack_score);
        }
        return @as(f64, @floatFromInt(m.score)) + leave_adjustment + other_adjustments;
    }

    /// What the leave map holds for a leave: its value plus the
    /// pre-endgame adjustment, or the endgame adjustment with the bag
    /// empty. `tiles_played` is the size of the rack less the leave's.
    /// Port of the values movegen's populateLeaveMap stores.
    pub fn leaveMapValue(self: *const Context, raw_leave: f64, leave_score: i32, leave_size: u32, tiles_played: u32) f64 {
        if (self.tiles_in_bag > 0) {
            var v = raw_leave;
            const bag_plus_seven = self.tiles_in_bag - @as(i32, @intCast(tiles_played)) + 7;
            if (bag_plus_seven >= 0 and bag_plus_seven < self.peg.len) v += self.peg[@intCast(bag_plus_seven)];
            return v;
        }
        return endgameAdjustment(leave_score, leave_size, self.opp_rack_score);
    }
};

/// Opening placement: a rough penalty for each vowel played next to a
/// double-letter square the opponent can use (from the Maven paper). Only
/// for the first play on an empty board. Port of placementAdjustment.
pub fn placementAdjustment(m: *const Move, board_dim: usize, ld: *const tiles.LetterDistribution) f64 {
    if (m.kind != .play) return 0;
    const start: usize = if (m.vertical) m.row else m.col;
    const v_penalty = -0.7;
    var penalty: f64 = 0.0;
    for (0..m.tiles_played) |i| {
        const j = start + i;
        if (!ld.isVowel(m.tiles[i])) continue;
        const near_dls = switch (board_dim) {
            15 => j == 2 or j == 6 or j == 8 or j == 12,
            21 => j == 5 or j == 9 or j == 11 or j == 15,
            else => false,
        };
        if (near_dls) penalty += v_penalty;
    }
    return penalty;
}

/// With the bag empty: going out gains twice the opponent's tiles; not
/// going out costs twice the leave's tiles and 10 more. Port of
/// endgameAdjustment.
pub fn endgameAdjustment(leave_score: i32, leave_size: u32, opp_rack_score: i32) f64 {
    if (leave_size != 0) {
        return @as(f64, @floatFromInt(-leave_score * endgame_non_outplay_leave_penalty_multiplier)) - endgame_non_outplay_constant_penalty;
    }
    return @floatFromInt(2 * opp_rack_score);
}

/// Whether `a` goes before `b` among moves of equal equity. A total order,
/// so the best move doesn't depend on the order moves are found in. Port
/// of move.TiebreaksBetter (MAGPIE's compare_moves).
pub fn tiebreaksBetter(a: *const Move, b: *const Move) bool {
    if (a.score != b.score) return a.score > b.score;
    if (a.kind != b.kind) return @backingInt(a.kind) < @backingInt(b.kind);
    if (a.row != b.row) return a.row < b.row;
    if (a.col != b.col) return a.col < b.col;
    if (a.vertical != b.vertical) return !a.vertical;
    if (a.tiles_played != b.tiles_played) return a.tiles_played < b.tiles_played;
    if (a.len != b.len) return a.len < b.len;
    for (a.tileSlice(), b.tileSlice()) |x, y| {
        if (x != y) return x < y;
    }
    return false;
}

/// Whether a move with equity `eq_a` ranks above one with `eq_b`.
pub fn better(a: *const Move, eq_a: f64, b: *const Move, eq_b: f64) bool {
    if (eq_a != eq_b) return eq_a > eq_b;
    return tiebreaksBetter(a, b);
}

// Tests ----------------------------------------------------------------------
// Ported from equity/calculators_test.go.

const testing = std.testing;
const testdata = @import("../testing/testdata.zig");

fn leaveOf(f: *const testdata.Fixture, text: []const u8) !tiles.Rack {
    var mls: [8]MachineLetter = undefined;
    return .fromLetters(mls[0..try f.ld.toMachineLetters(text, &mls)]);
}

test "leave values" {
    var f = try testdata.Fixture.load("NWL23");
    defer f.deinit();
    const bytes = try testdata.loadFile("lexica/gaddag/NWL23.klv2");
    defer testing.allocator.free(bytes);
    var klv = try kwg_mod.Klv.init(testing.allocator, bytes);
    defer klv.deinit(testing.allocator);
    const ctx: Context = .{ .ld = &f.ld, .klv = &klv, .board_dim = 15, .board_empty = false, .tiles_in_bag = 80, .opp_rack_score = 0 };

    const cases = [_]struct { []const u8, f64 }{
        .{ "?", 27.813936233520508 },      .{ "Q", -4.765872478485107 },
        .{ "?I", 27.964431762695312 },     .{ "I?", 27.964431762695312 },
        .{ "?DLQSV", 5.526261806488037 },  .{ "HMRRSS", -3.482311487197876 },
        .{ "AEINST", 34.272239685058594 }, .{ "SATINE", 34.272239685058594 },
        .{ "", 0 },                        .{ "ABCDEFG", 0 },
    };
    for (cases) |c| {
        const r = try leaveOf(&f, c[0]);
        var buf: [8]MachineLetter = undefined;
        try testing.expectEqual(c[1], ctx.leaveValue(buf[0..r.tiles(&buf)]));
    }
}

/// Every move for `rack_text` on `fen`, with its equity, best first; ties
/// in the order of their notation (as the Go tests sort them).
const Ranked = struct { m: Move, eq: f64, text: [32]u8, text_len: usize };

fn rankAll(f: *const testdata.Fixture, fen: ?[]const u8, rack_text: []const u8, ctx: *Context) ![]Ranked {
    const board_mod = @import("../board/board.zig");
    const cross_set = @import("../board/cross_set.zig");
    const movegen = @import("../movegen/movegen.zig");
    var b = if (fen) |t| try f.boardFromFen(t) else board_mod.Board.init(&board_mod.crossword_game);
    cross_set.generateAll(&b, f.kwg, &f.ld);
    b.updateAllAnchors();
    ctx.board_empty = b.tiles_played == 0;
    var rack = try leaveOf(f, rack_text);
    var gen = movegen.Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    try gen.genAll(&rack, false);
    const out = try testing.allocator.alloc(Ranked, gen.plays.items.len);
    for (gen.plays.items, out) |*m, *r| {
        const lv = m.leave(&rack);
        r.* = .{ .m = m.*, .eq = ctx.equity(m, &lv), .text = undefined, .text_len = 0 };
        var w: std.Io.Writer = .fixed(&r.text);
        try m.format(&f.ld, &w);
        r.text_len = w.end;
    }
    std.mem.sort(Ranked, out, {}, struct {
        fn lt(_: void, a: Ranked, b2: Ranked) bool {
            if (a.eq != b2.eq) return a.eq > b2.eq;
            return std.mem.lessThan(u8, a.text[0..a.text_len], b2.text[0..b2.text_len]);
        }
    }.lt);
    return out;
}

fn expectMove(r: Ranked, text: []const u8) !void {
    try testing.expectEqualStrings(text, r.text[0..r.text_len]);
}

test "endgame timing" {
    var f = try testdata.Fixture.load("NWL18");
    defer f.deinit();
    const opp = try leaveOf(&f, "EO");
    var ctx: Context = .{ .ld = &f.ld, .klv = null, .board_dim = 15, .board_empty = false, .tiles_in_bag = 0, .opp_rack_score = opp.scoreOn(&f.ld) };
    const ranked = try rankAll(&f, testdata.maven_vs_macondo, "AEEORS?", &ctx);
    defer testing.allocator.free(ranked);
    try testing.expectEqual(22.0, ranked[0].eq);
    // Use your blank.
    try expectMove(ranked[0], "L1.S..s");
    try expectMove(ranked[1], "M6.RE.EAmS");
    try expectMove(ranked[2], "M6.RE.EArS");
}

test "pre-endgame timing" {
    var f = try testdata.Fixture.load("NWL23");
    defer f.deinit();
    const klv_bytes = try testdata.loadFile("lexica/gaddag/NWL20.klv2");
    defer testing.allocator.free(klv_bytes);
    var klv = try kwg_mod.Klv.init(testing.allocator, klv_bytes);
    defer klv.deinit(testing.allocator);
    const peg_bytes = try testdata.loadFile("strategy/default/quackle_preendgame.json");
    defer testing.allocator.free(peg_bytes);
    const peg = try std.json.parseFromSlice([]f64, testing.allocator, peg_bytes, .{});
    defer peg.deinit();
    // Five tiles in the bag: playing all seven leaves 5 - 7 + 7 = 5, a
    // penalty of 3.5.
    var ctx: Context = .{ .ld = &f.ld, .klv = &klv, .peg = peg.value, .board_dim = 15, .board_empty = false, .tiles_in_bag = 5, .opp_rack_score = 0 };
    const ranked = try rankAll(&f, testdata.vs_oxy, "OXPBAZE", &ctx);
    defer testing.allocator.free(ranked);
    try testing.expectEqual(1780 - 3.5, ranked[0].eq);
}

test "opening placement" {
    var f = try testdata.Fixture.load("NWL23");
    defer f.deinit();
    const klv_bytes = try testdata.loadFile("lexica/gaddag/NWL20.klv2");
    defer testing.allocator.free(klv_bytes);
    var klv = try kwg_mod.Klv.init(testing.allocator, klv_bytes);
    defer klv.deinit(testing.allocator);
    var ctx: Context = .{ .ld = &f.ld, .klv = &klv, .board_dim = 15, .board_empty = true, .tiles_in_bag = 93, .opp_rack_score = 0 };
    const ranked = try rankAll(&f, null, "AEFLR", &ctx);
    defer testing.allocator.free(ranked);
    // FLARE and FARLE score 24 with no vowel next to a 2LS; FERAL scores
    // 24 too but puts a vowel there.
    try testing.expectEqual(24.0, ranked[0].eq);
    try testing.expectEqual(24.0, ranked[1].eq);
    try expectMove(ranked[2], "8D.FERAL");
    try testing.expectEqual(23.3, ranked[2].eq);
}

test "endgame adjustment" {
    // Going out with the opponent holding 9 points of tiles.
    try testing.expectEqual(18.0, endgameAdjustment(0, 0, 9));
    // Keeping a Q (10) and an A (1).
    try testing.expectEqual(-32.0, endgameAdjustment(11, 2, 9));
}
