//! Move generation, a port of Macondo's `movegen` package (Gordon's GADDAG
//! algorithm, "A Faster Scrabble Move Generation Algorithm").
//!
//! Moves are generated a row at a time; vertical moves are generated on the
//! transposed board. Anchors are the rightmost square of where a word can
//! be hooked, as in the Go code.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const Kwg = @import("kwg.zig").Kwg;
const kwg_mod = @import("kwg.zig");
const move_mod = @import("move.zig");
const equity = @import("equity.zig");
const leavemap = @import("leavemap.zig");
const shadow_mod = @import("shadow.zig");
const Board = board_mod.Board;
const Direction = board_mod.Direction;
const Move = move_mod.Move;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;

pub const rack_tile_limit = 7;
pub const bingo_bonus = 50;
pub const default_exchange_limit = 7;

/// How many tiles an exchange may throw in with `in_bag` tiles in the bag
/// after drawing, or 0 if exchanging isn't allowed. Port of
/// game.MaxCanExchange.
pub fn maxCanExchange(in_bag: i32, exchange_limit: i32) u8 {
    if (in_bag < exchange_limit) return 0;
    return @intCast(@min(in_bag, rack_tile_limit));
}

/// The exchange limit for a lexicon: Spanish games allow exchanges down to
/// one tile in the bag. Port of lexicon.IsSpanish's use in game rules.
pub fn exchangeLimit(lexicon: []const u8) i32 {
    var buf: [4]u8 = undefined;
    if (lexicon.len >= 4) {
        const p = std.ascii.lowerString(&buf, lexicon[0..4]);
        if (std.mem.eql(u8, p, "fise") or std.mem.eql(u8, p, "file")) return 1;
    }
    return default_exchange_limit;
}

/// The squares of the row being generated, read once from the board.
pub const CachedSquare = struct {
    cross_set: board_mod.CrossSet,
    /// The left extension set, for the shadow pass.
    left_ext: board_mod.CrossSet,
    cross_score: i32,
    letter: MachineLetter,
    letter_mul: u8,
    word_mul: u8,
};

pub const Generator = struct {
    board: *Board,
    kwg: Kwg,
    ld: *const tiles.LetterDistribution,
    plays: std.ArrayList(Move) = .empty,
    gpa: Allocator,

    max_can_exchange: u8 = default_exchange_limit,
    max_tile_usage: u32 = 100,
    gen_pass: bool = false,

    // Generation state.
    cur_row: usize = 0,
    cur_anchor_col: usize = 0,
    last_anchor_col: usize = 0,
    vertical: bool = false,
    tiles_played: u32 = 0,
    rack_bits: u64 = 0,
    strip: [board_mod.max_dim]MachineLetter = undefined,
    exchange_strip: [rack_tile_limit]MachineLetter = undefined,
    row: [board_mod.max_dim]CachedSquare = undefined,
    out_of_memory: bool = false,

    // The best-play search (genBest).
    best_only: bool = false,
    ctx: *const equity.Context = undefined,
    leave_map: leavemap.LeaveMap = .{},
    best: Move = undefined,
    best_eq: f64 = 0,
    have_best: bool = false,
    shadow: shadow_mod.State = .{},

    pub fn init(gpa: Allocator, b: *Board, kwg: Kwg, ld: *const tiles.LetterDistribution) Generator {
        return .{ .gpa = gpa, .board = b, .kwg = kwg, .ld = ld };
    }

    pub fn deinit(self: *Generator) void {
        self.plays.deinit(self.gpa);
        self.shadow.deinit(self.gpa);
        self.* = undefined;
    }

    /// Generates every move for `rack` into `plays` (cleared first): tile
    /// plays, then a pass if there are none (or gen_pass is set), then the
    /// exchanges if `add_exchange`. The board's cross-sets and anchors must
    /// be up to date.
    pub fn genAll(self: *Generator, rack: *Rack, add_exchange: bool) Allocator.Error!void {
        self.plays.clearRetainingCapacity();
        self.out_of_memory = false;
        self.best_only = false;
        self.leave_map.initialized = false;

        self.vertical = false;
        self.genByOrientation(rack, .horizontal);
        self.board.transpose();
        self.vertical = true;
        self.genByOrientation(rack, .vertical);
        self.board.transpose();

        if (self.plays.items.len == 0 or self.gen_pass) self.record(Move.pass());
        if (add_exchange) self.genExchanges(rack, 0, 0);
        if (self.out_of_memory) return error.OutOfMemory;
    }

    pub const Best = struct { move: Move, equity: f64 };

    /// The best move for `rack` by static equity (see equity.Context), as
    /// Go's GenBestStaticTurn finds it: exchanges first (if
    /// `add_exchange`), then the tile plays of each anchor, best shadow
    /// bound first, until no anchor left can beat the best move; a pass
    /// only if nothing else was found (or gen_pass is set). Ties go to
    /// equity.tiebreaksBetter. The rack must hold at most 7 tiles.
    pub fn genBest(self: *Generator, rack: *Rack, add_exchange: bool, ctx: *const equity.Context) Allocator.Error!Best {
        std.debug.assert(rack.num_tiles <= rack_tile_limit);
        self.out_of_memory = false;
        self.best_only = true;
        defer self.best_only = false;
        self.ctx = ctx;
        self.have_best = false;
        self.shadow.tiles_in_bag = ctx.tiles_in_bag;
        self.shadow.opp_rack_score = ctx.opp_rack_score;

        self.leave_map.init(rack);
        if (self.leave_map.initialized) {
            leavemap.populate(&self.leave_map, rack, ctx.klv, ctx, &self.shadow.best_leaves);
        } else {
            shadow_mod.bestLeavesWithoutMap(&self.shadow);
        }

        // The best exchange is a floor the anchors are measured against.
        self.shadow.anchor_floor = -std.math.inf(f64);
        if (add_exchange) {
            self.genExchanges(rack, 0, 0);
            if (self.have_best) self.shadow.anchor_floor = self.best_eq;
        }
        shadow_mod.genShadow(self, rack);
        shadow_mod.recordFromAnchors(self, rack);
        if (!self.have_best or self.gen_pass) self.recordBest(Move.pass(), rack, null);
        if (self.out_of_memory) return error.OutOfMemory;
        return .{ .move = self.best, .equity = self.best_eq };
    }

    /// Keeps `m` if it beats the best move so far. `rack` holds the leave;
    /// `fast_eq` is the equity from the leave map, if it was usable.
    fn recordBest(self: *Generator, m: Move, rack: *const Rack, fast_eq: ?f64) void {
        const eq = fast_eq orelse self.ctx.equity(&m, rack);
        if (!self.have_best or equity.better(&m, eq, &self.best, self.best_eq)) {
            self.best = m;
            self.best_eq = eq;
            self.have_best = true;
        }
    }

    /// The equity of the play or exchange being recorded from the leave
    /// map, or null where Go's recorder works it out in full (no leave
    /// map, or the opening).
    inline fn fastEquity(self: *const Generator, score: i32) ?f64 {
        if (!self.leave_map.initialized or self.ctx.board_empty) return null;
        return @as(f64, @floatFromInt(score)) + self.leave_map.currentValue();
    }

    pub fn genForAnchor(self: *Generator, rack: *Rack) void {
        const col = self.cur_anchor_col;
        self.recursiveGen(col, rack, self.kwg.gaddagRoot(), col, col, !self.vertical, 0, 0, 1);
    }

    fn record(self: *Generator, m: Move) void {
        self.plays.append(self.gpa, m) catch {
            self.out_of_memory = true;
        };
    }

    pub fn setRackBits(self: *Generator, rack: *const Rack) void {
        var bits: u64 = 0;
        for (rack.counts[0..self.ld.size], 0..) |n, ml| {
            if (n > 0) bits |= @as(u64, 1) << @intCast(ml);
        }
        self.rack_bits = bits;
    }

    pub fn loadRow(self: *Generator, row: usize, cs_dir: Direction) void {
        const b = self.board;
        for (0..b.dim) |col| {
            const sq = b.sqIdx(row, col);
            const bonus = b.bonuses[sq];
            self.row[col] = .{
                .cross_set = b.cross_sets[@backingInt(cs_dir)][sq],
                .left_ext = b.left_ext[@backingInt(cs_dir)][sq],
                .cross_score = b.cross_scores[@backingInt(cs_dir)][sq],
                .letter = b.squares[sq],
                .letter_mul = board_mod.letterMultiplier(bonus),
                .word_mul = board_mod.wordMultiplier(bonus),
            };
        }
    }

    fn genByOrientation(self: *Generator, rack: *Rack, dir: Direction) void {
        self.setRackBits(rack);
        // Plays along `dir` must fit the cross-sets of the other direction.
        const cs_dir = dir.other();
        for (0..self.board.dim) |row| {
            self.cur_row = row;
            self.loadRow(row, cs_dir);
            self.last_anchor_col = 100;
            for (0..self.board.dim) |col| {
                if (self.board.isAnchor(row, col, dir)) {
                    self.cur_anchor_col = col;
                    self.genForAnchor(rack);
                    self.last_anchor_col = col;
                }
            }
        }
    }

    /// Gordon's Gen.
    fn recursiveGen(self: *Generator, col: usize, rack: *Rack, node: u32, leftstrip: usize, rightstrip: usize, unique_play: bool, base_score: i32, cross_scores: i32, word_mul: i32) void {
        const sq = self.row[col];
        const nodes = self.kwg.nodes;
        if (sq.letter != 0) {
            const raw: u32 = tiles.unblank(sq.letter);
            var next: u32 = 0;
            var accepts = false;
            var i = node;
            while (true) : (i += 1) {
                const n = nodes[i];
                if (n >> kwg_mod.tile_shift == raw) {
                    next = n & kwg_mod.arc_mask;
                    accepts = n & kwg_mod.accepts_bit != 0;
                    break;
                }
                if (n & kwg_mod.is_end_bit != 0) break;
            }
            self.goOn(col, sq.letter, rack, next, accepts, leftstrip, rightstrip, unique_play, base_score + self.ld.score(sq.letter), cross_scores, word_mul);
            return;
        }
        if (rack.num_tiles == 0) return;

        const lm: i32 = sq.letter_mul;
        const cs = sq.cross_score;
        const wm: i32 = sq.word_mul;
        const empty_adjacent = sq.cross_set == board_mod.trivial_cross_set;
        const n_blank = rack.counts[0];
        // Letters the cross-set allows and, without a blank, the rack holds.
        var allowed: u64 = sq.cross_set & ~@as(u64, 1);
        if (n_blank == 0) allowed &= self.rack_bits;
        if (allowed == 0) return;

        var i = node;
        while (true) : (i += 1) {
            const n = nodes[i];
            const ml: MachineLetter = @intCast(n >> kwg_mod.tile_shift);
            if (allowed & (@as(u64, 1) << @intCast(ml)) != 0) {
                const n_ml = rack.counts[ml];
                const arc = n & kwg_mod.arc_mask;
                const accepts = n & kwg_mod.accepts_bit != 0;
                if (n_ml > 0) {
                    rack.take(ml);
                    if (n_ml == 1) self.rack_bits &= ~(@as(u64, 1) << @intCast(ml));
                    if (self.leave_map.initialized) self.leave_map.take(ml, n_ml - 1);
                    self.tiles_played += 1;
                    const sml = self.ld.score(ml);
                    var addl_cross: i32 = 0;
                    if (!empty_adjacent) {
                        addl_cross = if (wm > 1) wm * (cs + sml) else cs + sml * lm;
                    }
                    self.goOn(col, ml, rack, arc, accepts, leftstrip, rightstrip, unique_play, base_score + sml * lm, cross_scores + addl_cross, word_mul * wm);
                    self.tiles_played -= 1;
                    if (self.leave_map.initialized) self.leave_map.add(ml, n_ml - 1);
                    rack.add(ml);
                    self.rack_bits |= @as(u64, 1) << @intCast(ml);
                }
                if (n_blank > 0) {
                    rack.take(0);
                    if (self.leave_map.initialized) self.leave_map.take(0, n_blank - 1);
                    self.tiles_played += 1;
                    self.goOn(col, tiles.blank(ml), rack, arc, accepts, leftstrip, rightstrip, unique_play, base_score, cross_scores + cs * wm, word_mul * wm);
                    self.tiles_played -= 1;
                    if (self.leave_map.initialized) self.leave_map.add(0, n_blank - 1);
                    rack.add(0);
                }
            }
            if (n & kwg_mod.is_end_bit != 0) break;
        }
    }

    /// Gordon's GoOn.
    fn goOn(self: *Generator, cur_col: usize, letter: MachineLetter, rack: *Rack, new_node: u32, accepts: bool, leftstrip_in: usize, rightstrip_in: usize, unique_play_in: bool, base_score: i32, cross_scores: i32, word_mul: i32) void {
        var leftstrip = leftstrip_in;
        var rightstrip = rightstrip_in;
        var unique_play = unique_play_in;
        const bingo: i32 = if (self.tiles_played == rack_tile_limit) bingo_bonus else 0;
        const sq = self.row[cur_col];
        const dim = self.board.dim;

        if (sq.letter != 0) {
            self.strip[cur_col] = 0;
        } else {
            self.strip[cur_col] = letter;
            // A one-tile play is found both ways; keep it only once.
            if (self.vertical and sq.cross_set == board_mod.trivial_cross_set) unique_play = true;
        }

        if (cur_col <= self.cur_anchor_col) {
            leftstrip = cur_col;
            const no_letter_left = cur_col == 0 or self.row[cur_col - 1].letter == 0;
            if (accepts and no_letter_left and self.tiles_played > 0 and
                (unique_play or self.tiles_played > 1) and self.tiles_played <= self.max_tile_usage)
            {
                self.recordPlay(rack, leftstrip, rightstrip, base_score * word_mul + cross_scores + bingo);
            }
            if (new_node == 0) return;
            if (cur_col > 0 and cur_col - 1 != self.last_anchor_col) {
                self.recursiveGen(cur_col - 1, rack, new_node, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
            const sep = self.kwg.nextNodeIdx(new_node, 0);
            if (sep != 0 and no_letter_left and self.cur_anchor_col < dim - 1) {
                self.recursiveGen(self.cur_anchor_col + 1, rack, sep, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
        } else {
            rightstrip = cur_col;
            const no_letter_right = cur_col == dim - 1 or self.row[cur_col + 1].letter == 0;
            if (accepts and no_letter_right and self.tiles_played > 0 and
                (unique_play or self.tiles_played > 1) and self.tiles_played <= self.max_tile_usage)
            {
                self.recordPlay(rack, leftstrip, rightstrip, base_score * word_mul + cross_scores + bingo);
            }
            if (new_node != 0 and cur_col < dim - 1) {
                self.recursiveGen(cur_col + 1, rack, new_node, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
        }
    }

    fn recordPlay(self: *Generator, rack: *const Rack, leftstrip: usize, rightstrip: usize, score: i32) void {
        const len = rightstrip - leftstrip + 1;
        if (len < 2) return;
        var fast_eq: ?f64 = null;
        if (self.best_only) {
            fast_eq = self.fastEquity(score);
            // Turned away before the move is built.
            if (fast_eq) |eq| if (self.have_best and eq < self.best_eq) return;
        }
        var m: Move = .{
            .kind = .play,
            .vertical = self.vertical,
            .row = @intCast(if (self.vertical) leftstrip else self.cur_row),
            .col = @intCast(if (self.vertical) self.cur_row else leftstrip),
            .len = @intCast(len),
            .tiles_played = @intCast(self.tiles_played),
            .score = score,
        };
        @memcpy(m.tiles[0..len], self.strip[leftstrip .. rightstrip + 1]);
        if (self.best_only) return self.recordBest(m, rack, fast_eq);
        self.record(m);
    }

    /// Every distinct exchange (by tile counts). Port of andy-k's
    /// generateExchangeMoves.
    fn genExchanges(self: *Generator, rack: *Rack, ml_in: usize, strip_idx_in: usize) void {
        var ml = ml_in;
        var strip_idx = strip_idx_in;
        const size = self.ld.size;
        while (ml < size and rack.counts[ml] == 0) ml += 1;
        if (ml == size) {
            if (strip_idx == 0 or strip_idx > self.max_can_exchange) return;
            var fast_eq: ?f64 = null;
            if (self.best_only and self.leave_map.initialized and !self.ctx.board_empty) {
                // An exchange scores nothing: its equity is the leave's.
                fast_eq = self.leave_map.currentValue();
                if (self.have_best and fast_eq.? < self.best_eq) return;
            }
            var m: Move = .{ .kind = .exchange, .len = @intCast(strip_idx), .tiles_played = @intCast(strip_idx) };
            @memcpy(m.tiles[0..strip_idx], self.exchange_strip[0..strip_idx]);
            if (self.best_only) return self.recordBest(m, rack, fast_eq);
            self.record(m);
            return;
        }
        self.genExchanges(rack, ml + 1, strip_idx);
        const num = rack.counts[ml];
        var taken: u8 = 0;
        while (taken < num) : (taken += 1) {
            // More tiles than a rack holds: no exchange of all of them.
            if (strip_idx >= self.exchange_strip.len) break;
            self.exchange_strip[strip_idx] = @intCast(ml);
            strip_idx += 1;
            rack.take(@intCast(ml));
            if (self.leave_map.initialized) self.leave_map.take(@intCast(ml), rack.counts[ml]);
            self.genExchanges(rack, ml + 1, strip_idx);
        }
        for (0..taken) |_| {
            if (self.leave_map.initialized) self.leave_map.add(@intCast(ml), rack.counts[ml]);
            rack.add(@intCast(ml));
        }
    }
};

/// Whether exchanges are allowed and how many tiles may go, for the player
/// to move with `rack` when `unseen` tiles are in the bag or the
/// opponent's rack.
pub fn exchangeRules(unseen: i32, lexicon: []const u8) struct { allowed: bool, max: u8 } {
    const limit = exchangeLimit(lexicon);
    return .{
        .allowed = unseen - rack_tile_limit >= limit,
        .max = maxCanExchange(unseen - rack_tile_limit, limit),
    };
}

// Tests ----------------------------------------------------------------------
// Ported from movegen/movegen_test.go. America isn't published, so its tests
// skip without it.

const testing = std.testing;
const testdata = @import("testdata.zig");
const cross_set = @import("cross_set.zig");

const Counts = struct { scoring: usize, other: usize };

/// Generates for `rack_text` on `fen` (or an empty board) and returns the
/// generator, whose plays are sorted by score, highest first.
fn genFor(f: *const testdata.Fixture, b: *Board, fen: ?[]const u8, rack_text: []const u8, add_exchange: bool) !Generator {
    if (fen) |text| b.* = try f.boardFromFen(text);
    cross_set.generateAll(b, f.kwg, &f.ld);
    b.updateAllAnchors();
    var mls: [8]MachineLetter = undefined;
    const n = try f.ld.toMachineLetters(rack_text, &mls);
    var rack = Rack.fromLetters(mls[0..n]);
    var gen = Generator.init(testing.allocator, b, f.kwg, &f.ld);
    errdefer gen.deinit();
    try gen.genAll(&rack, add_exchange);
    std.mem.sort(Move, gen.plays.items, {}, struct {
        fn higher(_: void, x: Move, y: Move) bool {
            return x.score > y.score;
        }
    }.higher);
    return gen;
}

fn counts(gen: *const Generator) Counts {
    var c: Counts = .{ .scoring = 0, .other = 0 };
    for (gen.plays.items) |m| {
        if (m.kind == .play) c.scoring += 1 else c.other += 1;
    }
    return c;
}

test "single tile" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    var gen = try genFor(&f, &b, testdata.vs_matt, "A", false);
    defer gen.deinit();
    try testing.expectEqual(24, counts(&gen).scoring);
}

test "full rack" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    var gen = try genFor(&f, &b, testdata.vs_matt, "AABDELT", true);
    defer gen.deinit();
    try testing.expectEqual(Counts{ .scoring = 667, .other = 95 }, counts(&gen));
    for ([_]i32{ 38, 36, 36, 34, 34, 33, 30, 30, 30, 28 }, gen.plays.items[0..10]) |score, m| {
        try testing.expectEqual(score, m.score);
    }
}

test "full rack again" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    var gen = try genFor(&f, &b, testdata.vs_ed, "AFGIIIS", true);
    defer gen.deinit();
    try testing.expectEqual(Counts{ .scoring = 219, .other = 63 }, counts(&gen));
}

test "no plays: a pass" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    var gen = try genFor(&f, &b, testdata.vs_jeremy, "V", false);
    defer gen.deinit();
    try testing.expectEqual(1, gen.plays.items.len);
    try testing.expectEqual(.pass, gen.plays.items[0].kind);
}

test "exchanges" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    var mls: [8]MachineLetter = undefined;
    const n = try f.ld.toMachineLetters("ABCDEF?", &mls);
    var rack = Rack.fromLetters(mls[0..n]);
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    gen.genExchanges(&rack, 0, 0);
    try testing.expectEqual(127, gen.plays.items.len);
}

test "blanks (America)" {
    var f = try testdata.Fixture.load("America");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    {
        var gen = try genFor(&f, &b, testdata.vs_ed, "?", true);
        defer gen.deinit();
        try testing.expectEqual(Counts{ .scoring = 166, .other = 1 }, counts(&gen));
    }
    {
        var gen = try genFor(&f, &b, testdata.vs_ed, "??", true);
        defer gen.deinit();
        try testing.expectEqual(Counts{ .scoring = 1958, .other = 2 }, counts(&gen));
    }
    {
        var gen = try genFor(&f, &b, testdata.vs_jeremy, "DDESW??", false);
        defer gen.deinit();
        try testing.expectEqual(Counts{ .scoring = 8297, .other = 0 }, counts(&gen));
        try testing.expectEqual(106, gen.plays.items[0].score); // hEaDW(OR)DS
        try testing.expectEqual(7, gen.plays.items[0].tiles_played);
        for (gen.plays.items[2..9]) |m| try testing.expectEqual(32, m.score);
    }
}

test "giant 27-timer and empty board (America)" {
    var f = try testdata.Fixture.load("America");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    {
        var gen = try genFor(&f, &b, testdata.vs_oxy, "ABEOPXZ", false);
        defer gen.deinit();
        try testing.expectEqual(Counts{ .scoring = 519, .other = 0 }, counts(&gen));
        try testing.expectEqual(1780, gen.plays.items[0].score); // OXYPHENBUTAZONE
    }
    {
        b = Board.init(&board_mod.crossword_game);
        var gen = try genFor(&f, &b, null, "DEGORV?", true);
        defer gen.deinit();
        try testing.expectEqual(Counts{ .scoring = 3313, .other = 127 }, counts(&gen));
        try testing.expectEqual(80, gen.plays.items[0].score);
    }
}

/// The best move of the full move list, valued as genBest values moves
/// (from the leave map's values, except on the empty board), for checking
/// genBest's shortcuts.
fn bruteBest(gen: *Generator, rack: *Rack, add_exchange: bool, ctx: *const equity.Context) !Generator.Best {
    try gen.genAll(rack, add_exchange);
    var has_exchange = false;
    for (gen.plays.items) |m| has_exchange = has_exchange or m.kind == .exchange;
    var best: ?Generator.Best = null;
    for (gen.plays.items) |*m| {
        // genBest passes only if it found nothing else.
        if (m.kind == .pass and has_exchange) continue;
        const lv = m.leave(rack);
        var buf: [rack_tile_limit]MachineLetter = undefined;
        const raw = ctx.leaveValue(buf[0..lv.tiles(&buf)]);
        const lm = ctx.leaveMapValue(raw, lv.scoreOn(ctx.ld), lv.num_tiles, rack.num_tiles - lv.num_tiles);
        const eq = switch (m.kind) {
            .pass => ctx.equity(m, &lv),
            .play => if (ctx.board_empty) ctx.equity(m, &lv) else @as(f64, @floatFromInt(m.score)) + lm,
            .exchange => if (ctx.board_empty) ctx.equity(m, &lv) else lm,
        };
        if (best == null or equity.better(m, eq, &best.?.move, best.?.equity)) best = .{ .move = m.*, .equity = eq };
    }
    return best.?;
}

fn sameMove(a: *const Move, b: *const Move) bool {
    return a.kind == b.kind and a.vertical == b.vertical and a.row == b.row and a.col == b.col and
        a.score == b.score and std.mem.eql(MachineLetter, a.tileSlice(), b.tileSlice());
}

test "best play agrees with the full move list" {
    const position_mod = @import("position.zig");
    var store = testdata.store();
    defer store.deinit();
    const peg = [_]f64{ 0, -8, 0, -0.5, -2, -3.5, -2, 2, 10, 7, 4, -1, -2 };
    const cgps = [_][]const u8{
        "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;",
        "15/15/15/15/15/15/15/6KA7/15/15/15/15/15/15/15 / 0/0 0 lex CSW21;",
        "15/15/15/15/15/15/1PROTEAN7/3WINDY7/15/15/15/15/15/15/15 / 0/0 0 lex CSW21;",
        "15/15/15/15/15/15/15/6QI7/6I8/6S8/15/15/15/15/15 / 0/0 0 lex CSW21;",
        "14Z/5V7NA/5R4P2AN/5O1N1ARGUFY/5TOO2I2F1/4T1COWKS2E1/2REE1UN2A2R1/1RAVIoLI2G3Q/1EXON1IN2E1P1A/1C1L3GEM2AHI/BEMUD2SHUTTlED/1D1E8AI1/YET9IS1/ODA9ST1/W1JOLLER7 / 0/0 0 lex CSW21;",
        "5E1p7/1A1C1G1E5S1/1JIZ1G1R3V1P1/HO1APE1A3U1I1/OW1RAD1EXFIL1T1/MA2W2OI2G1T1/IN2K2N2MOBE1/E2FYRDS2o1ANS/3O6V1N1H/3U6ADD1E/3S6BOO1E/3T6LOR1L/INCITANT1AYRE2/3E5U5/3REQUITE5 / 0/0 0 lex CSW21;",
        testdata.vs_ed ++ " / 0/0 0 lex NWL20;",
        testdata.vs_matt ++ " / 0/0 0 lex NWL20;",
        testdata.vs_jeremy ++ " / 0/0 0 lex NWL20;",
        testdata.vs_oxy ++ " / 0/0 0 lex NWL20;",
        testdata.maven_vs_macondo ++ " / 0/0 0 lex NWL20;",
    };
    var prng: std.Random.DefaultPrng = .init(42);
    const random = prng.random();
    var checked: usize = 0;
    for (cgps) |text| {
        var p: position_mod.Position = try testdata.position(&store, text);
        var gen = Generator.init(testing.allocator, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();

        // The tiles not on the board.
        var pool: [200]MachineLetter = undefined;
        var n_pool: usize = 0;
        var on_board: [tiles.max_alphabet_size]u32 = @splat(0);
        for (p.board.squares[0 .. p.board.dim * p.board.dim]) |ml| {
            if (ml != 0) on_board[tiles.intrinsicIndex(ml)] += 1;
        }
        for (0..p.dist.size) |ml| {
            for (on_board[ml]..p.dist.counts[ml]) |_| {
                pool[n_pool] = @intCast(ml);
                n_pool += 1;
            }
        }

        for (0..60) |i| {
            random.shuffle(MachineLetter, pool[0..n_pool]);
            const rack_size = @min(n_pool, if (i % 5 == 4) 1 + random.uintLessThan(usize, 6) else 7);
            var rack: Rack = .fromLetters(pool[0..rack_size]);
            const opp: Rack = .fromLetters(pool[rack_size..@min(n_pool, rack_size + 7)]);
            // Mid-game, pre-endgame (with adjustments) and endgame.
            const bag: i32 = switch (i % 4) {
                0 => 60,
                1 => 1 + @as(i32, @intCast(random.uintLessThan(u32, 8))),
                2 => 3,
                else => 0,
            };
            const ctx: equity.Context = .{
                .ld = p.dist,
                .klv = p.leaves,
                .peg = if (i % 4 == 2) &peg else &.{},
                .board_dim = p.board.dim,
                .board_empty = p.board.tiles_played == 0,
                .tiles_in_bag = bag,
                .opp_rack_score = opp.scoreOn(p.dist),
            };
            const ex = exchangeRules(bag + opp.num_tiles, "NWL20");
            gen.max_can_exchange = ex.max;
            const got = try gen.genBest(&rack, ex.allowed, &ctx);
            const want = try bruteBest(&gen, &rack, ex.allowed, &ctx);
            if (!sameMove(&got.move, &want.move) or got.equity != want.equity) {
                var buf: [64]u8 = undefined;
                var w: std.Io.Writer = .fixed(&buf);
                try got.move.format(p.dist, &w);
                try w.print(" {d} vs ", .{got.equity});
                try want.move.format(p.dist, &w);
                try w.print(" {d}", .{want.equity});
                std.debug.print("{s} rack {d}: best {s}\n", .{ text, i, w.buffered() });
                return error.TestUnexpectedResult;
            }
            checked += 1;
        }
    }
    try testing.expectEqual(cgps.len * 60, checked);
}

test "shadow positions that once disagreed" {
    // From movegen/shadow_disagree_test.go, which replays seeded NWL23
    // games to these turns: the boards and racks it reaches. Scores only,
    // as there.
    var store = testdata.store();
    defer store.deinit();
    const cases = [_]struct { []const u8, []const u8, i32 }{
        .{ "6AAL2CERE/5WINO4ET/1V2QI6ODA/1U1AIT1PINTADAS/1G1D5OI1ER1/1HER1C7G1/2XI1OHM5U1/3FUZEES1JOYED/3T1Y2N1E4/8IFS4/8BAT4/9R5/2KOEL2WRUNG2/1bIPLANE1O5/VITA1BERMS5", "?LN", 13 },
        .{ "15/10N4/2ARROZ3O4/3U6V4/3SHA4A4/3KEX4T4/4JETLINER3/3PILEI2DEMOI/4R10/4AW9/4sO9/5W9/5I9/5N2OVERDO1/3BIGOTS2EASY", "?AIMNUU", 30 },
        .{ "5COZeS1MEMO/5R2RAPINI1/2uNTANNED5/5V9/4BE3U5/4AD3G5/4S4L5/OUTYIELD1I5/4F3HE5/1BUOYANCES5/1I5OW6/1G5TE6/1G5ID3PER/1E5J3FEW1/TREK2XANTHATES", "AAAIIQU", 26 },
    };
    for (cases) |c| {
        var buf: [256]u8 = undefined;
        var p = try testdata.position(&store, try std.fmt.bufPrint(&buf, "{s} {s}/ 0/0 0 lex NWL23;", .{ c[0], c[1] }));
        var gen = Generator.init(testing.allocator, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        const ctx: equity.Context = .{ .ld = p.dist, .klv = null, .board_dim = p.board.dim, .board_empty = false, .tiles_in_bag = 50, .opp_rack_score = 0 };
        var rack = p.racks[0];
        const best = try gen.genBest(&rack, false, &ctx);
        try testing.expectEqual(c[2], best.move.score);
        try testing.expectEqual(@as(f64, @floatFromInt(c[2])), best.equity);
        const brute = try bruteBest(&gen, &rack, false, &ctx);
        try testing.expect(sameMove(&best.move, &brute.move));
    }
}

test "top play only (America)" {
    // From TestTopPlayOnlyRecorder: hEaDW(OR)DS, with the leave and endgame
    // calculators.
    var store = testdata.store();
    defer store.deinit();
    var p = try testdata.position(&store, "7N6M/5ZOON4AA/7B5UN/2S4L3LADY/2T4E2QI1I1/2A2PORN3NOR/2BICE2AA1DA1E/6GUVS1OP1F/8ET1LA1U/5J3R1E1UT/4VOTE1I1R1NE/5G1MICKIES1/6FE1T1THEW/6OR3E1XI/6OY6G DDESW??/AHIILR 299/352 0 lex America;");
    var gen = Generator.init(testing.allocator, &p.board, p.lexicon.kwg, p.dist);
    defer gen.deinit();
    const ctx = p.equityContext();
    var rack = p.racks[0];
    const best = try gen.genBest(&rack, false, &ctx);
    try testing.expectEqual(106, best.move.score);
    try testing.expectEqual(.play, best.move.kind);
    try testing.expectEqual(7, best.move.tiles_played);
    var out: [32]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out);
    try best.move.format(p.dist, &w);
    try testing.expectEqualStrings("hEaDW..DS", w.buffered()[w.buffered().len - 9 ..]);
}

/// Gordon's Gen from one anchor of one row, as the Go row tests call
/// recursiveGen directly: horizontal unless the board is transposed and
/// `vertical` set. Appends to gen.plays.
fn rowGen(gen: *Generator, rack: *Rack, row: usize, anchor_col: usize, vertical: bool, unique: bool) void {
    gen.vertical = vertical;
    gen.cur_row = row;
    gen.cur_anchor_col = anchor_col;
    gen.setRackBits(rack);
    gen.loadRow(row, if (vertical) .horizontal else .vertical);
    gen.recursiveGen(anchor_col, rack, gen.kwg.gaddagRoot(), anchor_col, anchor_col, unique, 0, 0, 1);
}

fn rackFrom(f: *const testdata.Fixture, text: []const u8) !Rack {
    var mls: [8]MachineLetter = undefined;
    return .fromLetters(mls[0..try f.ld.toMachineLetters(text, &mls)]);
}

fn formatted(f: *const testdata.Fixture, m: *const Move, buf: []u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try m.format(&f.ld, &w);
    return w.buffered();
}

test "gen base: no cross-sets, no plays" {
    var f = try testdata.Fixture.load("NWL18");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    for (&b.cross_sets) |*cs| @memset(cs, 0);
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "AEINRST");
    rowGen(&gen, &rack, 4, 8, false, true);
    try testing.expectEqual(0, gen.plays.items.len);
}

test "simple row gen" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    const cases = [_]struct { []const u8, usize, usize, []const u8, usize }{
        .{ "P", 11, 2, "     REGNANT", 1 },
        .{ "O", 9, 2, "  PORTOLAN", 1 },
        .{ "S", 9, 2, "  PORTOLAN", 1 },
        .{ "?", 9, 2, "  PORTOLAN", 2 },
        .{ "TY", 7, 2, "  SOVRAN", 1 },
        .{ "ING", 6, 2, "  LAUGH", 1 },
        .{ "ZA", 3, 4, "  BE", 0 },
        .{ "AENPPSW", 14, 4, "        CHAWING", 1 }, // wappenschawing
        .{ "ABEHINT", 9, 4, "   THERMOS  A", 2 }, // nethermost, hithermost
        .{ "ABEHITT", 8, 4, "  THERMOS A   ", 1 }, // anchor on s: thermostat
        .{ "TT", 10, 4, "  THERMOS A   ", 3 }, // anchor on a: thermostat, at, att
        .{ "A", 1, 4, " B", 1 },
        .{ "A", 1, 4, " b", 1 },
    };
    var b = Board.init(&board_mod.crossword_game);
    for (cases) |c| {
        try f.setRowText(&b, c[2], c[3]);
        cross_set.generateAll(&b, f.kwg, &f.ld);
        var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
        defer gen.deinit();
        var rack = try rackFrom(&f, c[0]);
        rowGen(&gen, &rack, c[2], c[1], false, true);
        testing.expectEqual(c[4], gen.plays.items.len) catch |err| {
            std.debug.print("rack {s} on {s}\n", .{ c[0], c[3] });
            return err;
        };
    }
}

test "gen through, both ways, with allowed letters" {
    // HITHERMOST but not NETHERMOST.
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    try f.setRowText(&b, 4, "   THERMOS  A");
    cross_set.generateAll(&b, f.kwg, &f.ld);
    b.setCrossSet(4, 2, @as(u64, 1) << 9, .vertical); // only I
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "ABEHINT");
    rowGen(&gen, &rack, 4, 9, false, true);
    try testing.expectEqual(1, gen.plays.items.len);
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("5B.HI.......T", try formatted(&f, &gen.plays.items[0], &buf));
    const lv = gen.plays.items[0].leave(&rack);
    try testing.expectEqual(try rackFrom(&f, "ABEN"), lv);
}

test "row gen" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = try f.boardFromFen(testdata.vs_ed);
    cross_set.generateAll(&b, f.kwg, &f.ld);
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "AAEIRST");
    rowGen(&gen, &rack, 4, 8, false, true);
    // AIRGLOWS and REGLOWS only.
    try testing.expectEqual(2, gen.plays.items.len);
    const plays = gen.plays.items;
    const hi: usize = if (plays[0].score > plays[1].score) 0 else 1;
    try testing.expectEqual(try rackFrom(&f, "AEST"), plays[hi].leave(&rack));
    try testing.expectEqual(try rackFrom(&f, "AAIST"), plays[1 - hi].leave(&rack));
}

test "other row gen: played-through markers" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = try f.boardFromFen(testdata.vs_matt);
    cross_set.generateAll(&b, f.kwg, &f.ld);
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "A");
    rowGen(&gen, &rack, 14, 8, false, true);
    // AVENGED only.
    try testing.expectEqual(1, gen.plays.items.len);
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("15C.A......", try formatted(&f, &gen.plays.items[0], &buf));
}

test "one more row gen" {
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = try f.boardFromFen(testdata.vs_matt);
    cross_set.generateAll(&b, f.kwg, &f.ld);
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "A");
    rowGen(&gen, &rack, 0, 11, false, true);
    try testing.expectEqual(1, gen.plays.items.len);
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("1L..A", try formatted(&f, &gen.plays.items[0], &buf));
}

test "gen move just once" {
    // TAEL parallel to ABANDON, making RESPONDED, from several anchors.
    var f = try testdata.Fixture.load("NWL20");
    defer f.deinit();
    var b = try f.boardFromFen(testdata.vs_matt);
    cross_set.generateAll(&b, f.kwg, &f.ld);
    b.transpose();
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "AELT");
    gen.last_anchor_col = 100;
    for (8..13) |anchor| {
        rowGen(&gen, &rack, 10, anchor, true, false);
        gen.last_anchor_col = anchor;
    }
    try testing.expectEqual(34, gen.plays.items.len);
}

test "max tile usage (America)" {
    var f = try testdata.Fixture.load("America");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    cross_set.generateAll(&b, f.kwg, &f.ld);
    b.updateAllAnchors();
    var gen = Generator.init(testing.allocator, &b, f.kwg, &f.ld);
    defer gen.deinit();
    var rack = try rackFrom(&f, "VIVIFIC");
    // Seven placements of VIVIFIC and two of IF.
    try gen.genAll(&rack, false);
    try testing.expectEqual(9, gen.plays.items.len);
    // At most six tiles: just IF.
    gen.max_tile_usage = 6;
    try gen.genAll(&rack, false);
    try testing.expectEqual(2, gen.plays.items.len);
}
