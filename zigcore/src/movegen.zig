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
const CachedSquare = struct {
    cross_set: board_mod.CrossSet,
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

    pub fn init(gpa: Allocator, b: *Board, kwg: Kwg, ld: *const tiles.LetterDistribution) Generator {
        return .{ .gpa = gpa, .board = b, .kwg = kwg, .ld = ld };
    }

    pub fn deinit(self: *Generator) void {
        self.plays.deinit(self.gpa);
        self.* = undefined;
    }

    /// Generates every move for `rack` into `plays` (cleared first): tile
    /// plays, then a pass if there are none (or gen_pass is set), then the
    /// exchanges if `add_exchange`. The board's cross-sets and anchors must
    /// be up to date.
    pub fn genAll(self: *Generator, rack: *Rack, add_exchange: bool) Allocator.Error!void {
        self.plays.clearRetainingCapacity();
        self.out_of_memory = false;

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

    fn record(self: *Generator, m: Move) void {
        self.plays.append(self.gpa, m) catch {
            self.out_of_memory = true;
        };
    }

    fn setRackBits(self: *Generator, rack: *const Rack) void {
        var bits: u64 = 0;
        for (rack.counts[0..self.ld.size], 0..) |n, ml| {
            if (n > 0) bits |= @as(u64, 1) << @intCast(ml);
        }
        self.rack_bits = bits;
    }

    fn loadRow(self: *Generator, row: usize, cs_dir: Direction) void {
        const b = self.board;
        for (0..b.dim) |col| {
            const sq = b.sqIdx(row, col);
            const bonus = b.bonuses[sq];
            self.row[col] = .{
                .cross_set = b.cross_sets[@backingInt(cs_dir)][sq],
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
                    self.recursiveGen(col, rack, self.kwg.gaddagRoot(), col, col, !self.vertical, 0, 0, 1);
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
                    self.tiles_played += 1;
                    const sml = self.ld.score(ml);
                    var addl_cross: i32 = 0;
                    if (!empty_adjacent) {
                        addl_cross = if (wm > 1) wm * (cs + sml) else cs + sml * lm;
                    }
                    self.goOn(col, ml, rack, arc, accepts, leftstrip, rightstrip, unique_play, base_score + sml * lm, cross_scores + addl_cross, word_mul * wm);
                    self.tiles_played -= 1;
                    rack.add(ml);
                    self.rack_bits |= @as(u64, 1) << @intCast(ml);
                }
                if (n_blank > 0) {
                    rack.take(0);
                    self.tiles_played += 1;
                    self.goOn(col, tiles.blank(ml), rack, arc, accepts, leftstrip, rightstrip, unique_play, base_score, cross_scores + cs * wm, word_mul * wm);
                    self.tiles_played -= 1;
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
                self.recordPlay(leftstrip, rightstrip, base_score * word_mul + cross_scores + bingo);
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
                self.recordPlay(leftstrip, rightstrip, base_score * word_mul + cross_scores + bingo);
            }
            if (new_node != 0 and cur_col < dim - 1) {
                self.recursiveGen(cur_col + 1, rack, new_node, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
        }
    }

    fn recordPlay(self: *Generator, leftstrip: usize, rightstrip: usize, score: i32) void {
        const len = rightstrip - leftstrip + 1;
        if (len < 2) return;
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
            var m: Move = .{ .kind = .exchange, .len = @intCast(strip_idx), .tiles_played = @intCast(strip_idx) };
            @memcpy(m.tiles[0..strip_idx], self.exchange_strip[0..strip_idx]);
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
            self.genExchanges(rack, ml + 1, strip_idx);
        }
        for (0..taken) |_| rack.add(@intCast(ml));
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
