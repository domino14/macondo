//! Move generation, a port of Macondo's `movegen` package (Gordon's GADDAG
//! algorithm, "A Faster Scrabble Move Generation Algorithm").
//!
//! Moves are generated a row at a time; vertical moves are generated on the
//! transposed board. Anchors are the rightmost square of where a word can
//! be hooked, as in the Go code.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("../data/tiles.zig");
const board_mod = @import("../board/board.zig");
const Kwg = @import("../data/kwg.zig").Kwg;
const kwg_mod = @import("../data/kwg.zig");
const move_mod = @import("../board/move.zig");
const equity = @import("../equity/equity.zig");
const leavemap = @import("leavemap.zig");
const shadow_mod = @import("shadow.zig");
const carry_mod = @import("carry.zig");
const rit_mod = @import("../data/rit.zig");
const wmp = @import("../data/wmp.zig");
const wmp_movegen = @import("wmp_movegen.zig");
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
/// A square of the lane being generated: the board's square as the
/// lane's plays read it.
pub const CachedSquare = board_mod.Square;

/// What the generator does with the plays it finds: keeps them all,
/// keeps the best by equity (genBest), keeps the endgame's best by value
/// (genForAnchorSmallBest), or keeps them all as the endgame's small
/// moves (genLanes).
const SmallMove = @import("../endgame/endgame.zig").SmallMove;

const Record = enum { all, best, small_best, small_all };

/// An anchor's best play for the endgame: its move code (as the
/// endgame's SmallMove; 0 for none), value, score, tiles played and
/// length.
pub const SmallBest = struct {
    tm: u64 = 0,
    value: i32 = 0,
    score: i32 = 0,
    tiles_played: u8 = 0,
    len: u8 = 0,
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
    /// The squares of the row being generated: row_buf, or a lane the
    /// shadow pass loaded.
    row: *const [board_mod.max_dim]CachedSquare = undefined,
    row_buf: [board_mod.max_dim]CachedSquare = undefined,
    out_of_memory: bool = false,

    // The endgame's best play of an anchor (genForAnchorSmallBest).
    small_best: SmallBest = .{},
    small_out_bonus: i32 = 0,
    small_rack_tiles: u8 = 0,
    /// Where genLanes puts its plays.
    small_out: *std.ArrayList(SmallMove) = undefined,

    // The best-play search (genBest).
    best_only: bool = false,
    ctx: *const equity.Context = undefined,
    leave_map: leavemap.LeaveMap = .{},
    best: Move = undefined,
    best_eq: f64 = 0,
    have_best: bool = false,
    shadow: shadow_mod.State = .{},
    /// The word map, if genBest is to use it (it must be for this board
    /// size).
    word_map: ?*const wmp.Wmp = null,
    wmp_gen: wmp_movegen.MoveGen = .{},
    /// The rack genBest started with.
    full_rack: Rack = .{},
    /// The memory of best plays by rack (carry.zig), if genBest is to use
    /// it. It changes how fast genBest runs, not what it finds; it must be
    /// forgotten when the lexicon, leaves or word map change.
    carry: ?*carry_mod.Carry = null,
    /// The rack info table for the lexicon and leaves genBest values plays
    /// with, if it is to use one; and the entry for the rack of the
    /// generation in progress (full racks with tiles in the bag only).
    rit: ?*const rit_mod.Rit = null,
    rit_entry: ?*const rit_mod.Entry = null,

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
        // Word-map generation leaves it at its last anchor's count.
        self.tiles_played = 0;

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

    /// Lanes are rows 0..dim (horizontal plays) then columns dim..2*dim
    /// (vertical plays), the order genAll makes plays in.
    pub const max_lanes = 2 * board_mod.max_dim;

    /// Generates the plays of some anchors of some lanes into `plays`
    /// (cleared first), as genAll would make them: for each lane in
    /// `lanes`, the plays of the anchors in `wanted[lane]`, out of all the
    /// lane's anchors `anchors[lane]` (an anchor's plays start after the one
    /// before it). `anchor_of` gets each play's anchor (its index along the
    /// lane) and `ends[lane]` the count of plays after the lane. No pass or
    /// exchanges. The endgame's incremental move lists use it.
    /// The plays of the wanted anchors of `lanes` (rows, then columns),
    /// as the endgame's small moves, into `out`: lane by lane, anchor by
    /// anchor, each play's anchor in `anchor_of`, and the end of each
    /// lane's plays in `ends`.
    pub fn genLanes(self: *Generator, rack: *Rack, lanes: u64, wanted: *const [max_lanes]u32, anchors: *const [max_lanes]u32, ends: *[max_lanes]i32, anchor_of: *std.ArrayList(u8), out: *std.ArrayList(SmallMove)) Allocator.Error!void {
        out.clearRetainingCapacity();
        self.small_out = out;
        defer self.small_out = undefined;
        anchor_of.clearRetainingCapacity();
        self.out_of_memory = false;
        self.best_only = false;
        self.leave_map.initialized = false;
        self.tiles_played = 0;
        self.setRackBits(rack);
        const dim = self.board.dim;
        const row_lanes = lanes & ((@as(u64, 1) << @intCast(dim)) - 1);
        if (row_lanes != 0) {
            self.vertical = false;
            try self.genLaneSet(rack, .horizontal, row_lanes, 0, wanted, anchors, ends, anchor_of);
        }
        const col_lanes = lanes >> @intCast(dim);
        if (col_lanes != 0) {
            self.board.transpose();
            defer self.board.transpose();
            self.vertical = true;
            try self.genLaneSet(rack, .vertical, col_lanes, dim, wanted, anchors, ends, anchor_of);
        }
        if (self.out_of_memory) return error.OutOfMemory;
    }

    fn genLaneSet(self: *Generator, rack: *Rack, dir: Direction, rows: u64, offset: usize, wanted: *const [max_lanes]u32, anchors: *const [max_lanes]u32, ends: *[max_lanes]i32, anchor_of: *std.ArrayList(u8)) Allocator.Error!void {
        const out = self.small_out;
        for (0..self.board.dim) |row| {
            if (rows & (@as(u64, 1) << @intCast(row)) == 0) continue;
            const lane_anchors = anchors[offset + row];
            var want = wanted[offset + row] & lane_anchors;
            if (want != 0) {
                self.cur_row = row;
                self.loadRow(row, dir.other());
            }
            while (want != 0) : (want &= want - 1) {
                const col: usize = @ctz(want);
                const before = lane_anchors & ((@as(u32, 1) << @intCast(col)) - 1);
                self.last_anchor_col = if (before != 0) 31 - @clz(before) else 100;
                self.cur_anchor_col = col;
                self.recursiveGen(.small_all, col, rack, self.kwg.gaddagRoot(), col, col, !self.vertical, 0, 0, 1);
                try anchor_of.appendNTimes(self.gpa, @intCast(col), out.items.len - anchor_of.items.len);
            }
            ends[offset + row] = @intCast(out.items.len);
        }
    }

    pub const Best = struct { move: Move, equity: f64 };

    /// The best move for `rack` by static equity (see equity.Context), as
    /// Go's GenBestStaticTurn finds it: exchanges first (if
    /// `add_exchange`), then the tile plays of each anchor, best shadow
    /// bound first, until no anchor left can beat the best move; a pass
    /// only if nothing else was found (or gen_pass is set). Ties go to
    /// equity.tiebreaksBetter. The rack must hold at most 7 tiles.
    pub fn genBest(self: *Generator, rack: *Rack, add_exchange: bool, ctx: *const equity.Context) Allocator.Error!Best {
        self.bestStart(rack, add_exchange, ctx);
        self.bestSubracks(rack);
        shadow_mod.genShadow(self, rack);
        shadow_mod.recordFromAnchors(self, rack);
        return self.bestFinish(rack);
    }

    // genBest's steps, apart for timing them (the bench tool's phases).

    /// The leave of every subset of the rack, and with it (when they are
    /// allowed) the exchanges, whose best is then a floor the anchors are
    /// measured against.
    pub fn bestStart(self: *Generator, rack: *Rack, add_exchange: bool, ctx: *const equity.Context) void {
        std.debug.assert(rack.num_tiles <= rack_tile_limit);
        self.out_of_memory = false;
        self.best_only = true;
        self.ctx = ctx;
        self.have_best = false;
        self.shadow.tiles_in_bag = ctx.tiles_in_bag;
        self.shadow.opp_rack_score = ctx.opp_rack_score;
        self.shadow.anchor_floor = -std.math.inf(f64);
        if (self.carry) |c| c.cur = null;
        self.rit_entry = null;
        self.leave_map.init(rack);
        if (self.leave_map.initialized) {
            self.full_rack = rack.*;
            carry_mod.begin(self, rack);
            if (carry_mod.loadSetup(self, rack)) {
                // The rack's leave values are remembered, and its best
                // exchange if as many tiles can be exchanged as then.
                if (add_exchange and !carry_mod.loadExchange(self, self.max_can_exchange)) {
                    self.exchangesFromLeaveMap(rack);
                    carry_mod.saveExchange(self, self.max_can_exchange);
                }
            } else if (self.ritEntryFor(rack, ctx)) |e| {
                self.rit_entry = e;
                self.leavesFromRit(e, ctx);
                if (add_exchange) {
                    if (self.max_can_exchange >= rit_mod.rack_size and ctx.tiles_in_bag >= ctx.peg.len) {
                        self.recordRitExchange(e);
                    } else {
                        self.exchangesFromLeaveMap(rack);
                    }
                }
                carry_mod.saveSetup(self);
                if (add_exchange) carry_mod.saveExchange(self, self.max_can_exchange);
            } else {
                const visitor: ?leavemap.Visitor = if (add_exchange) .{ .ctx = self, .visit = visitExchange } else null;
                leavemap.populateVisiting(&self.leave_map, rack, ctx.klv, ctx, &self.shadow.best_leaves, visitor);
                carry_mod.saveSetup(self);
                if (add_exchange) carry_mod.saveExchange(self, self.max_can_exchange);
            }
        } else {
            shadow_mod.bestLeavesWithoutMap(&self.shadow);
            if (add_exchange) self.genExchanges(rack, 0, 0);
        }
        if (self.have_best) self.shadow.anchor_floor = self.best_eq;
    }

    /// The word map's look at the rack's subracks (it needs the leave
    /// map's values).
    pub fn bestSubracks(self: *Generator, rack: *const Rack) void {
        self.wmp_gen.init(self.ld, rack, if (self.leave_map.initialized) self.word_map else null);
        if (self.wmp_gen.isActive() and !carry_mod.loadWmp(self)) {
            if (self.rit_entry) |e| {
                self.wmp_gen.checkNonplaythroughWithRit(e, self.ctx, &self.leave_map);
            } else {
                self.wmp_gen.checkNonplaythroughExistence(self.ctx.tiles_in_bag > 0, &self.leave_map);
            }
            carry_mod.saveWmp(self);
        }
    }

    pub fn bestFinish(self: *Generator, rack: *const Rack) Allocator.Error!Best {
        defer self.best_only = false;
        if (!self.have_best or self.gen_pass) self.recordBest(Move.pass(), rack, null);
        // With the subrack entries looked up since.
        if (self.rit_entry != null and self.wmp_gen.isActive()) carry_mod.saveWmp(self);
        carry_mod.remember(self);
        if (self.out_of_memory) return error.OutOfMemory;
        return .{ .move = self.best, .equity = self.best_eq };
    }

    /// The exchange of everything but `leave`, recorded at the leave
    /// map's subset `leave` with its value.
    fn visitExchange(ctx: *anyopaque, leave: *const Rack, value: f64) void {
        const self: *Generator = @ptrCast(@alignCast(ctx));
        const n = self.full_rack.num_tiles - leave.num_tiles;
        if (n == 0 or n > self.max_can_exchange) return;
        const fast = !self.ctx.board_empty;
        if (fast and self.have_best and value < self.best_eq) return;
        var m: Move = .{ .kind = .exchange, .len = @intCast(n), .tiles_played = @intCast(n) };
        var k: usize = 0;
        for (self.full_rack.counts[0..self.ld.size], leave.counts[0..self.ld.size], 0..) |all, kept, ml| {
            for (kept..all) |_| {
                m.tiles[k] = @intCast(ml);
                k += 1;
            }
        }
        self.recordBest(m, leave, if (fast) value else null);
    }

    /// The rack info table's entry for `rack`, where genBest can use one:
    /// a full rack, tiles in the bag, leave values, a board with tiles.
    fn ritEntryFor(self: *const Generator, rack: *const Rack, ctx: *const equity.Context) ?*const rit_mod.Entry {
        const r = self.rit orelse return null;
        if (rack.num_tiles != rit_mod.rack_size or ctx.tiles_in_bag <= 0 or ctx.board_empty or ctx.klv == null) return null;
        return r.lookup(wmp.fromRack(rack));
    }

    /// The leave map's values and the best leaves from a table entry: its
    /// raw leaves plus the pre-endgame adjustment of each subset's size,
    /// as populate works them out.
    fn leavesFromRit(self: *Generator, e: *const rit_mod.Entry, ctx: *const equity.Context) void {
        const lm = &self.leave_map;
        if (ctx.tiles_in_bag >= ctx.peg.len) {
            // No subset gets an adjustment.
            for (&lm.values, e.leaves) |*d, v| d.* = v;
        } else {
            for (&lm.values, e.leaves, 0..) |*d, v, i| {
                const kept: u32 = @popCount(@as(u32, @intCast(i)));
                d.* = ctx.leaveMapValue(v, 0, kept, rit_mod.rack_size - kept);
            }
        }
        for (&self.shadow.best_leaves, e.best_leaves) |*d, v| d.* = v;
    }

    /// Records the table's best exchange (only when every exchange size is
    /// allowed and none gets a pre-endgame adjustment, as the table
    /// assumes).
    fn recordRitExchange(self: *Generator, e: *const rit_mod.Entry) void {
        if (e.exchange_len == 0) return;
        var m: Move = .{ .kind = .exchange, .len = e.exchange_len, .tiles_played = e.exchange_len };
        @memcpy(m.tiles[0..e.exchange_len], e.exchange[0..e.exchange_len]);
        var leave = self.full_rack;
        for (m.tileSlice()) |ml| leave.take(ml);
        var idx: u32 = 0;
        for (leave.counts[0..self.ld.size], 0..) |n, ml| {
            if (n > 0) idx |= ((@as(u32, 1) << @intCast(n)) - 1) << @intCast(self.leave_map.base[ml]);
        }
        self.recordBest(m, &leave, self.leave_map.values[idx]);
    }

    /// Records the exchanges from the leave map's values (filled for this
    /// rack), as populateVisiting's visitor does while filling it.
    fn exchangesFromLeaveMap(self: *Generator, rack: *const Rack) void {
        var letters: [rack_tile_limit]MachineLetter = undefined;
        var n: usize = 0;
        for (rack.counts[0..self.ld.size], 0..) |c, ml| {
            if (c > 0) {
                letters[n] = @intCast(ml);
                n += 1;
            }
        }
        var leave: Rack = .{};
        self.exchangesFrom(rack, letters[0..n], &leave, 0);
    }

    fn exchangesFrom(self: *Generator, rack: *const Rack, letters: []const MachineLetter, leave: *Rack, idx: u32) void {
        if (letters.len == 0) {
            visitExchange(self, leave, self.leave_map.values[idx]);
            return;
        }
        const ml = letters[0];
        const base = self.leave_map.base[ml];
        for (0..@as(usize, rack.counts[ml]) + 1) |kept| {
            leave.counts[ml] = @intCast(kept);
            leave.num_tiles += @intCast(kept);
            const bits = ((@as(u32, 1) << @intCast(kept)) - 1) << @intCast(base);
            self.exchangesFrom(rack, letters[1..], leave, idx | bits);
            leave.num_tiles -= @intCast(kept);
        }
        leave.counts[ml] = 0;
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
        if (self.best_only) {
            self.recursiveGen(.best, col, rack, self.kwg.gaddagRoot(), col, col, !self.vertical, 0, 0, 1);
        } else {
            self.recursiveGen(.all, col, rack, self.kwg.gaddagRoot(), col, col, !self.vertical, 0, 0, 1);
        }
    }

    /// The anchor's best tile play for the endgame (see SmallBest), with
    /// `out_bonus` added to a play of all `rack`'s tiles.
    pub fn genForAnchorSmallBest(self: *Generator, rack: *Rack, out_bonus: i32) SmallBest {
        const col = self.cur_anchor_col;
        self.small_best = .{};
        self.small_out_bonus = out_bonus;
        self.small_rack_tiles = rack.num_tiles;
        self.recursiveGen(.small_best, col, rack, self.kwg.gaddagRoot(), col, col, !self.vertical, 0, 0, 1);
        return self.small_best;
    }

    /// The endgame's encoding of the play at leftstrip..rightstrip
    /// (SmallMove.tm).
    inline fn smallCode(self: *const Generator, leftstrip: usize, rightstrip: usize) u64 {
        var code: u64 = 0;
        var t: usize = 0;
        var blanks: u64 = 0;
        for (self.strip[leftstrip .. rightstrip + 1]) |ml| {
            if (ml == 0) continue;
            if (tiles.isBlanked(ml)) blanks |= @as(u64, 1) << @intCast(t);
            code |= @as(u64, tiles.unblank(ml)) << @intCast(20 + 6 * t);
            t += 1;
        }
        const row = if (self.vertical) leftstrip else self.cur_row;
        const col = if (self.vertical) self.cur_row else leftstrip;
        return code | @intFromBool(self.vertical) | @as(u64, col) << 1 | @as(u64, row) << 6 | blanks << 12;
    }

    fn recordSmallAll(self: *Generator, leftstrip: usize, rightstrip: usize, score: i32) void {
        const len = rightstrip - leftstrip + 1;
        if (len < 2) return;
        self.small_out.append(self.gpa, .{ .tm = self.smallCode(leftstrip, rightstrip), .score = @intCast(score), .tiles_played = @intCast(self.tiles_played), .len = @intCast(len) }) catch {
            self.out_of_memory = true;
        };
    }

    /// Keeps the play if it beats the best so far: by value (the score,
    /// plus the out bonus for a play of the whole rack), ties to the
    /// larger move code. The code is built only for a play that can win.
    fn recordSmallBest(self: *Generator, leftstrip: usize, rightstrip: usize, score: i32) void {
        const len = rightstrip - leftstrip + 1;
        if (len < 2) return;
        const v = score + if (self.tiles_played == self.small_rack_tiles) self.small_out_bonus else 0;
        const sb = &self.small_best;
        if (sb.tm != 0 and v < sb.value) return;
        const code = self.smallCode(leftstrip, rightstrip);
        if (sb.tm != 0 and v == sb.value and code <= sb.tm) return;
        sb.* = .{ .tm = code, .value = v, .score = score, .tiles_played = @intCast(self.tiles_played), .len = @intCast(len) };
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

    /// Points the generator at row `row`'s squares, as the plays across
    /// `cs_dir` read them: the board's own lane where the transposition
    /// makes the row one, else a copy.
    pub fn loadRow(self: *Generator, row: usize, cs_dir: Direction) void {
        const p = cs_dir.other();
        if (self.board.isTransposed() == (p == .vertical)) {
            self.row = self.lanePtr(@backingInt(p), row);
        } else {
            self.loadRowInto(&self.row_buf, row, cs_dir);
            self.row = &self.row_buf;
        }
    }

    /// Lane `lane` of the plays of direction `side` (0: rows, 1: columns),
    /// in the board, whatever the transposition. A lane's squares are
    /// followed by the next lane's (or the views' spare room), so the
    /// pointer may cover max_dim of them.
    pub inline fn lanePtr(self: *const Generator, side: usize, lane: usize) *const [board_mod.max_dim]CachedSquare {
        return @ptrCast(&self.board.views[side][lane * self.board.dim]);
    }

    /// Reads row `row`'s squares, with the cross-sets of `cs_dir`, into
    /// `dest`.
    pub fn loadRowInto(self: *const Generator, dest: *[board_mod.max_dim]CachedSquare, row: usize, cs_dir: Direction) void {
        const b = self.board;
        const p = cs_dir.other();
        // The board is transposed for vertical plays: the row is the lane.
        if (b.isTransposed() == (p == .vertical)) {
            @memcpy(dest[0..b.dim], b.lane(p, row));
        } else {
            for (dest[0..b.dim], 0..) |*out, col| out.* = b.view(p, row, col).*;
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

    /// Gordon's Gen, for the best-play search (`best`: plays valued from
    /// the leave map as they are found) or for every move.
    fn recursiveGen(self: *Generator, comptime mode: Record, col: usize, rack: *Rack, node: u32, leftstrip: usize, rightstrip: usize, unique_play: bool, base_score: i32, cross_scores: i32, word_mul: i32) void {
        const sq = &self.row[col];
        const nodes = self.kwg.nodes;
        const list_tiles = self.kwg.sibling_tiles[node];
        const scores = &self.ld.score_table;
        if (sq.letter != 0) {
            const raw_bit = @as(u64, 1) << @intCast(tiles.unblank(sq.letter));
            var next: u32 = 0;
            var accepts = false;
            if (list_tiles & raw_bit != 0) {
                const n = nodes[Kwg.listNode(node, list_tiles, raw_bit)];
                next = n & kwg_mod.arc_mask;
                accepts = n & kwg_mod.accepts_bit != 0;
            }
            // Start loading the next list while this square is dealt with.
            @prefetch(&nodes[next], .{});
            self.goOn(mode, col, sq.letter, rack, next, accepts, leftstrip, rightstrip, unique_play, base_score + scores[sq.letter], cross_scores, word_mul);
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

        // Only the arcs whose tiles can go here, in the list's order.
        var placeable = allowed & list_tiles;
        while (placeable != 0) : (placeable &= placeable - 1) {
            const ml: MachineLetter = @intCast(@ctz(placeable));
            const n = nodes[Kwg.listNode(node, list_tiles, placeable & (~placeable +% 1))];
            {
                const n_ml = rack.counts[ml];
                const arc = n & kwg_mod.arc_mask;
                const accepts = n & kwg_mod.accepts_bit != 0;
                @prefetch(&nodes[arc], .{});
                if (n_ml > 0) {
                    rack.take(ml);
                    if (n_ml == 1) self.rack_bits &= ~(@as(u64, 1) << @intCast(ml));
                    if (mode == .best) self.leave_map.take(ml, n_ml - 1);
                    self.tiles_played += 1;
                    const sml = scores[ml];
                    var addl_cross: i32 = 0;
                    if (!empty_adjacent) {
                        addl_cross = if (wm > 1) wm * (cs + sml) else cs + sml * lm;
                    }
                    self.goOn(mode, col, ml, rack, arc, accepts, leftstrip, rightstrip, unique_play, base_score + sml * lm, cross_scores + addl_cross, word_mul * wm);
                    self.tiles_played -= 1;
                    if (mode == .best) self.leave_map.add(ml, n_ml - 1);
                    rack.add(ml);
                    self.rack_bits |= @as(u64, 1) << @intCast(ml);
                }
                if (n_blank > 0) {
                    rack.take(0);
                    if (mode == .best) self.leave_map.take(0, n_blank - 1);
                    self.tiles_played += 1;
                    self.goOn(mode, col, tiles.blank(ml), rack, arc, accepts, leftstrip, rightstrip, unique_play, base_score, cross_scores + cs * wm, word_mul * wm);
                    self.tiles_played -= 1;
                    if (mode == .best) self.leave_map.add(0, n_blank - 1);
                    rack.add(0);
                }
            }
        }
    }

    /// Gordon's GoOn.
    fn goOn(self: *Generator, comptime mode: Record, cur_col: usize, letter: MachineLetter, rack: *Rack, new_node: u32, accepts: bool, leftstrip_in: usize, rightstrip_in: usize, unique_play_in: bool, base_score: i32, cross_scores: i32, word_mul: i32) void {
        var leftstrip = leftstrip_in;
        var rightstrip = rightstrip_in;
        var unique_play = unique_play_in;
        const bingo: i32 = if (self.tiles_played == rack_tile_limit) bingo_bonus else 0;
        const sq = &self.row[cur_col];
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
                self.recordFound(mode, rack, leftstrip, rightstrip, base_score * word_mul + cross_scores + bingo);
            }
            if (new_node == 0) return;
            if (cur_col > 0 and cur_col - 1 != self.last_anchor_col) {
                self.recursiveGen(mode, cur_col - 1, rack, new_node, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
            // The separator is tile 0, so in a sorted list it can only come
            // first.
            const first = self.kwg.nodes[new_node];
            if (first >> kwg_mod.tile_shift == 0 and no_letter_left and self.cur_anchor_col < dim - 1) {
                const sep = first & kwg_mod.arc_mask;
                if (sep != 0) self.recursiveGen(mode, self.cur_anchor_col + 1, rack, sep, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
        } else {
            rightstrip = cur_col;
            const no_letter_right = cur_col == dim - 1 or self.row[cur_col + 1].letter == 0;
            if (accepts and no_letter_right and self.tiles_played > 0 and
                (unique_play or self.tiles_played > 1) and self.tiles_played <= self.max_tile_usage)
            {
                self.recordFound(mode, rack, leftstrip, rightstrip, base_score * word_mul + cross_scores + bingo);
            }
            if (new_node != 0 and cur_col < dim - 1) {
                self.recursiveGen(mode, cur_col + 1, rack, new_node, leftstrip, rightstrip, unique_play, base_score, cross_scores, word_mul);
            }
        }
    }

    inline fn recordFound(self: *Generator, comptime mode: Record, rack: *const Rack, leftstrip: usize, rightstrip: usize, score: i32) void {
        if (mode == .best) return self.recordPlay(rack, leftstrip, rightstrip, score);
        if (mode == .small_best) return self.recordSmallBest(leftstrip, rightstrip, score);
        if (mode == .small_all) return self.recordSmallAll(leftstrip, rightstrip, score);
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

    pub fn recordPlay(self: *Generator, rack: *const Rack, leftstrip: usize, rightstrip: usize, score: i32) void {
        var fast_eq: ?f64 = null;
        if (self.best_only) {
            fast_eq = self.fastEquity(score);
            // Turned away before the move is built.
            if (fast_eq) |eq| if (self.have_best and eq < self.best_eq) return;
        }
        self.recordPlayWithEquity(rack, leftstrip, rightstrip, score, fast_eq);
    }

    /// recordPlay with the play's equity already worked out from the leave
    /// map (null for one that must be valued in full).
    pub fn recordPlayWithEquity(self: *Generator, rack: *const Rack, leftstrip: usize, rightstrip: usize, score: i32, fast_eq: ?f64) void {
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
        if (self.best_only) return self.recordBest(m, rack, fast_eq);
        self.record(m);
    }

    /// Every distinct exchange (by tile counts). Port of andy-k's
    /// generateExchangeMoves.
    pub fn genExchanges(self: *Generator, rack: *Rack, ml_in: usize, strip_idx_in: usize) void {
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
const testdata = @import("../testing/testdata.zig");
const cross_set = @import("../board/cross_set.zig");

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
    const position_mod = @import("../board/position.zig");
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
        // With the lexicon's word map too, if it has been made.
        var name_buf: [64]u8 = undefined;
        const wmp_bytes: ?[]align(4) u8 = testdata.loadFile(try std.fmt.bufPrint(&name_buf, "lexica/{s}.wmp", .{p.lexiconName()})) catch null;
        defer if (wmp_bytes) |b| testing.allocator.free(b);
        var word_map: ?wmp.Wmp = if (wmp_bytes) |b| try wmp.Wmp.init(testing.allocator, b) else null;
        defer if (word_map) |*w| w.deinit(testing.allocator);

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
            gen.word_map = null;
            const got_kwg = try gen.genBest(&rack, ex.allowed, &ctx);
            gen.word_map = if (word_map) |*w| w else null;
            const got = if (word_map != null) try gen.genBest(&rack, ex.allowed, &ctx) else got_kwg;
            gen.word_map = null;
            const want = try bruteBest(&gen, &rack, ex.allowed, &ctx);
            if (!sameMove(&got_kwg.move, &want.move) or got_kwg.equity != want.equity or
                !sameMove(&got.move, &want.move) or got.equity != want.equity)
            {
                var buf: [128]u8 = undefined;
                var w: std.Io.Writer = .fixed(&buf);
                try got_kwg.move.format(p.dist, &w);
                try w.print(" {d}, word map ", .{got_kwg.equity});
                try got.move.format(p.dist, &w);
                try w.print(" {d}, all moves ", .{got.equity});
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

test "the endgame's small records agree with the full move list" {
    const position_mod = @import("../board/position.zig");
    var store = testdata.store();
    defer store.deinit();
    const cgps = [_][]const u8{
        "15/15/15/15/15/15/1PROTEAN7/3WINDY7/15/15/15/15/15/15/15 / 0/0 0 lex CSW21;",
        "5E1p7/1A1C1G1E5S1/1JIZ1G1R3V1P1/HO1APE1A3U1I1/OW1RAD1EXFIL1T1/MA2W2OI2G1T1/IN2K2N2MOBE1/E2FYRDS2o1ANS/3O6V1N1H/3U6ADD1E/3S6BOO1E/3T6LOR1L/INCITANT1AYRE2/3E5U5/3REQUITE5 / 0/0 0 lex CSW21;",
        testdata.vs_matt ++ " / 0/0 0 lex NWL20;",
        testdata.maven_vs_macondo ++ " / 0/0 0 lex NWL20;",
    };
    var prng: std.Random.DefaultPrng = .init(5);
    const random = prng.random();
    var small: std.ArrayList(SmallMove) = .empty;
    defer small.deinit(testing.allocator);
    var anchor_of: std.ArrayList(u8) = .empty;
    defer anchor_of.deinit(testing.allocator);
    for (cgps) |text| {
        var p: position_mod.Position = try testdata.position(&store, text);
        var gen = Generator.init(testing.allocator, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        const b = &p.board;
        const dim = b.dim;
        // Each lane's anchors: rows, then columns.
        var anchors: [Generator.max_lanes]u32 = @splat(0);
        for (0..dim) |r| {
            for (0..dim) |c| {
                if (b.isAnchor(r, c, .horizontal)) anchors[r] |= @as(u32, 1) << @intCast(c);
                if (b.isAnchor(r, c, .vertical)) anchors[dim + c] |= @as(u32, 1) << @intCast(r);
            }
        }
        const wanted: [Generator.max_lanes]u32 = @splat(~@as(u32, 0));
        var ends: [Generator.max_lanes]i32 = undefined;
        var pool: [100]MachineLetter = undefined;
        for (&pool, 0..) |*t, i| t.* = @intCast(if (i % 13 == 0) 0 else 1 + i % 26);
        for (0..12) |_| {
            random.shuffle(MachineLetter, &pool);
            var rack: Rack = .fromLetters(pool[0 .. 1 + random.uintLessThan(usize, 7)]);
            try gen.genAll(&rack, false);
            var want: std.ArrayList(SmallMove) = .empty;
            defer want.deinit(testing.allocator);
            for (gen.plays.items) |*mv| {
                if (mv.kind == .play) try want.append(testing.allocator, .fromMove(mv));
            }
            const all_lanes = (@as(u64, 1) << @intCast(2 * dim)) - 1;
            try gen.genLanes(&rack, all_lanes, &wanted, &anchors, &ends, &anchor_of, &small);
            try testing.expectEqual(want.items.len, small.items.len);
            for (want.items, small.items) |w, g| {
                try testing.expectEqual(w.tm, g.tm);
                try testing.expectEqual(w.score, g.score);
                try testing.expectEqual(w.tiles_played, g.tiles_played);
                try testing.expectEqual(w.len, g.len);
            }
            // Each anchor's best by value (an out bonus for the whole
            // rack), ties to the larger code, from its plays in the list.
            const out_bonus: i32 = random.intRangeAtMost(i32, 0, 30);
            var start: usize = 0;
            for (0..2 * dim) |lane| {
                const end: usize = @intCast(ends[lane]);
                defer start = end;
                const vertical = lane >= dim;
                const row = if (vertical) lane - dim else lane;
                if (vertical) b.transpose();
                defer if (vertical) b.transpose();
                gen.vertical = vertical;
                gen.cur_row = row;
                gen.loadRow(row, if (vertical) .horizontal else .vertical);
                gen.setRackBits(&rack);
                var as = anchors[lane];
                while (as != 0) : (as &= as - 1) {
                    const col: usize = @ctz(as);
                    const before = anchors[lane] & ((@as(u32, 1) << @intCast(col)) - 1);
                    gen.last_anchor_col = if (before != 0) 31 - @clz(before) else 100;
                    gen.cur_anchor_col = col;
                    gen.tiles_played = 0;
                    const got = gen.genForAnchorSmallBest(&rack, out_bonus);
                    var best: ?SmallMove = null;
                    var best_v: i32 = 0;
                    for (small.items[start..end], anchor_of.items[start..end]) |m, a| {
                        if (a != col) continue;
                        const v = m.score + if (m.tiles_played == rack.num_tiles) out_bonus else 0;
                        if (best == null or v > best_v or (v == best_v and m.tm > best.?.tm)) {
                            best = m;
                            best_v = v;
                        }
                    }
                    if (best) |m| {
                        try testing.expectEqual(m.tm, got.tm);
                        try testing.expectEqual(best_v, got.value);
                        try testing.expectEqual(@as(i32, m.score), got.score);
                    } else try testing.expectEqual(0, got.tm);
                }
            }
        }
    }
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
    gen.recursiveGen(.all, anchor_col, rack, gen.kwg.gaddagRoot(), anchor_col, anchor_col, unique, 0, 0, 1);
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
    for (&b.views) |*v| {
        for (v) |*sq| sq.cross_set = 0;
    }
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

test "Move.scoreOn scores every generated play as the generator did" {
    var store = testdata.store();
    defer store.deinit();
    const boards = [_][]const u8{
        "6AAL2CERE/5WINO4ET/1V2QI6ODA/1U1AIT1PINTADAS/1G1D5OI1ER1/1HER1C7G1/2XI1OHM5U1/3FUZEES1JOYED/3T1Y2N1E4/8IFS4/8BAT4/9R5/2KOEL2WRUNG2/1bIPLANE1O5/VITA1BERMS5",
        "15/10N4/2ARROZ3O4/3U6V4/3SHA4A4/3KEX4T4/4JETLINER3/3PILEI2DEMOI/4R10/4AW9/4sO9/5W9/5I9/5N2OVERDO1/3BIGOTS2EASY",
        "15/15/15/15/15/15/1PROTEAN7/3WINDY7/15/15/15/15/15/15/15",
    };
    var checked: usize = 0;
    for (boards) |fen| {
        for ([_][]const u8{ "?AEINST", "AEIRSTZ", "DGLOPRU" }) |rack_text| {
            var buf: [256]u8 = undefined;
            // (Skipping racks the board leaves no tiles for.)
            var p = testdata.position(&store, try std.fmt.bufPrint(&buf, "{s} {s}/ 0/0 0 lex NWL23;", .{ fen, rack_text })) catch |err| switch (err) {
                error.TooManyTiles => continue,
                else => |e| return e,
            };
            var gen = Generator.init(testing.allocator, &p.board, p.lexicon.kwg, p.dist);
            defer gen.deinit();
            var rack = p.racks[0];
            try gen.genAll(&rack, false);
            for (gen.plays.items) |*m| {
                if (m.kind != .play) continue;
                try testing.expectEqual(m.score, m.scoreOn(&p.board, p.dist));
                checked += 1;
            }
        }
    }
    try testing.expect(checked > 10_000);
}
