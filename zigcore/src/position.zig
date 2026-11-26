//! A game position: the board with its cross-sets, the racks, scores and
//! the data the rules need. Built from a CGP string, the way the Go
//! engine's cgp.ParseCGP followed by Game.RecalculateBoard builds one.

const std = @import("std");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const cross_set = @import("cross_set.zig");
const cgp_mod = @import("cgp.zig");
const data = @import("data.zig");
const equity = @import("equity.zig");
const kwg_mod = @import("kwg.zig");
const wmp = @import("wmp.zig");
const move_mod = @import("move.zig");
const MachineLetter = tiles.MachineLetter;

pub const Error = data.Error || cgp_mod.Error || error{
    UnknownLayout,
    UnsupportedVariant,
    BadRowCount,
    BadRowLength,
    UnknownLetter,
    BadRow,
    TooManyTiles,
    /// The lm opcode's move isn't on the board as written, or its score is
    /// wrong.
    BadLastMove,
};

/// The move that led to a position (CGP's lm opcode), made by the player
/// not on turn.
pub const LastMove = struct {
    /// Tiles along the play (0 where it played through a tile); for an
    /// exchange the tiles thrown in, 0 each when only their number is known.
    move: move_mod.Move,
    score: i32,
    /// A play challenged off: nothing changed but the turn.
    phony: bool = false,
};

pub const Position = struct {
    board: board_mod.Board,
    dist: *const tiles.LetterDistribution,
    lexicon: *const data.Lexicon,
    /// The leave values the lexicon is played with, if any.
    leaves: ?*const kwg_mod.Klv,
    /// Pre-endgame equity adjustments (see equity.Context.peg).
    peg: []const f64,
    /// The lexicon's word map, for the best-play search; null on boards
    /// it isn't made for, or when not asked for.
    word_map: ?*const wmp.Wmp = null,
    /// The rack info table for the lexicon and leaves, if one is used.
    rit: ?*const @import("rit.zig").Rit = null,
    /// Owned copies of the CGP's names.
    lexicon_name: [64]u8,
    lexicon_name_len: usize,
    racks: [2]tiles.Rack,
    scores: [2]i32,
    scoreless_turns: u32,
    max_scoreless_turns: u32,
    last_move: ?LastMove = null,

    /// Tiles player `who` can't see: the bag and the other rack.
    pub fn unseenBy(self: *const Position, who: usize) i32 {
        return @as(i32, @intCast(self.dist.total_tiles)) - @as(i32, @intCast(self.board.tiles_played)) -
            @as(i32, self.racks[who].num_tiles);
    }

    /// Tiles in the bag: those in neither rack nor on the board. With the
    /// opponent's rack unknown (empty in the CGP), its tiles count as in
    /// the bag, as in the Go engine.
    pub fn tilesInBag(self: *const Position) i32 {
        return self.unseenBy(0) - @as(i32, self.racks[1].num_tiles);
    }

    /// The equity context for the player to move (player 0).
    pub fn equityContext(self: *const Position) equity.Context {
        return .{
            .ld = self.dist,
            .klv = self.leaves,
            .peg = self.peg,
            .board_dim = self.board.dim,
            .board_empty = self.board.tiles_played == 0,
            .tiles_in_bag = self.tilesInBag(),
            .opp_rack_score = self.racks[1].scoreOn(self.dist),
        };
    }

    pub fn lexiconName(self: *const Position) []const u8 {
        return self.lexicon_name[0..self.lexicon_name_len];
    }

    /// Sets `self` up from a CGP string, loading data through `store`.
    /// On error `self` is left unchanged.
    pub fn fromCgp(self: *Position, text: []const u8, store: *data.Store, data_path: []const u8) Error!void {
        return self.fromCgpWith(text, store, data_path, .{});
    }

    /// What fromCgpWith loads besides what the rules need.
    pub const Extras = struct {
        /// The lexicon's word map (loaded, or made the first time).
        word_maps: bool = false,
        /// The rack info table for the lexicon and leaves (with word maps
        /// only).
        rack_tables: data.RackTables = .never,
    };

    /// fromCgp, with the extras the best-play search can use.
    pub fn fromCgpWith(self: *Position, text: []const u8, store: *data.Store, data_path: []const u8, extras: Extras) Error!void {
        const c = try cgp_mod.Cgp.parse(text);
        if (!(std.mem.eql(u8, c.variant, "classic") or std.mem.eql(u8, c.variant, "classic_super")))
            return error.UnsupportedVariant;
        const layout = board_mod.layoutByName(c.board_layout) orelse return error.UnknownLayout;
        if (c.lexicon.len > 64) return error.InvalidName;
        const dist = try store.distribution(data_path, c.letter_distribution);
        const lex = try store.lexicon(data_path, c.lexicon);
        const super = layout == &board_mod.super_crossword_game;
        const leaves = try store.leavesFor(data_path, c.lexicon, super);
        const peg = try store.preEndgameValues(data_path, c.lexicon);

        var p: Position = .{
            .board = .init(layout),
            .dist = dist,
            .lexicon = lex,
            .leaves = if (leaves) |l| &l.klv else null,
            .peg = peg,
            .lexicon_name = undefined,
            .lexicon_name_len = c.lexicon.len,
            .racks = .{ .{}, .{} },
            .scores = c.scores,
            .scoreless_turns = c.scoreless_turns,
            .max_scoreless_turns = c.max_scoreless_turns,
        };
        @memcpy(p.lexicon_name[0..c.lexicon.len], c.lexicon);

        if (c.num_rows != p.board.dim) return error.BadRowCount;
        // Count every tile in play against the distribution.
        var used: [tiles.max_alphabet_size]u32 = @splat(0);
        for (c.rows[0..c.num_rows], 0..) |row_text, r| {
            var row: [board_mod.max_dim]MachineLetter = undefined;
            const n = cgp_mod.decodeRow(row_text, dist, &row) catch |err| switch (err) {
                error.NoSpaceLeft => return error.BadRowLength,
                else => |e| return e,
            };
            if (n != p.board.dim) return error.BadRowLength;
            _ = p.board.setRow(r, row[0..n]);
            for (row[0..n]) |ml| {
                if (ml != 0) used[tiles.intrinsicIndex(ml)] += 1;
            }
        }
        for (c.racks, 0..) |rack_text, i| {
            var mls: [32]MachineLetter = undefined;
            const n = dist.toMachineLetters(rack_text, &mls) catch |err| switch (err) {
                error.UnknownLetter => return error.UnknownLetter,
                error.NoSpaceLeft => return error.TooManyTiles,
            };
            for (mls[0..n]) |ml| {
                // A rack holds undesignated tiles only.
                if (tiles.isBlanked(ml)) return error.UnknownLetter;
                used[ml] += 1;
            }
            p.racks[i] = .fromLetters(mls[0..n]);
        }
        for (used[0..dist.size], dist.counts[0..dist.size]) |u, have| {
            if (u > have) return error.TooManyTiles;
        }

        if (extras.word_maps) {
            if (try store.wordMap(data_path, c.lexicon, lex, dist)) |w| {
                if (w.board_dim == p.board.dim) p.word_map = w;
            }
        }
        if (p.word_map) |w| {
            if (leaves) |l| p.rit = try store.rackInfoTable(data_path, c.lexicon, lex, l, w, dist, extras.rack_tables);
        }
        p.board.countTiles();
        cross_set.generateAll(&p.board, lex.kwg, dist);
        p.board.updateAllAnchors();
        if (c.last_move.len > 0) p.last_move = try p.readLastMove(c.last_move, c.last_move_score);
        self.* = p;
    }

    /// Reads the lm opcode's move against the board (which shows it) and
    /// works out or checks its score.
    fn readLastMove(self: *const Position, text: []const u8, given_score: ?i32) Error!LastMove {
        const parsed = move_mod.parse(text, self.dist, self.board.dim) catch |err| switch (err) {
            error.UnknownLetter => return error.UnknownLetter,
            else => return error.BadLastMove,
        };
        var lm: LastMove = .{ .move = parsed.move, .score = 0, .phony = parsed.phony };
        if (lm.move.kind == .play) {
            const m = &lm.move;
            const b = &self.board;
            const dr: usize = if (m.vertical) 1 else 0;
            const dc: usize = 1 - dr;
            // The whole main word: empty (or the edge) on either side.
            if (m.col * dc + m.row * dr > 0) {
                const before_r = m.row - dr;
                const before_c = m.col - dc;
                if (b.hasLetter(before_r, before_c)) return error.BadLastMove;
            }
            const end_r = m.row + dr * m.len;
            const end_c = m.col + dc * m.len;
            if (end_r < b.dim and end_c < b.dim and b.hasLetter(end_r, end_c)) return error.BadLastMove;
            var before = b.*;
            for (m.tileSlice(), 0..) |ml, i| {
                const r = m.row + dr * i;
                const col = m.col + dc * i;
                const on = b.getLetter(r, col);
                if (on == 0) return error.BadLastMove;
                if (ml != 0) {
                    if (on != ml) return error.BadLastMove;
                    before.setLetter(r, col, 0);
                } else if (parsed.through[i] != 0 and parsed.through[i] != on) return error.BadLastMove;
            }
            before.countTiles();
            cross_set.generateAll(&before, self.lexicon.kwg, self.dist);
            // A legal play touches the tiles there were (plays through one,
            // or puts a tile next to one), or is the first and covers the
            // centre.
            var touches = false;
            const mid = b.dim / 2;
            for (m.tileSlice(), 0..) |ml, i| {
                const r = m.row + dr * i;
                const col = m.col + dc * i;
                if (ml == 0) touches = true;
                if (before.tiles_played == 0 and r == mid and col == mid) touches = true;
                if (r > 0 and before.hasLetter(r - 1, col)) touches = true;
                if (r + 1 < b.dim and before.hasLetter(r + 1, col)) touches = true;
                if (col > 0 and before.hasLetter(r, col - 1)) touches = true;
                if (col + 1 < b.dim and before.hasLetter(r, col + 1)) touches = true;
            }
            if (!touches) return error.BadLastMove;
            lm.score = m.scoreOn(&before, self.dist);
        }
        if (given_score) |sc| {
            if (sc != lm.score) return error.BadLastMove;
        }
        return lm;
    }

    /// The position before the last move, with its mover on turn: its
    /// tiles off the board, its score off the mover's, and the racks and
    /// scores swapped. The mover's rack is unknown (empty); the other
    /// player's is kept. Without a last move, null. The turns without a
    /// score before a scoring move aren't known; they are taken as 0.
    pub fn beforeLastMove(self: *const Position) ?Position {
        const lm = self.last_move orelse return null;
        var p = self.*;
        p.last_move = null;
        if (lm.move.kind == .play and !lm.phony) {
            const m = &lm.move;
            for (m.tileSlice(), 0..) |ml, i| {
                if (ml == 0) continue;
                p.board.setLetter(m.row + if (m.vertical) i else 0, m.col + if (m.vertical) 0 else i, 0);
            }
            p.board.countTiles();
            cross_set.generateAll(&p.board, p.lexicon.kwg, p.dist);
            p.board.updateAllAnchors();
        }
        p.racks = .{ .{}, self.racks[0] };
        p.scores = .{ self.scores[1] - lm.score, self.scores[0] };
        p.scoreless_turns = if (lm.score == 0) self.scoreless_turns -| 1 else 0;
        return p;
    }
};

test "last move: read, checked and taken back" {
    const testdata = @import("testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const cgp_text = "15/15/15/15/15/15/15/6SCATTER2/15/15/15/15/15/15/15 AEINRST/ 0/20 0 lex NWL23; lm 8G.S(CAT)TER 10;";
    const p = try testdata.position(&store, cgp_text);
    const lm = p.last_move.?;
    try std.testing.expectEqual(4, lm.move.tiles_played);
    try std.testing.expectEqual(10, lm.score);
    const before = p.beforeLastMove().?;
    try std.testing.expectEqual([2]i32{ 10, 0 }, before.scores);
    try std.testing.expectEqual(0, before.racks[0].num_tiles);
    try std.testing.expectEqual(7, before.racks[1].num_tiles);
    try std.testing.expect(!before.board.hasLetter(7, 6) and before.board.hasLetter(7, 7) and !before.board.hasLetter(7, 10));
    // Through squares as dots, and no score: worked out.
    const q = try testdata.position(&store, "15/15/15/15/15/15/15/6SCATTER2/15/15/15/15/15/15/15 AEINRST/ 0/20 0 lex NWL23; lm 8G.S...TER;");
    try std.testing.expectEqual(10, q.last_move.?.score);
    // A bingo through nothing.
    const r = try testdata.position(&store, "15/15/15/15/15/15/15/4RETINAS4/15/15/15/15/15/15/15 / 0/64 0 lex NWL23; lm 8E.RETINAS 64;");
    try std.testing.expectEqual(64, r.last_move.?.score);
    try std.testing.expectEqual(0, r.beforeLastMove().?.board.tiles_played);
    const x = try testdata.position(&store, "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AELNOQT/ 0/0 1 lex NWL18; lm ex.2;");
    try std.testing.expectEqual(.exchange, x.last_move.?.move.kind);
    try std.testing.expectEqual(0, x.beforeLastMove().?.scoreless_turns);
    for ([_][]const u8{
        "lm 8G.S(CAT)TER 11;", // wrong score
        "lm 8G.SCATTER 10;", // CAT not marked: read as placed, it scores 70
        "lm 8H.(CAT)TER;", // not the whole word
        "lm 8G.S(COT)TER;", // wrong through letters
        "lm H8.S(CAT)TER;", // wrong direction
        "lm 8G.s(CAT)TER;", // S is not a blank
    }) |op| {
        var buf: [160]u8 = undefined;
        const t = try std.fmt.bufPrint(&buf, "15/15/15/15/15/15/15/6SCATTER2/15/15/15/15/15/15/15 AEINRST/ 0/20 0 lex NWL23; {s}", .{op});
        try std.testing.expectError(error.BadLastMove, testdata.position(&store, t));
    }
    // A play touching none of the tiles there were.
    try std.testing.expectError(error.BadLastMove, testdata.position(&store, "15/QI13/15/15/15/15/15/6SCATTER2/15/15/15/15/15/15/15 AEINRST/ 0/20 0 lex NWL23; lm 2A.QI;"));
}
