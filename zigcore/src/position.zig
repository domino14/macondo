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
const MachineLetter = tiles.MachineLetter;

pub const Error = data.Error || cgp_mod.Error || error{
    UnknownLayout,
    UnsupportedVariant,
    BadRowCount,
    BadRowLength,
    UnknownLetter,
    BadRow,
    TooManyTiles,
};

pub const Position = struct {
    board: board_mod.Board,
    dist: *const tiles.LetterDistribution,
    lexicon: *const data.Lexicon,
    /// The leave values the lexicon is played with, if any.
    leaves: ?*const kwg_mod.Klv,
    /// Pre-endgame equity adjustments (see equity.Context.peg).
    peg: []const f64,
    /// Owned copies of the CGP's names.
    lexicon_name: [64]u8,
    lexicon_name_len: usize,
    racks: [2]tiles.Rack,
    scores: [2]i32,
    scoreless_turns: u32,
    max_scoreless_turns: u32,

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

        p.board.countTiles();
        cross_set.generateAll(&p.board, lex.kwg, dist);
        p.board.updateAllAnchors();
        self.* = p;
    }
};
