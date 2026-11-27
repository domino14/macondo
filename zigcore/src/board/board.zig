//! The game board: tiles, bonus squares, anchors, cross-sets, cross-scores
//! and the extension sets shadow move generation uses. Port of Macondo's
//! `board` package.
//!
//! The squares are kept as MAGPIE keeps them: twice, once for the plays of
//! each direction, each square a 32-byte struct of what such a play reads
//! there, lane by lane, so that a lane's squares lie together in either
//! direction. A tile is written to both; a cross-set (or cross-score, or
//! extension set) to the one of the plays it constrains. The letters are
//! also kept by themselves, row by row.
//!
//! Like the Go board, it is transposed by swapping the row and column
//! strides, so code written for horizontal lanes also handles vertical
//! ones. Every accessor that takes (row, col) goes through the strides.

const std = @import("std");
const tiles = @import("../data/tiles.zig");
const MachineLetter = tiles.MachineLetter;
const LetterDistribution = tiles.LetterDistribution;

pub const max_dim = 21;
pub const max_squares = max_dim * max_dim;

/// Every letter allowed.
pub const trivial_cross_set: CrossSet = (1 << tiles.max_alphabet_size) - 1;

/// The letters allowed on a square, a bit per machine letter.
pub const CrossSet = u64;

pub const Direction = enum(u1) {
    horizontal,
    vertical,

    pub fn other(d: Direction) Direction {
        return if (d == .horizontal) .vertical else .horizontal;
    }
};

pub const Layout = struct {
    name: []const u8,
    rows: []const []const u8,
};

pub const crossword_game: Layout = .{ .name = "CrosswordGame", .rows = &.{
    "=  '   =   '  =",
    " -   \"   \"   - ",
    "  -   ' '   -  ",
    "'  -   '   -  '",
    "    -     -    ",
    " \"   \"   \"   \" ",
    "  '   ' '   '  ",
    "=  '   -   '  =",
    "  '   ' '   '  ",
    " \"   \"   \"   \" ",
    "    -     -    ",
    "'  -   '   -  '",
    "  -   ' '   -  ",
    " -   \"   \"   - ",
    "=  '   =   '  =",
} };

pub const super_crossword_game: Layout = .{ .name = "SuperCrosswordGame", .rows = &.{
    "~  '   =  '  =   '  ~",
    " -  \"   -   -   \"  - ",
    "  -  ^   - -   ^  -  ",
    "'  =  '   =   '  =  '",
    " \"  -   \"   \"   -  \" ",
    "  ^  -   ' '   -  ^  ",
    "   '  -   '   -  '   ",
    "=      -     -      =",
    " -  \"   \"   \"   \"  - ",
    "  -  '   ' '   '  -  ",
    "'  =  '   -   '  =  '",
    "  -  '   ' '   '  -  ",
    " -  \"   \"   \"   \"  - ",
    "=      -     -      =",
    "   '  -   '   -  '   ",
    "  ^  -   ' '   -  ^  ",
    " \"  -   \"   \"   -  \" ",
    "'  =  '   =   '  =  '",
    "  -  ^   - -   ^  -  ",
    " -  \"   -   -   \"  - ",
    "~  '   =  '  =   '  ~",
} };

/// The layout with this name; "" is the standard board.
pub fn layoutByName(name: []const u8) ?*const Layout {
    if (name.len == 0 or std.mem.eql(u8, name, crossword_game.name)) return &crossword_game;
    if (std.mem.eql(u8, name, super_crossword_game.name)) return &super_crossword_game;
    return null;
}

pub fn letterMultiplier(bonus: u8) u8 {
    return switch (bonus) {
        '\'' => 2,
        '"' => 3,
        '^' => 4,
        else => 1,
    };
}

pub fn wordMultiplier(bonus: u8) u8 {
    return switch (bonus) {
        '-' => 2,
        '=' => 3,
        '~' => 4,
        else => 1,
    };
}

/// A square as the plays of one direction read it (MAGPIE's Square): its
/// cross-set and cross-score from the words across, its extension sets
/// from the words along, its letter and multipliers, and whether it
/// anchors such plays.
pub const Square = extern struct {
    cross_set: CrossSet,
    left_ext: u64,
    right_ext: u64,
    cross_score: i32,
    letter: MachineLetter,
    letter_mul: u8,
    word_mul: u8,
    anchor: bool,
};

comptime {
    std.debug.assert(@sizeOf(Square) == 32);
}

/// Everything a square holds that a move can change: its place (row by
/// row) and its place in the vertical plays' squares, and both squares.
pub const SavedSquare = struct {
    idx: u16,
    v_idx: u16,
    sq: [2]Square,
};

/// An undo record for a board (MAGPIE's MoveUndo): attached to the board,
/// it saves each square the first time it changes after a mark, so that
/// restoring to the mark puts back exactly what changed. The caller
/// reserves room before each move (`reserve`), as the setters cannot fail.
pub const Journal = struct {
    saved: std.ArrayList(SavedSquare) = .empty,
    /// Which mark last saved each square.
    stamp: []u32 = &.{},
    serial: u32 = 1,

    pub fn init(gpa: std.mem.Allocator, squares: usize) std.mem.Allocator.Error!Journal {
        const stamp = try gpa.alloc(u32, squares);
        @memset(stamp, 0);
        return .{ .stamp = stamp };
    }

    pub fn deinit(self: *Journal, gpa: std.mem.Allocator) void {
        self.saved.deinit(gpa);
        gpa.free(self.stamp);
    }

    /// Room for a move's changes: every square of the board.
    pub fn reserve(self: *Journal, gpa: std.mem.Allocator) std.mem.Allocator.Error!void {
        try self.saved.ensureUnusedCapacity(gpa, self.stamp.len);
    }

    /// Starts a new move's record; returns where it starts.
    pub fn mark(self: *Journal) usize {
        self.serial +%= 1;
        if (self.serial == 0) {
            @memset(self.stamp, 0);
            self.serial = 1;
        }
        return self.saved.items.len;
    }

    inline fn save(self: *Journal, b: *const Board, idx: usize, v_idx: usize) void {
        if (self.stamp[idx] == self.serial) return;
        self.stamp[idx] = self.serial;
        self.saved.appendAssumeCapacity(.{
            .idx = @intCast(idx),
            .v_idx = @intCast(v_idx),
            .sq = .{ b.views[0][idx], b.views[1][v_idx] },
        });
    }

    /// Puts back every square saved since `start` and forgets them; the
    /// squares can then be saved again.
    pub fn restore(self: *Journal, b: *Board, start: usize) void {
        for (self.saved.items[start..]) |*sq| {
            b.squares[sq.idx] = sq.sq[0].letter;
            b.views[0][sq.idx] = sq.sq[0];
            b.views[1][sq.v_idx] = sq.sq[1];
        }
        self.saved.shrinkRetainingCapacity(start);
        _ = self.mark();
    }
};

pub const Board = struct {
    dim: usize,
    row_mul: usize,
    col_mul: usize,
    tiles_played: u32,
    /// The letters, row by row.
    squares: [max_squares]MachineLetter,
    bonuses: [max_squares]u8,
    /// The squares as the plays of each direction read them, lane by lane:
    /// views[0] (horizontal plays) row by row, at the letters' places;
    /// views[1] (vertical plays) column by column. A square's cross-set
    /// and cross-score of direction d (the words across it in d) are in
    /// the view of the plays of the other direction, with its extension
    /// sets of d (letters that extend a word through it, in the direction
    /// of play, see cross_set.zig), and its anchor of d in view d.
    views: [2][max_squares]Square,
    /// While set, every change to a square through the setters (and
    /// anchor updates) records the square first: see Journal.
    journal: ?*Journal = null,

    pub fn init(layout: *const Layout) Board {
        const dim = layout.rows.len;
        std.debug.assert(dim <= max_dim);
        var b: Board = .{
            .dim = dim,
            .row_mul = dim,
            .col_mul = 1,
            .tiles_played = 0,
            .squares = @splat(0),
            .bonuses = @splat(' '),
            .views = @splat(@splat(.{ .cross_set = trivial_cross_set, .left_ext = trivial_cross_set, .right_ext = trivial_cross_set, .cross_score = 0, .letter = 0, .letter_mul = 1, .word_mul = 1, .anchor = false })),
        };
        for (layout.rows, 0..) |row, r| {
            std.debug.assert(row.len == dim);
            @memcpy(b.bonuses[r * dim ..][0..dim], row);
        }
        for (0..dim) |r| {
            for (0..dim) |c| {
                const bonus_char = b.bonuses[r * dim + c];
                inline for (.{ r * dim + c, c * dim + r }, 0..) |i, d| {
                    b.views[d][i].letter_mul = letterMultiplier(bonus_char);
                    b.views[d][i].word_mul = wordMultiplier(bonus_char);
                }
            }
        }
        b.clear();
        return b;
    }

    pub fn clear(self: *Board) void {
        std.debug.assert(self.journal == null);
        @memset(&self.squares, 0);
        for (&self.views) |*v| {
            for (v) |*sq| sq.letter = 0;
        }
        self.tiles_played = 0;
        self.setAllCrosses();
        self.resetCrossScores();
        self.updateAllAnchors();
    }

    pub fn setAllCrosses(self: *Board) void {
        std.debug.assert(self.journal == null);
        for (&self.views) |*v| {
            for (v) |*sq| {
                sq.cross_set = trivial_cross_set;
                sq.left_ext = trivial_cross_set;
                sq.right_ext = trivial_cross_set;
            }
        }
    }

    pub fn resetCrossScores(self: *Board) void {
        for (&self.views) |*v| {
            for (v) |*sq| sq.cross_score = 0;
        }
    }

    pub inline fn transpose(self: *Board) void {
        std.mem.swap(usize, &self.row_mul, &self.col_mul);
    }

    pub inline fn isTransposed(self: *const Board) bool {
        return self.col_mul != 1;
    }

    /// The square's place row by row: its letter's, and its place in
    /// views[0].
    pub inline fn sqIdx(self: *const Board, row: usize, col: usize) usize {
        return row * self.row_mul + col * self.col_mul;
    }

    /// The square's place in views[1], column by column.
    pub inline fn vIdx(self: *const Board, row: usize, col: usize) usize {
        return row * self.col_mul + col * self.row_mul;
    }

    /// The square at (row, col) as the plays of direction `p` read it.
    pub inline fn view(self: *const Board, p: Direction, row: usize, col: usize) *const Square {
        return &self.views[@backingInt(p)][if (p == .horizontal) self.sqIdx(row, col) else self.vIdx(row, col)];
    }

    inline fn viewMut(self: *Board, p: Direction, row: usize, col: usize) *Square {
        const idx = self.sqIdx(row, col);
        const v_idx = self.vIdx(row, col);
        self.touch(idx, v_idx);
        return &self.views[@backingInt(p)][if (p == .horizontal) idx else v_idx];
    }

    /// The lane `lane` of the plays of direction `p` (a row for horizontal
    /// plays, a column for vertical ones), whatever the transposition.
    pub inline fn lane(self: *const Board, p: Direction, l: usize) []const Square {
        return self.views[@backingInt(p)][l * self.dim ..][0..self.dim];
    }

    pub inline fn posExists(self: *const Board, row: isize, col: isize) bool {
        const d: isize = @intCast(self.dim);
        return row >= 0 and row < d and col >= 0 and col < d;
    }

    pub inline fn getLetter(self: *const Board, row: usize, col: usize) MachineLetter {
        return self.squares[self.sqIdx(row, col)];
    }

    pub inline fn setLetter(self: *Board, row: usize, col: usize, ml: MachineLetter) void {
        const idx = self.sqIdx(row, col);
        const v_idx = self.vIdx(row, col);
        self.touch(idx, v_idx);
        self.squares[idx] = ml;
        self.views[0][idx].letter = ml;
        self.views[1][v_idx].letter = ml;
    }

    pub inline fn hasLetter(self: *const Board, row: usize, col: usize) bool {
        return self.getLetter(row, col) != 0;
    }

    /// Whether the square is off the board or empty; for signed coordinates.
    inline fn emptyOrOff(self: *const Board, row: isize, col: isize) bool {
        if (!self.posExists(row, col)) return true;
        return !self.hasLetter(@intCast(row), @intCast(col));
    }

    pub inline fn bonus(self: *const Board, row: usize, col: usize) u8 {
        return self.bonuses[self.sqIdx(row, col)];
    }

    /// The cross-set of the words across the square in `dir`: what the
    /// plays of the other direction may put there.
    pub inline fn crossSet(self: *const Board, row: usize, col: usize, dir: Direction) CrossSet {
        return self.view(dir.other(), row, col).cross_set;
    }

    pub inline fn setCrossSet(self: *Board, row: usize, col: usize, cs: CrossSet, dir: Direction) void {
        self.viewMut(dir.other(), row, col).cross_set = cs;
    }

    /// Empties a square's cross-sets and cross-scores in both directions,
    /// as a filled square's are.
    pub inline fn clearCrossSets(self: *Board, row: usize, col: usize) void {
        const idx = self.sqIdx(row, col);
        const v_idx = self.vIdx(row, col);
        self.touch(idx, v_idx);
        inline for (.{ idx, v_idx }, 0..) |i, d| {
            self.views[d][i].cross_set = 0;
            self.views[d][i].cross_score = 0;
        }
    }

    pub inline fn crossScore(self: *const Board, row: usize, col: usize, dir: Direction) i32 {
        return self.view(dir.other(), row, col).cross_score;
    }

    pub inline fn setCrossScore(self: *Board, row: usize, col: usize, score: i32, dir: Direction) void {
        self.viewMut(dir.other(), row, col).cross_score = score;
    }

    pub inline fn isAnchor(self: *const Board, row: usize, col: usize, dir: Direction) bool {
        return self.view(dir, row, col).anchor;
    }

    /// The letters a tile played alone on the square may be: those both
    /// directions' cross-sets allow.
    pub inline fn loneTileSet(self: *const Board, row: usize, col: usize) CrossSet {
        return self.views[0][self.sqIdx(row, col)].cross_set & self.views[1][self.vIdx(row, col)].cross_set;
    }

    pub inline fn letterMul(self: *const Board, row: usize, col: usize) u8 {
        return self.views[0][self.sqIdx(row, col)].letter_mul;
    }

    pub inline fn wordMul(self: *const Board, row: usize, col: usize) u8 {
        return self.views[0][self.sqIdx(row, col)].word_mul;
    }

    pub inline fn leftExtSet(self: *const Board, row: usize, col: usize, dir: Direction) u64 {
        return self.view(dir.other(), row, col).left_ext;
    }

    pub inline fn rightExtSet(self: *const Board, row: usize, col: usize, dir: Direction) u64 {
        return self.view(dir.other(), row, col).right_ext;
    }

    pub inline fn setLeftExtSet(self: *Board, row: usize, col: usize, dir: Direction, es: u64) void {
        self.viewMut(dir.other(), row, col).left_ext = es;
    }

    pub inline fn setRightExtSet(self: *Board, row: usize, col: usize, dir: Direction, es: u64) void {
        self.viewMut(dir.other(), row, col).right_ext = es;
    }

    /// Records the square (its places row by row and in views[1]) in the
    /// journal, if there is one, before its first change since the
    /// journal's last mark.
    inline fn touch(self: *Board, idx: usize, v_idx: usize) void {
        if (self.journal) |j| j.save(self, idx, v_idx);
    }

    /// Whether the squares left and right of (row, col) are empty or off
    /// the board.
    pub fn leftAndRightEmpty(self: *const Board, row: usize, col: usize) bool {
        const r: isize = @intCast(row);
        const c: isize = @intCast(col);
        return self.emptyOrOff(r, c - 1) and self.emptyOrOff(r, c + 1);
    }

    /// The last column of the run of tiles from `col` going in `step`
    /// (-1 or 1). If (row, col) is empty, that is col - step.
    pub fn wordEdge(self: *const Board, row: usize, col: isize, step: isize) isize {
        var c = col;
        const r: isize = @intCast(row);
        while (!self.emptyOrOff(r, c)) c += step;
        return c - step;
    }

    /// The sum of the tile scores from (row, col) leftward to the first
    /// empty square.
    pub fn traverseBackwardsForScore(self: *const Board, row: usize, col: isize, ld: *const LetterDistribution) i32 {
        var score: i32 = 0;
        var c = col;
        const r: isize = @intCast(row);
        while (!self.emptyOrOff(r, c)) : (c -= 1) {
            score += ld.score(self.getLetter(row, @intCast(c)));
        }
        return score;
    }

    fn updateAnchors(self: *Board, row_in: usize, col_in: usize, vertical: bool) void {
        var row = row_in;
        var col = col_in;
        if (vertical) std.mem.swap(usize, &row, &col);
        const pos = self.sqIdx(row, col);
        const v_pos = self.vIdx(row, col);
        self.touch(pos, v_pos);
        const ah = &self.views[0][pos].anchor;
        const av = &self.views[1][v_pos].anchor;
        ah.* = false;
        av.* = false;
        const tile_above = row > 0 and self.hasLetter(row - 1, col);
        const tile_left = col > 0 and self.hasLetter(row, col - 1);
        const tile_below = row < self.dim - 1 and self.hasLetter(row + 1, col);
        const tile_right = col < self.dim - 1 and self.hasLetter(row, col + 1);
        if (self.hasLetter(row, col)) {
            // A filled square anchors only as the last square of a word.
            if (!tile_right) ah.* = true;
            if (!tile_below) av.* = true;
        } else {
            if (!tile_left and !tile_right and (tile_above or tile_below)) ah.* = true;
            if (!tile_above and !tile_below and (tile_left or tile_right)) av.* = true;
        }
    }

    /// Updates the anchors of (row, col) and its neighbours: what a tile
    /// placed there can change.
    pub fn updateAnchorsAround(self: *Board, row: usize, col: usize) void {
        self.updateAnchors(row, col, false);
        if (row > 0) self.updateAnchors(row - 1, col, false);
        if (row + 1 < self.dim) self.updateAnchors(row + 1, col, false);
        if (col > 0) self.updateAnchors(row, col - 1, false);
        if (col + 1 < self.dim) self.updateAnchors(row, col + 1, false);
    }

    pub fn updateAllAnchors(self: *Board) void {
        if (self.tiles_played > 0) {
            for (0..self.dim) |i| {
                for (0..self.dim) |j| self.updateAnchors(i, j, false);
            }
        } else {
            for (&self.views) |*v| {
                for (v) |*sq| sq.anchor = false;
            }
            const rc = self.dim / 2;
            self.views[0][self.sqIdx(rc, rc)].anchor = true;
        }
    }

    /// Sets a row's tiles. Returns the number of tiles placed.
    pub fn setRow(self: *Board, row: usize, mls: []const MachineLetter) u32 {
        std.debug.assert(self.journal == null);
        std.debug.assert(mls.len == self.dim);
        var n: u32 = 0;
        for (mls, 0..) |ml, col| {
            self.setLetter(row, col, ml);
            if (ml != 0) n += 1;
        }
        return n;
    }

    /// Recounts tiles_played from the squares.
    pub fn countTiles(self: *Board) void {
        var n: u32 = 0;
        for (self.squares[0 .. self.dim * self.dim]) |ml| n += @intFromBool(ml != 0);
        self.tiles_played = n;
    }
};

// Tests ----------------------------------------------------------------------

const testing = std.testing;

test "layout and multipliers" {
    const b = Board.init(&crossword_game);
    try testing.expectEqual(15, b.dim);
    try testing.expectEqual(3, wordMultiplier(b.bonus(0, 0)));
    try testing.expectEqual(2, letterMultiplier(b.bonus(0, 3)));
    try testing.expectEqual(3, letterMultiplier(b.bonus(1, 5)));
    try testing.expectEqual(2, wordMultiplier(b.bonus(7, 7)));
    try testing.expect(b.isAnchor(7, 7, .horizontal));
    try testing.expect(!b.isAnchor(7, 7, .vertical));
    const s = Board.init(&super_crossword_game);
    try testing.expectEqual(21, s.dim);
    try testing.expectEqual(4, wordMultiplier(s.bonus(0, 0)));
}

test "transpose" {
    var b = Board.init(&crossword_game);
    b.setLetter(3, 5, 1);
    b.transpose();
    try testing.expectEqual(1, b.getLetter(5, 3));
    b.transpose();
    try testing.expectEqual(1, b.getLetter(3, 5));
}
