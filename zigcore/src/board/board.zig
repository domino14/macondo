//! The game board: tiles, bonus squares, anchors, cross-sets, cross-scores
//! and the extension sets shadow move generation uses. Port of Macondo's
//! `board` package.
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

const PerDir = [2][max_squares]u64;

/// Everything a square holds that a move can change.
pub const SavedSquare = struct {
    idx: u16,
    letter: MachineLetter,
    anchors: [2]bool,
    cross_sets: [2]u64,
    cross_scores: [2]i32,
    left_ext: [2]u64,
    right_ext: [2]u64,
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

    inline fn save(self: *Journal, b: *const Board, idx: usize) void {
        if (self.stamp[idx] == self.serial) return;
        self.stamp[idx] = self.serial;
        self.saved.appendAssumeCapacity(.{
            .idx = @intCast(idx),
            .letter = b.squares[idx],
            .anchors = .{ b.anchors[0][idx], b.anchors[1][idx] },
            .cross_sets = .{ b.cross_sets[0][idx], b.cross_sets[1][idx] },
            .cross_scores = .{ b.cross_scores[0][idx], b.cross_scores[1][idx] },
            .left_ext = .{ b.left_ext[0][idx], b.left_ext[1][idx] },
            .right_ext = .{ b.right_ext[0][idx], b.right_ext[1][idx] },
        });
    }

    /// Puts back every square saved since `start` and forgets them; the
    /// squares can then be saved again.
    pub fn restore(self: *Journal, b: *Board, start: usize) void {
        for (self.saved.items[start..]) |sq| {
            b.squares[sq.idx] = sq.letter;
            inline for (0..2) |d| {
                b.anchors[d][sq.idx] = sq.anchors[d];
                b.cross_sets[d][sq.idx] = sq.cross_sets[d];
                b.cross_scores[d][sq.idx] = sq.cross_scores[d];
                b.left_ext[d][sq.idx] = sq.left_ext[d];
                b.right_ext[d][sq.idx] = sq.right_ext[d];
            }
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
    squares: [max_squares]MachineLetter,
    bonuses: [max_squares]u8,
    /// Each square's letter and word multipliers, from its bonus.
    letter_muls: [max_squares]u8,
    word_muls: [max_squares]u8,
    cross_sets: [2][max_squares]CrossSet,
    cross_scores: [2][max_squares]i32,
    anchors: [2][max_squares]bool,
    /// Letters that can extend a word leftward / rightward through a square,
    /// in the direction of play (see cross_set.zig).
    left_ext: PerDir,
    right_ext: PerDir,
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
            .letter_muls = @splat(1),
            .word_muls = @splat(1),
            .cross_sets = undefined,
            .cross_scores = undefined,
            .anchors = undefined,
            .left_ext = undefined,
            .right_ext = undefined,
        };
        for (layout.rows, 0..) |row, r| {
            std.debug.assert(row.len == dim);
            @memcpy(b.bonuses[r * dim ..][0..dim], row);
        }
        for (b.bonuses[0 .. dim * dim], 0..) |bonus_char, i| {
            b.letter_muls[i] = letterMultiplier(bonus_char);
            b.word_muls[i] = wordMultiplier(bonus_char);
        }
        b.clear();
        return b;
    }

    pub fn clear(self: *Board) void {
        std.debug.assert(self.journal == null);
        @memset(&self.squares, 0);
        self.tiles_played = 0;
        self.setAllCrosses();
        self.resetCrossScores();
        self.updateAllAnchors();
    }

    pub fn setAllCrosses(self: *Board) void {
        std.debug.assert(self.journal == null);
        for (0..2) |d| {
            @memset(&self.cross_sets[d], trivial_cross_set);
            @memset(&self.left_ext[d], trivial_cross_set);
            @memset(&self.right_ext[d], trivial_cross_set);
        }
    }

    pub fn resetCrossScores(self: *Board) void {
        for (0..2) |d| @memset(&self.cross_scores[d], 0);
    }

    pub inline fn transpose(self: *Board) void {
        std.mem.swap(usize, &self.row_mul, &self.col_mul);
    }

    pub inline fn isTransposed(self: *const Board) bool {
        return self.col_mul != 1;
    }

    pub inline fn sqIdx(self: *const Board, row: usize, col: usize) usize {
        return row * self.row_mul + col * self.col_mul;
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
        self.touch(idx);
        self.squares[idx] = ml;
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

    pub inline fn crossSet(self: *const Board, row: usize, col: usize, dir: Direction) CrossSet {
        return self.cross_sets[@backingInt(dir)][self.sqIdx(row, col)];
    }

    pub inline fn setCrossSet(self: *Board, row: usize, col: usize, cs: CrossSet, dir: Direction) void {
        const idx = self.sqIdx(row, col);
        self.touch(idx);
        self.cross_sets[@backingInt(dir)][idx] = cs;
    }

    pub inline fn addToCrossSet(self: *Board, row: usize, col: usize, ml: MachineLetter, dir: Direction) void {
        const idx = self.sqIdx(row, col);
        self.touch(idx);
        self.cross_sets[@backingInt(dir)][idx] |= @as(CrossSet, 1) << @intCast(ml);
    }

    pub inline fn crossScore(self: *const Board, row: usize, col: usize, dir: Direction) i32 {
        return self.cross_scores[@backingInt(dir)][self.sqIdx(row, col)];
    }

    pub inline fn setCrossScore(self: *Board, row: usize, col: usize, score: i32, dir: Direction) void {
        const idx = self.sqIdx(row, col);
        self.touch(idx);
        self.cross_scores[@backingInt(dir)][idx] = score;
    }

    pub inline fn isAnchor(self: *const Board, row: usize, col: usize, dir: Direction) bool {
        return self.anchors[@backingInt(dir)][self.sqIdx(row, col)];
    }

    pub inline fn leftExtSet(self: *const Board, row: usize, col: usize, dir: Direction) u64 {
        return self.left_ext[@backingInt(dir)][self.sqIdx(row, col)];
    }

    pub inline fn rightExtSet(self: *const Board, row: usize, col: usize, dir: Direction) u64 {
        return self.right_ext[@backingInt(dir)][self.sqIdx(row, col)];
    }

    pub inline fn setLeftExtSet(self: *Board, row: usize, col: usize, dir: Direction, es: u64) void {
        const idx = self.sqIdx(row, col);
        self.touch(idx);
        self.left_ext[@backingInt(dir)][idx] = es;
    }

    pub inline fn setRightExtSet(self: *Board, row: usize, col: usize, dir: Direction, es: u64) void {
        const idx = self.sqIdx(row, col);
        self.touch(idx);
        self.right_ext[@backingInt(dir)][idx] = es;
    }

    /// Records square `idx` in the journal, if there is one, before its
    /// first change since the journal's last mark.
    inline fn touch(self: *Board, idx: usize) void {
        if (self.journal) |j| j.save(self, idx);
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
        const h = @backingInt(Direction.horizontal);
        const v = @backingInt(Direction.vertical);
        self.touch(pos);
        self.anchors[h][pos] = false;
        self.anchors[v][pos] = false;
        const tile_above = row > 0 and self.hasLetter(row - 1, col);
        const tile_left = col > 0 and self.hasLetter(row, col - 1);
        const tile_below = row < self.dim - 1 and self.hasLetter(row + 1, col);
        const tile_right = col < self.dim - 1 and self.hasLetter(row, col + 1);
        if (self.hasLetter(row, col)) {
            // A filled square anchors only as the last square of a word.
            if (!tile_right) self.anchors[h][pos] = true;
            if (!tile_below) self.anchors[v][pos] = true;
        } else {
            if (!tile_left and !tile_right and (tile_above or tile_below)) self.anchors[h][pos] = true;
            if (!tile_above and !tile_below and (tile_left or tile_right)) self.anchors[v][pos] = true;
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
            for (0..2) |d| @memset(&self.anchors[d], false);
            const rc = self.dim / 2;
            self.anchors[@backingInt(Direction.horizontal)][self.sqIdx(rc, rc)] = true;
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
