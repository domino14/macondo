//! Cross-sets, cross-scores and extension sets from a GADDAG. Port of
//! Macondo's `cross_set` package (GaddagCrossSetGenerator).
//!
//! The write order matters: extension sets are written by the empty squares
//! next to a word, and a later square can overwrite what an earlier one
//! wrote. generateAll visits squares in the same order as the Go code so the
//! results match it exactly.

const std = @import("std");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const Kwg = @import("kwg.zig").Kwg;
const Board = board_mod.Board;
const Direction = board_mod.Direction;
const LetterDistribution = tiles.LetterDistribution;
const MachineLetter = tiles.MachineLetter;

/// Generates every cross-set on the board, horizontal then vertical (on the
/// transposed board).
pub fn generateAll(b: *Board, kwg: Kwg, ld: *const LetterDistribution) void {
    const n = b.dim;
    for (0..n) |i| {
        for (0..n) |j| generate(b, @intCast(i), @intCast(j), .horizontal, kwg, ld);
    }
    b.transpose();
    for (0..n) |i| {
        for (0..n) |j| generate(b, @intCast(i), @intCast(j), .vertical, kwg, ld);
    }
    b.transpose();
}

/// Brings cross-sets, extension sets and anchors up to date after `m`'s
/// tiles were put on the board (whose anchors were up to date before),
/// giving the same result as generateAll and updateAllAnchors. Port of
/// Go's updateForMove (MAGPIE's update_cross_set_for_move): only the
/// squares whose words changed are worked out again. Along the play, its
/// own squares and one on each side; across it, for each new tile, its
/// square and the squares just past each end of the word it makes.
/// Anchors look only at a square's neighbours.
pub fn updateForPlay(b: *Board, m: *const @import("move.zig").Move, kwg: Kwg, ld: *const LetterDistribution, was_empty: bool, cache: ?*Cache) void {
    const along: Direction = if (m.vertical) .vertical else .horizontal;
    const lane: isize = if (m.vertical) m.col else m.row;
    const start: isize = if (m.vertical) m.row else m.col;
    const len: isize = m.len;

    if (along == .vertical) b.transpose();
    var col = start - 1;
    while (col <= start + len) : (col += 1) generateCached(b, lane, col, along, kwg, ld, cache);
    if (along == .vertical) b.transpose();

    const across = along.other();
    if (across == .vertical) b.transpose();
    for (m.tileSlice(), 0..) |ml, i| {
        if (ml == 0) continue;
        const row: usize = @intCast(start + @as(isize, @intCast(i)));
        const right = b.wordEdge(row, lane, 1);
        const left = b.wordEdge(row, lane, -1);
        generateCached(b, @intCast(row), right + 1, across, kwg, ld, cache);
        generateCached(b, @intCast(row), left - 1, across, kwg, ld, cache);
        generateCached(b, @intCast(row), lane, across, kwg, ld, cache);
    }
    if (across == .vertical) b.transpose();

    if (was_empty) return b.updateAllAnchors();
    for (m.tileSlice(), 0..) |ml, i| {
        if (ml == 0) continue;
        const r = m.row + if (m.vertical) i else 0;
        const c = m.col + if (m.vertical) 0 else i;
        b.updateAnchorsAround(r, c);
    }
}

/// The result of walking the tiles leftward from a square.
const Traversal = struct { node: u32, valid: bool };

/// Walks the tiles from (row, col) leftward through the GADDAG, starting at
/// `node`. With `check_letter_set`, stops at `left_most_col` and reports
/// whether its letter ends a word there instead.
fn traverseBackwards(b: *const Board, row: usize, col_in: isize, node_in: u32, check_letter_set: bool, left_most_col: isize, kwg: Kwg) Traversal {
    var col = col_in;
    var node = node_in;
    while (b.posExists(@intCast(row), col)) {
        const ml = b.getLetter(row, @intCast(col));
        if (ml == 0) break;
        if (check_letter_set and col == left_most_col) {
            return .{ .node = node, .valid = kwg.inLetterSet(ml, node) };
        }
        node = kwg.nextNodeIdx(node, tiles.unblank(ml));
        // No path: a phony on the board, or a word with no extensions.
        if (node == 0) return .{ .node = 0, .valid = false };
        col -= 1;
    }
    return .{ .node = node, .valid = true };
}

/// Adds the blank (bit 0) to a non-empty extension set.
inline fn withBlank(es: u64) u64 {
    return if (es != 0) es | 1 else 0;
}

/// Generates the cross-set, cross-score and nearby extension sets of one
/// square, for plays in direction `dir` (the board may be transposed so
/// that `dir`'s cross words run along rows). Out-of-range squares are
/// ignored.
pub fn generate(b: *Board, row_s: isize, col_s: isize, dir: Direction, kwg: Kwg, ld: *const LetterDistribution) void {
    generateCached(b, row_s, col_s, dir, kwg, ld, null);
}

/// What generate works out for an empty square with tiles on at least
/// one side: its cross-set and cross-score, and the extension sets of the
/// words on its left and right (with the blank where not empty).
const Values = struct {
    cross_set: u64 = 0,
    leftside_leftx: u64 = 0,
    leftside_rightx: u64 = 0,
    rightside_leftx: u64 = 0,
    rightside_rightx: u64 = 0,
    score: i32 = 0,
};

/// A memory of Values by the words beside the square (MAGPIE's
/// CrossSetCache): they depend on nothing else, and the games of a sim or
/// an autoplay run meet the same words again and again. Direct-mapped by
/// the top bits of a 64-bit key, each 64-byte entry holding the whole key.
pub const Cache = struct {
    entries: []Entry,
    /// 64 less the index bits: the key's top bits pick the entry.
    shift: u6,

    const Entry = extern struct {
        key: u64 align(64) = 0,
        cross_set: u64 = 0,
        leftside_leftx: u64 = 0,
        leftside_rightx: u64 = 0,
        rightside_leftx: u64 = 0,
        rightside_rightx: u64 = 0,
        score: i32 = 0,
    };

    /// MAGPIE's size on machines with memory to spare: 2^14 entries, 1 MB.
    pub const default_index_bits = 14;

    pub fn init(gpa: std.mem.Allocator, index_bits: u6) std.mem.Allocator.Error!Cache {
        std.debug.assert(index_bits >= 1);
        const entries = try gpa.alloc(Entry, @as(usize, 1) << index_bits);
        @memset(entries, .{});
        return .{ .entries = entries, .shift = @intCast(@as(u7, 64) - index_bits) };
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        gpa.free(self.entries);
        self.* = undefined;
    }

    fn entry(self: *Cache, k: u64) *Entry {
        return &self.entries[k >> self.shift];
    }

    /// The key of the words from left_col to right_col beside the empty
    /// square at col, with the lexicon: a multiply-add per letter (blank
    /// flag included), a separator for the square, then MurmurHash3's
    /// finalizer. Never 0, which marks an empty entry.
    fn key(b: *const Board, row: usize, col: usize, left_col: usize, right_col: usize, kwg: Kwg) u64 {
        const multiplier: u64 = 0x9E3779B97F4A7C15;
        var k: u64 = @intFromPtr(kwg.nodes.ptr);
        for (left_col..right_col + 1) |c| {
            const symbol: u64 = if (c == col) 0x101 else @as(u64, b.getLetter(row, c)) + 1;
            k = k *% multiplier +% symbol;
        }
        k ^= k >> 33;
        k *%= 0xFF51AFD7ED558CCD;
        k ^= k >> 33;
        k *%= 0xC4CEB9FE1A85EC53;
        k ^= k >> 33;
        return k | 1;
    }
};

/// generate, taking the square's values from `cache` (if any) when it
/// has seen the words beside it.
pub fn generateCached(b: *Board, row_s: isize, col_s: isize, dir: Direction, kwg: Kwg, ld: *const LetterDistribution, cache: ?*Cache) void {
    if (!b.posExists(row_s, col_s)) return;
    const row: usize = @intCast(row_s);
    const col: usize = @intCast(col_s);
    // Extension sets are kept for the direction of play through the square.
    const t_dir = dir.other();

    // A filled square has no cross-set. Its extension sets are written by
    // the empty squares next to its word.
    if (b.hasLetter(row, col)) {
        b.setCrossSet(row, col, 0, dir);
        b.setCrossScore(row, col, 0, dir);
        return;
    }
    if (b.leftAndRightEmpty(row, col)) {
        b.setCrossSet(row, col, board_mod.trivial_cross_set, dir);
        b.setCrossScore(row, col, 0, dir);
        b.setLeftExtSet(row, col, t_dir, board_mod.trivial_cross_set);
        b.setRightExtSet(row, col, t_dir, board_mod.trivial_cross_set);
        return;
    }

    const right_col = b.wordEdge(row, col_s + 1, 1);
    const left_col = b.wordEdge(row, col_s - 1, -1);
    const nonempty_left = left_col < col_s;
    const nonempty_right = right_col > col_s;

    var v: Values = undefined;
    if (cache) |c| {
        const k = Cache.key(b, row, col, @intCast(left_col), @intCast(right_col), kwg);
        const e = c.entry(k);
        if (e.key == k) {
            v = .{
                .cross_set = e.cross_set,
                .leftside_leftx = e.leftside_leftx,
                .leftside_rightx = e.leftside_rightx,
                .rightside_leftx = e.rightside_leftx,
                .rightside_rightx = e.rightside_rightx,
                .score = e.score,
            };
        } else {
            v = computeValues(b, row, col_s, left_col, right_col, kwg, ld);
            e.* = .{
                .key = k,
                .cross_set = v.cross_set,
                .leftside_leftx = v.leftside_leftx,
                .leftside_rightx = v.leftside_rightx,
                .rightside_leftx = v.rightside_leftx,
                .rightside_rightx = v.rightside_rightx,
                .score = v.score,
            };
        }
    } else {
        v = computeValues(b, row, col_s, left_col, right_col, kwg, ld);
    }

    if (nonempty_left) {
        // On the last tile of the word to the left...
        b.setLeftExtSet(row, col - 1, t_dir, v.leftside_leftx);
        b.setRightExtSet(row, col - 1, t_dir, v.leftside_rightx);
        // ...and on the square before that word.
        if (left_col > 0) b.setLeftExtSet(row, @intCast(left_col - 1), t_dir, v.leftside_leftx);
    }
    if (nonempty_right) {
        // On the last tile of the word to the right (as in MAGPIE), and on
        // this square.
        b.setLeftExtSet(row, @intCast(right_col), t_dir, v.rightside_leftx);
        b.setRightExtSet(row, @intCast(right_col), t_dir, v.rightside_rightx);
        b.setLeftExtSet(row, col, t_dir, v.rightside_leftx);
    }
    b.setCrossScore(row, col, v.score, dir);
    b.setCrossSet(row, col, v.cross_set, dir);
}

/// The values of the empty square at col, with words from left_col to
/// col - 1 and from col + 1 to right_col (either may be empty).
fn computeValues(b: *const Board, row: usize, col_s: isize, left_col: isize, right_col: isize, kwg: Kwg, ld: *const LetterDistribution) Values {
    const nonempty_left = left_col < col_s;
    const nonempty_right = right_col > col_s;
    var v: Values = .{};
    var left: Traversal = .{ .node = 0, .valid = false };
    var right: Traversal = .{ .node = 0, .valid = false };
    var leftside_rightx: u64 = 0;

    if (nonempty_left) {
        left = traverseBackwards(b, row, col_s - 1, kwg.gaddagRoot(), false, 0, kwg);
        v.score += b.traverseBackwardsForScore(row, col_s - 1, ld);
        var leftside_leftx: u64 = 0;
        if (left.valid) {
            leftside_leftx = kwg.letterSets(left.node).extensions;
            const s = kwg.nextNodeIdx(left.node, 0);
            if (s != 0) leftside_rightx = kwg.letterSets(s).extensions;
        }
        v.leftside_leftx = withBlank(leftside_leftx);
        v.leftside_rightx = withBlank(leftside_rightx);
    }
    if (nonempty_right) {
        right = traverseBackwards(b, row, right_col, kwg.gaddagRoot(), false, 0, kwg);
        v.score += b.traverseBackwardsForScore(row, right_col, ld);
        var rightside_leftx: u64 = 0;
        var rightside_rightx: u64 = 0;
        if (right.valid) {
            rightside_leftx = kwg.letterSets(right.node).extensions;
            const s = kwg.nextNodeIdx(right.node, 0);
            if (s != 0) rightside_rightx = kwg.letterSets(s).extensions;
        }
        v.rightside_leftx = withBlank(rightside_leftx);
        v.rightside_rightx = withBlank(rightside_rightx);
    }

    if (nonempty_left and nonempty_right) {
        if (!left.valid or !right.valid) return v;
        // Try each letter that continues the right-hand word leftward and
        // can also follow the left-hand word, and check it joins them.
        var i = right.node;
        while (true) : (i += 1) {
            const t = kwg.tile(i);
            if (t != 0 and leftside_rightx & (@as(u64, 1) << @intCast(t)) != 0) {
                const joined = traverseBackwards(b, row, col_s - 1, kwg.arcIndex(i), true, left_col, kwg);
                if (joined.valid) v.cross_set |= @as(u64, 1) << @intCast(t);
            }
            if (kwg.isEnd(i)) break;
        }
    } else if (nonempty_left) {
        if (left.valid) v.cross_set = kwg.letterSet(kwg.nextNodeIdx(left.node, 0));
    } else {
        if (right.valid) v.cross_set = kwg.letterSet(right.node);
    }
    return v;
}

// Tests ----------------------------------------------------------------------
// Ported from cross_set/cross_set_test.go.

const testing = std.testing;
const testdata = @import("testdata.zig");
const CrossSet = board_mod.CrossSet;

const Fixture = testdata.Fixture;
const vs_ed = testdata.vs_ed;
const vs_matt = testdata.vs_matt;

const Case = struct { row: usize, col: usize, set: CrossSet, dir: Direction, score: i32 };

fn expectCase(b: *const Board, tc: Case) !void {
    testing.expectEqual(tc.set, b.crossSet(tc.row, tc.col, tc.dir)) catch |err| {
        std.debug.print("cross-set at {d},{d} {t}\n", .{ tc.row, tc.col, tc.dir });
        return err;
    };
    testing.expectEqual(tc.score, b.crossScore(tc.row, tc.col, tc.dir)) catch |err| {
        std.debug.print("cross-score at {d},{d} {t}\n", .{ tc.row, tc.col, tc.dir });
        return err;
    };
}

test "gen cross-set, loaded game" {
    var f = try Fixture.load("NWL18");
    defer f.deinit();
    var b = try f.boardFromFen(vs_matt);
    const cases = [_]Case{
        .{ .row = 10, .col = 10, .set = f.cs("E"), .dir = .horizontal, .score = 11 },
        .{ .row = 2, .col = 4, .set = f.cs("DKHLRSV"), .dir = .horizontal, .score = 9 },
        .{ .row = 8, .col = 7, .set = f.cs("S"), .dir = .horizontal, .score = 11 },
        // suffix, no hooks
        .{ .row = 12, .col = 8, .set = 0, .dir = .horizontal, .score = 11 },
        // prefix, no hooks
        .{ .row = 3, .col = 1, .set = 0, .dir = .horizontal, .score = 10 },
        // prefix and suffix, no path
        .{ .row = 6, .col = 8, .set = 0, .dir = .horizontal, .score = 5 },
        // in between
        .{ .row = 2, .col = 10, .set = f.cs("M"), .dir = .horizontal, .score = 2 },
    };
    for (cases) |tc| {
        generate(&b, @intCast(tc.row), @intCast(tc.col), tc.dir, f.kwg, &f.ld);
        try expectCase(&b, tc);
    }
}

test "gen cross-set, edges and phonies" {
    var f = try Fixture.load("NWL20");
    defer f.deinit();
    var b = Board.init(&board_mod.crossword_game);
    const Edge = struct { col: usize, row: []const u8, set: []const u8, score: i32 };
    const cases = [_]Edge{
        .{ .col = 0, .row = " A", .set = "ABDFHKLMNPTYZ", .score = 1 },
        .{ .col = 1, .row = "A", .set = "ABDEGHILMNRSTWXY", .score = 1 },
        .{ .col = 1, .row = "R VOTED", .set = "E", .score = 10 },
        .{ .col = 1, .row = "T UNFOLD", .set = "", .score = 11 },
        .{ .col = 7, .row = "      A VODOU I", .set = "", .score = 10 },
        .{ .col = 2, .row = "PA ABLE", .set = "RY", .score = 10 },
        .{ .col = 1, .row = "W SIWYG", .set = "Y", .score = 16 },
        .{ .col = 13, .row = "              F", .set = "EIO", .score = 4 },
        .{ .col = 14, .row = "             F ", .set = "AE", .score = 4 },
        .{ .col = 14, .row = "          WECH ", .set = "T", .score = 12 }, // phony
        .{ .col = 14, .row = "           ZZZ ", .set = "", .score = 30 },
        .{ .col = 14, .row = "       ZYZZYVA ", .set = "S", .score = 43 },
        .{ .col = 14, .row = "        ZYZZYV ", .set = "A", .score = 42 }, // phony
        .{ .col = 14, .row = "       Z Z Y A ", .set = "ABDEGHILMNRSTWXY", .score = 1 },
        .{ .col = 12, .row = "       z z Y A ", .set = "E", .score = 5 },
        .{ .col = 14, .row = "OxYpHeNbUTAzON ", .set = "E", .score = 15 },
        .{ .col = 6, .row = "OXYPHE BUTAZONE", .set = "N", .score = 40 },
        // The score is right even with no path in the GADDAG.
        .{ .col = 0, .row = " YHJKTKHKTLV", .set = "", .score = 42 },
        .{ .col = 14, .row = "   YHJKTKHKTLV ", .set = "", .score = 42 },
        .{ .col = 6, .row = "YHJKTK HKTLV", .set = "", .score = 42 },
    };
    const row = 4;
    for (cases) |tc| {
        try f.setRowText(&b, row, tc.row);
        generate(&b, row, @intCast(tc.col), .horizontal, f.kwg, &f.ld);
        try expectCase(&b, .{ .row = row, .col = tc.col, .set = f.cs(tc.set), .dir = .horizontal, .score = tc.score });
    }
}

test "gen all cross-sets" {
    var f = try Fixture.load("NWL18");
    defer f.deinit();
    const trivial = board_mod.trivial_cross_set;

    var b = try f.boardFromFen(vs_ed);
    generateAll(&b, f.kwg, &f.ld);
    for ([_]Case{
        .{ .row = 8, .col = 8, .set = f.cs("OS"), .dir = .horizontal, .score = 8 },
        .{ .row = 8, .col = 8, .set = f.cs("S"), .dir = .vertical, .score = 9 },
        .{ .row = 5, .col = 11, .set = f.cs("S"), .dir = .horizontal, .score = 5 },
        .{ .row = 5, .col = 11, .set = f.cs("AO"), .dir = .vertical, .score = 2 },
        .{ .row = 8, .col = 13, .set = f.cs("AEOU"), .dir = .horizontal, .score = 1 },
        .{ .row = 8, .col = 13, .set = f.cs("AEIMOUY"), .dir = .vertical, .score = 3 },
        .{ .row = 9, .col = 13, .set = f.cs("HMNPST"), .dir = .horizontal, .score = 1 },
        .{ .row = 9, .col = 13, .set = trivial, .dir = .vertical, .score = 0 },
        .{ .row = 14, .col = 14, .set = trivial, .dir = .horizontal, .score = 0 },
        .{ .row = 14, .col = 14, .set = trivial, .dir = .vertical, .score = 0 },
        .{ .row = 12, .col = 12, .set = 0, .dir = .horizontal, .score = 0 },
        .{ .row = 12, .col = 12, .set = 0, .dir = .vertical, .score = 0 },
    }) |tc| try expectCase(&b, tc);

    // More in-between (left and right) cross-sets.
    b = try f.boardFromFen(vs_matt);
    generateAll(&b, f.kwg, &f.ld);
    for ([_]Case{
        .{ .row = 8, .col = 7, .set = f.cs("S"), .dir = .horizontal, .score = 11 },
        .{ .row = 8, .col = 7, .set = 0, .dir = .vertical, .score = 12 },
        .{ .row = 5, .col = 11, .set = f.cs("BGOPRTWX"), .dir = .horizontal, .score = 2 },
        .{ .row = 5, .col = 11, .set = 0, .dir = .vertical, .score = 15 },
        .{ .row = 8, .col = 13, .set = trivial, .dir = .horizontal, .score = 0 },
        .{ .row = 8, .col = 13, .set = trivial, .dir = .vertical, .score = 0 },
        .{ .row = 11, .col = 4, .set = f.cs("DRS"), .dir = .horizontal, .score = 6 },
        .{ .row = 11, .col = 4, .set = f.cs("CGM"), .dir = .vertical, .score = 1 },
        .{ .row = 2, .col = 2, .set = trivial, .dir = .horizontal, .score = 0 },
        .{ .row = 2, .col = 2, .set = f.cs("AEI"), .dir = .vertical, .score = 2 },
        .{ .row = 7, .col = 12, .set = f.cs("AEIOY"), .dir = .horizontal, .score = 0 }, // a blank
        .{ .row = 7, .col = 12, .set = trivial, .dir = .vertical, .score = 0 },
        .{ .row = 11, .col = 8, .set = 0, .dir = .horizontal, .score = 4 },
        .{ .row = 11, .col = 8, .set = f.cs("AEOU"), .dir = .vertical, .score = 1 },
        .{ .row = 1, .col = 8, .set = f.cs("AEO"), .dir = .horizontal, .score = 1 },
        .{ .row = 1, .col = 8, .set = f.cs("DFHLMNRSTWX"), .dir = .vertical, .score = 1 },
        .{ .row = 10, .col = 10, .set = f.cs("E"), .dir = .horizontal, .score = 11 },
        .{ .row = 10, .col = 10, .set = trivial, .dir = .vertical, .score = 0 },
    }) |tc| try expectCase(&b, tc);
}
