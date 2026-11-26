//! Moves: tile plays, exchanges and passes. Port of Macondo's `move`
//! package, minus the leave (it is the rack minus the tiles, worked out
//! when needed).

const std = @import("std");
const tiles = @import("../data/tiles.zig");
const board = @import("board.zig");
const MachineLetter = tiles.MachineLetter;

pub const Kind = enum(u8) { play, exchange, pass };

pub const Move = struct {
    kind: Kind,
    vertical: bool = false,
    row: u8 = 0,
    col: u8 = 0,
    /// Tiles along the play, 0 for a square already holding a tile; for an
    /// exchange, the tiles thrown in.
    len: u8 = 0,
    tiles_played: u8 = 0,
    score: i32 = 0,
    tiles: [board.max_dim]MachineLetter = undefined,

    pub fn pass() Move {
        return .{ .kind = .pass };
    }

    pub fn tileSlice(self: *const Move) []const MachineLetter {
        return self.tiles[0..self.len];
    }

    /// What `rack` keeps after this move: the rack less the tiles played
    /// (a designated blank comes off as a blank) or thrown in.
    pub fn leave(self: *const Move, rack: *const tiles.Rack) tiles.Rack {
        var r = rack.*;
        if (self.kind == .pass) return r;
        for (self.tileSlice()) |ml| {
            if (self.kind == .play and ml == 0) continue;
            r.take(if (tiles.isBlanked(ml)) 0 else ml);
        }
        return r;
    }

    /// The score of play `m` on board `b`, which doesn't hold its tiles yet
    /// and whose cross-scores are up to date: the main word (multipliers
    /// on the new squares only), each cross word a new tile makes, and the
    /// bingo bonus.
    pub fn scoreOn(m: *const Move, b: *const board.Board, ld: *const tiles.LetterDistribution) i32 {
        if (m.kind != .play) return 0;
        // A play along one direction makes cross words in the other.
        const dir: board.Direction = if (m.vertical) .horizontal else .vertical;
        var main: i32 = 0;
        var word_mul: i32 = 1;
        var cross: i32 = 0;
        for (m.tileSlice(), 0..) |ml, i| {
            const r = m.row + if (m.vertical) i else 0;
            const c = m.col + if (m.vertical) 0 else i;
            if (ml == 0) {
                main += ld.score(b.getLetter(r, c));
                continue;
            }
            const sq = b.sqIdx(r, c);
            const lm: i32 = b.letter_muls[sq];
            const wm: i32 = b.word_muls[sq];
            const ts = ld.score(ml) * lm;
            main += ts;
            word_mul *= wm;
            const cs = b.crossScore(r, c, dir);
            if (b.crossSet(r, c, dir) != board.trivial_cross_set) cross += (cs + ts) * wm;
        }
        const bingo: i32 = if (m.tiles_played == rack_size) bingo_bonus else 0;
        return main * word_mul + cross + bingo;
    }

    /// UCGI notation: `8D.QUIZ` (row first = horizontal, column first =
    /// vertical; `.` for a tile already on the board, lower case for a
    /// designated blank), `exch.ABC`, `pass`.
    pub fn format(self: *const Move, ld: *const tiles.LetterDistribution, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self.kind) {
            .pass => return w.writeAll("pass"),
            .exchange => {
                try w.writeAll("exch.");
                for (self.tileSlice()) |ml| try w.writeAll(ld.letter(ml, false));
            },
            .play => {
                const col_letter: u8 = 'A' + self.col;
                if (self.vertical) {
                    try w.print("{c}{d}.", .{ col_letter, @as(u32, self.row) + 1 });
                } else {
                    try w.print("{d}{c}.", .{ @as(u32, self.row) + 1, col_letter });
                }
                for (self.tileSlice()) |ml| try w.writeAll(ld.letter(ml, true));
            },
        }
    }
};

/// CGP's lm notation for `m`, made on a board `b` that shows it: through
/// tiles in parentheses (`8G.S(CAT)TER`), `ex.ABC` (or `ex.3` with
/// `hide_exchange`), `pass`.
pub fn formatLastMove(m: *const Move, b: *const board.Board, ld: *const tiles.LetterDistribution, hide_exchange: bool, w: *std.Io.Writer) std.Io.Writer.Error!void {
    switch (m.kind) {
        .pass => return w.writeAll("pass"),
        .exchange => {
            try w.writeAll("ex.");
            if (hide_exchange) return w.print("{d}", .{m.len});
            for (m.tileSlice()) |ml| try w.writeAll(ld.letter(ml, false));
        },
        .play => {
            const col_letter: u8 = 'A' + m.col;
            if (m.vertical) {
                try w.print("{c}{d}.", .{ col_letter, @as(u32, m.row) + 1 });
            } else {
                try w.print("{d}{c}.", .{ @as(u32, m.row) + 1, col_letter });
            }
            var open = false;
            for (m.tileSlice(), 0..) |ml, i| {
                if (ml == 0) {
                    if (!open) try w.writeByte('(');
                    open = true;
                    const r = m.row + if (m.vertical) i else 0;
                    const c = m.col + if (m.vertical) 0 else i;
                    try w.writeAll(ld.letter(b.getLetter(r, c), true));
                    continue;
                }
                if (open) try w.writeByte(')');
                open = false;
                try w.writeAll(ld.letter(ml, true));
            }
            if (open) try w.writeByte(')');
        },
    }
}

const rack_size = 7;
const bingo_bonus = 50;

pub const ParseError = error{ BadMove, UnknownLetter, NoSpaceLeft };

/// A move read from text, with the letters of the squares it plays
/// through where the text gives them.
pub const Parsed = struct {
    move: Move,
    /// By position along the move: the letter of a square played through
    /// (0 where the move places a tile, or where the text marks the square
    /// without its letter).
    through: [board.max_dim]MachineLetter = @splat(0),
    /// An exchange written as a number of tiles (`ex.3`): its tiles are
    /// unknown.
    tiles_unknown: bool = false,
    /// A move challenged off (`phony.8D.WORD`): read as a pass.
    phony: bool = false,
};

/// Reads a move in the notation of CGP's lm opcode or of UCGI, on a board of
/// `dim`: `8G.S(CAT)TER` (through tiles in parentheses), `8G.S...TER`
/// (through squares as dots), `G8.WORD` (column first: vertical), lowercase
/// letters for designated blanks, `ex.ABC` or `ex.3`, `pass`, and
/// `phony.<move>`.
pub fn parse(text_in: []const u8, ld: *const tiles.LetterDistribution, dim: usize) ParseError!Parsed {
    var text = text_in;
    var res: Parsed = .{ .move = .pass() };
    if (std.mem.startsWith(u8, text, "phony.")) {
        _ = try parse(text["phony.".len..], ld, dim);
        res.phony = true;
        return res;
    }
    if (std.mem.eql(u8, text, "pass")) return res;
    if (std.mem.startsWith(u8, text, "ex.") or std.mem.startsWith(u8, text, "exch.")) {
        text = text[std.mem.indexOfScalar(u8, text, '.').? + 1 ..];
        res.move = .{ .kind = .exchange };
        if (std.fmt.parseInt(u8, text, 10)) |n| {
            if (n == 0 or n > rack_size) return error.BadMove;
            res.move.len = n;
            res.move.tiles_played = n;
            @memset(res.move.tiles[0..n], 0);
            res.tiles_unknown = true;
            return res;
        } else |_| {}
        var buf: [board.max_dim]MachineLetter = undefined;
        const n = try lettersOf(text, ld, &buf);
        if (n == 0 or n > rack_size) return error.BadMove;
        for (buf[0..n]) |ml| {
            if (ml == 0 or tiles.isBlanked(ml)) return error.BadMove;
        }
        @memcpy(res.move.tiles[0..n], buf[0..n]);
        res.move.len = @intCast(n);
        res.move.tiles_played = @intCast(n);
        return res;
    }

    // The coordinate: row first is horizontal, column first vertical.
    const dot = std.mem.indexOfScalar(u8, text, '.') orelse return error.BadMove;
    const coord = text[0..dot];
    if (coord.len < 2) return error.BadMove;
    var i: usize = 0;
    const vertical = !std.ascii.isDigit(coord[0]);
    var row: usize = 0;
    var col: usize = 0;
    if (vertical) {
        col = try columnNumber(coord, &i);
        row = try rowNumber(coord, &i);
    } else {
        row = try rowNumber(coord, &i);
        col = try columnNumber(coord, &i);
    }
    if (i != coord.len or row == 0 or col == 0 or row > dim or col > dim) return error.BadMove;
    var m: Move = .{ .kind = .play, .vertical = vertical, .row = @intCast(row - 1), .col = @intCast(col - 1) };

    const word = text[dot + 1 ..];
    var n: usize = 0;
    var buf: [board.max_dim]MachineLetter = undefined;
    var pos: usize = 0;
    while (pos < word.len) {
        if (word[pos] == '(') {
            const close = std.mem.indexOfScalarPos(u8, word, pos, ')') orelse return error.BadMove;
            const k = try lettersOf(word[pos + 1 .. close], ld, &buf);
            if (k == 0) return error.BadMove;
            for (buf[0..k]) |ml| {
                if (n == board.max_dim or ml == 0) return error.BadMove;
                m.tiles[n] = 0;
                res.through[n] = ml;
                n += 1;
            }
            pos = close + 1;
            continue;
        }
        const end = std.mem.indexOfScalarPos(u8, word, pos, '(') orelse word.len;
        if (std.mem.indexOfScalar(u8, word[pos..end], ')') != null) return error.BadMove;
        const k = try lettersOf(word[pos..end], ld, &buf);
        for (buf[0..k]) |ml| {
            if (n == board.max_dim) return error.BadMove;
            m.tiles[n] = ml;
            if (ml != 0) m.tiles_played += 1;
            n += 1;
        }
        pos = end;
    }
    if (n == 0 or m.tiles_played == 0) return error.BadMove;
    const start = if (vertical) m.row else m.col;
    if (start + n > dim) return error.BadMove;
    m.len = @intCast(n);
    res.move = m;
    return res;
}

/// Tiles from text, a tile of more than one character in brackets where
/// it could be read as its letters (`[CH]`), as in CGP rows.
fn lettersOf(text: []const u8, ld: *const tiles.LetterDistribution, out: []MachineLetter) ParseError!usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '[') {
            const close = std.mem.indexOfScalarPos(u8, text, i + 1, ']') orelse return error.BadMove;
            // The letters inside as one tile, or failing that the whole
            // thing with its brackets (Spanish writes RR as [RR]), as CGP
            // rows are read.
            var one: [8]MachineLetter = undefined;
            var k: usize = ld.toMachineLetters(text[i + 1 .. close], &one) catch 0;
            if (k != 1) k = ld.toMachineLetters(text[i .. close + 1], &one) catch return error.UnknownLetter;
            if (k != 1 or n == out.len) return error.BadMove;
            out[n] = one[0];
            n += 1;
            i = close + 1;
            continue;
        }
        const end = std.mem.indexOfScalarPos(u8, text, i, '[') orelse text.len;
        n += try ld.toMachineLetters(text[i..end], out[n..]);
        i = end;
    }
    return n;
}

fn rowNumber(s: []const u8, i: *usize) ParseError!usize {
    var n: usize = 0;
    const start = i.*;
    while (i.* < s.len and std.ascii.isDigit(s[i.*])) : (i.* += 1) {
        n = n * 10 + (s[i.*] - '0');
        if (n > 1000) return error.BadMove;
    }
    if (i.* == start) return error.BadMove;
    return n;
}

/// A column in "Excel column notation" (A = 1, Z = 26, AA = 27, ...).
fn columnNumber(s: []const u8, i: *usize) ParseError!usize {
    var n: usize = 0;
    const start = i.*;
    while (i.* < s.len and std.ascii.isUpper(s[i.*])) : (i.* += 1) {
        n = n * 26 + (s[i.*] - 'A' + 1);
        if (n > 1000) return error.BadMove;
    }
    if (i.* == start) return error.BadMove;
    return n;
}

test "parse moves in CGP and UCGI notation" {
    var f = try @import("../testing/testdata.zig").Fixture.load("NWL20");
    defer f.deinit();
    const ld = &f.ld;
    const a = try parse("8G.S(CAT)TER", ld, 15);
    try std.testing.expectEqual(Kind.play, a.move.kind);
    try std.testing.expect(!a.move.vertical);
    try std.testing.expectEqual(7, a.move.row);
    try std.testing.expectEqual(6, a.move.col);
    try std.testing.expectEqual(7, a.move.len);
    try std.testing.expectEqual(4, a.move.tiles_played);
    try std.testing.expectEqual(0, a.move.tiles[1]);
    try std.testing.expect(a.through[1] != 0 and a.through[4] == 0);
    const b = try parse("8G.S...TER", ld, 15);
    try std.testing.expectEqualSlices(MachineLetter, a.move.tileSlice(), b.move.tileSlice());
    const v = try parse("H4.ZeE", ld, 15);
    try std.testing.expect(v.move.vertical);
    try std.testing.expectEqual(3, v.move.row);
    try std.testing.expectEqual(7, v.move.col);
    try std.testing.expect(tiles.isBlanked(v.move.tiles[1]));
    const e = try parse("ex.QV", ld, 15);
    try std.testing.expectEqual(Kind.exchange, e.move.kind);
    try std.testing.expectEqual(2, e.move.len);
    const e3 = try parse("ex.3", ld, 15);
    try std.testing.expect(e3.tiles_unknown and e3.move.len == 3);
    try std.testing.expectEqual(Kind.pass, (try parse("pass", ld, 15)).move.kind);
    try std.testing.expect((try parse("phony.8D.QUIZ", ld, 15)).phony);
    for ([_][]const u8{ "8P.A", "16A.A", "8G.(CAT)", "8G.S(CAT", "8G.", "X.A", "ex.8", "8G.S)A(" }) |bad| {
        if (parse(bad, ld, 15)) |_| {
            std.debug.print("parsed {s}\n", .{bad});
            return error.TestUnexpectedResult;
        } else |_| {}
    }
}
