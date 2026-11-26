//! CGP (Compact Game Position) parsing. Port of Macondo's `cgp` package.
//!
//!     <board rows, / separated> <rack>/<rack> <score>/<score> <scoreless turns> [<op> <arg>; ...]
//!
//! Rows use FEN-style digits for runs of empty squares, lower case for
//! designated blanks, and [..] around a tile written with several letters.
//! Parsing happens in two steps: `parse` splits the text and reads the
//! options (which name the letter distribution), then `decodeRow` turns
//! rows into tiles with that distribution.

const std = @import("std");
const tiles = @import("tiles.zig");
const board = @import("board.zig");
const MachineLetter = tiles.MachineLetter;

pub const Error = error{
    TooFewFields,
    NotTwoPlayers,
    BadNumber,
    BadOption,
    TooManyRows,
};

pub const Cgp = struct {
    /// Slices of the parsed text.
    rows: [board.max_dim][]const u8,
    num_rows: usize,
    racks: [2][]const u8,
    scores: [2]i32,
    scoreless_turns: u32,
    lexicon: []const u8 = "NWL23",
    letter_distribution: []const u8 = "english",
    board_layout: []const u8 = "CrosswordGame",
    variant: []const u8 = "classic",
    max_scoreless_turns: u32 = 6,
    gid: []const u8 = "",
    /// The lm opcode: the move that led to the position, and its score if
    /// given.
    last_move: []const u8 = "",
    last_move_score: ?i32 = null,

    pub fn parse(text: []const u8) Error!Cgp {
        var fields: [5][]const u8 = undefined;
        var nf: usize = 0;
        var rest = std.mem.trim(u8, text, " \t\r\n");
        while (nf < 4) : (nf += 1) {
            const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse {
                fields[nf] = rest;
                nf += 1;
                rest = "";
                break;
            };
            fields[nf] = rest[0..sp];
            rest = rest[sp + 1 ..];
        }
        if (nf < 4) return error.TooFewFields;
        const ops = rest;

        var c: Cgp = .{
            .rows = undefined,
            .num_rows = 0,
            .racks = undefined,
            .scores = undefined,
            .scoreless_turns = 0,
        };
        var rows = std.mem.splitScalar(u8, fields[0], '/');
        while (rows.next()) |r| {
            if (c.num_rows == board.max_dim) return error.TooManyRows;
            c.rows[c.num_rows] = r;
            c.num_rows += 1;
        }
        if (!splitPair(fields[1], &c.racks)) return error.NotTwoPlayers;
        var score_text: [2][]const u8 = undefined;
        if (!splitPair(fields[2], &score_text)) return error.NotTwoPlayers;
        for (score_text, 0..) |s, i| c.scores[i] = std.fmt.parseInt(i32, s, 10) catch return error.BadNumber;
        c.scoreless_turns = std.fmt.parseInt(u32, fields[3], 10) catch return error.BadNumber;

        var it = std.mem.splitScalar(u8, ops, ';');
        while (it.next()) |raw| {
            const op = std.mem.trim(u8, raw, " ");
            if (op.len == 0) continue;
            const sp = std.mem.indexOfScalar(u8, op, ' ');
            const name = if (sp) |i| op[0..i] else op;
            const arg = if (sp) |i| op[i + 1 ..] else "";
            if (std.mem.eql(u8, name, "bdn")) {
                c.board_layout = try required(arg);
            } else if (std.mem.eql(u8, name, "gid")) {
                c.gid = try required(arg);
            } else if (std.mem.eql(u8, name, "ld")) {
                c.letter_distribution = try required(arg);
            } else if (std.mem.eql(u8, name, "lex")) {
                c.lexicon = try required(arg);
            } else if (std.mem.eql(u8, name, "mcnz")) {
                c.max_scoreless_turns = std.fmt.parseInt(u32, try required(arg), 10) catch return error.BadNumber;
            } else if (std.mem.eql(u8, name, "var")) {
                c.variant = try required(arg);
            } else if (std.mem.eql(u8, name, "lm")) {
                const a = try required(arg);
                var parts = std.mem.tokenizeScalar(u8, a, ' ');
                c.last_move = parts.next().?;
                if (parts.next()) |sc| c.last_move_score = std.fmt.parseInt(i32, sc, 10) catch return error.BadNumber;
                if (parts.next() != null) return error.BadOption;
            }
            // Other opcodes (tmr, ...) are ignored.
        }
        return c;
    }

    fn required(arg: []const u8) Error![]const u8 {
        if (arg.len == 0) return error.BadOption;
        return arg;
    }

    fn splitPair(s: []const u8, out: *[2][]const u8) bool {
        const slash = std.mem.indexOfScalar(u8, s, '/') orelse return false;
        if (std.mem.indexOfScalarPos(u8, s, slash + 1, '/') != null) return false;
        out[0] = s[0..slash];
        out[1] = s[slash + 1 ..];
        return true;
    }
};

/// Writes a CGP's first four fields: `b`'s rows, the racks and scores (the
/// player to move first) and the turns without a score. Tiles of more than
/// one character are bracketed.
pub fn writeFields(b: *const board.Board, ld: *const tiles.LetterDistribution, racks: [2]tiles.Rack, scores: [2]i32, scoreless_turns: u32, w: *std.Io.Writer) std.Io.Writer.Error!void {
    for (0..b.dim) |r| {
        if (r > 0) try w.writeByte('/');
        var empties: usize = 0;
        for (0..b.dim) |c| {
            const ml = b.getLetter(r, c);
            if (ml == 0) {
                empties += 1;
                continue;
            }
            if (empties > 0) try w.print("{d}", .{empties});
            empties = 0;
            const text = ld.letter(ml, true);
            const multi = (std.unicode.utf8CountCodepoints(text) catch 1) > 1;
            if (multi) try w.writeByte('[');
            try w.writeAll(text);
            if (multi) try w.writeByte(']');
        }
        if (empties > 0) try w.print("{d}", .{empties});
    }
    try w.writeByte(' ');
    for (racks, 0..) |rack, i| {
        if (i > 0) try w.writeByte('/');
        for (0..ld.size) |ml| {
            for (0..rack.counts[ml]) |_| try w.writeAll(ld.letter(@intCast(ml), false));
        }
    }
    try w.print(" {d}/{d} {d}", .{ scores[0], scores[1], scoreless_turns });
}

pub const RowError = error{ UnknownLetter, BadRow, NoSpaceLeft };

/// Decodes one CGP row into `out` and returns the number of squares.
pub fn decodeRow(row: []const u8, ld: *const tiles.LetterDistribution, out: []MachineLetter) RowError!usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < row.len) {
        const ch = row[i];
        if (ch >= '0' and ch <= '9') {
            var j = i;
            while (j < row.len and row[j] >= '0' and row[j] <= '9') j += 1;
            const empties = std.fmt.parseInt(usize, row[i..j], 10) catch return error.BadRow;
            if (n + empties > out.len) return error.NoSpaceLeft;
            @memset(out[n..][0..empties], 0);
            n += empties;
            i = j;
        } else if (ch == '[') {
            const close = std.mem.indexOfScalarPos(u8, row, i + 1, ']') orelse return error.BadRow;
            n += try decodeBracketed(row[i + 1 .. close], row[i .. close + 1], ld, out[n..]);
            i = close + 1;
        } else {
            // A run of letters, up to the next digit or bracket.
            var j = i;
            while (j < row.len and !(row[j] >= '0' and row[j] <= '9') and row[j] != '[') j += 1;
            n += ld.toMachineLetters(row[i..j], out[n..]) catch |err| switch (err) {
                error.UnknownLetter => return error.UnknownLetter,
                error.NoSpaceLeft => return error.NoSpaceLeft,
            };
            i = j;
        }
    }
    return n;
}

/// A bracketed tile is one tile: the letters inside it, or failing that the
/// whole thing with its brackets (Spanish writes [CH] as a tile).
fn decodeBracketed(inner: []const u8, whole: []const u8, ld: *const tiles.LetterDistribution, out: []MachineLetter) RowError!usize {
    if (out.len == 0) return error.NoSpaceLeft;
    var buf: [8]MachineLetter = undefined;
    if (ld.toMachineLetters(inner, &buf)) |k| {
        if (k == 1) {
            out[0] = buf[0];
            return 1;
        }
    } else |_| {}
    const k = ld.toMachineLetters(whole, &buf) catch return error.UnknownLetter;
    if (k != 1) return error.BadRow;
    out[0] = buf[0];
    return 1;
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;
const testdata = @import("testdata.zig");

test "parse fields and options" {
    const c = try Cgp.parse("15/15/15/15/15/15/15/7QI6/15/15/15/15/15/15/15 AEINRST/ 10/-3 2 lex CSW24; ld english ;tmr 1/2;mcnz 4;");
    try testing.expectEqual(15, c.num_rows);
    try testing.expectEqualStrings("7QI6", c.rows[7]);
    try testing.expectEqualStrings("AEINRST", c.racks[0]);
    try testing.expectEqualStrings("", c.racks[1]);
    try testing.expectEqual([2]i32{ 10, -3 }, c.scores);
    try testing.expectEqual(2, c.scoreless_turns);
    try testing.expectEqualStrings("CSW24", c.lexicon);
    try testing.expectEqualStrings("english", c.letter_distribution);
    try testing.expectEqual(4, c.max_scoreless_turns);

    const d = try Cgp.parse("15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0");
    try testing.expectEqualStrings("NWL23", d.lexicon);
    try testing.expectError(error.TooFewFields, Cgp.parse("15/15 A/B 0/0"));
    try testing.expectError(error.NotTwoPlayers, Cgp.parse("15 A/B/C 0/0/0 0"));
    try testing.expectError(error.BadNumber, Cgp.parse("15 A/B 0/x 0"));
    try testing.expectError(error.BadOption, Cgp.parse("15 A/B 0/0 0 lex"));
}

test "decode rows" {
    const bytes = try testdata.loadFile("letterdistributions/english");
    defer testing.allocator.free(bytes);
    var ld = try tiles.LetterDistribution.parse(testing.allocator, "english", bytes);
    defer ld.deinit(testing.allocator);
    var out: [board.max_dim]MachineLetter = undefined;

    var n = try decodeRow("3QuIz10", &ld, &out);
    try testing.expectEqualSlices(MachineLetter, &.{ 0, 0, 0, 17, tiles.blank(21), 9, tiles.blank(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, out[0..n]);
    try testing.expectEqual(17, n);
    n = try decodeRow("15", &ld, &out);
    try testing.expectEqual(15, n);
    try testing.expectError(error.UnknownLetter, decodeRow("3#", &ld, &out));
}

test "decode catalan rows with bracketed tiles" {
    const bytes = try testdata.loadFile("letterdistributions/catalan");
    defer testing.allocator.free(bytes);
    var ld = try tiles.LetterDistribution.parse(testing.allocator, "catalan", bytes);
    defer ld.deinit(testing.allocator);
    var out: [board.max_dim]MachineLetter = undefined;
    var ll: MachineLetter = 0;
    for (ld.letters[0..ld.size], 0..) |l, i| {
        if (std.mem.eql(u8, l, "L·L")) ll = @intCast(i);
    }
    // Digits right before a bracketed tile come first.
    const n = try decodeRow("2[L·L]A[l·l]10", &ld, &out);
    try testing.expectEqual(15, n);
    try testing.expectEqualSlices(MachineLetter, &.{ 0, 0, ll, 1, tiles.blank(ll) }, out[0..5]);
}
