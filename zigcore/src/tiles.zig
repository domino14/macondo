//! Tiles: machine letters, letter distributions and racks.
//!
//! Port of word-golib's `tilemapping` package. A tile is a byte: 0 is the
//! blank on a rack (and an empty square or a played-through marker
//! elsewhere), 1.. are the letters in distribution order, and a designated
//! blank is the letter with the high bit set.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MachineLetter = u8;
/// A bit per machine letter (unblanked), for cross-sets and the like.
pub const LetterSet = u64;

/// Fits a letter set in a u64.
pub const max_alphabet_size = 62;
pub const blank_mask: MachineLetter = 0x80;
pub const unblank_mask: MachineLetter = 0x7f;
pub const blank_token = "?";
pub const played_through_token = ".";

pub inline fn blank(ml: MachineLetter) MachineLetter {
    return ml | blank_mask;
}

pub inline fn unblank(ml: MachineLetter) MachineLetter {
    return ml & unblank_mask;
}

pub inline fn isBlanked(ml: MachineLetter) bool {
    return ml & blank_mask != 0;
}

/// The index a tile takes on a rack: a designated blank goes back to 0.
pub inline fn intrinsicIndex(ml: MachineLetter) MachineLetter {
    return if (isBlanked(ml)) 0 else ml;
}

pub const ParseError = error{ InvalidDistribution, TooManyLetters, OutOfMemory };
pub const ConvertError = error{UnknownLetter};

/// A letter distribution: the tiles of a game, their counts and scores,
/// read from the CSV files in data/letterdistributions
/// (`letter,count,score,vowel`, the blank first).
pub const LetterDistribution = struct {
    name: []u8,
    /// Size of the alphabet, including the blank at 0.
    size: u8,
    /// User-visible letters, upper case (the blank is "?").
    letters: [max_alphabet_size][]u8,
    /// The same letters in lower case, for designated blanks.
    lower: [max_alphabet_size][]u8,
    counts: [max_alphabet_size]u8,
    scores: [max_alphabet_size]i32,
    vowels: LetterSet,
    /// Total number of tiles, including blanks.
    total_tiles: u32,
    /// Longest letter, in code points (Catalan's L·L is 3).
    max_letter_len: usize,

    pub fn parse(gpa: Allocator, name: []const u8, csv: []const u8) ParseError!LetterDistribution {
        var ld: LetterDistribution = .{
            .name = try gpa.dupe(u8, name),
            .size = 0,
            .letters = undefined,
            .lower = undefined,
            .counts = @splat(0),
            .scores = @splat(0),
            .vowels = 0,
            .total_tiles = 0,
            .max_letter_len = 0,
        };
        errdefer ld.deinit(gpa);

        var lines = std.mem.splitScalar(u8, csv, '\n');
        while (lines.next()) |raw_line| {
            const line = std.mem.trimEnd(u8, raw_line, "\r");
            if (line.len == 0) continue;
            if (ld.size == max_alphabet_size) return error.TooManyLetters;
            var fields = std.mem.splitScalar(u8, line, ',');
            const glyph = fields.next() orelse return error.InvalidDistribution;
            const count = parseField(u8, fields.next()) orelse return error.InvalidDistribution;
            const points = parseField(i32, fields.next()) orelse return error.InvalidDistribution;
            const vowel = parseField(u8, fields.next()) orelse return error.InvalidDistribution;
            if (glyph.len == 0 or !std.unicode.utf8ValidateSlice(glyph)) return error.InvalidDistribution;

            const i = ld.size;
            ld.letters[i] = try gpa.dupe(u8, glyph);
            ld.size += 1;
            ld.lower[i] = toLowerAlloc(gpa, glyph) catch |err| {
                ld.size -= 1;
                gpa.free(ld.letters[i]);
                return err;
            };
            ld.counts[i] = count;
            ld.scores[i] = points;
            if (vowel == 1) ld.vowels |= @as(LetterSet, 1) << @intCast(i);
            ld.total_tiles += count;
            const len = std.unicode.utf8CountCodepoints(glyph) catch unreachable;
            ld.max_letter_len = @max(ld.max_letter_len, len);
        }
        if (ld.size == 0) return error.InvalidDistribution;
        return ld;
    }

    pub fn deinit(self: *LetterDistribution, gpa: Allocator) void {
        for (self.letters[0..self.size], self.lower[0..self.size]) |l, lo| {
            gpa.free(l);
            gpa.free(lo);
        }
        gpa.free(self.name);
        self.* = undefined;
    }

    /// The score of a tile; a designated blank scores as the blank.
    pub inline fn score(self: *const LetterDistribution, ml: MachineLetter) i32 {
        return self.scores[if (isBlanked(ml)) 0 else ml];
    }

    pub fn isVowel(self: *const LetterDistribution, ml: MachineLetter) bool {
        return self.vowels & (@as(LetterSet, 1) << @intCast(unblank(ml))) != 0;
    }

    /// The user-visible form of a tile: "?" for 0, lower case for a
    /// designated blank. With `played_through`, 0 is "." instead.
    pub fn letter(self: *const LetterDistribution, ml: MachineLetter, played_through: bool) []const u8 {
        if (ml == 0) return if (played_through) played_through_token else blank_token;
        if (isBlanked(ml)) return self.lower[unblank(ml)];
        return self.letters[ml];
    }

    /// Converts user-visible text to machine letters, longest tile first,
    /// like word-golib's ToMachineLetters: lower case is a designated blank,
    /// "?" is the blank, and " " or "." is 0. Returns the number written.
    pub fn toMachineLetters(self: *const LetterDistribution, text: []const u8, out: []MachineLetter) (ConvertError || error{NoSpaceLeft})!usize {
        var n: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            const match = self.matchAt(text, i) orelse blk: {
                if (text[i] == ' ' or text[i] == '.') break :blk Match{ .ml = 0, .len = 1 };
                return error.UnknownLetter;
            };
            if (n == out.len) return error.NoSpaceLeft;
            out[n] = match.ml;
            n += 1;
            i += match.len;
        }
        return n;
    }

    const Match = struct { ml: MachineLetter, len: usize };

    /// The longest tile starting at byte `start` of `text`, trying exact
    /// (upper case) matches before lower case ones at each length.
    fn matchAt(self: *const LetterDistribution, text: []const u8, start: usize) ?Match {
        // Byte offsets of the code point boundaries after `start`.
        var ends: [16]usize = undefined;
        var count: usize = 0;
        var pos = start;
        while (count < self.max_letter_len and count < ends.len and pos < text.len) {
            const cp_len = std.unicode.utf8ByteSequenceLength(text[pos]) catch return null;
            pos += cp_len;
            if (pos > text.len) return null;
            ends[count] = pos;
            count += 1;
        }
        var k = count;
        while (k > 0) : (k -= 1) {
            const candidate = text[start..ends[k - 1]];
            for (self.letters[0..self.size], 0..) |l, ml| {
                if (std.mem.eql(u8, l, candidate)) return .{ .ml = @intCast(ml), .len = candidate.len };
            }
            for (self.lower[0..self.size], self.letters[0..self.size], 0..) |lo, up, ml| {
                if (eqlIgnoreCase(lo, candidate) and !std.mem.eql(u8, up, candidate))
                    return .{ .ml = blank(@intCast(ml)), .len = candidate.len };
            }
        }
        return null;
    }
};

fn parseField(comptime T: type, field: ?[]const u8) ?T {
    const f = std.mem.trim(u8, field orelse return null, " ");
    return std.fmt.parseInt(T, f, 10) catch null;
}

/// Lower-cases the letters our distributions use: ASCII, Latin-1 and
/// Latin Extended-A. Anything else is left as it is.
pub fn toLowerCodepoint(cp: u21) u21 {
    return switch (cp) {
        'A'...'Z' => cp + 32,
        0xC0...0xD6, 0xD8...0xDE => cp + 32,
        0x100...0x137, 0x14A...0x177 => if (cp % 2 == 0) cp + 1 else cp,
        0x139...0x148, 0x179...0x17E => if (cp % 2 == 1) cp + 1 else cp,
        0x178 => 0xFF,
        else => cp,
    };
}

fn toLowerAlloc(gpa: Allocator, s: []const u8) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var it = (std.unicode.Utf8View.init(s) catch unreachable).iterator();
    while (it.nextCodepoint()) |cp| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(toLowerCodepoint(cp), &buf) catch unreachable;
        try out.appendSlice(gpa, buf[0..n]);
    }
    return out.toOwnedSlice(gpa);
}

/// Whether `s` lower-cases to `lower` (which is already lower case).
fn eqlIgnoreCase(lower: []const u8, s: []const u8) bool {
    var a = (std.unicode.Utf8View.init(lower) catch return false).iterator();
    var b = (std.unicode.Utf8View.init(s) catch return false).iterator();
    while (true) {
        const ca = a.nextCodepoint();
        const cb = b.nextCodepoint();
        if (ca == null or cb == null) return ca == null and cb == null;
        if (ca.? != toLowerCodepoint(cb.?)) return false;
    }
}

/// A rack as letter counts, the blank at 0.
pub const Rack = struct {
    counts: [max_alphabet_size]u8 = @splat(0),
    num_tiles: u8 = 0,

    pub fn fromLetters(mls: []const MachineLetter) Rack {
        var r: Rack = .{};
        for (mls) |ml| r.add(ml);
        return r;
    }

    pub inline fn add(self: *Rack, ml: MachineLetter) void {
        self.counts[ml] += 1;
        self.num_tiles += 1;
    }

    /// The caller makes sure the tile is on the rack.
    pub inline fn take(self: *Rack, ml: MachineLetter) void {
        self.counts[ml] -= 1;
        self.num_tiles -= 1;
    }

    pub inline fn has(self: *const Rack, ml: MachineLetter) bool {
        return self.counts[ml] > 0;
    }

    /// Writes the tiles in alphabetical (machine letter) order and returns
    /// how many there are.
    pub fn tiles(self: *const Rack, out: []MachineLetter) usize {
        var n: usize = 0;
        for (self.counts, 0..) |c, ml| {
            for (0..c) |_| {
                out[n] = @intCast(ml);
                n += 1;
            }
        }
        return n;
    }

    pub fn scoreOn(self: *const Rack, ld: *const LetterDistribution) i32 {
        var s: i32 = 0;
        for (self.counts[0..ld.size], 0..) |c, ml| s += ld.scores[ml] * c;
        return s;
    }
};

// Tests ----------------------------------------------------------------------

const testing = std.testing;
const testdata = @import("testdata.zig");

fn loadDist(name: []const u8) !LetterDistribution {
    var buf: [64]u8 = undefined;
    const bytes = try testdata.loadFile(try std.fmt.bufPrint(&buf, "letterdistributions/{s}", .{name}));
    defer testing.allocator.free(bytes);
    return LetterDistribution.parse(testing.allocator, name, bytes);
}

test "parse english" {
    var ld = try loadDist("english");
    defer ld.deinit(testing.allocator);
    try testing.expectEqual(27, ld.size);
    try testing.expectEqual(100, ld.total_tiles);
    try testing.expectEqualStrings("?", ld.letters[0]);
    try testing.expectEqualStrings("A", ld.letters[1]);
    try testing.expectEqualStrings("z", ld.lower[26]);
    try testing.expectEqual(10, ld.score(26));
    try testing.expectEqual(0, ld.score(blank(26)));
    try testing.expect(ld.isVowel(1) and ld.isVowel(blank(5)) and !ld.isVowel(2));
}

test "toMachineLetters english" {
    var ld = try loadDist("english");
    defer ld.deinit(testing.allocator);
    var out: [16]MachineLetter = undefined;
    const n = try ld.toMachineLetters("Qu?.Z ", &out);
    try testing.expectEqualSlices(MachineLetter, &.{ 17, blank(21), 0, 0, 26, 0 }, out[0..n]);
    try testing.expectError(error.UnknownLetter, ld.toMachineLetters("A1", &out));
    try testing.expectEqualStrings("u", ld.letter(blank(21), false));
    try testing.expectEqualStrings(".", ld.letter(0, true));
}

test "toMachineLetters catalan multi-letter tiles" {
    var ld = try loadDist("catalan");
    defer ld.deinit(testing.allocator);
    try testing.expectEqual(3, ld.max_letter_len);
    var out: [16]MachineLetter = undefined;
    const ll = idx: for (ld.letters[0..ld.size], 0..) |l, i| {
        if (std.mem.eql(u8, l, "L·L")) break :idx i;
    } else unreachable;
    const c_cedilla = idx: for (ld.letters[0..ld.size], 0..) |l, i| {
        if (std.mem.eql(u8, l, "Ç")) break :idx i;
    } else unreachable;
    const n = try ld.toMachineLetters("L·LLl·lçÇ", &out);
    const l = ll - 1; // L comes right before L·L
    try testing.expectEqualSlices(MachineLetter, &.{
        @intCast(ll), @intCast(l), blank(@intCast(ll)), blank(@intCast(c_cedilla)), @intCast(c_cedilla),
    }, out[0..n]);
}

test "rack" {
    var r = Rack.fromLetters(&.{ 5, 0, 1, 5 });
    try testing.expectEqual(4, r.num_tiles);
    r.take(5);
    var out: [8]MachineLetter = undefined;
    try testing.expectEqualSlices(MachineLetter, &.{ 0, 1, 5 }, out[0..r.tiles(&out)]);
}
