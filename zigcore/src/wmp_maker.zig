//! Builds word maps (see wmp.zig) from a word list or a KWG's DAWG. Port of
//! Macondo's wmp/maker.go (from MAGPIE's wmp_maker), with the same output
//! byte for byte. The engine needs its own: .wmp files are tens of times
//! the size of the KWGs they come from and aren't published, and an engine
//! with no Go beside it (the WASM build above all) has to make them.
//!
//! Go's maker radix-sorts racks, which keeps words of the same rack in list
//! order; sorting by (rack, list index) gives the same order.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const kwg_mod = @import("kwg.zig");
const board_mod = @import("board.zig");
const wmp = @import("wmp.zig");
const BitRack = wmp.BitRack;
const Entry = wmp.Entry;

const min_buckets = 16;

/// Words end to end, with their lengths.
pub const WordList = struct {
    letters: std.ArrayList(u8) = .empty,
    lengths: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *WordList, gpa: Allocator) void {
        self.letters.deinit(gpa);
        self.lengths.deinit(gpa);
    }

    pub fn append(self: *WordList, gpa: Allocator, word: []const u8) Allocator.Error!void {
        try self.letters.appendSlice(gpa, word);
        try self.lengths.append(gpa, @intCast(word.len));
    }

    /// The words of a KWG's DAWG up to `max_length` letters, in DAWG order.
    /// Port of ExtractWordsFromKWG.
    pub fn fromKwg(gpa: Allocator, k: kwg_mod.Kwg, max_length: usize) Allocator.Error!WordList {
        var list: WordList = .{};
        errdefer list.deinit(gpa);
        const root = k.dawgRoot();
        if (root == 0) return list;
        var prefix: [board_mod.max_dim]u8 = undefined;
        try extract(gpa, k, root, &prefix, 0, max_length, &list);
        return list;
    }

    fn extract(gpa: Allocator, k: kwg_mod.Kwg, node: u32, prefix: []u8, len: usize, max_length: usize, list: *WordList) Allocator.Error!void {
        var i = node;
        while (true) : (i += 1) {
            prefix[len] = k.tile(i);
            if (k.accepts(i)) try list.append(gpa, prefix[0 .. len + 1]);
            if (len + 1 < max_length and k.arcIndex(i) != 0) {
                try extract(gpa, k, k.arcIndex(i), prefix, len + 1, max_length, list);
            }
            if (k.isEnd(i)) break;
        }
    }
};

pub const Error = error{ BadBoardDim, IncompatibleDistribution } || Allocator.Error;

/// The word map file of the DAWG's words for a board of `dim` (Go and
/// MAGPIE make them for 15).
pub fn makeFromKwg(gpa: Allocator, k: kwg_mod.Kwg, ld: *const tiles.LetterDistribution, dim: usize) Error![]align(4) u8 {
    var list = try WordList.fromKwg(gpa, k, dim);
    defer list.deinit(gpa);
    return make(gpa, &list, ld, dim);
}

/// The word map file of `list`'s words for a board of `dim`. Words of
/// lengths outside 2 to `dim` are left out. Port of MakeFromWords.
pub fn make(gpa: Allocator, list: *const WordList, ld: *const tiles.LetterDistribution, dim: usize) Error![]align(4) u8 {
    if (dim < 2 or dim > board_mod.max_dim) return error.BadBoardDim;
    try wmp.checkCompatible(ld, dim);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var by_length: [board_mod.max_dim + 1]std.ArrayList(u8) = @splat(.empty);
    var off: usize = 0;
    for (list.lengths.items) |len| {
        if (len >= 2 and len <= dim) try by_length[len].appendSlice(arena, list.letters.items[off..][0..len]);
        off += len;
    }

    var m: wmp.Wmp = .{ .version = wmp.version, .board_dim = @intCast(dim), .max_word_lookup_bytes = 0, .wfls = @splat(.{}) };
    for (2..dim + 1) |len| {
        const unique = try buildWords(arena, &m.wfls[len], by_length[len].items, len);
        try buildBlanks(arena, &m.wfls[len], unique, false);
        try buildBlanks(arena, &m.wfls[len], unique, true);
    }
    m.max_word_lookup_bytes = maxWordLookupBytes(&m);
    return serialize(gpa, &m);
}

fn numBuckets(unique: usize) u32 {
    if (unique <= min_buckets) return min_buckets;
    return std.math.ceilPowerOfTwoAssert(u32, @intCast(unique));
}

inline fn bucketOf(br: BitRack, buckets: u32) u32 {
    return @as(u32, @truncate(wmp.mix(br))) & (buckets - 1);
}

/// Lays a table out: `racks` (sorted, distinct) go to their buckets in
/// order, and `fill` writes each one's entry, in rack order.
fn layOut(arena: Allocator, table: anytype, racks: []const BitRack, ctx: anytype, comptime fill: fn (@TypeOf(ctx), usize, *Entry) Allocator.Error!void) Allocator.Error!void {
    const buckets = numBuckets(racks.len);
    const starts = try arena.alloc(u32, buckets + 1);
    @memset(starts, 0);
    for (racks) |r| starts[bucketOf(r, buckets) + 1] += 1;
    for (1..starts.len) |b| starts[b] += starts[b - 1];
    const cursor = try arena.dupe(u32, starts[0..buckets]);
    const entries = try arena.alloc(Entry, racks.len);
    for (racks, 0..) |r, i| {
        const b = bucketOf(r, buckets);
        const e = &entries[cursor[b]];
        cursor[b] += 1;
        e.* = @splat(0);
        try fill(ctx, i, e);
        std.mem.writeInt(u128, e[wmp.inline_value_bytes..wmp.entry_size], r, .little);
    }
    table.* = .{ .num_buckets = buckets, .bucket_starts = starts, .entries = entries };
}

const WordPair = struct { rack: BitRack, index: u32 };

fn lessPair(_: void, a: WordPair, b: WordPair) bool {
    return if (a.rack != b.rack) a.rack < b.rack else a.index < b.index;
}

const Run = struct { start: u32, end: u32 };

const WordsCtx = struct {
    pairs: []const WordPair,
    runs: []const Run,
    words: []const u8,
    len: usize,
    letters: *std.ArrayList(u8),
    arena: Allocator,

    /// Up to 16 bytes of words go in the entry; more go to the word
    /// letters, and the entry says where.
    fn fill(c: WordsCtx, r: usize, e: *Entry) Allocator.Error!void {
        const run = c.runs[r];
        const n = run.end - run.start;
        if (n <= wmp.inline_value_bytes / c.len) {
            for (c.pairs[run.start..run.end], 0..) |p, k| {
                @memcpy(e[k * c.len ..][0..c.len], c.words[p.index * c.len ..][0..c.len]);
            }
            return;
        }
        std.mem.writeInt(u32, e[8..12], @intCast(c.letters.items.len), .little);
        std.mem.writeInt(u32, e[12..16], n, .little);
        for (c.pairs[run.start..run.end]) |p| try c.letters.appendSlice(c.arena, c.words[p.index * c.len ..][0..c.len]);
    }
};

/// The blankless table of one length; returns its racks, sorted.
fn buildWords(arena: Allocator, wfl: *wmp.ForLength, words: []const u8, len: usize) Allocator.Error![]BitRack {
    const n = words.len / len;
    const pairs = try arena.alloc(WordPair, n);
    for (pairs, 0..) |*p, i| p.* = .{ .rack = wmp.fromWord(words[i * len ..][0..len]), .index = @intCast(i) };
    std.mem.sortUnstable(WordPair, pairs, {}, lessPair);

    var runs: std.ArrayList(Run) = .empty;
    var racks: std.ArrayList(BitRack) = .empty;
    var i: usize = 0;
    while (i < n) {
        var j = i + 1;
        while (j < n and pairs[j].rack == pairs[i].rack) j += 1;
        try runs.append(arena, .{ .start = @intCast(i), .end = @intCast(j) });
        try racks.append(arena, pairs[i].rack);
        i = j;
    }
    var letters: std.ArrayList(u8) = .empty;
    try layOut(arena, &wfl.words, racks.items, WordsCtx{
        .pairs = pairs,
        .runs = runs.items,
        .words = words,
        .len = len,
        .letters = &letters,
        .arena = arena,
    }, WordsCtx.fill);
    wfl.word_letters = letters.items;
    return racks.items;
}

const BlankPair = struct { rack: BitRack, bit: u32 };

fn lessBlank(_: void, a: BlankPair, b: BlankPair) bool {
    return a.rack < b.rack;
}

fn letterMask(br: BitRack) u32 {
    var mask: u32 = 0;
    for (0..wmp.max_alphabet_size) |ml| {
        if (wmp.count(br, @intCast(ml)) != 0) mask |= @as(u32, 1) << @intCast(ml);
    }
    return mask;
}

fn fillBits(bits: []const u32, i: usize, e: *Entry) Allocator.Error!void {
    std.mem.writeInt(u32, e[8..12], bits[i], .little);
}

/// A blank table: each word rack with one of its letters made a blank,
/// holding the letters that blank can be; or (`double`) with two of them,
/// holding the letters the lower can be.
fn buildBlanks(arena: Allocator, wfl: *wmp.ForLength, unique: []const BitRack, double: bool) Allocator.Error!void {
    var pairs: std.ArrayList(BlankPair) = .empty;
    for (unique) |r| {
        var present1 = letterMask(r) & ~@as(u32, 1);
        while (present1 != 0) : (present1 &= present1 - 1) {
            const ml1: u8 = @intCast(@ctz(present1));
            var br = r;
            wmp.take(&br, ml1);
            wmp.add(&br, 0);
            const bit = @as(u32, 1) << @intCast(ml1);
            if (!double) {
                try pairs.append(arena, .{ .rack = br, .bit = bit });
                continue;
            }
            var present2 = letterMask(br) & ~(bit - 1) & ~@as(u32, 1);
            while (present2 != 0) : (present2 &= present2 - 1) {
                const ml2: u8 = @intCast(@ctz(present2));
                var br2 = br;
                wmp.take(&br2, ml2);
                wmp.add(&br2, 0);
                try pairs.append(arena, .{ .rack = br2, .bit = bit });
            }
        }
    }
    std.mem.sortUnstable(BlankPair, pairs.items, {}, lessBlank);
    var racks: std.ArrayList(BitRack) = .empty;
    var bits: std.ArrayList(u32) = .empty;
    for (pairs.items, 0..) |p, i| {
        if (i > 0 and p.rack == pairs.items[i - 1].rack) {
            bits.items[bits.items.len - 1] |= p.bit;
        } else {
            try racks.append(arena, p.rack);
            try bits.append(arena, p.bit);
        }
    }
    try layOut(arena, if (double) &wfl.double_blanks else &wfl.blanks, racks.items, @as([]const u32, bits.items), fillBits);
}

/// The most bytes a lookup can write: that of the two-blank rack with the
/// most words. Port of calculateMaxWordLookupBytes.
fn maxWordLookupBytes(m: *const wmp.Wmp) u32 {
    var most: u32 = 0;
    for (2..@as(usize, m.board_dim) + 1) |len| {
        const wfl = &m.wfls[len];
        for (wfl.double_blanks.entries) |*e| {
            var rack = wmp.entryRack(e);
            const first = wmp.blankLetters(e);
            var total: u32 = 0;
            wmp.setCount(&rack, 0, 1);
            var ml1: u8 = 1;
            while (ml1 < wmp.max_alphabet_size) : (ml1 += 1) {
                if (first & (@as(u32, 1) << @intCast(ml1)) == 0) continue;
                wmp.add(&rack, ml1);
                if (wfl.blankEntry(rack)) |be| {
                    const second = wmp.blankLetters(be);
                    var ml2 = ml1;
                    while (ml2 < wmp.max_alphabet_size) : (ml2 += 1) {
                        if (second & (@as(u32, 1) << @intCast(ml2)) == 0) continue;
                        wmp.add(&rack, ml2);
                        wmp.setCount(&rack, 0, 0);
                        if (wfl.hashedWordEntry(rack, wmp.mix(rack))) |we| total += @intCast(wfl.blanklessWords(we, len).len / len);
                        wmp.setCount(&rack, 0, 1);
                        wmp.take(&rack, ml2);
                    }
                }
                wmp.take(&rack, ml1);
            }
            most = @max(most, total * @as(u32, @intCast(len)));
        }
    }
    return most;
}

/// The word map's file bytes, aligned as data files are.
fn serialize(gpa: Allocator, m: *const wmp.Wmp) Allocator.Error![]align(4) u8 {
    var out: Out = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, &.{ m.version, m.board_dim });
    try appendInt(gpa, &out, m.max_word_lookup_bytes);
    for (2..@as(usize, m.board_dim) + 1) |len| {
        const wfl = &m.wfls[len];
        try appendTable(gpa, &out, &wfl.words);
        try appendInt(gpa, &out, @intCast(wfl.word_letters.len / len));
        try out.appendSlice(gpa, wfl.word_letters);
        try appendTable(gpa, &out, &wfl.blanks);
        try appendTable(gpa, &out, &wfl.double_blanks);
    }
    return out.toOwnedSlice(gpa);
}

const Out = std.ArrayListAligned(u8, .of(u32));

fn appendInt(gpa: Allocator, out: *Out, v: u32) Allocator.Error!void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(gpa, &b);
}

fn appendTable(gpa: Allocator, out: *Out, t: anytype) Allocator.Error!void {
    try appendInt(gpa, out, t.num_buckets);
    for (t.bucket_starts) |s| try appendInt(gpa, out, s);
    try appendInt(gpa, out, @intCast(t.entries.len));
    for (t.entries) |*e| try out.appendSlice(gpa, e);
}

// Tests ----------------------------------------------------------------------
// Ported from wmp/wmp_test.go (the ones that make word maps), plus a check
// that a whole lexicon's comes out as Go makes it.

const testing = std.testing;
const testdata = @import("testdata.zig");

fn englishDist() !tiles.LetterDistribution {
    const bytes = try testdata.loadFile("letterdistributions/english");
    defer testing.allocator.free(bytes);
    return tiles.LetterDistribution.parse(testing.allocator, "english", bytes);
}

fn listOf(words: []const []const u8) !WordList {
    var list: WordList = .{};
    for (words) |w| {
        var mls: [16]u8 = undefined;
        for (w, 0..) |c, i| mls[i] = c - 'A' + 1;
        try list.append(testing.allocator, mls[0..w.len]);
    }
    return list;
}

/// Looks `rack` (with ? for blanks) up in a word map made of `words`, and
/// checks the words found, sorted.
fn expectLookup(words: []const []const u8, rack: []const u8, want: []const []const u8) !void {
    var ld = try englishDist();
    defer ld.deinit(testing.allocator);
    var list = try listOf(words);
    defer list.deinit(testing.allocator);
    const bytes = try make(testing.allocator, &list, &ld, 15);
    defer testing.allocator.free(bytes);
    var m = try wmp.Wmp.init(testing.allocator, bytes);
    defer m.deinit(testing.allocator);

    var br: BitRack = 0;
    for (rack) |c| wmp.add(&br, if (c == '?') 0 else c - 'A' + 1);
    var buf: [wmp.result_buffer_size]u8 = undefined;
    const n = m.writeWords(br, rack.len, &buf);
    var got: std.ArrayList([]const u8) = .empty;
    defer got.deinit(testing.allocator);
    var i: usize = 0;
    while (i < n) : (i += rack.len) {
        for (buf[i..][0..rack.len]) |*c| c.* = c.* + 'A' - 1;
        try got.append(testing.allocator, buf[i..][0..rack.len]);
    }
    std.mem.sort([]const u8, got.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    try testing.expectEqual(want.len, got.items.len);
    for (want, got.items) |w, g| try testing.expectEqualStrings(w, g);
}

test "make and look up, no blanks" {
    const words = [_][]const u8{ "CAT", "ACT", "TAB", "BAT", "DOG", "GOD", "CARE", "RACE", "ACRE", "PEAR", "REAP", "AERIAL" };
    try expectLookup(&words, "CAT", &.{ "ACT", "CAT" });
    try expectLookup(&words, "CARE", &.{ "ACRE", "CARE", "RACE" });
    try expectLookup(&words, "PEAR", &.{ "PEAR", "REAP" });
    // Not inlined: 16 bytes hold two 6-letter words.
    try expectLookup(&words, "AERIAL", &.{"AERIAL"});
    try expectLookup(&words, "XYZ", &.{});
}

test "make and look up, blanks" {
    try expectLookup(&.{ "CAT", "BAT", "RAT", "EAT", "OAT" }, "?AT", &.{ "BAT", "CAT", "EAT", "OAT", "RAT" });
    try expectLookup(&.{ "CAT", "BAT", "RAT", "EAT" }, "??T", &.{ "BAT", "CAT", "EAT", "RAT" });
    try expectLookup(&.{ "RETINAS", "RETAINS", "RATINES", "RATINE" }, "RETINAS", &.{ "RATINES", "RETAINS", "RETINAS" });
}

test "made word maps read back" {
    var ld = try englishDist();
    defer ld.deinit(testing.allocator);
    var list = try listOf(&.{ "CAT", "ACT", "CARE", "RACE", "ACRE", "AERIAL" });
    defer list.deinit(testing.allocator);
    const bytes = try make(testing.allocator, &list, &ld, 15);
    defer testing.allocator.free(bytes);
    var m = try wmp.Wmp.init(testing.allocator, bytes);
    defer m.deinit(testing.allocator);
    try testing.expectEqual(wmp.version, m.version);
    try testing.expectEqual(15, m.board_dim);
    // Incompatible distributions and boards are refused.
    try testing.expectError(error.BadBoardDim, make(testing.allocator, &list, &ld, 1));
    ld.counts[0] = 3;
    try testing.expectError(error.IncompatibleDistribution, make(testing.allocator, &list, &ld, 15));
}

test "a lexicon's word map is Go's, byte for byte (NWL20)" {
    var ld = try englishDist();
    defer ld.deinit(testing.allocator);
    const kwg_bytes = try testdata.loadFile("lexica/gaddag/NWL20.kwg");
    defer testing.allocator.free(kwg_bytes);
    const go_made = try testdata.loadFile("lexica/NWL20.wmp");
    defer testing.allocator.free(go_made);
    const made = try makeFromKwg(testing.allocator, try kwg_mod.Kwg.init(kwg_bytes), &ld, 15);
    defer testing.allocator.free(made);
    try testing.expectEqual(go_made.len, made.len);
    try testing.expect(std.mem.eql(u8, go_made, made));
}
