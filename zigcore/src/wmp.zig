//! WordMap (WMP): an anagram table, from a multiset of tiles to the words
//! of a given length they make. MAGPIE's format and lookups (wmp.h); port
//! of Macondo's wmp package. Files are shared with both.
//!
//! A rack is a BitRack, 4 bits per letter. For each word length there are
//! three hash tables: racks without a blank (holding their words, inline
//! in the entry when they fit), racks with one blank (holding the letters
//! the blank can be), and racks with two (the letters the first blank can
//! be). Lookups read the file's bytes in place.
//!
//! File layout (little-endian): u8 version, u8 board dim, u32 max word
//! lookup bytes; then for each length 2..dim: the blankless table (u32
//! bucket count, bucket count + 1 u32 bucket starts, u32 entry count,
//! entries, u32 uninlined word count, the uninlined words), and the one-
//! and two-blank tables (bucket count, starts, entry count, entries).

const std = @import("std");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const Allocator = std.mem.Allocator;

pub const version = 3;
pub const earliest_supported_version = 3;
pub const inline_value_bytes = 16;
pub const entry_size = 32;
/// The most bytes one lookup writes.
pub const result_buffer_size = 10000;
pub const minimum_word_length = 2;
/// Letters a BitRack holds (128 bits, 4 per letter).
pub const max_alphabet_size = 32;
const max_blanks = 2;
const max_letter_count = 15;

/// A multiset of tiles: 4 bits of count per machine letter, letter 0 (the
/// blank) in the low bits. Its memory layout is the file's.
pub const BitRack = u128;

pub inline fn count(br: BitRack, ml: u8) u8 {
    return @intCast((br >> @intCast(@as(u32, ml) * 4)) & 0xf);
}

pub inline fn add(br: *BitRack, ml: u8) void {
    br.* += @as(BitRack, 1) << @intCast(@as(u32, ml) * 4);
}

pub inline fn take(br: *BitRack, ml: u8) void {
    br.* -= @as(BitRack, 1) << @intCast(@as(u32, ml) * 4);
}

pub inline fn setCount(br: *BitRack, ml: u8, n: u8) void {
    const shift: u7 = @intCast(@as(u32, ml) * 4);
    br.* = (br.* & ~(@as(BitRack, 0xf) << shift)) | (@as(BitRack, n) << shift);
}

pub fn fromWord(word: []const u8) BitRack {
    var br: BitRack = 0;
    for (word) |ml| add(&br, ml);
    return br;
}

pub fn fromRack(rack: *const tiles.Rack) BitRack {
    var br: BitRack = 0;
    for (rack.counts[0..max_alphabet_size], 0..) |n, ml| {
        if (n > 0) setCount(&br, @intCast(ml), n);
    }
    return br;
}

/// The 64-bit hash of a rack (MAGPIE's bit_rack_mix_to_64).
pub inline fn mix(br: BitRack) u64 {
    const high: u64 = @truncate(br >> 64);
    var low: u64 = @truncate(br);
    low ^= high;
    low ^= std.math.rotl(u64, high, 17);
    low ^= low >> 33;
    low *%= 0xff51afd7ed558ccd;
    low ^= low >> 33;
    low *%= 0xc4ceb9fe1a85ec53;
    low ^= low >> 33;
    return low;
}

/// Whether a word map can hold racks of `ld` for a board of `dim`: at most
/// 32 letters and 2 blanks, and no letter (counting the blanks as it)
/// more than 15 times in a word. Port of CheckCompatible.
pub fn checkCompatible(ld: *const tiles.LetterDistribution, dim: usize) error{IncompatibleDistribution}!void {
    if (ld.size > max_alphabet_size) return error.IncompatibleDistribution;
    const blanks = ld.counts[0];
    if (blanks > max_blanks) return error.IncompatibleDistribution;
    var most: usize = 0;
    for (0..ld.size) |ml| {
        var n: usize = ld.counts[ml];
        if (ml != 0) n += blanks;
        most = @max(most, n);
    }
    if (@min(most, dim) > max_letter_count) return error.IncompatibleDistribution;
}

/// One 32-byte entry: 16 bytes of inline words or metadata, then the
/// rack.
pub const Entry = [entry_size]u8;

pub inline fn entryRack(e: *const Entry) BitRack {
    return std.mem.readInt(u128, e[inline_value_bytes..entry_size], .little);
}

pub inline fn isInlined(e: *const Entry) bool {
    return e[0] != 0;
}

/// For a blankless entry not inlined: where its words start.
pub inline fn wordStart(e: *const Entry) u32 {
    return std.mem.readInt(u32, e[8..12], .little);
}

pub inline fn numWords(e: *const Entry) u32 {
    return std.mem.readInt(u32, e[12..16], .little);
}

/// For a blank entry: the letters the (first) blank can be.
pub inline fn blankLetters(e: *const Entry) u32 {
    return std.mem.readInt(u32, e[8..12], .little);
}

/// The bytes of words an inlined entry holds (it is padded with zeros).
fn inlinedBytes(e: *const Entry, word_length: usize) usize {
    var n = (inline_value_bytes / word_length) * word_length;
    while (n > word_length and e[n - 1] == 0) n -= word_length;
    return n;
}

const Table = struct {
    num_buckets: u32 = 0,
    bucket_starts: []align(1) const u32 = &.{},
    entries: []const Entry = &.{},

    fn scan(self: *const Table, bucket: u32, br: BitRack) ?*const Entry {
        const start = self.bucket_starts[bucket];
        const end = self.bucket_starts[bucket + 1];
        for (self.entries[start..end]) |*e| {
            if (entryRack(e) == br) return e;
        }
        return null;
    }

    fn lookup(self: *const Table, br: BitRack) ?*const Entry {
        if (self.num_buckets == 0) return null;
        return self.scan(@as(u32, @truncate(mix(br))) & (self.num_buckets - 1), br);
    }
};

pub const ForLength = struct {
    words: Table = .{},
    word_letters: []const u8 = &.{},
    blanks: Table = .{},
    double_blanks: Table = .{},
    /// A blocked Bloom filter over the blankless racks, built at load: two
    /// bits of one u64 per rack, chosen by its hash. Most racks looked up
    /// have no words, and this says so without reading the table.
    filter: []u64 = &.{},
    filter_shift: u6 = 0,

    const filter_bits_per_entry = 16;

    inline fn filterBits(hash: u64, shift: u6) u64 {
        return (@as(u64, 1) << @intCast((hash >> (shift - 6)) & 63)) |
            (@as(u64, 1) << @intCast((hash >> (shift - 12)) & 63));
    }

    fn buildFilter(self: *ForLength, gpa: Allocator) Allocator.Error!void {
        self.filter = &.{};
        if (self.words.num_buckets == 0 or self.words.entries.len == 0) return;
        var words: u64 = 64;
        while (words * 64 < @as(u64, self.words.entries.len) * filter_bits_per_entry) words <<= 1;
        const shift: u6 = @intCast(@as(u7, 64) - std.math.log2_int(u64, words));
        const filter = try gpa.alloc(u64, words);
        @memset(filter, 0);
        for (self.words.entries) |*e| {
            const hash = mix(entryRack(e));
            filter[hash >> shift] |= filterBits(hash, shift);
        }
        self.filter = filter;
        self.filter_shift = shift;
    }

    /// False if the blankless rack with this hash surely has no words.
    pub inline fn mayHaveWords(self: *const ForLength, hash: u64) bool {
        if (self.words.num_buckets == 0) return false;
        if (self.filter.len == 0) return true;
        const bits = filterBits(hash, self.filter_shift);
        return self.filter[hash >> self.filter_shift] & bits == bits;
    }

    pub fn wordEntry(self: *const ForLength, br: BitRack) ?*const Entry {
        const hash = mix(br);
        if (!self.mayHaveWords(hash)) return null;
        return self.words.scan(@as(u32, @truncate(hash)) & (self.words.num_buckets - 1), br);
    }

    /// wordEntry for a rack whose hash is known and passed mayHaveWords.
    pub fn hashedWordEntry(self: *const ForLength, br: BitRack, hash: u64) ?*const Entry {
        if (self.words.num_buckets == 0) return null;
        return self.words.scan(@as(u32, @truncate(hash)) & (self.words.num_buckets - 1), br);
    }

    pub fn blankEntry(self: *const ForLength, br: BitRack) ?*const Entry {
        return self.blanks.lookup(br);
    }

    pub fn doubleBlankEntry(self: *const ForLength, br: BitRack) ?*const Entry {
        return self.double_blanks.lookup(br);
    }

    /// The words of a blankless entry, where the word map keeps them.
    pub fn blanklessWords(self: *const ForLength, e: *const Entry, word_length: usize) []const u8 {
        if (isInlined(e)) return e[0..inlinedBytes(e, word_length)];
        const start = wordStart(e);
        return self.word_letters[start..][0 .. numWords(e) * word_length];
    }

    fn writeBlankless(self: *const ForLength, e: *const Entry, word_length: usize, buf: []u8) usize {
        const w = self.blanklessWords(e, word_length);
        @memcpy(buf[0..w.len], w);
        return w.len;
    }

    /// The words of a one-blank rack: for each letter the blank can be (from
    /// `min_ml` on), the words of the rack with that letter.
    fn writeBlank(self: *const ForLength, e: *const Entry, br_in: BitRack, word_length: usize, min_ml: u8, buf: []u8) usize {
        var n: usize = 0;
        var br = br_in;
        setCount(&br, 0, 0);
        const allowed = blankLetters(e);
        var ml = min_ml;
        while (ml < max_alphabet_size) : (ml += 1) {
            if (allowed & (@as(u32, 1) << @intCast(ml)) == 0) continue;
            add(&br, ml);
            // The entry says this makes a word: no need for the filter.
            if (self.hashedWordEntry(br, mix(br))) |be| n += self.writeBlankless(be, word_length, buf[n..]);
            take(&br, ml);
        }
        return n;
    }

    /// The words of a two-blank rack: for each letter the first blank can
    /// be, the one-blank rack's words with the second blank no lower.
    fn writeDoubleBlank(self: *const ForLength, e: *const Entry, br_in: BitRack, word_length: usize, buf: []u8) usize {
        var n: usize = 0;
        var br = br_in;
        setCount(&br, 0, 1);
        const allowed = blankLetters(e);
        var ml: u8 = 1;
        while (ml < max_alphabet_size) : (ml += 1) {
            if (allowed & (@as(u32, 1) << @intCast(ml)) == 0) continue;
            add(&br, ml);
            if (self.blankEntry(br)) |be| n += self.writeBlank(be, br, word_length, ml, buf[n..]);
            take(&br, ml);
        }
        return n;
    }
};

pub const Error = error{ InvalidWmp, UnsupportedVersion } || Allocator.Error;

pub const Wmp = struct {
    version: u8,
    board_dim: u8,
    max_word_lookup_bytes: u32,
    /// By word length; 0 and 1 unused.
    wfls: [board_mod.max_dim + 1]ForLength,

    /// Reads a word map from `bytes`, which must outlive it.
    pub fn init(gpa: Allocator, bytes: []const u8) Error!Wmp {
        var r: Reader = .{ .bytes = bytes };
        var w: Wmp = .{
            .version = try r.byte(),
            .board_dim = try r.byte(),
            .max_word_lookup_bytes = try r.int(),
            .wfls = @splat(.{}),
        };
        if (w.version < earliest_supported_version) return error.UnsupportedVersion;
        if (w.board_dim < 2 or w.board_dim > board_mod.max_dim) return error.InvalidWmp;
        errdefer w.deinit(gpa);
        for (2..@as(usize, w.board_dim) + 1) |len| {
            const wfl = &w.wfls[len];
            wfl.words = try r.table();
            const uninlined = try r.int();
            wfl.word_letters = try r.take(@as(usize, uninlined) * len);
            wfl.blanks = try r.table();
            wfl.double_blanks = try r.table();
            try wfl.buildFilter(gpa);
        }
        return w;
    }

    pub fn deinit(self: *Wmp, gpa: Allocator) void {
        for (&self.wfls) |*wfl| {
            if (wfl.filter.len > 0) gpa.free(wfl.filter);
            wfl.filter = &.{};
        }
    }

    /// The entry for rack `br` among the words of `word_length`, or null.
    pub fn entry(self: *const Wmp, br: BitRack, word_length: usize) ?*const Entry {
        if (word_length < minimum_word_length or word_length > self.board_dim) return null;
        const wfl = &self.wfls[word_length];
        return switch (count(br, 0)) {
            0 => wfl.wordEntry(br),
            1 => wfl.blankEntry(br),
            2 => wfl.doubleBlankEntry(br),
            else => null,
        };
    }

    /// Writes the words of `word_length` that rack `br` makes into `buf`
    /// (word_length bytes each, end to end) and returns the bytes written.
    pub fn writeWords(self: *const Wmp, br: BitRack, word_length: usize, buf: []u8) usize {
        const e = self.entry(br, word_length) orelse return 0;
        return self.writeWordsForEntry(e, br, word_length, buf);
    }

    /// writeWords for an entry already looked up.
    pub fn writeWordsForEntry(self: *const Wmp, e: *const Entry, br: BitRack, word_length: usize, buf: []u8) usize {
        if (word_length < minimum_word_length or word_length > self.board_dim) return 0;
        const wfl = &self.wfls[word_length];
        return switch (count(br, 0)) {
            0 => wfl.writeBlankless(e, word_length, buf),
            1 => wfl.writeBlank(e, br, word_length, 1, buf),
            2 => wfl.writeDoubleBlank(e, br, word_length, buf),
            else => 0,
        };
    }
};

/// Reads the file's fields in order, checking each fits.
const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) error{InvalidWmp}![]const u8 {
        if (n > self.bytes.len - self.pos) return error.InvalidWmp;
        defer self.pos += n;
        return self.bytes[self.pos..][0..n];
    }

    fn byte(self: *Reader) error{InvalidWmp}!u8 {
        return (try self.take(1))[0];
    }

    fn int(self: *Reader) error{InvalidWmp}!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn table(self: *Reader) error{InvalidWmp}!Table {
        const buckets = try self.int();
        // A power of two: buckets are found by masking the hash.
        if (buckets == 0 or buckets & (buckets - 1) != 0) return error.InvalidWmp;
        const starts = std.mem.bytesAsSlice(u32, try self.take((@as(usize, buckets) + 1) * 4));
        const n = try self.int();
        // Increasing, from 0 to the number of entries.
        if (starts[0] != 0 or starts[starts.len - 1] != n) return error.InvalidWmp;
        for (starts[1..], starts[0 .. starts.len - 1]) |b, a| {
            if (b < a) return error.InvalidWmp;
        }
        const entries = std.mem.bytesAsSlice(Entry, try self.take(@as(usize, n) * entry_size));
        return .{ .num_buckets = buckets, .bucket_starts = starts, .entries = entries };
    }
};

// Tests ----------------------------------------------------------------------
// Ported from wmp/wmp_test.go and filter_test.go. The Go tests build small
// word maps with the maker, which stays in Go; these read the files Go
// makes (cmd/make_wmp) and check lookups against the lexicon itself.

const testing = std.testing;
const testdata = @import("testdata.zig");
const kwg_mod = @import("kwg.zig");

fn letters(comptime word: []const u8) [word.len]u8 {
    var out: [word.len]u8 = undefined;
    for (word, 0..) |c, i| out[i] = if (c == '?') 0 else c - 'A' + 1;
    return out;
}

test "bit rack basics" {
    const cat = fromWord(&letters("CAT"));
    try testing.expectEqual(1, count(cat, 3));
    try testing.expectEqual(1, count(cat, 1));
    try testing.expectEqual(1, count(cat, 20));
    try testing.expectEqual(0, count(cat, 2));
    try testing.expectEqual(cat, fromWord(&letters("ACT")));
    try testing.expectEqual(cat, fromWord(&letters("TAC")));

    var br: BitRack = 0;
    for (0..3) |_| add(&br, 5);
    try testing.expectEqual(3, count(br, 5));
    take(&br, 5);
    try testing.expectEqual(2, count(br, 5));
    setCount(&br, 5, 0);
    try testing.expectEqual(0, br);
    // The last letters a BitRack holds use its high half.
    add(&br, 31);
    add(&br, 31);
    try testing.expectEqual(2, count(br, 31));
}

test "compatibility" {
    const bytes = try testdata.loadFile("letterdistributions/english");
    defer testing.allocator.free(bytes);
    var ld = try tiles.LetterDistribution.parse(testing.allocator, "english", bytes);
    defer ld.deinit(testing.allocator);
    try checkCompatible(&ld, 15);
    try checkCompatible(&ld, 21);
    // Three blanks.
    ld.counts[0] = 3;
    try testing.expectError(error.IncompatibleDistribution, checkCompatible(&ld, 15));
    // 17 E and 2 blanks: capped at 15 letters on a 15 board, 19 on a 21.
    ld.counts[0] = 2;
    ld.counts[5] = 17;
    try checkCompatible(&ld, 15);
    try testing.expectError(error.IncompatibleDistribution, checkCompatible(&ld, 21));
}

const Loaded = struct {
    bytes: []align(4) u8,
    wmp: Wmp,

    fn load(name: []const u8) !Loaded {
        var buf: [64]u8 = undefined;
        const bytes = try testdata.loadFile(try std.fmt.bufPrint(&buf, "lexica/{s}.wmp", .{name}));
        errdefer testing.allocator.free(bytes);
        return .{ .bytes = bytes, .wmp = try .init(testing.allocator, bytes) };
    }

    fn deinit(self: *Loaded) void {
        self.wmp.deinit(testing.allocator);
        testing.allocator.free(self.bytes);
    }
};

/// The words a lookup writes, sorted.
fn lookupSorted(w: *const Wmp, br: BitRack, len: usize, buf: []u8) []u8 {
    const n = w.writeWords(br, len, buf);
    const out = buf[0..n];
    var i: usize = len;
    // Insertion sort by word, fine for tests.
    while (i < n) : (i += len) {
        var j = i;
        while (j > 0 and std.mem.order(u8, out[j..][0..len], out[j - len ..][0..len]) == .lt) : (j -= len) {
            var tmp: [32]u8 = undefined;
            @memcpy(tmp[0..len], out[j..][0..len]);
            @memcpy(out[j..][0..len], out[j - len ..][0..len]);
            @memcpy(out[j - len ..][0..len], tmp[0..len]);
        }
    }
    return out;
}

test "load a word map and look up (CSW24)" {
    var l = try Loaded.load("CSW24");
    defer l.deinit();
    try testing.expectEqual(version, l.wmp.version);
    try testing.expectEqual(15, l.wmp.board_dim);
    try testing.expect(l.wmp.max_word_lookup_bytes > 0);

    var buf: [result_buffer_size]u8 = undefined;
    // RETINAS and its anagrams.
    const words = lookupSorted(&l.wmp, fromWord(&letters("RETINAS")), 7, &buf);
    var found = false;
    var i: usize = 0;
    while (i < words.len) : (i += 7) found = found or std.mem.eql(u8, words[i..][0..7], &letters("RETINAS"));
    try testing.expect(found and words.len >= 8 * 7);
    try testing.expectEqualSlices(u8, &(letters("ACT") ++ letters("CAT")), lookupSorted(&l.wmp, fromWord(&letters("CAT")), 3, &buf));
    try testing.expectEqual(0, l.wmp.writeWords(fromWord(&letters("XYZ")), 3, &buf));
    // Out of range lengths.
    try testing.expectEqual(0, l.wmp.writeWords(fromWord(&letters("A")), 1, &buf));
    try testing.expectEqual(0, l.wmp.writeWords(fromWord(&letters("AB")), 16, &buf));
}

/// Whether `word` can be made of `rack` (blanks standing in for anything).
fn makes(rack: BitRack, word: []const u8) bool {
    var r = rack;
    for (word) |ml| {
        if (count(r, ml) > 0) {
            take(&r, ml);
        } else if (count(r, 0) > 0) {
            take(&r, 0);
        } else return false;
    }
    return true;
}

test "lookups agree with the lexicon (NWL23)" {
    var l = try Loaded.load("NWL23");
    defer l.deinit();
    const kwg_bytes = try testdata.loadFile("lexica/gaddag/NWL23.kwg");
    defer testing.allocator.free(kwg_bytes);
    const k = try kwg_mod.Kwg.init(kwg_bytes);

    // Every word of lengths 2 to 8, by length.
    var by_len: [9]std.ArrayList(u8) = @splat(.empty);
    defer for (&by_len) |*b| b.deinit(testing.allocator);
    var prefix: [8]u8 = undefined;
    try collect(k, k.dawgRoot(), &prefix, 0, &by_len);

    var prng: std.Random.DefaultPrng = .init(7);
    const random = prng.random();
    var buf: [result_buffer_size]u8 = undefined;
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(testing.allocator);
    var checked: usize = 0;
    for (0..600) |i| {
        const len = 2 + i % 7;
        const words = by_len[len].items;
        // Half the racks are words' racks (so have words), half random.
        var rack: BitRack = 0;
        if (i % 2 == 0) {
            const w = random.uintLessThan(usize, words.len / len);
            rack = fromWord(words[w * len ..][0..len]);
        } else {
            for (0..len) |_| add(&rack, 1 + random.uintLessThan(u8, 26));
        }
        // And zero, one or two blanks.
        for (0..(i / 2) % 3) |_| {
            var ml: u8 = 1;
            while (count(rack, ml) == 0) ml += 1;
            take(&rack, ml);
            add(&rack, 0);
        }
        want.clearRetainingCapacity();
        var off: usize = 0;
        while (off < words.len) : (off += len) {
            if (makes(rack, words[off..][0..len])) try want.appendSlice(testing.allocator, words[off..][0..len]);
        }
        const got = lookupSorted(&l.wmp, rack, len, &buf);
        try testing.expectEqualSlices(u8, want.items, got);
        try testing.expect(got.len <= l.wmp.max_word_lookup_bytes or count(rack, 0) < 2);
        if (got.len > 0) checked += 1;
    }
    try testing.expect(checked > 300);
}

fn collect(k: kwg_mod.Kwg, node: u32, prefix: []u8, len: usize, by_len: *[9]std.ArrayList(u8)) !void {
    var i = node;
    while (true) : (i += 1) {
        prefix[len] = k.tile(i);
        if (k.accepts(i) and len + 1 >= 2) try by_len[len + 1].appendSlice(testing.allocator, prefix[0 .. len + 1]);
        if (len + 1 < 8 and k.arcIndex(i) != 0) try collect(k, k.arcIndex(i), prefix, len + 1, by_len);
        if (k.isEnd(i)) break;
    }
}

test "the filter lets every rack with words through" {
    var l = try Loaded.load("NWL23");
    defer l.deinit();
    for (2..16) |len| {
        const wfl = &l.wmp.wfls[len];
        for (wfl.words.entries) |*e| {
            try testing.expect(wfl.mayHaveWords(mix(entryRack(e))));
        }
    }
}

test "corrupt files are refused" {
    var l = try Loaded.load("NWL20");
    defer l.deinit();
    try testing.expectError(error.InvalidWmp, Wmp.init(testing.allocator, l.bytes[0 .. l.bytes.len - 1]));
    try testing.expectError(error.InvalidWmp, Wmp.init(testing.allocator, l.bytes[0..5]));
    var bad = try testing.allocator.dupe(u8, l.bytes[0..64]);
    defer testing.allocator.free(bad);
    bad[0] = 2;
    try testing.expectError(error.UnsupportedVersion, Wmp.init(testing.allocator, bad));
}
