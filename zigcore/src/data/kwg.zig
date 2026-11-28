//! KWG (Kurnia Word Graph) reader, a port of word-golib's `kwg` package.
//! See https://github.com/andy-k/wolges/blob/main/details.txt.
//!
//! A KWG is an array of little-endian u32 nodes:
//!
//!     bits  0..21  arc index (the first node of the child list)
//!     bit   22     last node of its list
//!     bit   23     accepts (the path to here spells a word)
//!     bits 24..31  tile (0 is the GADDAG separator)
//!
//! Node 0's arc is the DAWG root and node 1's the GADDAG root.

const std = @import("std");
const builtin = @import("builtin");
const tiles = @import("tiles.zig");
const MachineLetter = tiles.MachineLetter;
const LetterSet = tiles.LetterSet;

comptime {
    // The nodes are read in place.
    if (builtin.cpu.arch.endian() != .little) @compileError("KWG reading assumes a little-endian target");
}

pub const arc_mask: u32 = 0x3fffff;
pub const is_end_bit: u32 = 0x400000;
pub const accepts_bit: u32 = 0x800000;
pub const tile_shift = 24;

pub const Kwg = struct {
    nodes: []const u32,
    /// For each node, the tiles of the sibling list from it to the list's
    /// end, as a bit set: move generation visits only the arcs it can use
    /// and finds each by counting bits (lists are sorted by tile). MAGPIE's
    /// kwg sibling tiles. Empty until buildSiblingTiles.
    sibling_tiles: []const u64 = &.{},

    /// Uses `bytes` in place; they must outlive the Kwg.
    pub fn init(bytes: []align(4) const u8) error{InvalidKwg}!Kwg {
        if (bytes.len % 4 != 0 or bytes.len < 8) return error.InvalidKwg;
        return .{ .nodes = std.mem.bytesAsSlice(u32, bytes) };
    }

    /// The sibling tile sets of every node, for `sibling_tiles`; the
    /// caller owns them.
    pub fn buildSiblingTiles(self: Kwg, gpa: std.mem.Allocator) error{OutOfMemory}![]u64 {
        const out = try gpa.alloc(u64, self.nodes.len);
        var i = self.nodes.len;
        var next: u64 = 0;
        while (i > 0) {
            i -= 1;
            const n = self.nodes[i];
            const bit = @as(u64, 1) << @intCast((n >> tile_shift) & 63);
            next = bit | (if (n & is_end_bit != 0) 0 else next);
            out[i] = next;
        }
        return out;
    }

    /// The node of the list starting at `node` with tile `bit`, whose list
    /// tiles are `list_tiles` (which must hold it).
    pub inline fn listNode(node: u32, list_tiles: u64, bit: u64) u32 {
        return node + @popCount(list_tiles & (bit - 1));
    }

    pub inline fn tile(self: Kwg, i: u32) u8 {
        return @intCast(self.nodes[i] >> tile_shift);
    }

    pub inline fn accepts(self: Kwg, i: u32) bool {
        return self.nodes[i] & accepts_bit != 0;
    }

    pub inline fn isEnd(self: Kwg, i: u32) bool {
        return self.nodes[i] & is_end_bit != 0;
    }

    pub inline fn arcIndex(self: Kwg, i: u32) u32 {
        return self.nodes[i] & arc_mask;
    }

    pub inline fn gaddagRoot(self: Kwg) u32 {
        return self.arcIndex(1);
    }

    pub inline fn dawgRoot(self: Kwg) u32 {
        return self.arcIndex(0);
    }

    /// The child list reached from the list at `node` by `letter`, or 0.
    pub fn nextNodeIdx(self: Kwg, node: u32, letter: MachineLetter) u32 {
        var i = node;
        while (true) : (i += 1) {
            if (self.tile(i) == letter) return self.arcIndex(i);
            if (self.isEnd(i)) return 0;
        }
    }

    /// Whether `letter` (unblanked) ends a word in the list at `node`.
    pub fn inLetterSet(self: Kwg, letter: MachineLetter, node: u32) bool {
        const l = tiles.unblank(letter);
        var i = node;
        while (true) : (i += 1) {
            if (self.tile(i) == l) return self.accepts(i);
            if (self.isEnd(i)) return false;
        }
    }

    /// The letters that end a word in the list at `node`.
    pub fn letterSet(self: Kwg, node: u32) LetterSet {
        var ls: LetterSet = 0;
        var i = node;
        while (true) : (i += 1) {
            if (self.accepts(i)) ls |= @as(LetterSet, 1) << @intCast(self.tile(i));
            if (self.isEnd(i)) return ls;
        }
    }

    /// The letters that end a word (`letters`) and the letters that have
    /// children (`extensions`) in the list at `node`, both without the
    /// separator. Mirrors MAGPIE's kwg_get_letter_sets.
    pub fn letterSets(self: Kwg, node: u32) struct { letters: LetterSet, extensions: LetterSet } {
        var ls: LetterSet = 0;
        var es: LetterSet = 0;
        if (node == 0) return .{ .letters = 0, .extensions = 0 };
        var i = node;
        while (true) : (i += 1) {
            const t = self.tile(i);
            if (t != 0) {
                const bit = @as(LetterSet, 1) << @intCast(t);
                es |= bit;
                if (self.accepts(i)) ls |= bit;
            }
            if (self.isEnd(i)) return .{ .letters = ls, .extensions = es };
        }
    }

    /// Whether `word` is in the DAWG half of the graph.
    pub fn acceptsWord(self: Kwg, word: []const MachineLetter) bool {
        if (word.len == 0) return false;
        var node = self.dawgRoot();
        for (word[0 .. word.len - 1]) |ml| {
            node = self.nextNodeIdx(node, tiles.unblank(ml));
            if (node == 0) return false;
        }
        return self.inLetterSet(word[word.len - 1], node);
    }
};

/// Word counts below each node, for mapping a word to its index in the
/// DAWG (used by KLV leave files). Port of word-golib's CountWords.
pub const WordCounts = struct {
    counts: []i32,

    pub fn init(gpa: std.mem.Allocator, kwg: Kwg) !WordCounts {
        const counts = try gpa.alloc(i32, kwg.nodes.len);
        @memset(counts, 0);
        var wc: WordCounts = .{ .counts = counts };
        var p = kwg.nodes.len;
        while (p > 0) {
            p -= 1;
            _ = wc.countAt(kwg, @intCast(p));
        }
        return wc;
    }

    pub fn deinit(self: *WordCounts, gpa: std.mem.Allocator) void {
        gpa.free(self.counts);
        self.* = undefined;
    }

    fn countAt(self: *WordCounts, kwg: Kwg, p: u32) i32 {
        if (p >= self.counts.len) return 0;
        // -1 marks a node being counted; the graph has no cycles.
        std.debug.assert(self.counts[p] != -1);
        if (self.counts[p] == 0) {
            self.counts[p] = -1;
            var n: i32 = @intFromBool(kwg.accepts(p));
            if (kwg.arcIndex(p) != 0) n += self.countAt(kwg, kwg.arcIndex(p));
            if (!kwg.isEnd(p)) n += self.countAt(kwg, p + 1);
            self.counts[p] = n;
        }
        return self.counts[p];
    }

    /// The index of `word` among the words below `node`, or -1.
    pub fn wordIndex(self: WordCounts, kwg: Kwg, start: u32, word: []const MachineLetter) i32 {
        var node = start;
        var idx: i32 = 0;
        var li: usize = 0;
        while (node != 0) {
            idx += self.counts[node];
            while (kwg.tile(node) != word[li]) {
                if (kwg.isEnd(node)) return -1;
                node += 1;
            }
            idx -= self.counts[node];
            li += 1;
            if (li == word.len) return if (kwg.accepts(node)) idx else -1;
            if (kwg.accepts(node)) idx += 1;
            node = kwg.arcIndex(node);
        }
        return -1;
    }
};

/// KLV2 leave values: a u32 node count, a KWG of leaves (sorted racks), a
/// u32 leave count and one little-endian f32 per leave. Port of
/// equity.ReadKLV.
pub const Klv = struct {
    kwg: Kwg,
    counts: WordCounts,
    values: []const f32,

    /// Uses `bytes` in place; they must outlive the Klv.
    pub fn init(gpa: std.mem.Allocator, bytes: []align(4) const u8) !Klv {
        const words = std.mem.bytesAsSlice(u32, bytes[0 .. bytes.len - bytes.len % 4]);
        if (words.len < 1) return error.InvalidKlv;
        const num_nodes = words[0];
        if (words.len < 2 + @as(usize, num_nodes)) return error.InvalidKlv;
        const kwg = try Kwg.init(@alignCast(std.mem.sliceAsBytes(words[1 .. 1 + num_nodes])));
        const num_leaves = words[1 + num_nodes];
        const values_start = 2 + @as(usize, num_nodes);
        if (words.len < values_start + num_leaves) return error.InvalidKlv;
        const values = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(words[values_start .. values_start + num_leaves]));
        return .{ .kwg = kwg, .counts = try .init(gpa, kwg), .values = @alignCast(values) };
    }

    pub fn deinit(self: *Klv, gpa: std.mem.Allocator) void {
        self.counts.deinit(gpa);
        self.* = undefined;
    }

    /// The value of a leave, whose tiles must be sorted. 0 if unknown.
    pub fn leaveValue(self: Klv, leave: []const MachineLetter) f64 {
        if (leave.len == 0) return 0;
        const idx = self.counts.wordIndex(self.kwg, self.kwg.dawgRoot(), leave);
        if (idx < 0 or idx >= self.values.len) return 0;
        return self.values[@intCast(idx)];
    }
};

// Tests ----------------------------------------------------------------------

const testing = std.testing;
const testdata = @import("../testing/testdata.zig");

test "NWL20 words and letter sets" {
    const bytes = try testdata.loadFile("lexica/gaddag/NWL20.kwg");
    defer testing.allocator.free(bytes);
    const k = try Kwg.init(bytes);

    // Letters: A=1 ... Z=26.
    try testing.expect(k.acceptsWord(&.{ 17, 9 })); // QI
    try testing.expect(k.acceptsWord(&.{ 26, 25, 26, 26, 25, 22, 1 })); // ZYZZYVA
    try testing.expect(!k.acceptsWord(&.{ 26, 26 })); // ZZ
    try testing.expect(!k.acceptsWord(&.{17})); // Q

    // GADDAG: from the root, "Q" reversed then the separator then nothing:
    // letters that follow Q-reversed and end a word are those X where XQ? No:
    // in a GADDAG, path [I, Q] (reversed prefix "QI") accepts.
    const after_i = k.nextNodeIdx(k.gaddagRoot(), 9);
    try testing.expect(after_i != 0);
    try testing.expect(k.inLetterSet(17, after_i)); // QI read backwards
    try testing.expect(k.letterSet(after_i) & (1 << 17) != 0);
}

test "NWL20 leave values" {
    const bytes = try testdata.loadFile("lexica/gaddag/NWL20.klv2");
    defer testing.allocator.free(bytes);
    var klv = try Klv.init(testing.allocator, bytes);
    defer klv.deinit(testing.allocator);
    // Blanks and S are good, Q alone is bad, an empty leave is 0.
    try testing.expect(klv.leaveValue(&.{0}) > 20);
    try testing.expect(klv.leaveValue(&.{19}) > 5);
    try testing.expect(klv.leaveValue(&.{17}) < 0);
    try testing.expectEqual(0, klv.leaveValue(&.{}));
}

test "reachable sibling lists are sorted by tile" {
    // Move generation finds the GADDAG separator (tile 0) by looking at a
    // list's first node only. Every list reachable from the roots is
    // checked (a file may end with nodes no arc leads to).
    for ([_][]const u8{ "lexica/gaddag/NWL20.kwg", "lexica/gaddag/CSW24.kwg", "lexica/gaddag/FRA20.kwg", "lexica/gaddag/OSPS49.kwg" }) |path| {
        const bytes = testdata.loadFile(path) catch continue;
        defer testing.allocator.free(bytes);
        const k = try Kwg.init(bytes);
        var seen = try std.DynamicBitSet.initEmpty(testing.allocator, k.nodes.len);
        defer seen.deinit();
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(testing.allocator);
        try stack.appendSlice(testing.allocator, &.{ k.dawgRoot(), k.gaddagRoot() });
        while (stack.pop()) |start| {
            if (start == 0 or seen.isSet(start)) continue;
            seen.set(start);
            var i = start;
            while (true) : (i += 1) {
                if (k.arcIndex(i) != 0) try stack.append(testing.allocator, k.arcIndex(i));
                if (k.isEnd(i)) break;
                try testing.expect(k.tile(i) < k.tile(i + 1));
            }
        }
    }
}
