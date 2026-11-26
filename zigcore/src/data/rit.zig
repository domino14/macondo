//! The rack info table: what the best-play search works out for a full rack
//! before it looks at the board, precomputed for every rack of a lexicon
//! and leave file. Port of MAGPIE's rack info table (ent/rack_info_table.h,
//! impl/rack_info_table_maker.c), in the same layout but for one thing:
//! MAGPIE rounds leave values to thousandths, and this table keeps the leave
//! file's exact values, so that equities stay those of Macondo's Go engine
//! to the last bit.
//!
//! A full (7-tile) rack's entry holds:
//!   - the raw leave value of each of its 128 subsets, by leave map index,
//!     and the best of each size (the pre-endgame adjustment, which depends
//!     on the bag, is added when the entry is used);
//!   - its best exchange, when every exchange size is allowed and no
//!     pre-endgame adjustment applies;
//!   - which word lengths its subracks make alone, and the best leave of
//!     the subracks that do;
//!   - for each number of tiles played, the letters that make a word with
//!     some subrack of that size (one board tile played through);
//!   - for 6 and 7 tiles played (blankless racks), by word length, the
//!     letters some word made of a subrack and more letters has outside it
//!     (several board tiles played through).
//! The last three only let the search skip placements where no word can
//! exist, so the moves found are the same with or without the table.
//!
//! The file is the table as it lies in memory (little-endian), so it is
//! used in place, mapped or read.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const kwg_mod = @import("kwg.zig");
const wmp = @import("wmp.zig");
const equity = @import("../equity/equity.zig");
const leavemap = @import("../movegen/leavemap.zig");
const Move = @import("../board/move.zig").Move;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const BitRack = wmp.BitRack;

comptime {
    if (builtin.cpu.arch.endian() != .little) @compileError("the rack info table is little-endian");
}

pub const rack_size = 7;
pub const leaves_per_entry = 1 << rack_size;
/// Multi-playthrough masks: for 7 tiles played, word lengths 8 to 15; for
/// 6, 7 to 15 (a 15x15 board, the only one word maps are made for).
pub const tp7_min_length = rack_size + 1;
pub const tp7_lengths = 8;
pub const tp6_min_length = rack_size;
pub const tp6_lengths = 9;

pub const magic = "ZRIT".*;
pub const version: u32 = 1;

pub const Header = extern struct {
    magic: [4]u8 = magic,
    version: u32 = version,
    rack_size: u8 = rack_size,
    alphabet_size: u8,
    pad: [2]u8 = @splat(0),
    num_buckets: u32,
    num_entries: u32,
    pad2: u32 = 0,
    /// What the table was made from: Wyhash of the word graph's and the
    /// leave file's bytes.
    kwg_hash: u64,
    klv_hash: u64,
    pad3: [24]u8 = @splat(0),
};

pub const Entry = extern struct {
    rack: BitRack,
    /// By leave map index (see leavemap.LeaveMap): the leave file's value of
    /// the subset kept, 0 if it has none.
    leaves: [leaves_per_entry]f32,
    /// By leave size: the best of leaves.
    best_leaves: [rack_size + 1]f32,
    /// By leave size: the best leave of a subrack whose played tiles make a
    /// word alone; -inf if none does.
    np_best_leaves: [rack_size + 1]f32,
    /// By leave size: letters L such that some subrack of the size played
    /// makes a word with L.
    playthrough_union: [rack_size + 1]u32,
    multi_pt_tp7: [tp7_lengths]u32,
    multi_pt_tp6: [tp6_lengths]u32,
    /// Bit n: some subrack of n tiles makes a word alone.
    np_has_word: u8,
    /// The best exchange: how many tiles, and which (machine-letter
    /// order).
    exchange_len: u8,
    exchange: [rack_size]MachineLetter,
    pad: [3]u8 = @splat(0),
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64);
    std.debug.assert(@sizeOf(Entry) % 16 == 0);
}

pub const Error = error{InvalidRit};

pub const Rit = struct {
    header: *const Header,
    bucket_starts: []const u32,
    entries: []const Entry,

    /// The table in `bytes` (16-byte aligned), which must outlive it.
    pub fn init(bytes: []align(16) const u8) Error!Rit {
        if (bytes.len < @sizeOf(Header)) return error.InvalidRit;
        const h: *const Header = @ptrCast(bytes.ptr);
        if (!std.mem.eql(u8, &h.magic, &magic) or h.version != version or h.rack_size != rack_size) return error.InvalidRit;
        if (h.num_buckets == 0 or !std.math.isPowerOfTwo(h.num_buckets)) return error.InvalidRit;
        const starts_end = bucketsEnd(h.num_buckets);
        const size = std.math.add(usize, starts_end, std.math.mul(usize, h.num_entries, @sizeOf(Entry)) catch return error.InvalidRit) catch return error.InvalidRit;
        if (bytes.len != size) return error.InvalidRit;
        const starts: [*]const u32 = @ptrCast(@alignCast(bytes.ptr + @sizeOf(Header)));
        const entries: [*]const Entry = @ptrCast(@alignCast(bytes.ptr + starts_end));
        const r: Rit = .{
            .header = h,
            .bucket_starts = starts[0 .. h.num_buckets + 1],
            .entries = entries[0..h.num_entries],
        };
        if (r.bucket_starts[0] != 0 or r.bucket_starts[h.num_buckets] != h.num_entries) return error.InvalidRit;
        for (r.bucket_starts[0..h.num_buckets], r.bucket_starts[1..]) |a, b| {
            if (a > b) return error.InvalidRit;
        }
        return r;
    }

    /// Whether the table was made from these files.
    pub fn madeFrom(self: *const Rit, kwg_bytes: []const u8, klv_bytes: []const u8) bool {
        return self.header.kwg_hash == hashBytes(kwg_bytes) and self.header.klv_hash == hashBytes(klv_bytes);
    }

    pub fn lookup(self: *const Rit, br: BitRack) ?*const Entry {
        const bucket = @as(u32, @truncate(wmp.mix(br))) & (self.header.num_buckets - 1);
        for (self.entries[self.bucket_starts[bucket]..self.bucket_starts[bucket + 1]]) |*e| {
            if (e.rack == br) return e;
        }
        return null;
    }
};

pub fn hashBytes(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0, bytes);
}

/// Where the entries start: after the header and the bucket starts,
/// rounded up to 16 bytes.
fn bucketsEnd(num_buckets: u32) usize {
    return std.mem.alignForward(usize, @sizeOf(Header) + (@as(usize, num_buckets) + 1) * 4, 16);
}

// The maker -----------------------------------------------------------------

/// What a table is made from. The word map must be for the word graph's
/// lexicon (it makes the word lists; the graph is only hashed).
pub const Inputs = struct {
    ld: *const tiles.LetterDistribution,
    kwg_bytes: []const u8,
    klv: *const kwg_mod.Klv,
    klv_bytes: []const u8,
    word_map: *const wmp.Wmp,
};

pub const MakeError = Allocator.Error || error{ IncompatibleDistribution, ThreadFailed };

/// Makes the table for every full rack `inputs.ld` can draw, on `threads`
/// threads (1 for none). Returns the file's bytes.
pub fn make(gpa: Allocator, inputs: Inputs, threads: usize) MakeError![]align(16) u8 {
    if (inputs.ld.size > wmp.max_alphabet_size) return error.IncompatibleDistribution;
    var racks: std.ArrayList(BitRack) = .empty;
    defer racks.deinit(gpa);
    try enumerateRacks(gpa, inputs.ld, 0, rack_size, 0, &racks);
    return build(gpa, inputs, racks.items, threads, true);
}

/// A table of `racks` only (full racks of the distribution), with the
/// same entries a whole table has: for tests, where making the whole
/// table takes too long. The multi-playthrough masks are worked out rack
/// by rack instead of by the flipped walk.
pub fn makeFor(gpa: Allocator, inputs: Inputs, racks: []const BitRack) MakeError![]align(16) u8 {
    if (inputs.ld.size > wmp.max_alphabet_size) return error.IncompatibleDistribution;
    return build(gpa, inputs, racks, 1, false);
}

fn build(gpa: Allocator, inputs: Inputs, racks_in: []const BitRack, threads: usize, whole: bool) MakeError![]align(16) u8 {
    const ld = inputs.ld;
    const racks: struct { items: []const BitRack } = .{ .items = racks_in };
    const n: u32 = @intCast(racks.items.len);
    const num_buckets: u32 = @max(1, std.math.ceilPowerOfTwoAssert(u32, @max(n, 1)));

    const starts_end = bucketsEnd(num_buckets);
    const bytes = try gpa.alignedAlloc(u8, .@"16", starts_end + @as(usize, n) * @sizeOf(Entry));
    errdefer gpa.free(bytes);
    @memset(bytes[0..starts_end], 0);
    const h: *Header = @ptrCast(bytes.ptr);
    h.* = .{
        .alphabet_size = ld.size,
        .num_buckets = num_buckets,
        .num_entries = n,
        .kwg_hash = hashBytes(inputs.kwg_bytes),
        .klv_hash = hashBytes(inputs.klv_bytes),
    };
    const starts: [*]u32 = @ptrCast(@alignCast(bytes.ptr + @sizeOf(Header)));
    const entries: [*]Entry = @ptrCast(@alignCast(bytes.ptr + starts_end));

    // Racks by bucket.
    const counts = try gpa.alloc(u32, num_buckets);
    defer gpa.free(counts);
    @memset(counts, 0);
    for (racks.items) |br| counts[bucketOf(br, num_buckets)] += 1;
    var offset: u32 = 0;
    for (0..num_buckets) |b| {
        starts[b] = offset;
        offset += counts[b];
    }
    starts[num_buckets] = offset;
    @memset(counts, 0);
    for (racks.items) |br| {
        const b = bucketOf(br, num_buckets);
        entries[starts[b] + counts[b]].rack = br;
        counts[b] += 1;
    }

    // Each entry on its own (in parallel), then the multi-playthrough
    // masks over the whole table.
    const all = entries[0..n];
    const workers = @max(1, @min(threads, 256));
    if (workers == 1) {
        computeEntries(inputs, all);
    } else {
        var handles: [256]std.Thread = undefined;
        var spawned: usize = 0;
        defer for (handles[0..spawned]) |t| t.join();
        const chunk = (all.len + workers - 1) / workers;
        var i: usize = 0;
        while (i < all.len) : (i += chunk) {
            const part = all[i..@min(all.len, i + chunk)];
            handles[spawned] = std.Thread.spawn(.{}, computeEntries, .{ inputs, part }) catch return error.ThreadFailed;
            spawned += 1;
        }
    }
    const rit = Rit.init(bytes) catch unreachable;
    if (whole) {
        try addMultiPlaythroughMasks(gpa, inputs, &rit, all);
    } else {
        addMultiPlaythroughMasksFor(inputs, all);
    }
    return bytes;
}

/// The multi-playthrough masks of `entries`, word by word: for each
/// distinct subrack S of 6 or 7 tiles of a blankless rack and each
/// blankless word M of a covered length containing S, the letters of M
/// outside S.
fn addMultiPlaythroughMasksFor(inputs: Inputs, entries: []Entry) void {
    const w = inputs.word_map;
    for (entries) |*e| {
        e.multi_pt_tp6 = @splat(0);
        e.multi_pt_tp7 = @splat(0);
    }
    var len: usize = tp6_min_length;
    while (len < tp6_min_length + tp6_lengths and len <= w.board_dim) : (len += 1) {
        for (w.wfls[len].words.entries) |*we| {
            const word = wmp.entryRack(we);
            if (wmp.count(word, 0) != 0) continue;
            const word_letters = letterSet(word);
            for (entries) |*e| {
                if (wmp.count(e.rack, 0) != 0) continue;
                // A subrack of 6 or 7 tiles has every letter of the rack
                // but at most one.
                const rack_letters = letterSet(e.rack);
                const missing = rack_letters & ~word_letters;
                if (@popCount(missing) > 1) continue;
                if (len >= tp7_min_length and missing == 0 and within(e.rack, word)) {
                    e.multi_pt_tp7[len - tp7_min_length] |= outside(word, e.rack);
                }
                var letters = rack_letters;
                while (letters != 0) : (letters &= letters - 1) {
                    const ml: u8 = @intCast(@ctz(letters));
                    var sub = e.rack;
                    wmp.take(&sub, ml);
                    if (letterSet(sub) & ~word_letters != 0 or !within(sub, word)) continue;
                    e.multi_pt_tp6[len - tp6_min_length] |= outside(word, sub);
                }
            }
        }
    }
}

/// The letters a multiset has.
fn letterSet(br: BitRack) u32 {
    var set: u32 = 0;
    for (0..wmp.max_alphabet_size) |ml| {
        if (wmp.count(br, @intCast(ml)) != 0) set |= @as(u32, 1) << @intCast(ml);
    }
    return set;
}

/// The letters of `word` beyond those of `sub` (not the blank).
fn outside(word: BitRack, sub: BitRack) u32 {
    var mask: u32 = 0;
    for (1..wmp.max_alphabet_size) |l| {
        if (wmp.count(word, @intCast(l)) > wmp.count(sub, @intCast(l))) mask |= @as(u32, 1) << @intCast(l);
    }
    return mask;
}

/// Whether multiset `a` is within `b`.
fn within(a: BitRack, b: BitRack) bool {
    for (0..wmp.max_alphabet_size) |ml| {
        if (wmp.count(a, @intCast(ml)) > wmp.count(b, @intCast(ml))) return false;
    }
    return true;
}

fn bucketOf(br: BitRack, num_buckets: u32) u32 {
    return @as(u32, @truncate(wmp.mix(br))) & (num_buckets - 1);
}

/// Every multiset of `remaining` more tiles from letters `ml` on, within
/// the distribution's counts.
fn enumerateRacks(gpa: Allocator, ld: *const tiles.LetterDistribution, ml: usize, remaining: u32, cur: BitRack, out: *std.ArrayList(BitRack)) Allocator.Error!void {
    if (remaining == 0) return out.append(gpa, cur);
    if (ml >= ld.size) return;
    var br = cur;
    const most = @min(ld.counts[ml], remaining);
    var k: u32 = 0;
    while (true) : (k += 1) {
        try enumerateRacks(gpa, ld, ml + 1, remaining - k, br, out);
        if (k == most) break;
        wmp.add(&br, @intCast(ml));
    }
}

fn computeEntries(inputs: Inputs, entries: []Entry) void {
    for (entries) |*e| computeEntry(inputs, e);
}

/// What the leave walk visits a rack's subsets with.
const EntryBuilder = struct {
    inputs: Inputs,
    entry: *Entry,
    full: Rack,
    full_br: BitRack,
    best_exchange: Move = undefined,
    best_exchange_eq: f64 = 0,
    have_exchange: bool = false,

    fn visit(ctx: *anyopaque, leave: *const Rack, value: f64) void {
        const self: *EntryBuilder = @ptrCast(@alignCast(ctx));
        const e = self.entry;
        const ld = self.inputs.ld;
        const leave_size = leave.num_tiles;
        const played_size = rack_size - leave_size;

        // The exchange of everything but the leave, ranked as the search
        // ranks exchanges.
        if (played_size > 0) {
            var m: Move = .{ .kind = .exchange, .len = @intCast(played_size), .tiles_played = @intCast(played_size) };
            var k: usize = 0;
            for (self.full.counts[0..ld.size], leave.counts[0..ld.size], 0..) |all, kept, ml| {
                for (kept..all) |_| {
                    m.tiles[k] = @intCast(ml);
                    k += 1;
                }
            }
            if (!self.have_exchange or equity.better(&m, value, &self.best_exchange, self.best_exchange_eq)) {
                self.best_exchange = m;
                self.best_exchange_eq = value;
                self.have_exchange = true;
            }
        }

        const played = self.full_br - wmp.fromRack(leave);
        const w = self.inputs.word_map;
        if (played_size >= wmp.minimum_word_length and w.entry(played, played_size) != null) {
            e.np_has_word |= @as(u8, 1) << @intCast(played_size);
            const v: f32 = @floatCast(value);
            if (v > e.np_best_leaves[leave_size]) e.np_best_leaves[leave_size] = v;
        }
        if (played_size >= 1) {
            var mask: u32 = 0;
            for (1..ld.size) |l| {
                var q = played;
                wmp.add(&q, @intCast(l));
                if (w.entry(q, played_size + 1) != null) mask |= @as(u32, 1) << @intCast(l);
            }
            e.playthrough_union[leave_size] |= mask;
        }
    }
};

fn computeEntry(inputs: Inputs, e: *Entry) void {
    const ld = inputs.ld;
    var rack: Rack = .{};
    for (0..ld.size) |ml| {
        for (0..wmp.count(e.rack, @intCast(ml))) |_| rack.add(@intCast(ml));
    }
    const br = e.rack;
    e.* = std.mem.zeroes(Entry);
    e.rack = br;
    e.np_best_leaves = @splat(-std.math.inf(f32));

    // The raw values: a bag big enough for no pre-endgame adjustment.
    const ctx: equity.Context = .{
        .ld = ld,
        .klv = inputs.klv,
        .peg = &.{},
        .board_dim = 15,
        .board_empty = false,
        .tiles_in_bag = 100,
        .opp_rack_score = 0,
    };
    var lm: leavemap.LeaveMap = .{};
    lm.init(&rack);
    // Indices no subset has stay 0, so that a table is the same every time
    // it is made.
    lm.values = @splat(0);
    var b: EntryBuilder = .{ .inputs = inputs, .entry = e, .full = rack, .full_br = br };
    var best: leavemap.BestLeaves = undefined;
    var r = rack;
    leavemap.populateVisiting(&lm, &r, inputs.klv, &ctx, &best, .{ .ctx = &b, .visit = EntryBuilder.visit });
    for (&e.leaves, lm.values) |*d, v| d.* = @floatCast(v);
    for (&e.best_leaves, best) |*d, v| d.* = @floatCast(v);
    if (b.have_exchange) {
        e.exchange_len = b.best_exchange.len;
        @memcpy(e.exchange[0..b.best_exchange.len], b.best_exchange.tileSlice());
    }
}

/// The multi-playthrough masks (MAGPIE's flipped walk): for each blankless
/// word M of a covered length and each subrack S of 6 or 7 tiles inside it,
/// the letters of M outside S go into S's result for M's length; then each
/// S's result goes to every full rack that contains S.
fn addMultiPlaythroughMasks(gpa: Allocator, inputs: Inputs, rit: *const Rit, entries: []Entry) Allocator.Error!void {
    const ld = inputs.ld;
    const w = inputs.word_map;
    const Result = [tp6_lengths]u32;
    inline for (.{ rack_size - 1, rack_size }) |subrack_size| {
        const min_len = if (subrack_size == rack_size) tp7_min_length else tp6_min_length;
        const num_lens = if (subrack_size == rack_size) tp7_lengths else tp6_lengths;
        var results: std.AutoHashMapUnmanaged(BitRack, Result) = .empty;
        defer results.deinit(gpa);
        var len: usize = min_len;
        while (len < min_len + num_lens and len <= w.board_dim) : (len += 1) {
            const table = &w.wfls[len].words;
            for (table.entries) |*we| {
                const word = wmp.entryRack(we);
                if (wmp.count(word, 0) != 0) continue;
                var counts: [wmp.max_alphabet_size]u8 = undefined;
                for (&counts, 0..) |*c, ml| c.* = wmp.count(word, @intCast(ml));
                try walkSubracks(gpa, ld, &counts, 1, subrack_size, 0, &results, len - min_len);
            }
        }
        var it = results.iterator();
        while (it.next()) |kv| {
            var any: u32 = 0;
            for (kv.value_ptr[0..num_lens]) |x| any |= x;
            if (any == 0) continue;
            supersets(ld, rit, entries, kv.key_ptr.*, rack_size - subrack_size, 1, kv.value_ptr, subrack_size == rack_size);
        }
    }
}

/// The subracks of `remaining` more tiles of a word (letter counts
/// `counts`) from letter `ml` on, within the distribution; at each, the
/// word's other letters go into the subrack's result for length slot `slot`.
fn walkSubracks(gpa: Allocator, ld: *const tiles.LetterDistribution, counts: *const [wmp.max_alphabet_size]u8, ml: usize, remaining: u32, sub: BitRack, results: anytype, slot: usize) Allocator.Error!void {
    if (remaining == 0) {
        var mask: u32 = 0;
        for (1..wmp.max_alphabet_size) |l| {
            if (counts[l] > wmp.count(sub, @intCast(l))) mask |= @as(u32, 1) << @intCast(l);
        }
        const gop = try results.getOrPut(gpa, sub);
        if (!gop.found_existing) gop.value_ptr.* = @splat(0);
        gop.value_ptr[slot] |= mask;
        return;
    }
    if (ml >= ld.size) return;
    const most = @min(@min(counts[ml], ld.counts[ml]), remaining);
    var s = sub;
    var k: u32 = 0;
    while (true) : (k += 1) {
        try walkSubracks(gpa, ld, counts, ml + 1, remaining - k, s, results, slot);
        if (k == most) break;
        wmp.add(&s, @intCast(ml));
    }
}

/// ORs `result` into the masks of every full rack made of `cur` and `left`
/// more tiles (no blanks, letters from `ml` on, within the distribution).
fn supersets(ld: *const tiles.LetterDistribution, rit: *const Rit, entries: []Entry, cur: BitRack, left: u32, ml: usize, result: *const [tp6_lengths]u32, tp7: bool) void {
    if (left == 0) {
        const found = rit.lookup(cur) orelse return;
        const e = &entries[found - rit.entries.ptr];
        if (tp7) {
            for (&e.multi_pt_tp7, result[0..tp7_lengths]) |*d, r| d.* |= r;
        } else {
            for (&e.multi_pt_tp6, result) |*d, r| d.* |= r;
        }
        return;
    }
    var l = ml;
    while (l < ld.size) : (l += 1) {
        if (wmp.count(cur, @intCast(l)) >= ld.counts[l]) continue;
        var next = cur;
        wmp.add(&next, @intCast(l));
        supersets(ld, rit, entries, next, left - 1, l, result, tp7);
    }
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

test "entry layout" {
    try testing.expectEqual(704, @sizeOf(Entry));
}

const testdata = @import("../testing/testdata.zig");
const game_mod = @import("../game/game.zig");
const movegen = @import("../movegen/movegen.zig");
const build_options = @import("build_options");

/// NWL20 positions with a full rack to move and tiles in the bag, from
/// seeded static games, and a table for their racks.
const Fixture = struct {
    store: @import("data.zig").Store,
    pos: @import("../board/position.zig").Position,
    games: std.ArrayList(game_mod.Game) = .empty,
    bytes: []align(16) u8 = &.{},
    rit: Rit = undefined,

    fn init(self: *Fixture) !void {
        self.store = testdata.store();
        errdefer self.store.deinit();
        self.pos = try testdata.position(&self.store, "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;");
        const lex = self.pos.lexicon;
        self.pos.word_map = (try self.store.wordMap(build_options.data_path, "NWL20", lex, self.pos.dist)) orelse return error.SkipZigTest;
        const leaves = (try self.store.leavesFor(build_options.data_path, "NWL20", false)) orelse return error.SkipZigTest;

        var gen = movegen.Generator.init(testing.allocator, &self.pos.board, lex.kwg, self.pos.dist);
        defer gen.deinit();
        gen.word_map = self.pos.word_map;
        var racks: std.ArrayList(BitRack) = .empty;
        defer racks.deinit(testing.allocator);
        for (0..4) |seed| {
            var g: game_mod.Game = .start(&self.pos, @splat(@intCast(seed + 40)), false);
            gen.board = &g.pos.board;
            while (g.playing) {
                const rack = g.pos.racks[g.on_turn];
                if (rack.num_tiles == rack_size and g.bag.len > 0 and g.pos.board.tiles_played > 0) {
                    try self.games.append(testing.allocator, g);
                    const br = wmp.fromRack(&rack);
                    if (std.mem.indexOfScalar(BitRack, racks.items, br) == null) try racks.append(testing.allocator, br);
                }
                _ = try game_mod.playStaticTurn(&g, &gen);
            }
        }
        self.bytes = try makeFor(testing.allocator, .{
            .ld = self.pos.dist,
            .kwg_bytes = lex.bytes,
            .klv = &leaves.klv,
            .klv_bytes = leaves.bytes,
            .word_map = self.pos.word_map.?,
        }, racks.items);
        self.rit = try Rit.init(self.bytes);
    }

    fn deinit(self: *Fixture) void {
        testing.allocator.free(self.bytes);
        self.games.deinit(testing.allocator);
        self.store.deinit();
    }
};

/// Whether leave map index `i` is a subset of `rack`: of each letter, a
/// run of low bits.
fn canonicalIndex(lm: *const leavemap.LeaveMap, rack: *const Rack, i: usize) bool {
    var idx: u32 = @intCast(i);
    for (rack.counts, 0..) |c, ml| {
        if (c == 0) continue;
        const bits = (idx >> @intCast(lm.base[ml])) & ((@as(u32, 1) << @intCast(c)) - 1);
        if (bits & (bits + 1) != 0) return false;
        idx &= ~(((@as(u32, 1) << @intCast(c)) - 1) << @intCast(lm.base[ml]));
    }
    return idx == 0;
}

test "table entries are what the generator works out" {
    var f: Fixture = .{ .store = undefined, .pos = undefined };
    try f.init();
    defer f.deinit();
    try testing.expect(f.rit.entries.len > 50);
    try testing.expect(f.rit.madeFrom(f.pos.lexicon.bytes, (try f.store.leavesFor(build_options.data_path, "NWL20", false)).?.bytes));

    var gen = movegen.Generator.init(testing.allocator, &f.pos.board, f.pos.lexicon.kwg, f.pos.dist);
    defer gen.deinit();
    gen.word_map = f.pos.word_map;
    for (f.rit.entries) |*e| {
        try testing.expectEqual(e, f.rit.lookup(e.rack).?);
        var rack: Rack = .{};
        for (0..f.pos.dist.size) |ml| {
            for (0..wmp.count(e.rack, @intCast(ml))) |_| rack.add(@intCast(ml));
        }
        // No pre-endgame adjustment, every exchange allowed: the leave
        // walk's values and best exchange.
        const ctx: equity.Context = .{ .ld = f.pos.dist, .klv = f.pos.leaves, .board_dim = 15, .board_empty = false, .tiles_in_bag = 50, .opp_rack_score = 0 };
        gen.max_can_exchange = rack_size;
        var r = rack;
        gen.bestStart(&r, true, &ctx);
        for (gen.leave_map.values, e.leaves, 0..) |want, got, i| if (canonicalIndex(&gen.leave_map, &rack, i)) testing.expectEqual(want, @as(f64, got)) catch |err| {
            std.debug.print("leave {d} of {any}\n", .{ i, rack.counts[0..f.pos.dist.size] });
            return err;
        };
        for (gen.shadow.best_leaves, e.best_leaves) |want, got| try testing.expectEqual(want, @as(f64, got));
        try testing.expect(gen.have_best);
        try testing.expectEqual(gen.best.len, e.exchange_len);
        try testing.expectEqualSlices(MachineLetter, gen.best.tileSlice(), e.exchange[0..e.exchange_len]);
        // The word lengths subracks make alone, and their best leaves.
        gen.bestSubracks(&r);
        for (wmp.minimum_word_length..rack_size + 1) |n| {
            try testing.expectEqual(gen.wmp_gen.has_word_of_length[n], e.np_has_word & (@as(u8, 1) << @intCast(n)) != 0);
            if (gen.wmp_gen.has_word_of_length[n]) try testing.expectEqual(gen.wmp_gen.best_leave_values[rack_size - n], @as(f64, e.np_best_leaves[rack_size - n]));
        }
        gen.best_only = false;
    }
}

test "the best move is the same with the table" {
    var f: Fixture = .{ .store = undefined, .pos = undefined };
    try f.init();
    defer f.deinit();
    const plain = try testing.allocator.create(movegen.Generator);
    defer testing.allocator.destroy(plain);
    plain.* = .init(testing.allocator, &f.pos.board, f.pos.lexicon.kwg, f.pos.dist);
    defer plain.deinit();
    plain.word_map = f.pos.word_map;
    const with = try testing.allocator.create(movegen.Generator);
    defer testing.allocator.destroy(with);
    with.* = .init(testing.allocator, &f.pos.board, f.pos.lexicon.kwg, f.pos.dist);
    defer with.deinit();
    with.word_map = f.pos.word_map;
    with.rit = &f.rit;

    // Pre-endgame adjustments for every bag size, to take the table's
    // other paths too.
    var peg: [101]f64 = undefined;
    for (&peg, 0..) |*v, i| v.* = @as(f64, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.37;
    var used: usize = 0;
    var anchors_plain: usize = 0;
    var anchors_with: usize = 0;
    for (f.games.items) |*g| {
        for ([_]bool{ false, true }) |with_peg| {
            for ([_]u8{ rack_size, 3 }) |max_exchange| {
                var ctx = g.equityContext();
                if (with_peg) ctx.peg = &peg;
                const ex = g.exchangeRules();
                plain.board = &g.pos.board;
                with.board = &g.pos.board;
                plain.max_can_exchange = @min(ex.max, max_exchange);
                with.max_can_exchange = @min(ex.max, max_exchange);
                var r1 = g.pos.racks[g.on_turn];
                var r2 = r1;
                const want = try plain.genBest(&r1, ex.allowed, &ctx);
                anchors_plain += plain.shadow.heap.anchors.items.len;
                const got = try with.genBest(&r2, ex.allowed, &ctx);
                anchors_with += with.shadow.heap.anchors.items.len;
                if (with.rit_entry != null) used += 1;
                testing.expectEqual(want.equity, got.equity) catch |err| {
                    std.debug.print("equity {d} vs {d}\n", .{ want.equity, got.equity });
                    return err;
                };
                try testing.expect(!equity.tiebreaksBetter(&want.move, &got.move) and !equity.tiebreaksBetter(&got.move, &want.move));
            }
        }
    }
    try testing.expectEqual(4 * f.games.items.len, used);
    // The table rules placements out.
    try testing.expect(anchors_with < anchors_plain);
}
