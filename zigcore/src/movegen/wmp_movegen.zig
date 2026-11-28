//! The word-map move generator's state: the rack's subracks with the words
//! they make, and what the shadow pass finds per (playthrough blocks, tiles
//! played) of each square. Port of Macondo's wmp/movegen.go (MAGPIE's
//! wmp_move_gen).
//!
//! The flow, per generation: init with the rack; checkNonplaythroughExistence
//! enumerates every subrack and looks it up (which word lengths can be played
//! without board tiles, and the best leave of each); during the shadow pass
//! the playthrough tiles of the current extent are tracked and each
//! placement recorded into its anchor slot; then wordmap generation asks for
//! the words of each subrack of an anchor.

const std = @import("std");
const tiles = @import("../data/tiles.zig");
const wmp = @import("../data/wmp.zig");
const leavemap = @import("leavemap.zig");
pub const BitRack = wmp.BitRack;
const rit_mod = @import("../data/rit.zig");
pub const Entry = wmp.Entry;

pub const rack_size = 7;
/// The most runs of board tiles one word can cross on a 15 board.
pub const max_playthrough_blocks = 15 / 2 + 1;
pub const max_anchors = (rack_size + 1) * (max_playthrough_blocks + 1);
/// Where each subrack size starts in the flat arrays of 128 subracks:
/// C(7,0), C(7,1), ... summed.
const combination_offsets = [rack_size + 1]u8{ 0, 1, 8, 29, 64, 99, 120, 127 };
pub const total_subracks = 1 << rack_size;
/// The blank word cache's size.
const blank_word_cache_bytes = 4 * wmp.result_buffer_size;
pub const score_base_count = 2;

/// What the shadow found for one (playthrough blocks, tiles to play) pair
/// of the square it is on. tiles_to_play is 0 for an empty slot.
pub const AnchorSlot = struct {
    highest_equity: f64 = 0,
    highest_score: i32 = 0,
    score_bases: [score_base_count]i32 = .{ 0, 0 },
    tiles_to_play: u8 = 0,
    playthrough_blocks: u8 = 0,
    word_length: u8 = 0,
    leftmost_start_col: u8 = 0,
    rightmost_start_col: u8 = 0,
};

pub const MoveGen = struct {
    wmp: ?*const wmp.Wmp = null,
    player: BitRack = 0,
    full_rack_size: u32 = 0,

    // The board tiles the current extent plays through.
    playthrough: BitRack = 0,
    num_played_through: u32 = 0,
    playthrough_blocks: u32 = 0,
    playthrough_copy: BitRack = 0,
    num_played_through_copy: u32 = 0,
    playthrough_blocks_copy: u32 = 0,

    // The subracks of the rack, by size (from combination_offsets), with
    // their entries, the leave map's value of what they leave and the sum
    // of their tile scores.
    np_racks: [total_subracks]BitRack = undefined,
    np_entries: [total_subracks]?*const Entry = undefined,
    /// Which np_entries have been looked up (all, without a rack info
    /// table: see npEntry).
    np_known: u128 = 0,
    pt_racks: [total_subracks]BitRack = undefined,
    pt_entries: [total_subracks]?*const Entry = undefined,
    // Three spare entries at the end for leaveKept's last vector.
    leave_values: [total_subracks + 3]f64 = @splat(0),
    tile_score_sums: [total_subracks + 3]i32 = @splat(0),
    /// The hashes of the playthrough subracks candidateSubracks let through.
    pt_hashes: [64]u64 = undefined,
    count_by_size: [rack_size + 1]u8 = @splat(0),
    /// By leave size: the best leave of the subracks of the other size that
    /// make a word.
    best_leave_values: [rack_size + 1]f64 = @splat(0),
    has_word_of_length: [rack_size + 1]bool = @splat(false),

    anchors: [max_anchors]AnchorSlot = @splat(.{}),
    /// The slots touched since resetAnchors.
    dirty: [max_anchors]u8 = undefined,
    num_dirty: usize = 0,
    tile_scores: [wmp.max_alphabet_size]i32 = @splat(0),

    // Word retrieval.
    buffer: [wmp.result_buffer_size]u8 = undefined,
    words: []const u8 = &.{},
    num_words: usize = 0,
    tiles_to_play: usize = 0,
    word_length: usize = 0,
    /// The word lists of the rack's blank subracks without playthrough,
    /// kept for the generation: by subrack, where the list starts plus one
    /// (0 for none) and its word count.
    blank_words: [blank_word_cache_bytes]u8 = undefined,
    blank_used: usize = 0,
    blank_offsets: [total_subracks]u16 = @splat(0),
    blank_counts: [total_subracks]u16 = @splat(0),

    /// Sets up for `rack`. Inactive without a word map.
    pub fn init(self: *MoveGen, ld: *const tiles.LetterDistribution, rack: *const tiles.Rack, w: ?*const wmp.Wmp) void {
        self.wmp = w;
        if (w == null) return;
        for (&self.tile_scores, 0..) |*t, ml| t.* = if (ml < ld.size) ld.scores[ml] else 0;
        self.player = wmp.fromRack(rack);
        self.full_rack_size = rack.num_tiles;
        self.has_word_of_length = @splat(false);
    }

    pub inline fn isActive(self: *const MoveGen) bool {
        return self.wmp != null;
    }

    pub fn resetAnchors(self: *MoveGen) void {
        for (self.dirty[0..self.num_dirty]) |i| self.anchors[i] = .{};
        self.num_dirty = 0;
    }

    pub fn dirtyAnchors(self: *const MoveGen) []const u8 {
        return self.dirty[0..self.num_dirty];
    }

    pub fn resetPlaythrough(self: *MoveGen) void {
        self.playthrough = 0;
        self.num_played_through = 0;
        self.playthrough_blocks = 0;
    }

    pub inline fn hasPlaythrough(self: *const MoveGen) bool {
        return self.num_played_through > 0;
    }

    pub inline fn addPlaythroughLetter(self: *MoveGen, ml: u8) void {
        wmp.add(&self.playthrough, ml);
        self.num_played_through += 1;
    }

    pub inline fn incrementPlaythroughBlocks(self: *MoveGen) void {
        self.playthrough_blocks += 1;
    }

    pub fn savePlaythroughState(self: *MoveGen) void {
        self.playthrough_copy = self.playthrough;
        self.num_played_through_copy = self.num_played_through;
        self.playthrough_blocks_copy = self.playthrough_blocks;
    }

    pub fn restorePlaythroughState(self: *MoveGen) void {
        self.playthrough = self.playthrough_copy;
        self.num_played_through = self.num_played_through_copy;
        self.playthrough_blocks = self.playthrough_blocks_copy;
    }

    /// Whether the whole rack and the playthrough tiles make a word (of
    /// their combined length). Keeps the entry for word retrieval.
    pub fn checkPlaythroughFullRackExistence(self: *MoveGen) bool {
        const idx = combination_offsets[self.full_rack_size];
        const br = self.player + self.playthrough;
        self.pt_racks[idx] = br;
        self.pt_entries[idx] = self.wmp.?.entry(br, self.full_rack_size + self.num_played_through);
        return self.pt_entries[idx] != null;
    }

    /// Enumerates the rack's subracks, with the leave map's value of each
    /// one's leave (the leave map must hold the rack's values), and looks
    /// up which make words. `check_leaves` also works out the best leave
    /// of each size among those that do.
    pub fn checkNonplaythroughExistence(self: *MoveGen, check_leaves: bool, lm: *leavemap.LeaveMap) void {
        lm.current = (@as(u32, 1) << @intCast(self.full_rack_size)) - 1;
        self.resetBlankWordCache();
        self.count_by_size = @splat(0);
        self.has_word_of_length = @splat(false);
        var rl: RackLetters = .{};
        for (0..wmp.max_alphabet_size) |ml| {
            const n = wmp.count(self.player, @intCast(ml));
            if (n == 0) continue;
            rl.letters[rl.n] = @intCast(ml);
            rl.counts[rl.n] = n;
            rl.scores[rl.n] = self.tile_scores[ml];
            rl.n += 1;
        }
        self.enumerate(&rl, 0, 0, 0, 0, lm);
        self.np_known = 0;
        var size: usize = wmp.minimum_word_length;
        while (size <= self.full_rack_size) : (size += 1) self.checkNonplaythroughsOfSize(size, check_leaves);
    }

    /// The rack's distinct letters with their counts and scores, for
    /// enumerate (which walks only those, as MAGPIE's compact enumeration).
    const RackLetters = struct {
        letters: [rack_size]u8 = undefined,
        counts: [rack_size]u32 = undefined,
        scores: [rack_size]i32 = undefined,
        n: usize = 0,
    };

    fn enumerate(self: *MoveGen, rl: *const RackLetters, current: BitRack, idx: usize, size: u32, score_sum: i32, lm: *leavemap.LeaveMap) void {
        if (idx == rl.n) {
            const i = combination_offsets[size] + self.count_by_size[size];
            self.np_racks[i] = current;
            self.leave_values[i] = lm.currentValue();
            self.tile_score_sums[i] = score_sum;
            self.count_by_size[size] += 1;
            return;
        }
        const ml = rl.letters[idx];
        const max_this = rl.counts[idx];
        const tile_score = rl.scores[idx];
        var cur = current;
        for (0..max_this) |k| {
            const i: u32 = @intCast(k);
            self.enumerate(rl, cur, idx + 1, size + i, score_sum + @as(i32, @intCast(i)) * tile_score, lm);
            wmp.add(&cur, ml);
            lm.complementAdd(ml, i, max_this);
        }
        self.enumerate(rl, cur, idx + 1, size + max_this, score_sum + @as(i32, @intCast(max_this)) * tile_score, lm);
        var k = max_this;
        while (k > 0) {
            k -= 1;
            lm.complementTake(ml, k, max_this);
        }
    }

    /// checkNonplaythroughExistence with a rack info table entry for the
    /// rack: the subracks are enumerated with their leave values, but
    /// which sizes make words and the best leaves of those that do come
    /// from the entry (its raw best plus the pre-endgame adjustment of the
    /// size, which is the most of the subracks' values), and each
    /// subrack's word map entry is looked up only when needed.
    pub fn checkNonplaythroughWithRit(self: *MoveGen, e: *const rit_mod.Entry, ctx: *const @import("../equity/equity.zig").Context, lm: *leavemap.LeaveMap) void {
        lm.current = (@as(u32, 1) << @intCast(self.full_rack_size)) - 1;
        self.resetBlankWordCache();
        self.count_by_size = @splat(0);
        var rl: RackLetters = .{};
        for (0..wmp.max_alphabet_size) |ml| {
            const n = wmp.count(self.player, @intCast(ml));
            if (n == 0) continue;
            rl.letters[rl.n] = @intCast(ml);
            rl.counts[rl.n] = n;
            rl.scores[rl.n] = self.tile_scores[ml];
            rl.n += 1;
        }
        self.enumerate(&rl, 0, 0, 0, 0, lm);
        self.np_known = 0;
        self.has_word_of_length = @splat(false);
        var size: usize = wmp.minimum_word_length;
        while (size <= self.full_rack_size) : (size += 1) {
            const leave_size = self.full_rack_size - size;
            const has = e.np_has_word & (@as(u8, 1) << @intCast(size)) != 0;
            self.has_word_of_length[size] = has;
            self.best_leave_values[leave_size] = if (has)
                ctx.leaveMapValue(e.np_best_leaves[leave_size], 0, @intCast(leave_size), @intCast(size))
            else
                -std.math.inf(f64);
        }
    }

    /// The word map entry of subrack `i` (of `size` tiles) without
    /// playthrough, looked up the first time it is asked for.
    inline fn npEntry(self: *MoveGen, i: usize, size: usize) ?*const Entry {
        const bit = @as(u128, 1) << @intCast(i);
        if (self.np_known & bit == 0) {
            self.np_entries[i] = self.wmp.?.entry(self.np_racks[i], size);
            self.np_known |= bit;
        }
        return self.np_entries[i];
    }

    fn checkNonplaythroughsOfSize(self: *MoveGen, size: usize, check_leaves: bool) void {
        const leave_size = self.full_rack_size - size;
        self.best_leave_values[leave_size] = if (check_leaves) -std.math.inf(f64) else 0;
        const offset = combination_offsets[size];
        for (0..self.count_by_size[size]) |k| {
            const i = offset + k;
            self.np_entries[i] = self.wmp.?.entry(self.np_racks[i], size);
            self.np_known |= @as(u128, 1) << @intCast(i);
            if (self.np_entries[i] == null) continue;
            self.has_word_of_length[size] = true;
            if (check_leaves and self.leave_values[i] > self.best_leave_values[leave_size]) {
                self.best_leave_values[leave_size] = self.leave_values[i];
            }
        }
    }

    /// Whether some subrack of `len` tiles makes a word.
    pub inline fn wordOfLengthExists(self: *const MoveGen, len: usize) bool {
        if (len > rack_size) return false;
        return self.has_word_of_length[len];
    }

    /// Folds one shadow placement into the slot of the current (blocks,
    /// tiles played) pair: widens its start columns, raises its score and
    /// equity bounds, and updates its score bases. A score base bounds the
    /// scores of the slot's plays by their own tiles: for every play,
    /// score <= base[b] + (b+1) * (sum of the play's tile scores). See
    /// wmp.MoveGen.RecordPlacement for why.
    pub fn recordPlacement(self: *MoveGen, tiles_played: u32, word_length: u32, start_col: usize, score: i32, eq: f64, shadow_tile_score_sum: i32, smallest_multiplier: i32) void {
        const idx = self.playthrough_blocks * (rack_size + 1) + tiles_played;
        const a = &self.anchors[idx];
        const base0 = score - @min(1, smallest_multiplier) * shadow_tile_score_sum;
        const base1 = score - @min(2, smallest_multiplier) * shadow_tile_score_sum;
        if (a.tiles_to_play == 0) {
            a.* = .{
                .highest_equity = eq,
                .highest_score = score,
                .score_bases = .{ base0, base1 },
                .tiles_to_play = @intCast(tiles_played),
                .playthrough_blocks = @intCast(self.playthrough_blocks),
                .word_length = @intCast(word_length),
                .leftmost_start_col = @intCast(start_col),
                .rightmost_start_col = @intCast(start_col),
            };
            self.dirty[self.num_dirty] = @intCast(idx);
            self.num_dirty += 1;
            return;
        }
        a.word_length = @intCast(word_length);
        const c: u8 = @intCast(start_col);
        if (c < a.leftmost_start_col) {
            a.leftmost_start_col = c;
        } else if (c > a.rightmost_start_col) {
            a.rightmost_start_col = c;
        }
        if (eq > a.highest_equity) a.highest_equity = eq;
        if (score > a.highest_score) a.highest_score = score;
        if (base0 > a.score_bases[0]) a.score_bases[0] = base0;
        if (base1 > a.score_bases[1]) a.score_bases[1] = base1;
    }

    /// Sets up word retrieval for an anchor.
    pub fn playthroughSubracksInit(self: *MoveGen, tiles_to_play: usize, word_length: usize) void {
        self.word_length = word_length;
        self.num_played_through = @intCast(word_length - tiles_to_play);
        self.tiles_to_play = tiles_to_play;
    }

    pub inline fn tileScoreSum(self: *const MoveGen, idx: usize) i32 {
        return self.tile_score_sums[combination_offsets[self.tiles_to_play] + idx];
    }

    pub inline fn leaveValue(self: *const MoveGen, idx: usize) f64 {
        return self.leave_values[combination_offsets[self.tiles_to_play] + idx];
    }

    pub inline fn nonplaythroughSubrack(self: *const MoveGen, idx: usize) BitRack {
        return self.np_racks[combination_offsets[self.tiles_to_play] + idx];
    }

    /// The least of the anchor's score bound and what its score bases
    /// allow a subrack with this tile score sum.
    pub fn subrackScoreBound(score_bound: i32, bases: *const [score_base_count]i32, tile_score_sum: i32) i32 {
        return @min(score_bound, bases[0] + tile_score_sum, bases[1] + 2 * tile_score_sum);
    }

    /// The subracks of the anchor's size that may have plays, as a bit set
    /// by index: with `prune`, those whose score bound plus leave reaches
    /// `cutoff`; without playthrough tiles, those with words; with them,
    /// those with a blank, or that the filter lets through with the
    /// playthrough tiles.
    pub fn candidateSubracks(self: *MoveGen, prune: bool, score_bound: i32, bases: *const [score_base_count]i32, cutoff: f64) u64 {
        const offset = combination_offsets[self.tiles_to_play];
        const n = self.count_by_size[self.tiles_to_play];
        var kept: u64 = (@as(u64, 1) << @intCast(n)) - 1;
        if (prune) kept = self.leaveKept(offset, n, score_bound, bases, cutoff);
        var candidates: u64 = 0;
        if (self.word_length <= self.tiles_to_play) {
            while (kept != 0) : (kept &= kept - 1) {
                const i = @ctz(kept);
                if (self.npEntry(offset + i, self.tiles_to_play) != null) candidates |= @as(u64, 1) << @intCast(i);
            }
            return candidates;
        }
        const wfl = &self.wmp.?.wfls[self.word_length];
        while (kept != 0) : (kept &= kept - 1) {
            const i = @ctz(kept);
            const br = self.np_racks[offset + i] + self.playthrough;
            self.pt_racks[offset + i] = br;
            if (wmp.count(br, 0) != 0) {
                candidates |= @as(u64, 1) << @intCast(i);
                continue;
            }
            const hash = wmp.indexHash(br);
            if (wfl.mayHaveWords(hash)) {
                self.pt_hashes[i] = hash;
                candidates |= @as(u64, 1) << @intCast(i);
            }
        }
        return candidates;
    }

    /// The subracks of size starting at `offset` (`n` of them) whose bound
    /// plus leave reaches `cutoff`, as a bit set: four at a time.
    fn leaveKept(self: *const MoveGen, offset: usize, n: usize, score_bound: i32, bases: *const [score_base_count]i32, cutoff: f64) u64 {
        const V = 4;
        const VI = @Vector(V, i32);
        const VF = @Vector(V, f64);
        const sb: VI = @splat(score_bound);
        const b0: VI = @splat(bases[0]);
        const b1: VI = @splat(bases[1]);
        const cut: VF = @splat(cutoff);
        var kept: u64 = 0;
        var i: usize = 0;
        while (i < n) : (i += V) {
            // Past the last subrack the arrays hold others' values (the
            // arrays have room), and the bits are cleared below.
            const ts: VI = self.tile_score_sums[offset + i ..][0..V].*;
            const lv: VF = self.leave_values[offset + i ..][0..V].*;
            const bound = @min(sb, @min(b0 + ts, b1 + ts + ts));
            const keep = !(@as(VF, @floatFromInt(bound)) + lv < cut);
            kept |= @as(u64, @as(u4, @bitCast(keep))) << @intCast(i);
        }
        return kept & ((@as(u64, 1) << @intCast(n)) - 1);
    }

    pub fn resetBlankWordCache(self: *MoveGen) void {
        if (self.blank_used != 0) {
            self.blank_offsets = @splat(0);
            self.blank_used = 0;
        }
    }

    /// Finds the words of subrack `idx` of the anchor, one candidateSubracks
    /// marked; false if it has none. words then holds them.
    pub fn subrackWords(self: *MoveGen, idx: usize) bool {
        const si = combination_offsets[self.tiles_to_play] + idx;
        const is_playthrough = self.word_length > self.tiles_to_play;
        const w = self.wmp.?;
        const wfl = &w.wfls[self.word_length];
        var br: BitRack = undefined;
        var entry: ?*const Entry = undefined;
        if (is_playthrough) {
            br = self.pt_racks[si];
            entry = switch (wmp.count(br, 0)) {
                0 => wfl.hashedWordEntry(br, self.pt_hashes[idx & 63]),
                1 => wfl.blankEntry(br),
                2 => wfl.doubleBlankEntry(br),
                else => null,
            };
            self.pt_entries[si] = entry;
        } else {
            br = self.np_racks[si];
            entry = self.npEntry(si, self.tiles_to_play);
        }
        const e = entry orelse return false;
        if (wmp.count(br, 0) == 0) {
            self.words = wfl.blanklessWords(e, self.word_length);
            self.num_words = self.words.len / self.word_length;
            return self.num_words > 0;
        }
        if (!is_playthrough and self.blank_offsets[si] != 0) {
            const off = self.blank_offsets[si] - 1;
            self.num_words = self.blank_counts[si];
            self.words = self.blank_words[off..][0 .. self.num_words * self.word_length];
            return true;
        }
        // A list is cached while the cache has room for the longest.
        const cache = !is_playthrough and self.blank_used <= blank_word_cache_bytes - wmp.result_buffer_size;
        const buf: []u8 = if (cache) self.blank_words[self.blank_used..] else &self.buffer;
        const n = w.writeWordsForEntry(e, br, self.word_length, buf);
        if (n == 0 or n % self.word_length != 0) return false;
        self.num_words = n / self.word_length;
        self.words = buf[0..n];
        if (cache and self.num_words <= std.math.maxInt(u16)) {
            self.blank_offsets[si] = @intCast(self.blank_used + 1);
            self.blank_counts[si] = @intCast(self.num_words);
            self.blank_used += n;
        }
        return true;
    }
};

// Tests ----------------------------------------------------------------------
// Ported from wmp/movegen_test.go (MAGPIE's wmp_move_gen_test.c).

const testing = std.testing;
const testdata = @import("../testing/testdata.zig");

const Fixture = struct {
    bytes: []align(4) u8,
    wmp: wmp.Wmp,
    f: testdata.Fixture,

    fn load() !Fixture {
        var f = try testdata.Fixture.load("CSW24");
        errdefer f.deinit();
        const bytes = try testdata.loadFile("lexica/CSW24.wmp");
        errdefer testing.allocator.free(bytes);
        return .{ .bytes = bytes, .wmp = try .init(testing.allocator, bytes), .f = f };
    }

    fn deinit(self: *Fixture) void {
        self.wmp.deinit(testing.allocator);
        testing.allocator.free(self.bytes);
        self.f.deinit();
    }

    fn rack(self: *const Fixture, text: []const u8) !tiles.Rack {
        var mls: [8]tiles.MachineLetter = undefined;
        return .fromLetters(mls[0..try self.f.ld.toMachineLetters(text, &mls)]);
    }
};

test "word map move gen inactive without a word map" {
    var mg: MoveGen = .{};
    const r: tiles.Rack = .{};
    var ld: tiles.LetterDistribution = undefined;
    ld.size = 0;
    mg.init(&ld, &r, null);
    try testing.expect(!mg.isActive());
}

test "nonplaythrough existence" {
    var fx = try Fixture.load();
    defer fx.deinit();
    var rack = try fx.rack("VIVIFIC");
    var lm: leavemap.LeaveMap = .{};
    lm.init(&rack);
    // Each leave is worth its number of tiles.
    for (&lm.values, 0..) |*v, i| v.* = @floatFromInt(@popCount(i));

    var mg: MoveGen = .{};
    mg.init(&fx.f.ld, &rack, &fx.wmp);
    mg.resetPlaythrough();
    try testing.expect(mg.isActive() and !mg.hasPlaythrough());

    for ([_]bool{ false, true }) |check_leaves| {
        mg.checkNonplaythroughExistence(check_leaves, &lm);
        // IF and VIVIFIC.
        try testing.expect(mg.wordOfLengthExists(2));
        for (3..7) |len| try testing.expect(!mg.wordOfLengthExists(len));
        try testing.expect(mg.wordOfLengthExists(7));
        for (2..8) |len| {
            const leave_size = 7 - len;
            if (!check_leaves) {
                try testing.expectEqual(0, mg.best_leave_values[leave_size]);
            } else if (mg.wordOfLengthExists(len)) {
                try testing.expectEqual(@as(f64, @floatFromInt(leave_size)), mg.best_leave_values[leave_size]);
            }
        }
        // The leave map is back at the whole rack.
        try testing.expectEqual(127, lm.current);
    }
}

test "playthrough bingo existence" {
    var fx = try Fixture.load();
    defer fx.deinit();
    var rack = try fx.rack("CHEESE?");
    var mg: MoveGen = .{};
    mg.init(&fx.f.ld, &rack, &fx.wmp);
    mg.resetPlaythrough();
    try testing.expect(mg.isActive() and !mg.hasPlaythrough());
    const N = 14;
    const P = 16;
    const Q = 17;
    const I = 9;

    mg.addPlaythroughLetter(N);
    try testing.expect(mg.hasPlaythrough());
    try testing.expect(mg.checkPlaythroughFullRackExistence()); // ENCHEErS
    mg.savePlaythroughState();
    mg.addPlaythroughLetter(P);
    try testing.expect(mg.checkPlaythroughFullRackExistence()); // NIPCHEESE
    mg.addPlaythroughLetter(Q);
    try testing.expect(!mg.checkPlaythroughFullRackExistence());
    mg.restorePlaythroughState();
    mg.addPlaythroughLetter(I);
    try testing.expect(mg.checkPlaythroughFullRackExistence()); // NIpCHEESE
    mg.savePlaythroughState();
    mg.addPlaythroughLetter(P);
    try testing.expect(mg.checkPlaythroughFullRackExistence()); // NIPCHEESEs
    mg.addPlaythroughLetter(Q);
    try testing.expect(!mg.checkPlaythroughFullRackExistence());
}
