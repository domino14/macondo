//! The best tile play of a rack by value (the score, plus a bonus for a
//! play of the whole rack), for the endgame's greedy playouts: MAGPIE's
//! MOVE_RECORD_BEST_SMALL generation with its lane shadow cache.
//!
//! The anchors are taken best shadow bound first until no anchor left can
//! beat the best play found. Across an endgame's playouts the same lanes
//! come back with the same racks (the boards differ in a few lanes, and a
//! side's racks are subsets of one rack), so two caches remember work by
//! the lane's contents: a lane's anchors with their bounds, and an anchor's
//! best play. A lane whose key is known is not shadowed again, nor an
//! anchor whose key is known generated again.
//!
//! Ties go to the larger move code, so the play found does not depend on
//! the order anchors are taken in.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("../data/tiles.zig");
const board_mod = @import("../board/board.zig");
const movegen = @import("movegen.zig");
const shadow = @import("shadow.zig");
const equity = @import("../equity/equity.zig");
const SmallMove = @import("../endgame/endgame.zig").SmallMove;
const Generator = movegen.Generator;
const Direction = board_mod.Direction;
const Rack = tiles.Rack;

const max_lane_anchors = 9;

/// One 64-byte line: a lookup touches one.
const LaneEntry = extern struct {
    key: u64 align(64) = 0,
    count: u8 = 0,
    cols: [max_lane_anchors]u8 = undefined,
    lasts: [max_lane_anchors]u8 = undefined,
    /// The bounds are whole points (scores and the bonus).
    bounds: [max_lane_anchors]i32 = undefined,
};

/// Half a line.
const PlayEntry = extern struct {
    key: u64 align(32) = 0,
    /// The best play's move code (0: the anchor has none), its score and
    /// value, its tiles played and length.
    tm: u64 = 0,
    value: i32 = 0,
    score: i16 = 0,
    tiles_played: u8 = 0,
    len: u8 = 0,
    _: u64 = 0,
};

comptime {
    std.debug.assert(@sizeOf(LaneEntry) == 64);
    std.debug.assert(@sizeOf(PlayEntry) == 32);
}

pub const Cache = struct {
    lanes: []align(64) LaneEntry,
    plays: []align(32) PlayEntry,
    lane_shift: u6,
    play_shift: u6,
    lane_hits: u64 = 0,
    lane_misses: u64 = 0,
    play_hits: u64 = 0,
    play_misses: u64 = 0,

    pub fn init(gpa: Allocator, lane_bits: u6, play_bits: u6) Allocator.Error!Cache {
        const lanes = try gpa.alignedAlloc(LaneEntry, .@"64", @as(usize, 1) << lane_bits);
        errdefer gpa.free(lanes);
        @memset(lanes, .{});
        const plays = try gpa.alignedAlloc(PlayEntry, .@"32", @as(usize, 1) << play_bits);
        @memset(plays, .{});
        return .{ .lanes = lanes, .plays = plays, .lane_shift = @intCast(@as(u7, 64) - lane_bits), .play_shift = @intCast(@as(u7, 64) - play_bits) };
    }

    pub fn deinit(self: *Cache, gpa: Allocator) void {
        gpa.free(self.lanes);
        gpa.free(self.plays);
        self.* = undefined;
    }
};

inline fn mix(k: u64, v: u64) u64 {
    return (k ^ v) *% 0x9E3779B97F4A7C15;
}

inline fn finish(k_in: u64) u64 {
    var k = k_in;
    k ^= k >> 33;
    k *%= 0xFF51AFD7ED558CCD;
    k ^= k >> 33;
    return if (k == 0) 1 else k;
}

/// Lanes are rows 0..dim (horizontal plays), then columns dim..2*dim.
pub const max_lanes = Generator.max_lanes;

/// The key of what a lane's plays and shadow bounds depend on, the rack
/// aside: its squares' tiles, the cross-sets, cross-scores and extension
/// sets of the direction across, and its anchors, each square's part
/// (squareKey) xored in, so that a move changes only its squares'. A
/// board's lanes keep theirs (see the endgame's solver).
pub fn laneKey(b: *const board_mod.Board, lane: usize) u64 {
    var k: u64 = finish(lane +% 1);
    for (0..b.dim) |i| k ^= squareKey(b, lane, i);
    return k;
}

/// Square `i` (along the lane)'s part of lane `lane`'s key.
pub fn squareKey(b: *const board_mod.Board, lane: usize, i: usize) u64 {
    const dim = b.dim;
    const vertical = lane >= dim;
    const fixed = if (vertical) lane - dim else lane;
    const idx = if (vertical) i * dim + fixed else fixed * dim + i;
    const cs = acrossIndex(vertical);
    const an = alongIndex(vertical);
    return squarePart(lane, i, b.squares[idx], b.anchors[an][idx], b.cross_sets[cs][idx], b.cross_scores[cs][idx], b.left_ext[cs][idx], b.right_ext[cs][idx]);
}

/// The direction of the cross-sets (and extension sets) a lane's plays
/// read, and of its anchors.
inline fn acrossIndex(vertical: bool) usize {
    return if (vertical) @backingInt(Direction.horizontal) else @backingInt(Direction.vertical);
}
inline fn alongIndex(vertical: bool) usize {
    return if (vertical) @backingInt(Direction.vertical) else @backingInt(Direction.horizontal);
}

fn squarePart(lane: usize, i: usize, letter: u8, anchor: bool, cross_set: u64, cross_score: i32, left_ext: u64, right_ext: u64) u64 {
    // The square's place hashed on its own first: xored straight into its
    // contents, two squares could trade contents (cross-sets 2 and 0 at
    // places 1 and 3, say) and leave the lane's key as it was.
    var k = mix(finish(@as(u64, lane) << 8 | i), cross_set);
    k = mix(k, @as(u64, letter) | @as(u64, @intFromBool(anchor)) << 8 | @as(u64, @as(u32, @bitCast(cross_score))) << 16);
    k = mix(k, left_ext);
    k = mix(k, right_ext);
    return finish(k);
}

/// Updates the keys of the two lanes through a square the journal saved
/// (`sq`, as it was) to the board's square as it is.
pub fn updateKeys(b: *const board_mod.Board, keys: *[max_lanes]u64, sq: *const board_mod.SavedSquare) void {
    const dim = b.dim;
    const r = sq.idx / dim;
    const c = sq.idx % dim;
    inline for (.{ false, true }) |vertical| {
        const lane = if (vertical) dim + c else r;
        const i = if (vertical) r else c;
        const cs = acrossIndex(vertical);
        const an = alongIndex(vertical);
        keys[lane] ^= squarePart(lane, i, sq.letter, sq.anchors[an], sq.cross_sets[cs], sq.cross_scores[cs], sq.left_ext[cs], sq.right_ext[cs]);
        keys[lane] ^= squareKey(b, lane, i);
    }
}

/// The best tile play for `rack` by score plus, for a play of the whole
/// rack, `out_bonus`, ties to the larger move code; null if there is none.
/// The rack must hold 1 to 7 tiles.
/// `lane_keys` are the board's lanes' keys (laneKey).
pub fn best(gen: *Generator, rack: *Rack, out_bonus: i32, cache: *Cache, lane_keys: *const [max_lanes]u64) Allocator.Error!?SmallMove {
    return bestIn(gen, rack, out_bonus, cache, lane_keys, ~@as(u64, 0), null, 0);
}

/// As best, among `incumbent` (of value `incumbent_value`), if any, and
/// the plays in the lanes of `lanes` (rows 0..dim, then columns).
pub fn bestIn(gen: *Generator, rack: *Rack, out_bonus: i32, cache: *Cache, lane_keys: *const [max_lanes]u64, lanes: u64, incumbent: ?SmallMove, incumbent_value: i32) Allocator.Error!?SmallMove {
    if (lanes == 0) return incumbent;
    const eq_ctx: equity.Context = .{ .ld = gen.ld, .klv = null, .board_dim = gen.board.dim, .board_empty = false, .tiles_in_bag = 1, .opp_rack_score = 0 };
    gen.ctx = &eq_ctx;
    defer gen.ctx = undefined;
    gen.out_of_memory = false;
    gen.best_only = false;
    gen.wmp_gen.init(gen.ld, rack, null);
    const s = &gen.shadow;
    s.tiles_in_bag = 1;
    s.opp_rack_score = 0;
    s.anchor_floor = -std.math.inf(f64);
    s.best_leaves = @splat(0);
    s.best_leaves[0] = @floatFromInt(out_bonus);
    shadow.initShadow(gen, rack);

    var rack_key: u64 = mix(@intFromPtr(gen.kwg.nodes.ptr), @as(u64, @bitCast(@as(i64, out_bonus))));
    for (0..gen.ld.size) |ml| rack_key = mix(rack_key, rack.counts[ml]);
    rack_key = finish(rack_key);

    const dim = gen.board.dim;
    // Each lane's key with the rack's, and which lanes are loaded. The
    // keys first, their entries asked for, then the entries read.
    // (The lane keys cover the anchors, so a lane the cache knows, with
    // anchors or none, needs no look at the board.)
    var keys: [2][board_mod.max_dim]u64 = undefined;
    var loaded: [2]u32 = .{ 0, 0 };
    for (0..2) |side| {
        for (0..dim) |row| {
            if (lanes >> @intCast(side * dim + row) & 1 == 0) continue;
            const key = finish(mix(lane_keys[side * dim + row], rack_key));
            keys[side][row] = key;
            @prefetch(&cache.lanes[key >> cache.lane_shift], .{});
        }
    }
    inline for (.{ Direction.horizontal, Direction.vertical }, 0..) |dir, side| {
        if (dir == .vertical) gen.board.transpose();
        gen.vertical = dir == .vertical;
        for (0..dim) |row| {
            if (lanes >> @intCast(side * dim + row) & 1 == 0) continue;
            const lane = &s.lanes[side][row];
            const key = keys[side][row];
            const e = &cache.lanes[key >> cache.lane_shift];
            if (e.key == key) {
                cache.lane_hits += 1;
                for (0..e.count) |i| {
                    // Its best play will be wanted if it is taken.
                    @prefetch(&cache.plays[finish(mix(key, @as(u64, e.cols[i]) << 8 | e.lasts[i])) >> cache.play_shift], .{});
                    s.heap.anchors.appendBounded(.{ .highest_equity = @floatFromInt(e.bounds[i]), .highest_score = 0, .row = @intCast(row), .col = e.cols[i], .last_anchor_col = e.lasts[i], .dir = dir }) catch {
                        gen.out_of_memory = true;
                    };
                }
                continue;
            }
            const mask = shadow.anchorMask(gen, row, dir);
            if (mask == 0) {
                e.key = key;
                e.count = 0;
                continue;
            }
            cache.lane_misses += 1;
            gen.loadRowInto(lane, row, dir.other());
            loaded[side] |= @as(u32, 1) << @intCast(row);
            const first = s.heap.anchors.items.len;
            shadow.shadowLane(gen, row, mask, dir);
            const added = s.heap.anchors.items[first..];
            if (added.len <= max_lane_anchors) {
                e.key = key;
                e.count = @intCast(added.len);
                for (added, 0..) |a, i| {
                    e.cols[i] = a.col;
                    e.lasts[i] = a.last_anchor_col;
                    // Whole points, unless an anchor has no play at all.
                    e.bounds[i] = if (a.highest_equity < -1e9) std.math.minInt(i32) else @intFromFloat(@ceil(a.highest_equity));
                }
            }
        }
        if (dir == .vertical) gen.board.transpose();
    }
    if (gen.out_of_memory) return error.OutOfMemory;
    try s.heap.heapifyAll(gen.gpa);

    gen.setRackBits(rack);
    var result: ?SmallMove = incumbent;
    var best_v: i32 = incumbent_value;
    var transposed = false;
    while (s.heap.keys.items.len > 0) {
        const a = s.heap.extractMax().anchor;
        if (result != null and a.highest_equity < @as(f64, @floatFromInt(best_v))) break;
        const side: usize = @intFromBool(a.dir == .vertical);
        const pkey = finish(mix(keys[side][a.row], @as(u64, a.col) << 8 | a.last_anchor_col));
        const pe = &cache.plays[pkey >> cache.play_shift];
        var cand: ?SmallMove = null;
        var cand_v: i32 = 0;
        if (pe.key == pkey) {
            cache.play_hits += 1;
            if (pe.tm != 0) {
                cand = .{ .tm = pe.tm, .score = pe.score, .tiles_played = pe.tiles_played, .len = pe.len };
                cand_v = pe.value;
            }
        } else {
            cache.play_misses += 1;
            const need_transposed = a.dir == .vertical;
            if (need_transposed != transposed) {
                gen.board.transpose();
                transposed = need_transposed;
            }
            gen.vertical = need_transposed;
            gen.cur_row = a.row;
            gen.cur_anchor_col = a.col;
            gen.last_anchor_col = a.last_anchor_col;
            gen.row = &s.lanes[side][a.row];
            if (loaded[side] & (@as(u32, 1) << @intCast(a.row)) == 0) {
                gen.loadRowInto(gen.row, a.row, if (need_transposed) .horizontal else .vertical);
                loaded[side] |= @as(u32, 1) << @intCast(a.row);
            }
            gen.tiles_played = 0;
            gen.plays.clearRetainingCapacity();
            gen.genForAnchor(rack);
            for (gen.plays.items) |*mv| {
                const m: SmallMove = .fromMove(mv);
                const v: i32 = m.score + (if (m.tiles_played == rack.num_tiles) out_bonus else 0);
                if (cand == null or v > cand_v or (v == cand_v and m.tm > cand.?.tm)) {
                    cand = m;
                    cand_v = v;
                }
            }
            const c = cand orelse SmallMove{};
            pe.* = .{ .key = pkey, .tm = c.tm, .value = cand_v, .score = c.score, .tiles_played = c.tiles_played, .len = c.len };
        }
        if (cand) |c| {
            if (result == null or cand_v > best_v or (cand_v == best_v and c.tm > result.?.tm)) {
                result = c;
                best_v = cand_v;
            }
        }
    }
    if (transposed) gen.board.transpose();
    gen.plays.clearRetainingCapacity();
    gen.tiles_played = 0;
    if (gen.out_of_memory) return error.OutOfMemory;
    return result;
}

const testing = std.testing;

test "a lane's key changes when two of its squares trade contents" {
    var b = board_mod.Board.init(&board_mod.crossword_game);
    // Column E (lane dim + 4) of a Spanish endgame: cross-sets 2 and 0,
    // with the same cross-score, at rows 1 and 3. Hashing a square's place
    // into its cross-set with a plain xor gave both orders the same key.
    const lane = b.dim + 4;
    const h: usize = @backingInt(Direction.horizontal);
    b.cross_sets[h][1 * b.dim + 4] = 2;
    b.cross_sets[h][3 * b.dim + 4] = 0;
    b.cross_scores[h][1 * b.dim + 4] = 8;
    b.cross_scores[h][3 * b.dim + 4] = 8;
    const before = laneKey(&b, lane);
    b.cross_sets[h][1 * b.dim + 4] = 0;
    b.cross_sets[h][3 * b.dim + 4] = 2;
    try testing.expect(laneKey(&b, lane) != before);
}

test "changing any one square of a lane changes its key" {
    var prng: std.Random.DefaultPrng = .init(11);
    const r = prng.random();
    var b = board_mod.Board.init(&board_mod.crossword_game);
    for (0..300) |_| {
        // A random board's lane and square, and one of its fields changed.
        for (0..2) |d| {
            for (0..b.dim * b.dim) |i| {
                b.cross_sets[d][i] = r.int(u64) & r.int(u64);
                b.cross_scores[d][i] = r.intRangeAtMost(i32, 0, 20);
                b.anchors[d][i] = r.boolean();
            }
        }
        for (0..b.dim * b.dim) |i| b.squares[i] = if (r.uintLessThan(u8, 3) == 0) r.intRangeAtMost(u8, 1, 26) else 0;
        const lane = r.uintLessThan(usize, 2 * b.dim);
        const before = laneKey(&b, lane);
        const i = r.uintLessThan(usize, b.dim);
        const idx = if (lane >= b.dim) i * b.dim + (lane - b.dim) else lane * b.dim + i;
        const cs: usize = if (lane >= b.dim) @backingInt(Direction.horizontal) else @backingInt(Direction.vertical);
        const an: usize = if (lane >= b.dim) @backingInt(Direction.vertical) else @backingInt(Direction.horizontal);
        switch (r.uintLessThan(u8, 4)) {
            0 => b.cross_sets[cs][idx] ^= @as(u64, 1) << r.int(u6),
            1 => b.cross_scores[cs][idx] += 1,
            2 => b.anchors[an][idx] = !b.anchors[an][idx],
            else => b.squares[idx] +%= 1,
        }
        try testing.expect(laneKey(&b, lane) != before);
    }
}
