//! Shadow play: an upper bound on the equity of the plays through each
//! anchor, worked out without the word graph, so that the best-play search
//! can look at anchors best first and stop once no anchor left can beat the
//! best play found. Port of Macondo's movegen/shadow.go, which follows
//! MAGPIE's (the algorithm is from wolges; see details.txt there).
//!
//! The bound places the rack's tiles where they would score most: tiles a
//! square allows only one letter for are placed there, and the rest go to
//! the remaining squares highest score to highest multiplier.

const std = @import("std");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const movegen = @import("movegen.zig");
const equity = @import("equity.zig");
const leavemap = @import("leavemap.zig");
const Generator = movegen.Generator;
const Direction = board_mod.Direction;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const rack_tile_limit = movegen.rack_tile_limit;
const trivial = board_mod.trivial_cross_set;

pub const Anchor = struct {
    highest_equity: f64,
    highest_score: i32,
    row: u8,
    col: u8,
    last_anchor_col: u8,
    dir: Direction,
};

/// The anchors of a generation, handed out highest bound first. What is
/// heaped is one 64-bit key per anchor: its bound as an integer that orders
/// as the bound does, with the low bits replaced by the anchor's index.
/// Bounds differing by less than about one part in 10^9 may come out in
/// either order, which changes when the search can stop, not what it finds.
pub const AnchorHeap = struct {
    anchors: std.ArrayList(Anchor) = .empty,
    keys: std.ArrayList(u64) = .empty,

    const idx_bits = 20;
    const idx_mask: u64 = (1 << idx_bits) - 1;

    pub fn deinit(self: *AnchorHeap, gpa: std.mem.Allocator) void {
        self.anchors.deinit(gpa);
        self.keys.deinit(gpa);
    }

    /// `eq` as an integer, without its low bits: key(a) < key(b) only if
    /// a < b.
    pub fn key(eq: f64) u64 {
        var bits: u64 = @bitCast(eq + 0); // -0 becomes +0, which it equals
        if (bits >> 63 != 0) bits = ~bits else bits |= 1 << 63;
        return bits & ~idx_mask;
    }

    fn reset(self: *AnchorHeap) void {
        self.anchors.clearRetainingCapacity();
        self.keys.clearRetainingCapacity();
    }

    fn heapifyAll(self: *AnchorHeap, gpa: std.mem.Allocator) error{OutOfMemory}!void {
        const n = self.anchors.items.len;
        std.debug.assert(n <= idx_mask + 1);
        try self.keys.resize(gpa, n);
        for (self.anchors.items, self.keys.items, 0..) |a, *k, i| k.* = key(a.highest_equity) | i;
        var i = n / 2;
        while (i > 0) {
            i -= 1;
            self.heapifyDown(i);
        }
    }

    fn heapifyDown(self: *AnchorHeap, start: usize) void {
        const keys = self.keys.items;
        const n = keys.len;
        var parent = start;
        const k = keys[parent];
        while (true) {
            var child = parent * 2 + 1;
            if (child >= n) break;
            if (child + 1 < n and keys[child + 1] > keys[child]) child += 1;
            if (keys[child] <= k) break;
            keys[parent] = keys[child];
            parent = child;
        }
        keys[parent] = k;
    }

    /// Removes the anchor with the highest key; returns it with its key.
    fn extractMax(self: *AnchorHeap) struct { anchor: *const Anchor, key: u64 } {
        const keys = self.keys.items;
        const max = keys[0];
        const n = keys.len - 1;
        keys[0] = keys[n];
        self.keys.items.len = n;
        if (n > 0) self.heapifyDown(0);
        return .{ .anchor = &self.anchors.items[@intCast(max & idx_mask)], .key = max };
    }

    /// Whether an anchor with this key, and so every anchor still in the
    /// heap, has a bound below `eq`.
    fn keyBelow(k: u64, eq: f64) bool {
        return k & ~idx_mask < key(eq);
    }
};

const CrossWordMul = struct { multiplier: i32, col: u8 };

pub const State = struct {
    heap: AnchorHeap = .{},
    /// The equity of the best play found before the shadow pass (the best
    /// exchange), or -inf. Anchors bounded below it aren't kept.
    anchor_floor: f64 = -std.math.inf(f64),

    left_col: usize = 0,
    right_col: usize = 0,
    anchor_left_ext: u64 = 0,
    anchor_right_ext: u64 = 0,

    mainword_restricted_score: i32 = 0,
    perp_additional_score: i32 = 0,
    word_multiplier: i32 = 0,

    highest_equity: f64 = 0,
    highest_score: i32 = 0,
    max_tiles_to_play: u32 = 0,

    full_rack_desc_tile_scores: [rack_tile_limit]i32 = @splat(0),
    desc_tile_scores: [rack_tile_limit]i32 = @splat(0),
    desc_tile_scores_copy: [rack_tile_limit]i32 = @splat(0),

    num_unrestricted_muls: usize = 0,
    last_word_multiplier: i32 = 1,
    desc_eff_letter_muls: [rack_tile_limit]i32 = @splat(0),
    desc_eff_letter_muls_copy: [rack_tile_limit]i32 = @splat(0),
    desc_cross_word_muls: [rack_tile_limit]CrossWordMul = undefined,
    desc_cross_word_muls_copy: [rack_tile_limit]CrossWordMul = undefined,

    rack: [tiles.max_alphabet_size]u8 = @splat(0),
    /// The tiles tryRestrict has taken from `rack` for the anchor being
    /// shadowed, in order, so they can be put back.
    restricted: [rack_tile_limit + 1]u8 = undefined,
    num_restricted: usize = 0,
    rack_cross_set: u64 = 0,
    num_letters_on_rack: u32 = 0,

    /// Every lane's squares as the shadow pass loaded them (rows, then
    /// columns as rows of the transposed board), for the recording pass.
    lanes: [2][board_mod.max_dim][board_mod.max_dim]movegen.CachedSquare = undefined,

    best_leaves: leavemap.BestLeaves = undefined,
    tiles_in_bag: i32 = 0,
    opp_rack_score: i32 = 0,
    /// The non-score part of the bound, by tiles played (see
    /// setShadowEquities).
    equity_by_tiles_played: [rack_tile_limit + 1]f64 = undefined,

    pub fn deinit(self: *State, gpa: std.mem.Allocator) void {
        self.heap.deinit(gpa);
    }

    fn noteRestricted(self: *State, ml: MachineLetter) void {
        self.restricted[self.num_restricted] = ml;
        self.num_restricted += 1;
    }

    fn unrestrictTo(self: *State, n: usize) void {
        while (self.num_restricted > n) {
            self.num_restricted -= 1;
            self.rack[self.restricted[self.num_restricted]] += 1;
        }
    }

    /// The letters the rack can put on a square allowing `cross_set`: any
    /// of them with a blank.
    inline fn rackPossible(self: *const State, cross_set: u64) u64 {
        var possible = cross_set & self.rack_cross_set;
        if (self.rack[0] > 0) possible |= cross_set;
        return possible;
    }
};

/// The best leave value of each size when there is no leave map (an empty
/// rack): 0 with tiles in the bag, else the endgame adjustment.
pub fn bestLeavesWithoutMap(s: *State) void {
    s.best_leaves = @splat(-std.math.inf(f64));
    s.best_leaves[0] = if (s.tiles_in_bag > 0) 0 else @floatFromInt(2 * s.opp_rack_score);
}

fn initShadow(gen: *Generator, rack: *const Rack) void {
    const s = &gen.shadow;
    s.heap.reset();
    s.rack_cross_set = 0;
    s.num_letters_on_rack = rack.num_tiles;
    var n: usize = 0;
    for (0..gen.ld.size) |ml| {
        s.rack[ml] = rack.counts[ml];
        if (rack.counts[ml] > 0) {
            s.rack_cross_set |= @as(u64, 1) << @intCast(ml);
            const sc = gen.ld.scores[ml];
            for (0..rack.counts[ml]) |_| {
                s.full_rack_desc_tile_scores[n] = sc;
                n += 1;
            }
        }
    }
    @memset(s.full_rack_desc_tile_scores[n..], 0);
    std.mem.sort(i32, s.full_rack_desc_tile_scores[0..n], {}, std.sort.desc(i32));
    setShadowEquities(s, gen.ctx);
}

/// The non-score part of the bound depends only on how many tiles a play
/// uses, so it is worked out once per generation (MAGPIE's
/// gen_set_shadow_equities and static_eval_get_shadow_equity): with tiles
/// in the bag, the best leave of the size left plus the pre-endgame
/// adjustment; with the bag empty, going out's bonus, or the penalty for
/// keeping the lowest-scoring tiles.
///
/// Every term of a play's equity left out of the bound must never be
/// positive (the opening placement adjustment is not). Go's shadow leaves
/// out the pre-endgame adjustment, which can be positive; this follows
/// MAGPIE.
fn setShadowEquities(s: *State, ctx: *const equity.Context) void {
    const n = s.num_letters_on_rack;
    for (0..n + 1) |tp| {
        var eq: f64 = 0;
        if (s.tiles_in_bag > 0) {
            eq += s.best_leaves[n - tp];
            const bag_plus_seven = s.tiles_in_bag - @as(i32, @intCast(tp)) + rack_tile_limit;
            if (bag_plus_seven < ctx.peg.len) eq += ctx.peg[@intCast(bag_plus_seven)];
        } else if (tp < n) {
            var lowest: i32 = 0;
            for (tp..n) |i| lowest += s.full_rack_desc_tile_scores[i];
            eq += @as(f64, @floatFromInt(-equity.endgame_non_outplay_leave_penalty_multiplier * lowest)) - equity.endgame_non_outplay_constant_penalty;
        } else {
            eq += @floatFromInt(2 * s.opp_rack_score);
        }
        s.equity_by_tiles_played[tp] = eq;
    }
}

/// Works out the bound of every anchor, both directions, and heaps them.
pub fn genShadow(gen: *Generator, rack: *const Rack) void {
    initShadow(gen, rack);
    const dim = gen.board.dim;
    inline for (.{ Direction.horizontal, Direction.vertical }, 0..) |dir, lane| {
        if (dir == .vertical) gen.board.transpose();
        gen.vertical = dir == .vertical;
        for (0..dim) |row| {
            if (!rowHasAnchor(gen, row, dir)) continue;
            gen.cur_row = row;
            gen.last_anchor_col = 100;
            gen.loadRow(row, dir.other());
            gen.shadow.lanes[lane][row] = gen.row;
            for (0..dim) |col| {
                if (gen.board.isAnchor(row, col, dir)) {
                    playForAnchor(gen, row, col, gen.last_anchor_col, dir);
                    gen.last_anchor_col = col;
                    if (gen.board.hasLetter(row, col)) gen.last_anchor_col += 1;
                }
            }
        }
        if (dir == .vertical) gen.board.transpose();
    }
    gen.shadow.heap.heapifyAll(gen.gpa) catch {
        gen.out_of_memory = true;
    };
}

fn rowHasAnchor(gen: *const Generator, row: usize, dir: Direction) bool {
    for (0..gen.board.dim) |col| {
        if (gen.board.isAnchor(row, col, dir)) return true;
    }
    return false;
}

fn playForAnchor(gen: *Generator, row: usize, col: usize, last_anchor_col: usize, dir: Direction) void {
    const s = &gen.shadow;
    const cs_dir = dir.other();
    s.left_col = col;
    s.right_col = col;
    s.anchor_left_ext = gen.row[col].left_ext;
    s.anchor_right_ext = gen.board.rightExtSet(row, col, cs_dir);
    if (s.anchor_left_ext | s.anchor_right_ext == 0) return;

    s.num_unrestricted_muls = 0;
    s.desc_eff_letter_muls = @splat(0);
    s.last_word_multiplier = 1;
    s.desc_tile_scores = s.full_rack_desc_tile_scores;
    s.mainword_restricted_score = 0;
    s.perp_additional_score = 0;
    s.word_multiplier = 1;
    s.highest_equity = -std.math.inf(f64);
    s.highest_score = 0;
    s.max_tiles_to_play = 0;
    gen.tiles_played = 0;
    gen.cur_anchor_col = col;

    s.num_restricted = 0;
    const orig_rack_cross_set = s.rack_cross_set;
    const cur = gen.row[col].letter;
    if (cur == 0) {
        startNonplaythrough(gen, dir);
    } else {
        startPlaythrough(gen, last_anchor_col, cur, dir);
    }
    s.unrestrictTo(0);
    s.rack_cross_set = orig_rack_cross_set;

    if (s.max_tiles_to_play == 0) return;
    if (s.highest_equity >= s.anchor_floor) {
        s.heap.anchors.append(gen.gpa, .{
            .highest_equity = s.highest_equity,
            .highest_score = s.highest_score,
            .row = @intCast(row),
            .col = @intCast(col),
            .last_anchor_col = @intCast(last_anchor_col),
            .dir = dir,
        }) catch {
            gen.out_of_memory = true;
        };
    }
}

fn startNonplaythrough(gen: *Generator, dir: Direction) void {
    const s = &gen.shadow;
    const col = s.left_col;
    const sq = gen.row[col];
    const possible = s.rackPossible(sq.cross_set);
    if (possible == 0) return;
    const lm: i32 = sq.letter_mul;
    const wm: i32 = sq.word_mul;
    const cs = sq.cross_score;
    const is_cross_word = cs > 0 or sq.cross_set != trivial;

    // A single tile: no main word yet.
    s.word_multiplier = 0;
    s.perp_additional_score = cs * wm;
    if (!tryRestrict(gen, possible, lm, wm, col, is_cross_word)) insertUnrestricted(gen, col, lm, wm, is_cross_word);
    gen.tiles_played += 1;
    // One-tile plays are recorded horizontally only.
    if (dir == .horizontal) record(gen);

    s.word_multiplier = wm;
    maybeRecalcEffMuls(gen);
    nonplaythroughPlayLeft(gen, dir == .horizontal);
}

fn startPlaythrough(gen: *Generator, last_anchor_col: usize, first: MachineLetter, dir: Direction) void {
    const s = &gen.shadow;
    var cur = first;
    // The tiles already on the board to the left.
    while (true) {
        s.mainword_restricted_score += gen.ld.score(cur);
        if (s.left_col == 0 or s.left_col == last_anchor_col + 1) break;
        s.left_col -= 1;
        cur = gen.row[s.left_col].letter;
        if (cur == 0) {
            s.left_col += 1;
            break;
        }
    }
    playthroughPlayLeft(gen, dir == .horizontal);
}

fn nonplaythroughPlayLeft(gen: *Generator, is_unique: bool) void {
    const s = &gen.shadow;
    while (true) {
        if (s.rackPossible(s.anchor_right_ext) != 0) playRight(gen, is_unique);
        s.anchor_right_ext = trivial;

        if (s.left_col == 0 or s.left_col == gen.last_anchor_col + 1 or gen.tiles_played >= s.num_letters_on_rack) return;
        if (s.rackPossible(s.anchor_left_ext) == 0) return;
        s.anchor_left_ext = trivial;

        s.left_col -= 1;
        gen.tiles_played += 1;
        const sq = gen.row[s.left_col];
        const possible = s.rackPossible(sq.cross_set);
        if (possible == 0) return;
        const lm: i32 = sq.letter_mul;
        const wm: i32 = sq.word_mul;
        const cs = sq.cross_score;
        const is_cross_word = cs > 0 or sq.cross_set != trivial;
        s.perp_additional_score += cs * wm;
        s.word_multiplier *= wm;
        if (!tryRestrict(gen, possible, lm, wm, s.left_col, is_cross_word)) insertUnrestricted(gen, s.left_col, lm, wm, is_cross_word);
        record(gen);
    }
}

fn playthroughPlayLeft(gen: *Generator, is_unique_in: bool) void {
    const s = &gen.shadow;
    var is_unique = is_unique_in;
    while (true) {
        if (s.rackPossible(s.anchor_right_ext) != 0) playRight(gen, is_unique);
        s.anchor_right_ext = trivial;

        const possible_left = s.rackPossible(s.anchor_left_ext);
        s.anchor_left_ext = trivial;

        if (s.left_col == 0 or s.left_col == gen.last_anchor_col + 1 or gen.tiles_played >= s.num_letters_on_rack) break;
        if (possible_left == 0) break;

        s.left_col -= 1;
        gen.tiles_played += 1;
        const sq = gen.row[s.left_col];
        const possible = s.rackPossible(sq.cross_set);
        if (possible == 0) break;
        const lm: i32 = sq.letter_mul;
        const wm: i32 = sq.word_mul;
        const cs = sq.cross_score;
        const is_cross_word = cs > 0 or sq.cross_set != trivial;
        s.perp_additional_score += cs * wm;
        s.word_multiplier *= wm;
        if (!tryRestrict(gen, possible, lm, wm, s.left_col, is_cross_word)) insertUnrestricted(gen, s.left_col, lm, wm, is_cross_word);
        if (sq.cross_set == trivial) is_unique = true;
        if (gen.tiles_played > 0 and (is_unique or gen.tiles_played > 1)) record(gen);
    }
}

/// Extends to the right of the current extent: over empty squares, each
/// time taking in the tiles already on the board right after.
fn playRight(gen: *Generator, is_unique_in: bool) void {
    const s = &gen.shadow;
    var is_unique = is_unique_in;

    const orig_main = s.mainword_restricted_score;
    const orig_perp = s.perp_additional_score;
    const orig_word_mul = s.word_multiplier;
    const orig_tiles_played = gen.tiles_played;
    const orig_right_col = s.right_col;
    const orig_num_unrestricted = s.num_unrestricted_muls;
    // What a restriction or an unrestricted square changes is saved just
    // before the first one, as most calls stop at their first square. The
    // multipliers are saved with the word multiplier they were worked out
    // for, and go back together.
    const orig_rack_cross_set = s.rack_cross_set;
    const orig_num_restricted = s.num_restricted;
    var saved_tile_scores = false;
    var changed_muls = false;
    var last_word_mul_at_save: i32 = 0;

    const dim = gen.board.dim;
    while (s.right_col < dim - 1 and gen.tiles_played < s.num_letters_on_rack) {
        s.right_col += 1;
        gen.tiles_played += 1;
        const sq = gen.row[s.right_col];
        const possible = s.rackPossible(sq.cross_set & sq.left_ext & s.anchor_right_ext);
        s.anchor_right_ext = trivial;
        if (possible == 0) break;

        const lm: i32 = sq.letter_mul;
        const wm: i32 = sq.word_mul;
        const cs = sq.cross_score;
        const is_cross_word = cs > 0 or sq.cross_set != trivial;
        s.perp_additional_score += cs * wm;
        s.word_multiplier *= wm;

        if (!saved_tile_scores and possible & (possible -% 1) == 0) {
            // One letter: tryRestrict may take a tile.
            s.desc_tile_scores_copy = s.desc_tile_scores;
            saved_tile_scores = true;
        }
        if (!tryRestrict(gen, possible, lm, wm, s.right_col, is_cross_word)) {
            if (!changed_muls) {
                s.desc_cross_word_muls_copy = s.desc_cross_word_muls;
                s.desc_eff_letter_muls_copy = s.desc_eff_letter_muls;
                last_word_mul_at_save = s.last_word_multiplier;
                changed_muls = true;
            }
            insertUnrestricted(gen, s.right_col, lm, wm, is_cross_word);
        }
        if (sq.cross_set == trivial) is_unique = true;

        // The tiles on the board right after.
        while (s.right_col + 1 < dim) {
            const next = gen.row[s.right_col + 1].letter;
            if (next == 0) break;
            s.mainword_restricted_score += gen.ld.score(next);
            s.right_col += 1;
        }

        if (gen.tiles_played > 0 and (is_unique or gen.tiles_played > 1)) {
            maybeRecalcEffMuls(gen);
            record(gen);
        }
    }

    s.mainword_restricted_score = orig_main;
    s.perp_additional_score = orig_perp;
    s.word_multiplier = orig_word_mul;
    s.unrestrictTo(orig_num_restricted);
    s.rack_cross_set = orig_rack_cross_set;
    if (saved_tile_scores) s.desc_tile_scores = s.desc_tile_scores_copy;
    if (changed_muls) {
        s.num_unrestricted_muls = orig_num_unrestricted;
        s.desc_cross_word_muls = s.desc_cross_word_muls_copy;
        s.desc_eff_letter_muls = s.desc_eff_letter_muls_copy;
        s.last_word_multiplier = last_word_mul_at_save;
    }
    s.right_col = orig_right_col;
    gen.tiles_played = orig_tiles_played;
    maybeRecalcEffMuls(gen);
}

/// The bound of the current extent: the best placement of the unrestricted
/// tiles, the restricted ones where they go, and the rest of the equity at
/// its best for this many tiles played. Keeps the highest seen for the
/// anchor.
fn record(gen: *Generator) void {
    const s = &gen.shadow;
    var tiles_played_score: i32 = 0;
    for (s.desc_tile_scores, s.desc_eff_letter_muls) |t, m| tiles_played_score += t * m;
    const bingo: i32 = if (gen.tiles_played == rack_tile_limit) movegen.bingo_bonus else 0;
    const score = tiles_played_score + s.mainword_restricted_score * s.word_multiplier + s.perp_additional_score + bingo;

    const eq = @as(f64, @floatFromInt(score)) + s.equity_by_tiles_played[gen.tiles_played];

    if (eq > s.highest_equity) s.highest_equity = eq;
    if (score > s.highest_score) s.highest_score = score;
    if (gen.tiles_played > s.max_tiles_to_play) s.max_tiles_to_play = gen.tiles_played;
}

/// If only one letter can go on the square, takes a tile for it (a blank
/// when the rack has none of the letter) and scores it there. Not when the
/// rack has both the letter and a blank: which one goes here is then left
/// to the best arrangement.
fn tryRestrict(gen: *Generator, possible: u64, letter_mul: i32, word_mul: i32, col: usize, is_cross_word: bool) bool {
    _ = col;
    const s = &gen.shadow;
    if (possible == 0 or possible & (possible - 1) != 0) return false;
    const ml: MachineLetter = @intCast(@ctz(possible));
    var tile_score: i32 = 0;
    if (s.rack[ml] > 0) {
        if (ml != 0 and s.rack[0] > 0) return false;
        s.rack[ml] -= 1;
        s.noteRestricted(ml);
        if (s.rack[ml] == 0) s.rack_cross_set &= ~possible;
        tile_score = gen.ld.score(ml);
    } else {
        s.rack[0] -= 1;
        s.noteRestricted(0);
        if (s.rack[0] == 0) s.rack_cross_set &= ~@as(u64, 1);
    }
    removeFromDescTileScores(s, tile_score);
    const lsm = tile_score * letter_mul;
    s.mainword_restricted_score += lsm;
    if (is_cross_word) s.perp_additional_score += lsm * word_mul;
    return true;
}

/// Takes `score` out of the descending tile scores. Looks among the
/// positive ones only, as Go and MAGPIE do: a blank's 0 stays.
fn removeFromDescTileScores(s: *State, score: i32) void {
    var total: isize = 0;
    for (s.desc_tile_scores) |t| {
        if (t > 0) total += 1;
    }
    const new_size = total - 1;
    var i = new_size;
    while (i >= 0) : (i -= 1) {
        const u: usize = @intCast(i);
        if (s.desc_tile_scores[u] == score) {
            const n: usize = @intCast(new_size);
            for (u..n) |j| s.desc_tile_scores[j] = s.desc_tile_scores[j + 1];
            s.desc_tile_scores[n] = 0;
            return;
        }
    }
}

fn insertUnrestricted(gen: *Generator, col: usize, letter_mul: i32, word_mul: i32, is_cross_word: bool) void {
    const s = &gen.shadow;
    maybeRecalcEffMuls(gen);
    const cross_mul: i32 = if (is_cross_word) letter_mul * word_mul else 0;

    var idx = s.num_unrestricted_muls;
    while (idx > 0 and s.desc_cross_word_muls[idx - 1].multiplier < cross_mul) : (idx -= 1) {
        s.desc_cross_word_muls[idx] = s.desc_cross_word_muls[idx - 1];
    }
    s.desc_cross_word_muls[idx] = .{ .multiplier = cross_mul, .col = @intCast(col) };

    const eff = s.word_multiplier * letter_mul + cross_mul;
    idx = s.num_unrestricted_muls;
    while (idx > 0 and s.desc_eff_letter_muls[idx - 1] < eff) : (idx -= 1) {
        s.desc_eff_letter_muls[idx] = s.desc_eff_letter_muls[idx - 1];
    }
    s.desc_eff_letter_muls[idx] = eff;
    s.num_unrestricted_muls += 1;
}

/// Works the effective multipliers out again if the word multiplier has
/// changed since they were.
fn maybeRecalcEffMuls(gen: *Generator) void {
    const s = &gen.shadow;
    if (s.last_word_multiplier == s.word_multiplier) return;
    s.last_word_multiplier = s.word_multiplier;
    const n = s.num_unrestricted_muls;
    s.num_unrestricted_muls = 0;
    for (0..n) |i| {
        const xw = s.desc_cross_word_muls[i];
        const eff = s.word_multiplier * @as(i32, gen.row[xw.col].letter_mul) + xw.multiplier;
        var idx = s.num_unrestricted_muls;
        while (idx > 0 and s.desc_eff_letter_muls[idx - 1] < eff) : (idx -= 1) {
            s.desc_eff_letter_muls[idx] = s.desc_eff_letter_muls[idx - 1];
        }
        s.desc_eff_letter_muls[idx] = eff;
        s.num_unrestricted_muls += 1;
    }
}

/// Generates the plays of the heaped anchors, best bound first, until no
/// anchor left can beat the best play found.
pub fn recordFromAnchors(gen: *Generator, rack: *Rack) void {
    gen.setRackBits(rack);
    gen.tiles_played = 0;
    var transposed = false;
    var loaded: ?struct { row: u8, dir: Direction } = null;
    const s = &gen.shadow;
    while (s.heap.keys.items.len > 0) {
        const top = s.heap.extractMax();
        const a = top.anchor;
        if (gen.have_best and a.highest_equity < gen.best_eq) {
            // So is every anchor left.
            if (AnchorHeap.keyBelow(top.key, gen.best_eq)) break;
            continue;
        }
        const need_transposed = a.dir == .vertical;
        if (need_transposed != transposed) {
            gen.board.transpose();
            transposed = need_transposed;
        }
        gen.vertical = need_transposed;
        gen.cur_row = a.row;
        gen.cur_anchor_col = a.col;
        gen.last_anchor_col = a.last_anchor_col;
        if (loaded == null or loaded.?.row != a.row or loaded.?.dir != a.dir) {
            gen.row = s.lanes[@intFromBool(need_transposed)][a.row];
            loaded = .{ .row = a.row, .dir = a.dir };
        }
        gen.genForAnchor(rack);
    }
    if (transposed) gen.board.transpose();
}

// Tests ----------------------------------------------------------------------
// Ported from movegen/shadow_test.go (from MAGPIE's test/shadow_test.c).

const testing = std.testing;
const testdata = @import("testdata.zig");

/// The anchors of the shadow pass alone for `rack_text` on the CGP's
/// board, highest score bound first, as Go's RunShadowOnly returns them.
fn shadowOnly(cgp_text: []const u8, rack_text: []const u8) ![]Anchor {
    var store = testdata.store();
    defer store.deinit();
    var p = try testdata.position(&store, cgp_text);
    var mls: [8]MachineLetter = undefined;
    const rack: Rack = .fromLetters(mls[0..try p.dist.toMachineLetters(rack_text, &mls)]);
    var gen = Generator.init(testing.allocator, &p.board, p.lexicon.kwg, p.dist);
    defer gen.deinit();
    // Scores only: no leaves.
    const ctx: equity.Context = .{ .ld = p.dist, .klv = null, .board_dim = p.board.dim, .board_empty = false, .tiles_in_bag = 1, .opp_rack_score = 0 };
    gen.ctx = &ctx;
    gen.shadow.tiles_in_bag = 1;
    gen.shadow.best_leaves = @splat(0);
    gen.shadow.anchor_floor = -std.math.inf(f64);
    genShadow(&gen, &rack);
    if (gen.out_of_memory) return error.OutOfMemory;
    const anchors = try testing.allocator.dupe(Anchor, gen.shadow.heap.anchors.items);
    std.mem.sort(Anchor, anchors, {}, struct {
        fn higher(_: void, a: Anchor, b: Anchor) bool {
            return a.highest_score > b.highest_score;
        }
    }.higher);
    return anchors;
}

/// Each anchor's bound is at least the expected score: a bound below the
/// best play's score would be a bug (one above only prunes less).
fn expectShadow(cgp_text: []const u8, rack_text: []const u8, expected: []const i32) !void {
    const anchors = try shadowOnly(cgp_text, rack_text);
    defer testing.allocator.free(anchors);
    if (anchors.len < expected.len) {
        std.debug.print("{s} {s}: {d} anchors, want at least {d}\n", .{ cgp_text, rack_text, anchors.len, expected.len });
        return error.TestUnexpectedResult;
    }
    for (expected, anchors[0..expected.len], 0..) |want, a, i| {
        if (a.highest_score < want) {
            std.debug.print("{s} {s}: anchor {d} bound {d}, want at least {d}\n", .{ cgp_text, rack_text, i, a.highest_score, want });
            return error.TestUnexpectedResult;
        }
    }
}

const empty_cgp = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex CSW21;";
const ka_opening = "15/15/15/15/15/15/15/6KA7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/4 0 lex CSW21;";
const aa_opening = "15/15/15/15/15/15/15/6AA7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/4 0 lex CSW21;";
const doug_v_emely = "15/15/15/15/15/15/15/3WINDY7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;";
const triple_letters = "15/15/15/15/15/15/1PROTEAN7/3WINDY7/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;";
const triple_double = "15/15/15/15/15/15/15/15/15/2PAV10/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;";
const bottom_left_re = "15/15/15/15/15/15/15/15/15/15/15/15/15/15/RE13 ADEEGIL/AEILOUY 0/32 0 lex CSW21;";
const later_between_dws = "15/15/15/15/5LATER5/15/15/15/15/15/15/15/15/15/15 ADEEGIL/AEILOUY 0/32 0 lex CSW21;";
const vs_oxy_cgp = "1PACIFYING5/1IS12/YE13/1REQUALIFIED3/H1L12/EDS12/NO3T9/1RAINWASHING3/UM3O9/T2E1O9/1WAKEnERS6/1OnETIME7/OOT2E1B7/N6U7/1JACULATING4 / 0/0 0 lex NWL20;";
const qi_qis = "15/15/15/15/15/15/15/6QI7/6I8/6S8/15/15/15/15/15 FRUITED/EGGCUPS 22/12 0 lex CSW21;";
const shuttled_ravioli = "14Z/5V7NA/5R4P2AN/5O1N1ARGUFY/5TOO2I2F1/4T1COWKS2E1/2REE1UN2A2R1/1RAVIoLI2G3Q/1EXON1IN2E1P1A/1C1L3GEM2AHI/BEMUD2SHUTTlED/1D1E8AI1/YET9IS1/ODA9ST1/W1JOLLER7 BII/EO 477/388 0 lex CSW21;";
const toeless = "15/15/15/15/15/15/5Q2J6/5UVAE6/5I2U6/5Z9/15/15/15/15/15 TOELESS/EEGIPRW 42/38 0 lex CSW21;";
const addle = "5E1p7/1A1C1G1E5S1/1JIZ1G1R3V1P1/HO1APE1A3U1I1/OW1RAD1EXFIL1T1/MA2W2OI2G1T1/IN2K2N2MOBE1/E2FYRDS2o1ANS/3O6V1N1H/3U6ADD1E/3S6BOO1E/3T6LOR1L/INCITANT1AYRE2/3E5U5/3REQUITE5 L/I 467/473 0 lex CSW21;";
const magpies = "15/15/15/15/15/15/15/MAGPIE2LUMBAGO/15/15/15/15/15/15/15 SS/Q 0/0 0 lex CSW21;";

test "shadow score bounds" {
    const cases = [_]struct { []const u8, []const u8, []const i32 }{
        .{ empty_cgp, "OU", &.{4} },                            .{ empty_cgp, "ID", &.{6} },
        .{ empty_cgp, "AX", &.{18} },                           .{ empty_cgp, "BD", &.{10} },
        .{ empty_cgp, "QK", &.{30} },                           .{ empty_cgp, "AESR", &.{8} },
        .{ empty_cgp, "TNCL", &.{12} },                         .{ empty_cgp, "AAAAA", &.{12} },
        .{ empty_cgp, "CAAAA", &.{20} },                        .{ empty_cgp, "CAKAA", &.{32} },
        .{ empty_cgp, "AIERZ", &.{48} },                        .{ empty_cgp, "AIERZN", &.{50} },
        .{ empty_cgp, "AIERZNL", &.{102} },                     .{ empty_cgp, "?", &.{0} },
        .{ empty_cgp, "??", &.{0} },                            .{ empty_cgp, "??OU", &.{4} },
        .{ empty_cgp, "??OUA", &.{8} },                         .{ ka_opening, "EE", &.{ 10, 9, 8, 5, 5, 3 } },
        .{ ka_opening, "E?", &.{ 8, 8, 8, 7, 7, 7, 3, 3, 2 } }, .{ ka_opening, "J", &.{ 21, 14, 9, 9 } },
        .{ aa_opening, "JF", &.{ 42, 25, 25, 18, 14, 13 } },    .{ aa_opening, "JFU", &.{44} },
        .{ ka_opening, "JFU", &.{32} },                         .{ aa_opening, "JFUG", &.{47} },
        .{ aa_opening, "JFUGX", &.{61} },                       .{ aa_opening, "JFUGXL", &.{102} },
        .{ doug_v_emely, "Q", &.{ 14, 11 } },                   .{ doug_v_emely, "BD", &.{14} },
        .{ doug_v_emely, "BOH", &.{24} },                       .{ doug_v_emely, "BOHGX", &.{44} },
        .{ doug_v_emely, "BOHGXZ", &.{116} },                   .{ doug_v_emely, "BOHGXZQ", &.{206} },
        .{ triple_letters, "A", &.{6} },                        .{ triple_letters, "Z", &.{33} },
        .{ triple_letters, "ZLW", &.{73} },                     .{ triple_letters, "ZLW?", &.{79} },
        .{ triple_letters, "QZLW", &.{85} },                    .{ triple_double, "K", &.{13} },
        .{ triple_double, "KT", &.{20} },                       .{ triple_double, "KT?", &.{24} },
        .{ bottom_left_re, "M", &.{8} },                        .{ bottom_left_re, "MN", &.{16} },
        .{ bottom_left_re, "MNA", &.{20} },                     .{ bottom_left_re, "MNAU", &.{22} },
        .{ bottom_left_re, "MNAUT", &.{30} },                   .{ bottom_left_re, "MNAUTE", &.{39} },
        .{ later_between_dws, "Z", &.{21} },                    .{ later_between_dws, "ZL", &.{64} },
        .{ later_between_dws, "ZLI", &.{68} },                  .{ later_between_dws, "ZLIE", &.{72} },
        .{ later_between_dws, "ZLIER", &.{76} },                .{ later_between_dws, "ZLIERA", &.{80} },
        .{ later_between_dws, "ZLIERAI", &.{212} },             .{ vs_oxy_cgp, "A", &.{18} },
        .{ vs_oxy_cgp, "O", &.{63} },                           .{ vs_oxy_cgp, "E", &.{72} },
        .{ vs_oxy_cgp, "PB", &.{156} },                         .{ vs_oxy_cgp, "PA", &.{50} },
        .{ vs_oxy_cgp, "PBA", &.{174} },                        .{ vs_oxy_cgp, "Z", &.{42} },
        .{ vs_oxy_cgp, "ZE", &.{160} },                         .{ vs_oxy_cgp, "AZE", &.{184} },
        .{ vs_oxy_cgp, "AZEB", &.{484} },                       .{ vs_oxy_cgp, "AZEBP", &.{604} },
        .{ vs_oxy_cgp, "AZEBPX", &.{686} },                     .{ vs_oxy_cgp, "AZEBPXO", &.{1780} },
        .{ vs_oxy_cgp, "AZEBPQO", &.{1836} },                   .{ qi_qis, "FRUITED", &.{ 128, 103, 88, 0, 0, 0, 0, 0 } },
        .{ shuttled_ravioli, "BII", &.{ 25, 16, 15 } },         .{ toeless, "TOELESS", &.{86} },
        .{ addle, "L", &.{ 17, 17, 16, 14, 10, 9 } },           .{ magpies, "SS", &.{15} },
    };
    for (cases) |c| try expectShadow(c[0], c[1], c[2]);
}
