//! A best-play generator's memory of what it found for a rack on earlier
//! boards: Macondo's movegen/carry.go, after MAGPIE's best_move_carry.
//!
//! In a sim the same rack is generated for again and again on boards that
//! differ in a few lanes: every candidate play is answered from the same
//! opponent rack, and the racks drawn after often match too. Which plays a
//! lane (a row, or a column) holds for a rack, and what each is worth, is
//! decided by the lane alone: its tiles, its anchors, and its squares'
//! cross-sets and cross-scores, with the bag's size for the leave values.
//! So when a generation searched a lane and its best play overall was M, a
//! later generation for the same rack and bag that finds the lane unchanged
//! knows no play of the lane beats M, and if its best play so far is at
//! least M it doesn't look at the lane.
//!
//! The best plays are remembered too, a few per rack. One that is still a
//! play of the new board (see revalidate) is scored there, and the
//! generation starts from the best of them, so anchors that can't reach it
//! are dropped and more lanes are skipped. So is what the generation works
//! out for the rack before it looks at the board: the leave map, the best
//! leaves by size and the word map's subracks.
//!
//! Lanes are matched on their whole contents, not a hash, and "at least M"
//! is the generator's own order (equity.better), so the play found is the
//! one found without the memory.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("../data/tiles.zig");
const board_mod = @import("../board/board.zig");
const equity = @import("../equity/equity.zig");
const leavemap = @import("leavemap.zig");
const movegen = @import("movegen.zig");
const wmp_movegen = @import("wmp_movegen.zig");
const Move = @import("../board/move.zig").Move;
const Generator = movegen.Generator;
const CachedSquare = movegen.CachedSquare;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const max_dim = board_mod.max_dim;

/// Racks remembered, each in the slot its hash picks.
pub const num_racks = 64;
/// Best plays kept per rack, the oldest replaced first.
pub const moves_per_rack = 8;
/// Contents a lane is remembered with, the oldest replaced first.
pub const bounds_per_lane = 4;
/// Tile types of a rack that can be remembered.
const max_alphabet = 64;

/// With these contents, the lane holds no play better than the rack's
/// remembered move `move_idx`.
const Bound = struct {
    hash: u64 = 0,
    epoch: u32 = 0,
    stamp: u32 = 0,
    anchors: u32 = 0,
    move_idx: u8 = 0,
};

/// What a lane's plays depend on, square by square.
const LaneSquare = struct {
    cross_set: u64,
    cross_score: i32,
    letter: MachineLetter,

    fn of(sq: *const CachedSquare) LaneSquare {
        return .{ .cross_set = sq.cross_set, .cross_score = sq.cross_score, .letter = sq.letter };
    }

    fn eql(self: LaneSquare, sq: *const CachedSquare) bool {
        return self.cross_set == sq.cross_set and self.cross_score == sq.cross_score and self.letter == sq.letter;
    }
};

/// The word map generator's look at a rack's subracks
/// (checkNonplaythroughExistence), kept with the rack.
pub const WmpRackState = struct {
    np_racks: [wmp_movegen.total_subracks]wmp_movegen.BitRack,
    np_entries: [wmp_movegen.total_subracks]?*const wmp_movegen.Entry,
    np_known: u128,
    leave_values: [wmp_movegen.total_subracks + 3]f64,
    tile_score_sums: [wmp_movegen.total_subracks + 3]i32,
    count_by_size: [wmp_movegen.rack_size + 1]u8,
    best_leave_values: [wmp_movegen.rack_size + 1]f64,
    has_word_of_length: [wmp_movegen.rack_size + 1]bool,

    fn save(self: *WmpRackState, g: *const wmp_movegen.MoveGen) void {
        self.np_racks = g.np_racks;
        self.np_entries = g.np_entries;
        self.np_known = g.np_known;
        self.leave_values = g.leave_values;
        self.tile_score_sums = g.tile_score_sums;
        self.count_by_size = g.count_by_size;
        self.best_leave_values = g.best_leave_values;
        self.has_word_of_length = g.has_word_of_length;
    }

    fn load(self: *const WmpRackState, g: *wmp_movegen.MoveGen) void {
        g.np_racks = self.np_racks;
        g.np_entries = self.np_entries;
        g.np_known = self.np_known;
        g.leave_values = self.leave_values;
        g.tile_score_sums = self.tile_score_sums;
        g.count_by_size = self.count_by_size;
        g.best_leave_values = self.best_leave_values;
        g.has_word_of_length = self.has_word_of_length;
        g.resetBlankWordCache();
    }
};

const RackEntry = struct {
    /// 0 for an unused entry. Bounds written under another epoch are dead.
    epoch: u32 = 0,
    bag_key: i32 = 0,
    counts: [max_alphabet]u8 = @splat(0),
    moves: [moves_per_rack]Move = undefined,
    equities: [moves_per_rack]f64 = undefined,
    /// The tiles of each move's lane under it when it was last a
    /// generation's best play (its own tiles mark squares played through
    /// only as such, and on another board they may hold other tiles).
    through: [moves_per_rack][max_dim]MachineLetter = undefined,
    /// Changes whenever the move is replaced.
    stamps: [moves_per_rack]u32 = @splat(0),
    num_moves: usize = 0,
    next_move: usize = 0,
    next_bound: [2][max_dim]u8 = @splat(@splat(0)),
    has_setup: bool = false,
    has_wmp: bool = false,
    /// The best exchange (if any) when at most `exchange_max` tiles could
    /// be exchanged, or exchange_max -1 when none was worked out.
    exchange_max: i16 = -1,
    exchange: ?Move = null,
    exchange_eq: f64 = 0,
    leave_map: leavemap.LeaveMap = .{},
    best_leaves: leavemap.BestLeaves = undefined,
    /// The rack's rack info table entry, if the generator has a table.
    rit_entry: ?*const @import("../data/rit.zig").Entry = null,
    wmp_state: WmpRackState = undefined,
    /// bounds[(dir*dim+lane)*bounds_per_lane+i] is a lane's i'th bound;
    /// its contents are the dim squares from that index times dim.
    bounds: []Bound = &.{},
    squares: []LaneSquare = &.{},

    fn bound(self: *RackEntry, dim: usize, dir: usize, lane: usize, i: usize) struct { b: *Bound, sq: []LaneSquare } {
        const idx = (dir * dim + lane) * bounds_per_lane + i;
        return .{ .b = &self.bounds[idx], .sq = self.squares[idx * dim ..][0..dim] };
    }
};

pub const Carry = struct {
    gpa: Allocator,
    racks: [num_racks]RackEntry = @splat(.{}),
    next_epoch: u32 = 0,
    dim: usize = 0,
    bounds_buf: []Bound = &.{},
    squares_buf: []LaneSquare = &.{},

    // The generation in progress: the rack's entry (null when it doesn't
    // use the memory), and by lane its hash (0 without anchors), anchors,
    // remembered bound with the same contents (-1 for none) and whether
    // it is skipped.
    cur: ?*RackEntry = null,
    hashes: [2][max_dim]u64 = undefined,
    anchors: [2][max_dim]u32 = undefined,
    found: [2][max_dim]i8 = undefined,
    skip: [2][max_dim]bool = undefined,

    // Counts, for tests and measurements.
    generations: u64 = 0,
    rack_hits: u64 = 0,
    lanes_seen: u64 = 0,
    lanes_skipped: u64 = 0,
    seeded: u64 = 0,

    pub fn init(gpa: Allocator) Carry {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Carry) void {
        self.gpa.free(self.bounds_buf);
        self.gpa.free(self.squares_buf);
        self.* = undefined;
    }

    /// Forgets everything (the generator's lexicon, word map or leaves
    /// changed).
    pub fn forget(self: *Carry) void {
        for (&self.racks) |*e| e.epoch = 0;
        self.cur = null;
    }

    fn layOut(self: *Carry, dim: usize) Allocator.Error!void {
        self.forget();
        const per_rack = 2 * dim * bounds_per_lane;
        const b = try self.gpa.alloc(Bound, num_racks * per_rack);
        errdefer self.gpa.free(b);
        const s = try self.gpa.alloc(LaneSquare, num_racks * per_rack * dim);
        self.gpa.free(self.bounds_buf);
        self.gpa.free(self.squares_buf);
        self.bounds_buf = b;
        self.squares_buf = s;
        @memset(b, .{});
        for (&self.racks, 0..) |*e, i| {
            e.bounds = b[i * per_rack ..][0..per_rack];
            e.squares = s[i * per_rack * dim ..][0 .. per_rack * dim];
        }
        self.dim = dim;
    }
};

/// Decides whether this generation uses the memory, and if so finds or
/// starts the rack's entry. Only where a play is priced as its score plus
/// the leave map's value, which with tiles in the bag depends on nothing
/// but the rack, the leave and the bag's size.
pub fn begin(gen: *Generator, rack: *const Rack) void {
    const c = gen.carry orelse return;
    c.cur = null;
    const ctx = gen.ctx;
    if (ctx.klv == null or rack.num_tiles == 0 or rack.num_tiles > leavemap.max_rack_size or
        ctx.board_empty or ctx.tiles_in_bag <= 0 or gen.ld.size > max_alphabet) return;
    if (c.dim != gen.board.dim) c.layOut(gen.board.dim) catch return;
    // A bag with at least as many tiles as there are pre-endgame values
    // gives every play its leave value alone, however many it has.
    const bag_key: i32 = @min(ctx.tiles_in_bag, @as(i32, @intCast(ctx.peg.len)));
    var hash: u64 = @as(u64, @intCast(bag_key)) +% 0x9E3779B97F4A7C15;
    for (rack.counts[0..gen.ld.size]) |n| {
        hash = (hash ^ n) *% 0xFF51AFD7ED558CCD;
        hash ^= hash >> 29;
    }
    const entry = &c.racks[(hash >> 32) & (num_racks - 1)];
    c.generations += 1;
    if (entry.epoch != 0 and entry.bag_key == bag_key and
        std.mem.eql(u8, entry.counts[0..gen.ld.size], rack.counts[0..gen.ld.size]))
    {
        c.rack_hits += 1;
    } else {
        c.next_epoch +%= 1;
        if (c.next_epoch == 0) {
            // Wrapped: old bounds could pass for new ones.
            for (&c.racks) |*e| e.epoch = 0;
            c.next_epoch = 1;
        }
        entry.epoch = c.next_epoch;
        entry.bag_key = bag_key;
        @memcpy(entry.counts[0..gen.ld.size], rack.counts[0..gen.ld.size]);
        entry.num_moves = 0;
        entry.next_move = 0;
        entry.next_bound = @splat(@splat(0));
        entry.has_setup = false;
        entry.has_wmp = false;
        entry.exchange_max = -1;
    }
    c.cur = entry;
}

/// The remembered leave map and best leaves for the rack, or null if the
/// generation must work them out (and then call saveSetup).
pub fn loadSetup(gen: *Generator, rack: *const Rack) bool {
    const c = gen.carry orelse return false;
    const entry = c.cur orelse return false;
    if (!entry.has_setup) return false;
    gen.leave_map = entry.leave_map;
    gen.leave_map.current = (@as(u32, 1) << @intCast(rack.num_tiles)) - 1;
    gen.shadow.best_leaves = entry.best_leaves;
    gen.rit_entry = entry.rit_entry;
    return true;
}

pub fn saveSetup(gen: *Generator) void {
    const c = gen.carry orelse return;
    const entry = c.cur orelse return;
    if (!gen.leave_map.initialized) {
        // Not a generation the memory is for after all.
        c.cur = null;
        return;
    }
    entry.leave_map = gen.leave_map;
    entry.best_leaves = gen.shadow.best_leaves;
    entry.rit_entry = gen.rit_entry;
    entry.has_setup = true;
}

/// Records the rack's remembered best exchange for at most `max` tiles,
/// if there is one; false if the exchanges must be looked at (and then
/// saveExchange called).
pub fn loadExchange(gen: *Generator, max: u8) bool {
    const c = gen.carry orelse return false;
    const entry = c.cur orelse return false;
    if (entry.exchange_max != max) return false;
    if (entry.exchange) |m| {
        gen.best = m;
        gen.best_eq = entry.exchange_eq;
        gen.have_best = true;
    }
    return true;
}

/// Remembers the generation's best move, just after the exchanges were
/// looked at (and nothing else), as the rack's best exchange.
pub fn saveExchange(gen: *Generator, max: u8) void {
    const c = gen.carry orelse return;
    const entry = c.cur orelse return;
    entry.exchange_max = max;
    entry.exchange = if (gen.have_best) gen.best else null;
    entry.exchange_eq = gen.best_eq;
}

/// Puts back the word map generator's subracks for the rack; false if
/// they must be worked out (and then saveWmp called).
pub fn loadWmp(gen: *Generator) bool {
    const c = gen.carry orelse return false;
    const entry = c.cur orelse return false;
    if (!entry.has_wmp) return false;
    entry.wmp_state.load(&gen.wmp_gen);
    return true;
}

pub fn saveWmp(gen: *Generator) void {
    const c = gen.carry orelse return;
    const entry = c.cur orelse return;
    entry.wmp_state.save(&gen.wmp_gen);
    entry.has_wmp = true;
}

/// Whether the generation in progress uses the memory.
pub inline fn active(gen: *const Generator) bool {
    const c = gen.carry orelse return false;
    return c.cur != null;
}

/// Hashes a lane's contents: its anchors, and each square's tile,
/// cross-set and cross-score. Never 0. Each square's terms are multiplied
/// by constants of its own and summed, so that the multiplies don't wait
/// on one another; a hash only finds a remembered lane, which is then
/// compared whole, so collisions cost time, not correctness.
fn hashLane(squares: []const CachedSquare, anchors: u32) u64 {
    var a: u64 = anchors;
    var b: u64 = 0;
    for (squares, 0..) |*sq, i| {
        a +%= sq.cross_set *% lane_hash_keys[2 * i];
        b +%= (@as(u64, @as(u32, @bitCast(sq.cross_score))) | @as(u64, sq.letter) << 32) *% lane_hash_keys[2 * i + 1];
    }
    var h = (a ^ std.math.rotl(u64, b, 32)) *% 0xC4CEB9FE1A85EC53;
    h ^= h >> 32;
    return h | 1;
}

/// Odd constants for hashLane, two per square (splitmix64's outputs).
const lane_hash_keys = blk: {
    var keys: [2 * max_dim]u64 = undefined;
    var x: u64 = 0x9E3779B97F4A7C15;
    for (&keys) |*k| {
        x +%= 0x9E3779B97F4A7C15;
        var z = x;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        k.* = (z ^ (z >> 31)) | 1;
    }
    break :blk keys;
};

/// Notes lane (dir, lane) for the generation, with its anchors, and finds
/// its remembered bound.
pub fn noteLane(gen: *Generator, dir: usize, lane: usize, anchors: u32) void {
    const c = gen.carry.?;
    c.anchors[dir][lane] = anchors;
    c.found[dir][lane] = -1;
    c.skip[dir][lane] = false;
    if (anchors == 0) {
        c.hashes[dir][lane] = 0;
        return;
    }
    const dim = gen.board.dim;
    const squares = gen.lanePtr(dir, lane)[0..dim];
    const hash = hashLane(squares, anchors);
    c.hashes[dir][lane] = hash;
    c.lanes_seen += 1;
    const entry = c.cur.?;
    for (0..bounds_per_lane) |i| {
        const bd = entry.bound(dim, dir, lane, i);
        if (bd.b.hash != hash or bd.b.epoch != entry.epoch or bd.b.anchors != anchors or
            entry.stamps[bd.b.move_idx] != bd.b.stamp) continue;
        const same = for (bd.sq, squares) |k, *sq| {
            if (!k.eql(sq)) break false;
        } else true;
        if (same) {
            c.found[dir][lane] = @intCast(i);
            return;
        }
    }
}

/// The score of `m`'s tiles laid in `squares` (Go's scoreLanePlay).
fn scoreInLane(ld: *const tiles.LetterDistribution, squares: []const CachedSquare, m: *const Move, start: usize) i32 {
    var played: i32 = 0;
    var playthrough: i32 = 0;
    var hooked_cross: i32 = 0;
    var played_cross: i32 = 0;
    var word_mul: i32 = 1;
    for (m.tileSlice(), start..) |ml, col| {
        const sq = &squares[col];
        if (ml == 0) {
            playthrough += ld.score(sq.letter);
            continue;
        }
        const tile_score = ld.score(ml);
        const wm: i32 = sq.word_mul;
        const lm: i32 = sq.letter_mul;
        hooked_cross += sq.cross_score * wm;
        if (sq.cross_score > 0 or sq.cross_set != board_mod.trivial_cross_set) played_cross += tile_score * wm * lm;
        played += tile_score * lm;
        word_mul *= wm;
    }
    const bingo: i32 = if (m.tiles_played == movegen.rack_tile_limit) movegen.bingo_bonus else 0;
    return (played + playthrough) * word_mul + hooked_cross + played_cross + bingo;
}

/// The remembered play i scored on this board, if it is a play of it: its
/// lane has the tiles it was played through and none just before or after
/// it, the squares it fills are empty and allow its tiles, and it covers
/// an anchor. It then spells the word it spelled, with valid cross words,
/// and the generator finds every such placement. A one-tile play down a
/// column is taken only if its tile makes no word across: otherwise the
/// generator makes it in its horizontal form only, and the vertical one
/// could win a tie against it.
fn revalidate(gen: *const Generator, rack: *const Rack, entry: *const RackEntry, i: usize) ?struct { score: i32, eq: f64 } {
    const c = gen.carry.?;
    const m = &entry.moves[i];
    if (m.kind != .play) return null;
    const dir: usize = @intFromBool(m.vertical);
    const lane: usize = if (m.vertical) m.col else m.row;
    const start: usize = if (m.vertical) m.row else m.col;
    const dim = gen.board.dim;
    const end = start + m.len;
    const anchors = c.anchors[dir][lane];
    if (end > dim or anchors & ((@as(u32, 1) << @intCast(end)) - (@as(u32, 1) << @intCast(start))) == 0) return null;
    const squares = gen.lanePtr(dir, lane)[0..dim];
    if (start > 0 and squares[start - 1].letter != 0) return null;
    if (end < dim and squares[end].letter != 0) return null;
    const one_tile_down = m.vertical and m.tiles_played == 1;
    var used: [max_alphabet]u8 = @splat(0);
    for (m.tileSlice(), start..) |ml, col| {
        const sq = &squares[col];
        if (ml == 0) {
            if (sq.letter == 0 or sq.letter != entry.through[i][col - start]) return null;
            continue;
        }
        if (sq.letter != 0 or (sq.cross_set >> @intCast(tiles.unblank(ml))) & 1 == 0) return null;
        if (one_tile_down and sq.cross_set != board_mod.trivial_cross_set) return null;
        used[if (tiles.isBlanked(ml)) 0 else ml] += 1;
    }
    // The leave's index in the leave map: of each tile type, the low bits
    // of its slot, as many as are left.
    var idx: u32 = 0;
    for (rack.counts[0..gen.ld.size], 0..) |n, ml| {
        if (n == 0) continue;
        if (used[ml] > n) return null;
        const left = n - used[ml];
        idx |= ((@as(u32, 1) << @intCast(left)) - 1) << @intCast(gen.leave_map.base[ml]);
    }
    const score = scoreInLane(gen.ld, squares, m, start);
    return .{ .score = score, .eq = @as(f64, @floatFromInt(score)) + gen.leave_map.values[idx] };
}

/// Called once every lane is noted: starts the generation from the best
/// remembered play that is a play of this board, if it beats what the
/// generation has (an exchange, or nothing), then marks every unchanged
/// lane whose bound the generation's best play is no worse than.
pub fn seedAndSkip(gen: *Generator, rack: *const Rack) void {
    const c = gen.carry.?;
    const entry = c.cur.?;
    const dim = gen.board.dim;
    var seeded = false;
    for (0..entry.num_moves) |i| {
        const r = revalidate(gen, rack, entry, i) orelse continue;
        if (gen.have_best and r.eq < gen.best_eq) continue;
        var m = entry.moves[i];
        m.score = r.score;
        if (!gen.have_best or equity.better(&m, r.eq, &gen.best, gen.best_eq)) {
            gen.best = m;
            gen.best_eq = r.eq;
            gen.have_best = true;
            seeded = true;
        }
    }
    if (seeded) c.seeded += 1;
    if (!gen.have_best) return;
    if (gen.best_eq > gen.shadow.anchor_floor) gen.shadow.anchor_floor = gen.best_eq;
    for (0..2) |dir| {
        for (0..dim) |lane| {
            const f = c.found[dir][lane];
            if (f < 0) continue;
            const bd = entry.bound(dim, dir, lane, @intCast(f));
            const k = bd.b.move_idx;
            if (!equity.better(&entry.moves[k], entry.equities[k], &gen.best, gen.best_eq)) {
                c.skip[dir][lane] = true;
                c.lanes_skipped += 1;
            }
        }
    }
}

/// Called when the generation is over, with its best move: that move
/// bounds every lane the generation searched.
pub fn remember(gen: *Generator) void {
    const c = gen.carry orelse return;
    const entry = c.cur orelse return;
    c.cur = null;
    if (!gen.have_best) return;
    const best = &gen.best;
    const best_eq = gen.best_eq;
    // The move's slot: the one already holding it, or the oldest.
    var move_idx: usize = for (0..entry.num_moves) |i| {
        const m = &entry.moves[i];
        if (entry.equities[i] == best_eq and !equity.tiebreaksBetter(m, best) and !equity.tiebreaksBetter(best, m)) break i;
    } else moves_per_rack;
    if (move_idx == moves_per_rack) {
        move_idx = entry.next_move;
        entry.next_move = (entry.next_move + 1) % moves_per_rack;
        if (entry.num_moves < moves_per_rack) entry.num_moves += 1;
        entry.stamps[move_idx] +%= 1;
        entry.moves[move_idx] = best.*;
        entry.equities[move_idx] = best_eq;
    }
    const dim = gen.board.dim;
    if (best.kind == .play) {
        // Its squares, as the board holds them.
        const dir: usize = @intFromBool(best.vertical);
        const lane: usize = if (best.vertical) best.col else best.row;
        const start: usize = if (best.vertical) best.row else best.col;
        for (0..best.len) |k| entry.through[move_idx][k] = gen.lanePtr(dir, lane)[start + k].letter;
    }
    for (0..2) |dir| {
        for (0..dim) |lane| {
            if (c.hashes[dir][lane] == 0 or c.skip[dir][lane]) continue;
            const f = c.found[dir][lane];
            if (f >= 0) {
                // The same contents are remembered with another bound;
                // keep the lower of the two (a bound whose move was just
                // replaced is rewritten too).
                const b = entry.bound(dim, dir, lane, @intCast(f)).b;
                const k = b.move_idx;
                if (b.stamp != entry.stamps[k] or equity.better(&entry.moves[k], entry.equities[k], best, best_eq)) {
                    b.move_idx = @intCast(move_idx);
                    b.stamp = entry.stamps[move_idx];
                }
                continue;
            }
            const slot = entry.next_bound[dir][lane];
            entry.next_bound[dir][lane] = (slot + 1) % bounds_per_lane;
            const bd = entry.bound(dim, dir, lane, slot);
            bd.b.* = .{
                .hash = c.hashes[dir][lane],
                .epoch = entry.epoch,
                .stamp = entry.stamps[move_idx],
                .anchors = c.anchors[dir][lane],
                .move_idx = @intCast(move_idx),
            };
            for (bd.sq, gen.lanePtr(dir, lane)[0..dim]) |*k, *sq| k.* = .of(sq);
        }
    }
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

test "lane hashes are never 0 and see every field" {
    var squares: [15]CachedSquare = @splat(.{ .cross_set = 3, .left_ext = 0, .right_ext = 0, .cross_score = 2, .letter = 0, .letter_mul = 1, .word_mul = 1, .anchor = false });
    const h = hashLane(&squares, 5);
    try testing.expect(h != 0);
    try testing.expect(hashLane(&squares, 4) != h);
    squares[3].letter = 7;
    const h2 = hashLane(&squares, 5);
    try testing.expect(h2 != h);
    squares[3].cross_score = 9;
    try testing.expect(hashLane(&squares, 5) != h2);
}
