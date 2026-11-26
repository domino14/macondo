//! Incremental move lists for the endgame search (port of
//! endgame/negamax/pathlists.go).
//!
//! A row's or a column's plays (a lane's) depend only on that lane's
//! squares and on the rack, and a move changes those for a handful of
//! lanes. So the lists of the positions along the search's path are kept by
//! lane, and a position's list is derived from the nearest ancestor's for
//! the same player: the lanes the moves in between touched are generated
//! again (anchor by anchor where only part of a lane changed), the others
//! copied, less the plays of tiles the player has since played. The result
//! is the list a full generation makes, in the same order, without the
//! pass. The lists also keep each player's best play of a lone tile.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("../data/tiles.zig");
const board_mod = @import("../board/board.zig");
const movegen = @import("../movegen/movegen.zig");
const endgame = @import("endgame.zig");
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const Board = board_mod.Board;
const SmallMove = endgame.SmallMove;

pub const max_lanes = movegen.Generator.max_lanes;
const max_tile_types = 48;
const high_bits: u64 = 0x8888888888888888;

/// A multiset of tiles, 4 bits a type (the blank is type 0); a rack's has
/// the top bit of every field set for `holds`.
const TileCounts = [3]u64;

fn rackTileCounts(rack: *const Rack, num_letters: usize) TileCounts {
    var c: TileCounts = @splat(high_bits);
    for (0..num_letters) |ml| {
        const n = rack.counts[ml];
        if (n > 0) c[ml >> 4] += @as(u64, n) << @intCast((ml & 15) * 4);
    }
    return c;
}

fn playTileCounts(m: SmallMove) TileCounts {
    var c: TileCounts = @splat(0);
    for (0..m.tiles_played) |i| {
        const ml: usize = m.rackTile(i);
        c[ml >> 4] += @as(u64, 1) << @intCast((ml & 15) * 4);
    }
    return c;
}

/// Whether the rack (rackTileCounts) holds `used`: subtracting a count no
/// larger than a field's keeps its top bit, a larger one clears it.
fn holds(rack: *const TileCounts, used: *const TileCounts) bool {
    for (0..3) |i| {
        if ((rack[i] -% used[i]) & high_bits != high_bits) return false;
    }
    return true;
}

/// For each tile type with a play alone in a list, the best: the highest
/// score, and of those the first in row order (as scanLoneTile has it).
const SoloTable = struct {
    found: u64 = 0,
    score: [max_tile_types]i16 = undefined,
    row: [max_tile_types]i8 = undefined,
    col: [max_tile_types]i8 = undefined,
};

pub const MoveList = struct {
    moves: std.ArrayList(SmallMove) = .empty,
    used: std.ArrayList(TileCounts) = .empty,
    /// The anchor (its index along the lane) each play came from.
    anchor: std.ArrayList(u8) = .empty,
    lane_begin: [max_lanes + 1]i32 = undefined,
    valid: bool = false,
    solo: SoloTable = .{},
    solo_valid: bool = false,
    /// The plays by tiles played, most first, and within, by score, best
    /// first (tiles << 56 | score << 32 | index): made when first wanted
    /// (aloneBest). The plays of t tiles are
    /// by_score[tiles_begin[t + 1]..tiles_begin[t]].
    by_score: std.ArrayList(u64) = .empty,
    tiles_begin: [max_tiles + 2]u32 = undefined,
    by_score_valid: bool = false,

    const max_tiles = movegen.rack_tile_limit;

    fn deinit(l: *MoveList, gpa: Allocator) void {
        l.moves.deinit(gpa);
        l.used.deinit(gpa);
        l.anchor.deinit(gpa);
        l.by_score.deinit(gpa);
    }

    fn orderByScore(l: *MoveList, gpa: Allocator) Allocator.Error!void {
        if (l.by_score_valid) return;
        const moves = l.moves.items;
        try l.by_score.resize(gpa, moves.len);
        for (moves, l.by_score.items, 0..) |m, *k, i| {
            k.* = @as(u64, m.tiles_played) << 56 | @as(u64, @as(u24, @intCast(m.score))) << 32 | i;
        }
        std.sort.pdq(u64, l.by_score.items, {}, std.sort.desc(u64));
        // tiles_begin[t] counts the plays of t tiles or more.
        var i: usize = 0;
        var t: usize = max_tiles + 1;
        while (t > 0) {
            t -= 1;
            while (i < l.by_score.items.len and l.by_score.items[i] >> 56 > t) i += 1;
            l.tiles_begin[t + 1] = @intCast(i);
        }
        l.by_score_valid = true;
    }

    pub fn soloPlays(l: *MoveList, b: *const Board) *const SoloTable {
        const t = &l.solo;
        if (l.solo_valid) return t;
        l.solo_valid = true;
        t.found = 0;
        for (l.moves.items) |m| {
            if (m.tiles_played != 1) continue;
            const tile: usize = m.rackTile(0);
            const score = m.score;
            const known = t.found & (@as(u64, 1) << @intCast(tile)) != 0;
            if (known and score < t.score[tile]) continue;
            var row = m.row();
            var col = m.col();
            var j: usize = 0;
            while (j < m.len and b.hasLetter(row, col)) : (j += 1) {
                if (m.vertical()) row += 1 else col += 1;
            }
            if (!known or score > t.score[tile] or row < @as(usize, @intCast(t.row[tile])) or
                (row == @as(usize, @intCast(t.row[tile])) and col < @as(usize, @intCast(t.col[tile]))))
            {
                t.found |= @as(u64, 1) << @intCast(tile);
                t.score[tile] = score;
                t.row[tile] = @intCast(row);
                t.col[tile] = @intCast(col);
            }
        }
        return t;
    }
};

pub const LonePlay = struct {
    valid: bool = false,
    found: bool = false,
    tile: MachineLetter = 0,
    row: u8 = 0,
    col: u8 = 0,
    score: i32 = 0,
};

const max_changed = 16;

/// What the lists keep for one position on the path.
const Level = struct {
    lists: [2]MoveList = .{ .{}, .{} },
    /// The lanes the move played from this position changes, and in each
    /// the squares.
    hits: u64 = 0,
    lane_hits: [max_lanes]u32 = undefined,
    mover: u1 = 0,
    tile_play: bool = false,
    /// The empty squares that move changes (row << 8 | col): what a
    /// one-tile play depends on.
    changed: [max_changed]u16 = undefined,
    n_changed: u8 = 0,
    lone: [2]LonePlay = .{ .{}, .{} },
    /// The opponent's tile types with a play here, once worked out.
    stuck_types: ?u64 = null,
    /// Each player's best plays here by lane, shared by the positions one
    /// move down in the lanes their move leaves alone (siblingBest).
    sibling: [2]Sibling = .{ .{}, .{} },
};

/// A lane's best play of the player's whole rack and best other play (by
/// score, ties to the larger move code), each a pass if there is none.
const LaneBest = struct { whole: SmallMove = .{}, partial: SmallMove = .{} };

const Sibling = struct {
    /// The lanes whose bests are known.
    valid: u64 = 0,
    best: [max_lanes]LaneBest = undefined,
};

/// The solver's view the lists need: its board, racks and move generator.
pub const Host = struct {
    board: *Board,
    /// The board's occupancy, a mask a row.
    occ: *const [board_mod.max_dim]u32,
    racks: *[2]Rack,
    on_turn: *const u1,
    gen: *movegen.Generator,
    dist: *const tiles.LetterDistribution,
};

pub const PathLists = struct {
    gpa: Allocator,
    host: Host,
    levels: std.ArrayList(Level) = .empty,
    /// The stack depth the lists were started at: nothing is derived across
    /// it.
    base: usize = 0,
    dim: usize,
    ends: [max_lanes]i32 = undefined,
    hit: [max_lanes]u32 = undefined,
    anchors: [max_lanes]u32 = undefined,
    wanted: [max_lanes]u32 = undefined,
    gen_anchors: std.ArrayList(u8) = .empty,
    generated: std.ArrayList(SmallMove) = .empty,
    /// A list of some lanes' plays (siblingBest).
    scratch: MoveList = .{},
    /// Set while a leaf is played out.
    in_playout: bool = false,
    /// Compare every list with a full generation (tests).
    check: bool = false,

    pub fn init(gpa: Allocator, host: Host) PathLists {
        return .{ .gpa = gpa, .host = host, .dim = host.board.dim };
    }

    pub fn deinit(p: *PathLists) void {
        for (p.levels.items) |*lv| {
            lv.lists[0].deinit(p.gpa);
            lv.lists[1].deinit(p.gpa);
        }
        p.levels.deinit(p.gpa);
        p.gen_anchors.deinit(p.gpa);
        p.generated.deinit(p.gpa);
        p.scratch.deinit(p.gpa);
    }

    /// Forgets everything, the lists starting at stack depth `base`.
    pub fn reset(p: *PathLists, base: usize) void {
        p.base = base;
        p.in_playout = false;
        for (p.levels.items) |*lv| {
            lv.lists[0].valid = false;
            lv.lists[1].valid = false;
            lv.lone = .{ .{}, .{} };
            lv.stuck_types = null;
            lv.sibling[0].valid = 0;
            lv.sibling[1].valid = 0;
        }
    }

    fn grow(p: *PathLists, n: usize) Allocator.Error!void {
        while (p.levels.items.len < n) try p.levels.append(p.gpa, .{});
    }

    fn numLetters(p: *const PathLists) usize {
        return p.host.dist.size;
    }

    fn loneTile(p: *const PathLists, rack: *const Rack) MachineLetter {
        for (0..p.numLetters()) |ml| {
            if (rack.counts[ml] > 0) return @intCast(ml);
        }
        return 0;
    }

    /// Before `m` is played from the position at stack depth `n`.
    pub fn onMove(p: *PathLists, n: usize, m: SmallMove) Allocator.Error!void {
        try p.grow(n + 2);
        const lv = &p.levels.items[n];
        lv.mover = p.host.on_turn.*;
        lv.tile_play = !m.isPass();
        for (0..2) |player| {
            const rack = &p.host.racks[player];
            if (rack.num_tiles == 1) {
                if (!p.in_playout or (n > p.base and p.levels.items[n - 1].lone[player].valid)) {
                    _ = p.lonePlayAt(n, @intCast(player), p.loneTile(rack));
                }
            }
        }
        if (lv.tile_play) {
            const mover = lv.mover;
            const rack = &p.host.racks[mover];
            const n_tiles = rack.num_tiles;
            if (n_tiles >= 2 and m.tiles_played == n_tiles - 1) {
                const l = &lv.lists[mover];
                if (l.valid) {
                    const tile = keptTile(rack, m, p.numLetters());
                    const c = &lv.lone[mover];
                    if (!c.valid or c.tile != tile) p.noteLoneFromList(c, l, tile);
                }
            }
        }
        lv.hits = p.moveHits(m, lv);
        const next = &p.levels.items[n + 1];
        next.lists[0].valid = false;
        next.lists[1].valid = false;
        next.lone = .{ .{}, .{} };
        next.stuck_types = null;
        next.sibling[0].valid = 0;
        next.sibling[1].valid = 0;
    }

    /// The player's best play at stack depth `n` as siblingBest finds it,
    /// for a position without siblings (MAGPIE's
    /// path_move_lists_best_play_alone): among the plays of their list two
    /// moves up that still stand (which their rack holds, covering no
    /// square either move changed), and those of the lanes the two moves
    /// changed, which are returned to be searched. Null without that list.
    pub fn aloneBest(p: *PathLists, n: usize, player: u1, out_bonus: i32) Allocator.Error!?Shared {
        if (n < p.base + 2 or n > p.levels.items.len) return null;
        const l = &p.levels.items[n - 2].lists[player];
        if (!l.valid) return null;
        try l.orderByScore(p.gpa);
        const num_lanes = 2 * p.dim;
        @memset(p.hit[0..num_lanes], 0);
        var touched: u64 = 0;
        for (n - 2..n) |j| {
            const lv = &p.levels.items[j];
            touched |= lv.hits;
            var lanes = lv.hits;
            while (lanes != 0) : (lanes &= lanes - 1) {
                const lane = @ctz(lanes);
                p.hit[lane] |= lv.lane_hits[lane];
            }
        }
        const rack = &p.host.racks[player];
        const n_tiles = rack.num_tiles;
        const counts = rackTileCounts(rack, p.numLetters());
        var best: ?SmallMove = null;
        var best_v: i32 = std.math.minInt(i32);
        var t: usize = @min(n_tiles, MoveList.max_tiles);
        while (t >= 1) : (t -= 1) {
            const bonus: i32 = if (t == n_tiles) out_bonus else 0;
            for (l.by_score.items[l.tiles_begin[t + 1]..l.tiles_begin[t]]) |key| {
                const score: i32 = @intCast(key >> 32 & 0xffffff);
                if (score + bonus < best_v) break;
                const idx: usize = @intCast(key & 0xffffffff);
                const m = l.moves.items[idx];
                const lane = if (m.vertical()) p.dim + m.col() else m.row();
                if (playSpan(m) & p.hit[lane] != 0) continue;
                if (!holds(&counts, &l.used.items[idx])) continue;
                const v = score + bonus;
                if (best == null or v > best_v or (v == best_v and m.tm > best.?.tm)) {
                    best = m;
                    best_v = v;
                }
            }
        }
        return .{ .best = best, .value = best_v, .own = touched };
    }

    /// What siblingBest finds.
    pub const Shared = struct {
        /// The best play in the lanes the last move left alone, if any, and
        /// its value.
        best: ?SmallMove,
        value: i32,
        /// The lanes the last move changed, to be searched.
        own: u64,
    };

    /// The player's best play at stack depth `n` by score plus, for one of
    /// their whole rack, `out_bonus`, ties to the larger move code, among
    /// the lanes the last move (from n - 1, by the other player) left
    /// alone, and those it changed. The plays in the lanes it left alone
    /// are the player's plays one move up, whose best by lane all the
    /// positions after n - 1 share (MAGPIE's path_move_lists_best_play).
    /// Null at the start of the path.
    pub fn siblingBest(p: *PathLists, n: usize, player: u1, out_bonus: i32) Allocator.Error!?Shared {
        if (n <= p.base or n > p.levels.items.len) return null;
        const up = &p.levels.items[n - 1];
        if (up.tile_play and up.mover == player) return null;
        const num_lanes = 2 * p.dim;
        const own = up.hits;
        const shared = ((@as(u64, 1) << @intCast(num_lanes)) - 1) & ~own;
        const sib = &up.sibling[player];
        const n_tiles = p.host.racks[player].num_tiles;
        const missing = shared & ~sib.valid;
        if (missing != 0) {
            const l = if (up.lists[player].valid) &up.lists[player] else blk: {
                try p.derive(n - 1, player, missing, &p.scratch);
                break :blk &p.scratch;
            };
            var lanes = missing;
            while (lanes != 0) : (lanes &= lanes - 1) {
                const lane = @ctz(lanes);
                var lb: LaneBest = .{};
                for (l.moves.items[@intCast(l.lane_begin[lane])..@intCast(l.lane_begin[lane + 1])]) |m| {
                    const kept = if (m.tiles_played == n_tiles) &lb.whole else &lb.partial;
                    if (kept.isPass() or m.score > kept.score or (m.score == kept.score and m.tm > kept.tm)) kept.* = m;
                }
                sib.best[lane] = lb;
            }
            sib.valid |= missing;
        }
        var best: ?SmallMove = null;
        var best_v: i32 = 0;
        var lanes = shared;
        while (lanes != 0) : (lanes &= lanes - 1) {
            const lane = @ctz(lanes);
            const lb = &sib.best[lane];
            inline for (.{ true, false }) |whole| {
                const m = if (whole) lb.whole else lb.partial;
                if (!m.isPass()) {
                    const v = m.score + (if (whole) out_bonus else 0);
                    if (best == null or v > best_v or (v == best_v and m.tm > best.?.tm)) {
                        best = m;
                        best_v = v;
                    }
                }
            }
        }
        return .{ .best = best, .value = best_v, .own = own };
    }

    /// Notes the opponent's tile types with a play at stack depth `n`.
    pub fn noteStuckTypes(p: *PathLists, n: usize, types: u64) void {
        if (n < p.levels.items.len) p.levels.items[n].stuck_types = types;
    }

    /// The opponent's tile types with a play one move up from stack depth
    /// `n`, if they were worked out.
    pub fn stuckTypesBefore(p: *const PathLists, n: usize) ?u64 {
        if (n <= p.base or n > p.levels.items.len) return null;
        return p.levels.items[n - 1].stuck_types;
    }

    /// Sets p.hit to the squares the last move (from stack depth n - 1)
    /// changed, and returns its lanes.
    pub fn lastMoveHits(p: *PathLists, n: usize) u64 {
        @memset(p.hit[0 .. 2 * p.dim], 0);
        const lv = &p.levels.items[n - 1];
        var lanes = lv.hits;
        while (lanes != 0) : (lanes &= lanes - 1) {
            const lane = @ctz(lanes);
            p.hit[lane] = lv.lane_hits[lane];
        }
        return lv.hits;
    }

    /// The tile types of `wanted` that a play of the player's places at
    /// stack depth `n` (the position now), as far as their nearest list up
    /// the path tells: the plays of it that still stand, which their rack
    /// holds and which cover no square a move since has changed (in p.hit,
    /// the lanes of `touched`). The plays it cannot tell are those of the
    /// anchors in `touched` whose reach covers such a square: see
    /// tileTypesGenerated. Null without a list up the path (MAGPIE's
    /// path_move_lists_standing_tile_types).
    pub fn standingTileTypes(p: *PathLists, n: usize, player: u1, wanted: u64) ?struct { found: u64, touched: u64 } {
        if (n >= p.levels.items.len or n < p.base) return null;
        const num_lanes = 2 * p.dim;
        @memset(p.hit[0..num_lanes], 0);
        var touched: u64 = 0;
        var j = n;
        const l = while (true) {
            if (p.levels.items[j].lists[player].valid) break &p.levels.items[j].lists[player];
            if (j == p.base) return null;
            j -= 1;
            const lv = &p.levels.items[j];
            touched |= lv.hits;
            var lanes = lv.hits;
            while (lanes != 0) : (lanes &= lanes - 1) {
                const lane = @ctz(lanes);
                p.hit[lane] |= lv.lane_hits[lane];
            }
        };
        const counts = rackTileCounts(&p.host.racks[player], p.numLetters());
        var found: u64 = 0;
        for (l.moves.items, l.used.items) |m, *used| {
            var types: u64 = 0;
            for (0..m.tiles_played) |i| types |= @as(u64, 1) << @intCast(m.rackTile(i));
            if (types & wanted & ~found == 0) continue;
            if (!holds(&counts, used)) continue;
            const lane = if (m.vertical()) p.dim + m.col() else m.row();
            if (playSpan(m) & p.hit[lane] != 0) continue;
            found |= types & wanted;
            if (found == wanted) break;
        }
        return .{ .found = found, .touched = touched };
    }

    /// The tile types of `wanted` that the plays of the player's from the
    /// anchors in the lanes of `lanes` whose reach covers a square of p.hit
    /// (as standingTileTypes leaves it) place; stops once it has them all.
    pub fn tileTypesGenerated(p: *PathLists, player: u1, lanes: u64, wanted: u64) Allocator.Error!u64 {
        const rack = &p.host.racks[player];
        const n_tiles = rack.num_tiles;
        // The anchors to generate from, worked out on the board as it
        // stands (before the generator transposes it for the columns).
        var lane_set = lanes;
        while (lane_set != 0) : (lane_set &= lane_set - 1) {
            const lane = @ctz(lane_set);
            const masks = p.laneMasks(lane);
            const full = (@as(u32, 1) << @intCast(p.dim)) - 1;
            const empties = ~masks.occupied & full;
            const blocked = empties & ~p.laneFits(lane, rack);
            const hit = p.hit[lane];
            var wanted_anchors: u32 = 0;
            var prev: ?usize = null;
            var as = masks.anchors;
            while (as != 0) : (as &= as - 1) {
                const a: usize = @ctz(as);
                if (spanReach(empties, blocked, a, prev, n_tiles, p.dim) & hit != 0) wanted_anchors |= @as(u32, 1) << @intCast(a);
                prev = a;
            }
            p.anchors[lane] = masks.anchors;
            p.wanted[lane] = wanted_anchors;
        }
        const gen = p.host.gen;
        gen.plays.clearRetainingCapacity();
        gen.out_of_memory = false;
        gen.best_only = false;
        gen.leave_map.initialized = false;
        gen.tiles_played = 0;
        gen.setRackBits(rack);
        defer gen.plays.clearRetainingCapacity();
        var found: u64 = 0;
        const dim = p.dim;
        inline for (.{ false, true }) |vertical| {
            const side_lanes = if (vertical) lanes >> @intCast(dim) else lanes & ((@as(u64, 1) << @intCast(dim)) - 1);
            if (side_lanes != 0 and found != wanted) {
                if (vertical) gen.board.transpose();
                defer if (vertical) gen.board.transpose();
                gen.vertical = vertical;
                var rows = side_lanes;
                while (rows != 0 and found != wanted) : (rows &= rows - 1) {
                    const row: usize = @ctz(rows);
                    const lane = if (vertical) dim + row else row;
                    const lane_anchors = p.anchors[lane];
                    var want = p.wanted[lane];
                    if (want == 0) continue;
                    gen.cur_row = row;
                    gen.loadRow(row, if (vertical) .horizontal else .vertical);
                    while (want != 0 and found != wanted) : (want &= want - 1) {
                        const col: usize = @ctz(want);
                        const before = lane_anchors & ((@as(u32, 1) << @intCast(col)) - 1);
                        gen.last_anchor_col = if (before != 0) 31 - @clz(before) else 100;
                        gen.cur_anchor_col = col;
                        gen.genForAnchor(rack);
                        if (gen.out_of_memory) return error.OutOfMemory;
                        for (gen.plays.items) |*mv| {
                            const m: SmallMove = .fromMove(mv);
                            for (0..m.tiles_played) |i| found |= (@as(u64, 1) << @intCast(m.rackTile(i))) & wanted;
                        }
                        gen.plays.clearRetainingCapacity();
                    }
                }
            }
        }
        return found;
    }

    /// The best play of the player's one tile at stack depth `n`.
    pub fn bestLoneTilePlay(p: *PathLists, n: usize, player: u1, solver: anytype) Allocator.Error!?SmallMove {
        try p.grow(n + 1);
        const tile = p.loneTile(&p.host.racks[player]);
        const c = p.lonePlayAt(n, player, tile);
        if (!c.found) return null;
        return solver.loneTileMove(tile, c.row, c.col, c.score);
    }

    /// Notes, at stack depth `n`, the best play of `tile` alone for the
    /// player, who holds it among others and is about to play the rest.
    pub fn noteLonePlay(p: *PathLists, n: usize, player: u1, tile: MachineLetter) Allocator.Error!void {
        const c = &p.levels.items[n].lone[player];
        if (c.valid and c.tile == tile) return;
        const l = try p.get(n, player);
        p.noteLoneFromList(&p.levels.items[n].lone[player], l, tile);
    }

    fn noteLoneFromList(p: *PathLists, c: *LonePlay, l: *MoveList, tile: MachineLetter) void {
        const t = l.soloPlays(p.host.board);
        c.valid = true;
        c.tile = tile;
        c.found = t.found & (@as(u64, 1) << @intCast(tile)) != 0;
        if (c.found) {
            c.row = @intCast(t.row[tile]);
            c.col = @intCast(t.col[tile]);
            c.score = t.score[tile];
        }
    }

    fn lonePlayAt(p: *PathLists, n: usize, player: u1, tile: MachineLetter) *LonePlay {
        const c = &p.levels.items[n].lone[player];
        if (c.valid and c.tile == tile) return c;
        const b = p.host.board;
        c.valid = true;
        c.tile = tile;
        var src: ?usize = null;
        var j = n;
        while (j > p.base) {
            j -= 1;
            const lv = &p.levels.items[j];
            if (lv.lone[player].valid and lv.lone[player].tile == tile) {
                src = j;
                break;
            }
            if (lv.tile_play and lv.mover == player) break;
        }
        if (src) |s| {
            const old = p.levels.items[s].lone[player];
            var derived = true;
            if (old.found) {
                if (b.hasLetter(old.row, old.col)) derived = false;
                const key = @as(u16, old.row) << 8 | old.col;
                var k = s;
                while (k < n and derived) : (k += 1) {
                    const lv = &p.levels.items[k];
                    for (lv.changed[0..lv.n_changed]) |sq| {
                        if (sq == key) {
                            derived = false;
                            break;
                        }
                    }
                }
            }
            if (derived) {
                c.found = old.found;
                c.row = old.row;
                c.col = old.col;
                c.score = old.score;
                const face: i32 = if (tile == 0) 0 else p.host.dist.score_table[tile];
                for (s..n) |k| {
                    const lv = &p.levels.items[k];
                    for (lv.changed[0..lv.n_changed]) |sq| {
                        const r: usize = sq >> 8;
                        const col: usize = sq & 0xff;
                        if (b.hasLetter(r, col)) continue;
                        const score = loneTileAt(b, tile, face, r, col) orelse continue;
                        if (!c.found or score > c.score or (score == c.score and (r < c.row or (r == c.row and col < c.col)))) {
                            c.found = true;
                            c.row = @intCast(r);
                            c.col = @intCast(col);
                            c.score = score;
                        }
                    }
                }
                return c;
            }
        }
        if (endgame.scanLoneTileOn(b, p.host.occ, p.host.dist, tile)) |f| {
            c.found = true;
            c.row = @intCast(f.r);
            c.col = @intCast(f.c);
            c.score = f.score;
        } else {
            c.found = false;
        }
        return c;
    }

    /// moveHits, noting the changed empty squares in `lv` too.
    fn moveHits(p: *PathLists, m: SmallMove, lv: *Level) u64 {
        lv.n_changed = 0;
        if (m.isPass()) return 0;
        const hits = endgame.moveHits(p.host.board, m, &lv.lane_hits);
        const b = p.host.board;
        const dim = p.dim;
        const v = m.vertical();
        const fixed: usize = if (v) m.col() else m.row();
        const start: usize = if (v) m.row() else m.col();
        const len: usize = m.len;
        const Note = struct {
            fn f(l: *Level, vert: bool, lane: usize, pos: usize) void {
                const r = if (vert) pos else lane;
                const c = if (vert) lane else pos;
                l.changed[l.n_changed] = @intCast(r << 8 | c);
                l.n_changed += 1;
            }
            fn has(bb: *const Board, vert: bool, lane: usize, pos: usize) bool {
                return if (vert) bb.hasLetter(pos, lane) else bb.hasLetter(lane, pos);
            }
        };
        for (0..len) |i| {
            const pos = start + i;
            if (Note.has(b, v, fixed, pos)) continue;
            var above: isize = @as(isize, @intCast(fixed)) - 1;
            while (above >= 0 and Note.has(b, v, @intCast(above), pos)) above -= 1;
            if (above >= 0) Note.f(lv, v, @intCast(above), pos);
            var below = fixed + 1;
            while (below < dim and Note.has(b, v, below, pos)) below += 1;
            if (below < dim) Note.f(lv, v, below, pos);
        }
        if (start > 0) Note.f(lv, v, fixed, start - 1);
        if (start + len < dim) Note.f(lv, v, fixed, start + len);
        return hits;
    }

    /// The squares of `lane` where one of the rack's tiles fits (by the
    /// cross-sets the lane's plays read), as the generator tests it.
    fn laneFits(p: *const PathLists, lane: usize, rack: *const Rack) u32 {
        const b = p.host.board;
        const dim = p.dim;
        var rack_bits: u64 = 0;
        for (1..p.numLetters()) |ml| {
            if (rack.counts[ml] > 0) rack_bits |= @as(u64, 1) << @intCast(ml);
        }
        const blank = rack.counts[0] > 0;
        // A row's plays read the cross-sets of words down, a column's of
        // words across.
        const cs: usize = if (lane < dim) @backingInt(board_mod.Direction.vertical) else @backingInt(board_mod.Direction.horizontal);
        var fits: u32 = 0;
        for (0..dim) |i| {
            const idx = if (lane < dim) lane * dim + i else i * dim + (lane - dim);
            const allowed = b.cross_sets[cs][idx] & ~@as(u64, 1);
            const ok = if (blank) allowed != 0 else allowed & rack_bits != 0;
            fits |= @as(u32, @intFromBool(ok)) << @intCast(i);
        }
        return fits;
    }

    fn laneMasks(p: *const PathLists, lane: usize) struct { anchors: u32, occupied: u32 } {
        const b = p.host.board;
        const dim = p.dim;
        var anchors: u32 = 0;
        var occupied: u32 = 0;
        for (0..dim) |i| {
            const idx = if (lane < dim) lane * dim + i else i * dim + (lane - dim);
            const d: usize = if (lane < dim) 0 else 1;
            if (b.anchors[d][idx]) anchors |= @as(u32, 1) << @intCast(i);
            if (b.squares[idx] != 0) occupied |= @as(u32, 1) << @intCast(i);
        }
        return .{ .anchors = anchors, .occupied = occupied };
    }

    /// The player's tile plays at stack depth `n` (the solver's position),
    /// without the pass. The list stays as it is until a move is played
    /// into this depth again.
    pub fn get(p: *PathLists, n: usize, player: u1) Allocator.Error!*MoveList {
        try p.grow(n + 1);
        const l = &p.levels.items[n].lists[player];
        if (l.valid) return l;
        try p.derive(n, player, ~@as(u64, 0), l);
        l.valid = true;
        l.by_score_valid = false;
        if (p.check) try p.checkAgainstFullGeneration(l, player);
        return l;
    }

    /// Makes into `l` the player's plays at stack depth `n` in the lanes of
    /// `restrict` (the others left empty), from their nearest list up the
    /// path. The lanes of `restrict` must stand on the board as they did at
    /// `n`, and the player's rack must be as it was there.
    fn derive(p: *PathLists, n: usize, player: u1, restrict: u64, l: *MoveList) Allocator.Error!void {
        var src: ?*MoveList = null;
        var mask: u64 = 0;
        var rack_changed = false;
        const num_lanes = 2 * p.dim;
        @memset(p.hit[0..num_lanes], 0);
        var j = n;
        while (j > p.base) {
            j -= 1;
            const lv = &p.levels.items[j];
            mask |= lv.hits;
            var lanes = lv.hits;
            while (lanes != 0) : (lanes &= lanes - 1) {
                const lane = @ctz(lanes);
                p.hit[lane] |= lv.lane_hits[lane];
            }
            if (lv.tile_play and lv.mover == player) rack_changed = true;
            if (lv.lists[player].valid) {
                src = &lv.lists[player];
                break;
            }
        }
        const all_lanes = ((@as(u64, 1) << @intCast(num_lanes)) - 1) & restrict;
        if (src == null) mask = all_lanes;
        mask &= all_lanes;
        const rack = &p.host.racks[player];
        var partial: u64 = 0;
        var gen_mask = mask;
        if (src != null) {
            const full = (@as(u32, 1) << @intCast(p.dim)) - 1;
            const n_tiles = rack.num_tiles;
            var lanes = mask;
            while (lanes != 0) : (lanes &= lanes - 1) {
                const lane = @ctz(lanes);
                const hit = p.hit[lane] & full;
                const masks = p.laneMasks(lane);
                if (hit == full) {
                    p.anchors[lane] = masks.anchors;
                    p.wanted[lane] = ~@as(u32, 0);
                    continue;
                }
                const empties = ~masks.occupied & full;
                const blocked = empties & ~p.laneFits(lane, rack);
                var wanted: u32 = 0;
                var prev: ?usize = null;
                var as = masks.anchors;
                while (as != 0) : (as &= as - 1) {
                    const a: usize = @ctz(as);
                    const reach = spanReach(empties, blocked, a, prev, n_tiles, p.dim);
                    if (hit & reach != 0) wanted |= @as(u32, 1) << @intCast(a);
                    prev = a;
                }
                p.anchors[lane] = masks.anchors;
                p.wanted[lane] = wanted;
                partial |= @as(u64, 1) << @intCast(lane);
                if (wanted == 0) gen_mask &= ~(@as(u64, 1) << @intCast(lane));
            }
        } else {
            var lanes = all_lanes;
            while (lanes != 0) : (lanes &= lanes - 1) {
                const lane = @ctz(lanes);
                p.wanted[lane] = ~@as(u32, 0);
                p.anchors[lane] = p.laneMasks(lane).anchors;
            }
        }
        p.generated.clearRetainingCapacity();
        if (gen_mask != 0) {
            const gen = p.host.gen;
            try gen.genLanes(rack, gen_mask, &p.wanted, &p.anchors, &p.ends, &p.gen_anchors);
            try p.generated.ensureTotalCapacity(p.gpa, gen.plays.items.len);
            for (gen.plays.items) |*mv| p.generated.appendAssumeCapacity(.fromMove(mv));
        }
        const generated = p.generated.items;
        const gen_anchors = p.gen_anchors.items;
        const rack_counts = rackTileCounts(rack, p.numLetters());
        l.solo_valid = false;
        var room = generated.len;
        if (src) |s| room += s.moves.items.len;
        try l.moves.resize(p.gpa, room);
        try l.used.resize(p.gpa, room);
        try l.anchor.resize(p.gpa, room);
        const moves = l.moves.items;
        const used = l.used.items;
        const anchor = l.anchor.items;
        var k: usize = 0;
        var gen_pos: usize = 0;
        var lane: usize = 0;
        while (lane < num_lanes) : (lane += 1) {
            const lane_bit = @as(u64, 1) << @intCast(lane);
            if (all_lanes & lane_bit == 0) {
                l.lane_begin[lane] = @intCast(k);
                continue;
            }
            if (mask & lane_bit == 0 and !rack_changed) {
                const s = src.?;
                var end = lane + 1;
                while (end < num_lanes and mask & (@as(u64, 1) << @intCast(end)) == 0 and all_lanes & (@as(u64, 1) << @intCast(end)) != 0) end += 1;
                const from: usize = @intCast(s.lane_begin[lane]);
                const to: usize = @intCast(s.lane_begin[end]);
                while (lane < end) : (lane += 1) l.lane_begin[lane] = s.lane_begin[lane] + @as(i32, @intCast(k)) - @as(i32, @intCast(from));
                lane -= 1;
                @memcpy(moves[k..][0 .. to - from], s.moves.items[from..to]);
                @memcpy(used[k..][0 .. to - from], s.used.items[from..to]);
                @memcpy(anchor[k..][0 .. to - from], s.anchor.items[from..to]);
                k += to - from;
                continue;
            }
            l.lane_begin[lane] = @intCast(k);
            var gen_end = gen_pos;
            if (gen_mask & lane_bit != 0) gen_end = @intCast(p.ends[lane]);
            if (partial & lane_bit != 0) {
                const s = src.?;
                var from: usize = @intCast(s.lane_begin[lane]);
                const to: usize = @intCast(s.lane_begin[lane + 1]);
                const wanted = p.wanted[lane];
                var as = p.anchors[lane];
                while (as != 0) : (as &= as - 1) {
                    const a: u8 = @intCast(@ctz(as));
                    if (wanted & (@as(u32, 1) << @intCast(a)) != 0) {
                        while (gen_pos < gen_end and gen_anchors[gen_pos] == a) : (gen_pos += 1) {
                            moves[k] = generated[gen_pos];
                            used[k] = playTileCounts(generated[gen_pos]);
                            anchor[k] = a;
                            k += 1;
                        }
                        continue;
                    }
                    while (from < to and s.anchor.items[from] < a) from += 1;
                    const hit = p.hit[lane];
                    while (from < to and s.anchor.items[from] == a) : (from += 1) {
                        // A play over a square a move changed may not stand
                        // (its reach now short of the change, as through a
                        // square no tile of the rack fits any more).
                        if (playSpan(s.moves.items[from]) & hit != 0) continue;
                        if (!rack_changed or holds(&rack_counts, &s.used.items[from])) {
                            moves[k] = s.moves.items[from];
                            used[k] = s.used.items[from];
                            anchor[k] = a;
                            k += 1;
                        }
                    }
                }
                gen_pos = gen_end;
                continue;
            }
            if (mask & lane_bit != 0) {
                for (gen_pos..gen_end) |i| {
                    moves[k] = generated[i];
                    used[k] = playTileCounts(generated[i]);
                    anchor[k] = gen_anchors[i];
                    k += 1;
                }
                gen_pos = gen_end;
                continue;
            }
            const s = src.?;
            const from: usize = @intCast(s.lane_begin[lane]);
            const to: usize = @intCast(s.lane_begin[lane + 1]);
            for (from..to) |i| {
                if (holds(&rack_counts, &s.used.items[i])) {
                    moves[k] = s.moves.items[i];
                    used[k] = s.used.items[i];
                    anchor[k] = s.anchor.items[i];
                    k += 1;
                }
            }
        }
        l.moves.shrinkRetainingCapacity(k);
        l.used.shrinkRetainingCapacity(k);
        l.anchor.shrinkRetainingCapacity(k);
        l.lane_begin[num_lanes] = @intCast(k);
    }

    fn checkAgainstFullGeneration(p: *PathLists, l: *const MoveList, player: u1) Allocator.Error!void {
        const gen = p.host.gen;
        const saved = gen.gen_pass;
        gen.gen_pass = false;
        defer gen.gen_pass = saved;
        try gen.genAll(&p.host.racks[player], false);
        var n = gen.plays.items.len;
        if (n > 0 and gen.plays.items[n - 1].kind == .pass) n -= 1;
        if (n != l.moves.items.len) std.debug.panic("incremental move list has {d} plays, full generation {d}", .{ l.moves.items.len, n });
        for (gen.plays.items[0..n], l.moves.items) |*mv, m| {
            const want: SmallMove = .fromMove(mv);
            if (want.tm != m.tm or want.score != m.score or want.len != m.len or want.tiles_played != m.tiles_played)
                std.debug.panic("incremental move list differs from full generation", .{});
        }
    }
};

/// The squares along its lane that a play covers.
fn playSpan(m: SmallMove) u32 {
    const start: u5 = @intCast(if (m.vertical()) m.row() else m.col());
    return ((@as(u32, 1) << @intCast(m.len)) - 1) << start;
}

/// The squares along a lane that a play from the anchor at `a` can cover
/// (MAGPIE's anchor_span_reach): past the previous anchor `prev` (the
/// generator does not go back onto it); each way no further than the
/// empty squares (of `empties`) the rack's `n_tiles` tiles can fill, less
/// the anchor's own if it is empty; and short of an empty square no tile
/// of the rack fits (`blocked`). An anchor on such a square anchors no
/// play, and reaches only itself.
fn spanReach(empties: u32, blocked: u32, a: usize, prev: ?usize, n_tiles: usize, dim: usize) u32 {
    const bit = @as(u32, 1) << @intCast(a);
    if (blocked & bit != 0) return bit;
    const spare = n_tiles -| @intFromBool(empties & bit != 0);
    // Right: the first empty square past the ones the spare tiles fill, or
    // a blocked one.
    var right_empties = empties & ~((bit << 1) -% 1);
    for (0..spare) |_| right_empties &= right_empties -% 1;
    var right_stop: usize = if (right_empties != 0) @ctz(right_empties) else dim;
    const right_blocked = blocked & ~((bit << 1) -% 1);
    if (right_blocked != 0) right_stop = @min(right_stop, @ctz(right_blocked));
    // Left: likewise, and past the previous anchor.
    var left_empties = empties & (bit - 1);
    for (0..spare) |_| {
        if (left_empties == 0) break;
        left_empties &= ~(@as(u32, 1) << @intCast(31 - @clz(left_empties)));
    }
    var left: usize = if (left_empties != 0) 32 - @clz(left_empties) else 0;
    const left_blocked = blocked & (bit - 1);
    if (left_blocked != 0) left = @max(left, 32 - @clz(left_blocked));
    if (prev) |pa| left = @max(left, pa + 1);
    const right = right_stop - 1;
    return (((@as(u32, 2) << @intCast(right)) -% 1) & ~((@as(u32, 1) << @intCast(left)) - 1)) | bit;
}

/// The tile of the rack that `m`, which places all the others, keeps.
pub fn keptTile(rack: *const Rack, m: SmallMove, num_letters: usize) MachineLetter {
    var sum: usize = 0;
    for (0..num_letters) |ml| sum += ml * rack.counts[ml];
    for (0..m.tiles_played) |i| sum -= m.rackTile(i);
    return @intCast(sum);
}

/// The score of `tile` played alone on the empty square (r, c), if legal
/// (loneTileAt).
pub fn loneTileAt(b: *const Board, tile: MachineLetter, face: i32, r: usize, c: usize) ?i32 {
    const dim = b.dim;
    const has_h = (c > 0 and b.hasLetter(r, c - 1)) or (c + 1 < dim and b.hasLetter(r, c + 1));
    const has_v = (r > 0 and b.hasLetter(r - 1, c)) or (r + 1 < dim and b.hasLetter(r + 1, c));
    if (!has_h and !has_v) return null;
    const idx = r * dim + c;
    const combined = b.cross_sets[0][idx] & b.cross_sets[1][idx];
    if (tile == 0) {
        if (combined >> 1 == 0) return null;
    } else if (combined & (@as(u64, 1) << @intCast(tile)) == 0) return null;
    const ls = face * b.letter_muls[idx];
    const wm: i32 = b.word_muls[idx];
    var score: i32 = 0;
    if (has_h) score += (ls + b.cross_scores[0][idx]) * wm;
    if (has_v) score += (ls + b.cross_scores[1][idx]) * wm;
    return score;
}

test "an anchor's reach covers every span a play from it can cover" {
    var prng: std.Random.DefaultPrng = .init(5);
    const r = prng.random();
    const dim = 15;
    const full = (@as(u32, 1) << dim) - 1;
    for (0..20000) |_| {
        const occupied = r.int(u32) & r.int(u32) & full;
        const empties = ~occupied & full;
        const blocked = empties & r.int(u32) & r.int(u32) & r.int(u32);
        const a = r.uintLessThan(usize, dim);
        const prev: ?usize = if (a > 0 and r.boolean()) r.uintLessThan(usize, a) else null;
        const n_tiles = r.intRangeAtMost(usize, 1, 7);
        const reach = spanReach(empties, blocked, a, prev, n_tiles, dim);
        // Every run [lo, hi] through the anchor, past the previous anchor,
        // over no blocked square, with no more empty squares than tiles.
        for (0..a + 1) |lo| {
            if (prev) |pa| if (lo <= pa) continue;
            for (a..dim) |hi| {
                const span = ((@as(u32, 2) << @intCast(hi)) - 1) & ~((@as(u32, 1) << @intCast(lo)) - 1);
                if (span & blocked != 0) continue;
                if (@popCount(span & empties) > n_tiles) continue;
                try std.testing.expect(span & ~reach == 0);
            }
        }
    }
}

test "a play's span is the squares it covers along its lane" {
    var m: SmallMove = .{ .len = 4 };
    m.tm = (@as(u64, 3) << 6) | (@as(u64, 5) << 1); // row 3, column 5, across
    try std.testing.expectEqual(@as(u32, 0b1111) << 5, playSpan(m));
    m.tm |= 1; // down: its lane is column 5, from row 3
    try std.testing.expectEqual(@as(u32, 0b1111) << 3, playSpan(m));
}
