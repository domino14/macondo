//! The endgame solver: port of endgame/negamax (the single-threaded
//! search: iterative deepening, negamax with negascout, the transposition
//! table and the greedy playout that values the positions the search
//! stops at).
//!
//! It works on a position with the bag empty, the player to move first.
//! Moves are played and unplayed on one board: a play saves the squares it
//! changes and an unplay puts them back, as the Go engine's board journal
//! does. Values are spreads from the side to move's point of view; the
//! solve reports the change in the solving player's spread.
//!
//! Go's endgame mode is kept: two turns in a row without a score end the
//! game (each player losing their rack's value), and the count the
//! position comes with is remapped as Game.SetEndgameMode does. So is a
//! quirk of its undo: taking a move back restores the count of turns
//! without a score but not the count before it, which move ordering reads.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("../data/tiles.zig");
const board_mod = @import("../board/board.zig");
const cross_set = @import("../board/cross_set.zig");
const movegen = @import("../movegen/movegen.zig");
const move_mod = @import("../board/move.zig");
const position = @import("../board/position.zig");
const zobrist_mod = @import("zobrist.zig");
const ttable = @import("ttable.zig");
const abdada_mod = @import("abdada.zig");
const gosort = @import("../gocompat/gosort.zig");
const pathlists = @import("pathlists.zig");
const wordprune = @import("wordprune.zig");
const bestsmall = @import("../movegen/bestsmall.zig");
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const Board = board_mod.Board;
const Move = move_mod.Move;

pub const huge: i16 = 32000;
pub const max_variant = 25;
const max_playout = max_variant;
const aspiration_window = 25;
const stuck_memo_size = 1 << 14;
const StuckEntry = struct { key: u64 = 0, types: u64 = 0 };
/// What a play that goes out adds to its value for the first reply (see
/// negamax): enough to put every such play first.
const first_reply_out_priority = 100000;
const early_pass_bf: i16 = 1 << 13;
const hash_move_bf: i16 = 1 << 12;
const going_out_bf: i16 = 1 << 11;
const conservation_tile_weight = 7;
/// Salts the table key of a leaf valued with a capped playout (see
/// Options.capped_first_depth), so that no deeper search reads its value.
const capped_leaf_salt: u64 = 0x9E3779B97F4A7C15;
/// The moves a node takes best first by selection before it sorts the
/// rest (MAGPIE's LAZY_SELECTION_LIMIT).
const lazy_selection_limit = 8;
const conservation_value_weight = 2;
/// Turns without a score that end the game in endgame mode.
const max_scoreless = 2;

/// A move as the search keeps it (tinymove.SmallMove): `tm` encodes the
/// position and the tiles placed (TinyMove; 0 is a pass).
pub const SmallMove = struct {
    tm: u64 = 0,
    score: i16 = 0,
    est: i16 = 0,
    tiles_played: u8 = 0,
    len: u8 = 0,

    pub inline fn isPass(m: SmallMove) bool {
        return m.tm == 0;
    }
    pub inline fn row(m: SmallMove) usize {
        return @intCast((m.tm & 0b111_1100_0000) >> 6);
    }
    pub inline fn col(m: SmallMove) usize {
        return @intCast((m.tm & 0b11_1110) >> 1);
    }
    pub inline fn vertical(m: SmallMove) bool {
        return m.tm & 1 != 0;
    }
    /// The i-th tile placed, a blank with its letter marked.
    pub inline fn tile(m: SmallMove, i: usize) MachineLetter {
        const ml: MachineLetter = @intCast((m.tm >> @intCast(20 + 6 * i)) & 63);
        return if (m.tm & (@as(u64, 1) << @intCast(12 + i)) != 0) tiles.blank(ml) else ml;
    }
    /// The rack tile the i-th tile placed came from (0 for a blank).
    pub inline fn rackTile(m: SmallMove, i: usize) MachineLetter {
        if (m.tm & (@as(u64, 1) << @intCast(12 + i)) != 0) return 0;
        return @intCast((m.tm >> @intCast(20 + 6 * i)) & 63);
    }

    /// The encoding of a move from the move generator (MoveToTinyMove).
    pub fn fromMove(mv: *const Move) SmallMove {
        if (mv.kind != .play) return .{};
        var code: u64 = 0;
        var t: usize = 0;
        var blanks: u64 = 0;
        for (mv.tileSlice()) |ml| {
            if (ml == 0) continue;
            if (tiles.isBlanked(ml)) blanks |= @as(u64, 1) << @intCast(t);
            code |= @as(u64, tiles.unblank(ml)) << @intCast(20 + 6 * t);
            t += 1;
        }
        if (mv.vertical) code |= 1;
        code |= @as(u64, mv.col) << 1;
        code |= @as(u64, mv.row) << 6;
        code |= blanks << 12;
        return .{ .tm = code, .score = @intCast(mv.score), .tiles_played = mv.tiles_played, .len = mv.len };
    }
};

/// Sorts `moves` by estimate, best first, keeping equal estimates in
/// order (MAGPIE's sort_small_moves_range): insertion sort for short runs,
/// else each half sorted and merged through `scratch`, which holds half.
fn sortByEstimateStable(moves: []SmallMove, scratch: []SmallMove) void {
    const n = moves.len;
    if (n <= 16) {
        if (n < 2) return;
        for (1..n) |i| {
            const m = moves[i];
            var pos = i;
            while (pos > 0 and moves[pos - 1].est < m.est) : (pos -= 1) moves[pos] = moves[pos - 1];
            moves[pos] = m;
        }
        return;
    }
    const half = n / 2;
    sortByEstimateStable(moves[0..half], scratch);
    sortByEstimateStable(moves[half..], scratch);
    if (moves[half - 1].est >= moves[half].est) return;
    @memcpy(scratch[0..half], moves[0..half]);
    var left: usize = 0;
    var right = half;
    var out: usize = 0;
    while (left < half and right < n) : (out += 1) {
        if (scratch[left].est >= moves[right].est) {
            moves[out] = scratch[left];
            left += 1;
        } else {
            moves[out] = moves[right];
            right += 1;
        }
    }
    @memcpy(moves[out..][0 .. half - left], scratch[left..half]);
}

/// MAGPIE's lazy move ordering: puts `list`'s moves in place, best
/// estimate first, through index `through`, the moves before `ordered.*`
/// being in place already. The first lazy_selection_limit slots are each
/// given the first of the best estimates left, swapped with the move it
/// held; the slot after that sorts the rest stably. Returns whether the
/// rest is sorted. Equal estimates keep MAGPIE's order, which the swaps
/// make other than a stable sort's.
fn orderThrough(list: []SmallMove, ordered: *usize, through: usize, scratch: []SmallMove) bool {
    while (ordered.* <= through) {
        const slot = ordered.*;
        ordered.* += 1;
        if (slot >= lazy_selection_limit) {
            sortByEstimateStable(list[slot..], scratch);
            return true;
        }
        var best = slot;
        for (slot + 1..list.len) |i| {
            if (list[i].est > list[best].est) best = i;
        }
        std.mem.swap(SmallMove, &list[slot], &list[best]);
    }
    return false;
}

/// A line of play, best first (PVLine).
pub const Line = struct {
    moves: [max_variant]SmallMove = undefined,
    len: usize = 0,
    /// How many of the leading moves the search found; the rest are the
    /// greedy playout's.
    searched: usize = 0,
    score: i16 = 0,

    fn clear(l: *Line) void {
        l.len = 0;
        l.searched = 0;
    }

    fn update(l: *Line, m: SmallMove, child: *const Line, score: i16) void {
        l.moves[0] = m;
        const n = @min(child.len, max_variant - 1);
        @memcpy(l.moves[1..][0..n], child.moves[0..n]);
        l.len = n + 1;
        l.searched = @min(child.searched + 1, l.len);
        l.score = score;
    }

    fn setPlayout(l: *Line, ms: []const SmallMove) void {
        const n = @min(ms.len, max_variant);
        @memcpy(l.moves[0..n], ms[0..n]);
        l.len = n;
        l.searched = 0;
    }
};

pub const Options = struct {
    /// Value the positions the search stops at by playing them out
    /// greedily (else by their spread).
    heuristics: bool = true,
    iterative_deepening: bool = true,
    /// Search the window (-1, 1) only: whether the solving player wins.
    first_win: bool = false,
    /// The transposition table's size in bytes (rounded down to a power
    /// of two entries, at least 2^24).
    tt_bytes: u64 = 256 << 20,
    /// Value a child that goes out, or that the table answers, without
    /// playing it (speed only).
    child_shortcuts: bool = true,
    /// Skip moves that keep one tile and can't beat alpha (speed only).
    keep_one: bool = true,
    /// Skip moves the opponent can answer by going out for a result that
    /// doesn't beat alpha (speed only).
    out_reply: bool = true,
    /// Go's choice of going-out replies for that bound (its best eight
    /// plays) rather than MAGPIE's (each lane's best, or the best three
    /// drops of a lone tile).
    go_out_reply: bool = false,
    /// Derive move lists from the ancestors' (speed only).
    lists: bool = true,
    /// Compare every derived list with a full generation (tests).
    check_lists: bool = false,
    /// Order moves by MAGPIE's estimate (assignEstimatesHeuristic): the
    /// score less the face value of the tiles played, and when the
    /// opponent has stuck tiles the conservation bonus and build chains.
    /// Go's default is its own ordering, which searches more positions
    /// on typical endgames (a third more at 6 plies on 100 from static
    /// games) though fewer on some deep, wide ones.
    magpie_ordering: bool = true,
    /// Generate moves and cross-sets over a word graph of only the words
    /// this endgame's tiles can make (speed only).
    pruned_kwg: bool = true,
    /// The greedy playout picks plays by score plus, for a play that goes
    /// out, twice the opponent's rack (MAGPIE's pick; Go's is the score
    /// alone). It changes the values of searches that stop short of the
    /// end of the game.
    playout_out_bonus: bool = true,
    /// One ply above the leaves, search the best play by score first and
    /// make the move list only if it does not cut off (MAGPIE's; speed
    /// only, but the search's order changes, and with it the positions
    /// searched).
    first_reply: bool = true,
    /// The first replies and the playouts' first moves take the best plays
    /// of the lanes the last move left alone from the lists, shared by the
    /// last move's siblings, and search only the lanes it changed.
    sibling: bool = true,
    /// With MAGPIE's ordering, take a node's first moves best first as the
    /// search reaches them, and sort the rest (stably) only if it gets
    /// past them, as MAGPIE does (most nodes cut off on their first move),
    /// and sort the root's moves stably; else sort whole lists at once, as
    /// Go does. Equal estimates fall in other orders, so the positions
    /// searched differ; not the value.
    lazy_order: bool = true,
    /// In the first depth's search of a deeper solve, play the leaves out
    /// one move only, and keep their values apart from deeper depths'
    /// (MAGPIE's capped leaves): that depth only orders the root's moves
    /// for the next.
    capped_first_depth: bool = true,
    /// Search each depth after the first in a window around the last
    /// depth's value, widened on failure (MAGPIE's aspiration windows; Go's
    /// single-threaded search has none). The value is the same; the
    /// positions searched differ.
    aspiration: bool = true,
    /// Clear the table (to `tt_bytes`) before solving; off when the
    /// caller has just done so (benchmarks time the solve alone).
    clear_tt: bool = true,
    /// Threads to search with: the solver's own, and threads - 1 helpers
    /// on the solver's `io` (without one the solver searches alone).
    threads: usize = 1,
    /// How the threads share a search.
    parallel: Parallel = .pv_split,
    /// Stop a search past this many positions (0: no limit): the solve
    /// then answers from the last depth it finished.
    node_limit: u64 = 0,
    /// Search the best line on past the solve's depth for the result
    /// (off where only the value is wanted, as for a sim's endgame leaf).
    complete_line: bool = true,
};

/// How several threads share an endgame search.
pub const Parallel = enum {
    /// Every thread searches every depth, sharing the transposition table;
    /// a move another thread is searching is put off (abdada.zig).
    abdada,
    /// PV splitting (Marsland and Campbell, 1982; Go's pvsplit.go): this
    /// solver walks the principal variation; at each of its nodes it
    /// searches the first move alone, then hands the others to all the
    /// threads, a null-window search each at the best bound so far. A
    /// move that beats it becomes the new principal variation. A thread
    /// with no move left joins one still being searched, and they put off
    /// each other's moves as ABDADA's threads do.
    pv_split,
};

/// PV splitting's smallest depth for a split: below it a node's moves are
/// cheaper to search than to hand out.
const pv_split_min_depth = 2;

/// The caches a solver searches with.
pub const Caches = struct {
    best: bestsmall.Cache,
    cross: cross_set.Cache,

    fn create(gpa: Allocator) Allocator.Error!*Caches {
        const c = try gpa.create(Caches);
        errdefer gpa.destroy(c);
        c.best = try .init(gpa, 18, 19);
        errdefer c.best.deinit(gpa);
        c.cross = try .init(gpa, cross_set.Cache.default_index_bits);
        return c;
    }

    fn destroy(c: *Caches, gpa: Allocator) void {
        c.best.deinit(gpa);
        c.cross.deinit(gpa);
        gpa.destroy(c);
    }
};

/// What solves keep between them: each thread's caches, made once rather
/// than for every solve (tens of megabytes to clear a thread). Each solve
/// salts their keys, so that none reads an earlier one's entries.
pub const Workspace = struct {
    gpa: Allocator,
    caches: std.ArrayList(*Caches) = .empty,
    solves: u64 = 0,

    pub fn init(gpa: Allocator) Workspace {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Workspace) void {
        for (self.caches.items) |c| c.destroy(self.gpa);
        self.caches.deinit(self.gpa);
        self.* = undefined;
    }

    /// Thread `index`'s caches, made if new.
    fn get(self: *Workspace, index: usize) Allocator.Error!*Caches {
        while (self.caches.items.len <= index) {
            const c = try Caches.create(self.gpa);
            errdefer c.destroy(self.gpa);
            try self.caches.append(self.gpa, c);
        }
        return self.caches.items[index];
    }

    /// Salts every thread's caches for a new solve.
    fn newSolve(self: *Workspace) void {
        self.solves += 1;
        const salt = (self.solves *% 0x9E3779B97F4A7C15) | 1;
        for (self.caches.items) |c| {
            c.best.salt = salt;
            c.cross.salt = salt;
        }
    }
};

/// One round of null-window searches of a PV node's moves (Parallel.pv_split).
const SplitPoint = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    /// The moves from the root to the node.
    path: []const SmallMove,
    node_key: u64,
    depth: i32,
    children: []const SmallMove,
    /// The children to search this round, by index.
    pending: []const u16,
    /// Indexed like pending.
    moves: []Claim,
    /// Bounds on each child's value (by child index): from below where
    /// fail_high, else from above.
    vals: []i16,
    fail_high: []bool,
    /// The best lower bound on the node's value; a move that fails high
    /// raises it for the moves still being searched.
    alpha: std.atomic.Value(i32),
    beta: i16,
    /// Set when the round is over: a cutoff, or an error.
    over: std.atomic.Value(bool) = .init(false),
    /// The first pending move nobody has taken.
    next: usize = 0,
    impure: bool = false,
    max_reach: u64 = 0,
    err: ?Error = null,

    const Claim = struct {
        /// Set when the move has its result, or the round is over: its
        /// searchers stop.
        stop: std.atomic.Value(bool) = .init(false),
        searchers: u32 = 0,
        done: bool = false,
    };

    /// A move to search: the next nobody has taken, else the unfinished
    /// one with the fewest threads on it; its place in pending.
    fn claim(sp: *SplitPoint) ?usize {
        sp.mutex.lockUncancelable(sp.io);
        defer sp.mutex.unlock(sp.io);
        var n: ?usize = null;
        if (sp.next < sp.pending.len) {
            n = sp.next;
            sp.next += 1;
        } else {
            for (sp.moves, 0..) |*mv, i| {
                if (!mv.done and (n == null or mv.searchers < sp.moves[n.?].searchers)) n = i;
            }
        }
        if (n) |i| sp.moves[i].searchers += 1;
        return n;
    }

    /// Records move n's null-window result `v` at bound `a`, unless
    /// another thread got there first. False if it must be searched
    /// again: it beat `a`, but the bound has risen past it since.
    fn finish(sp: *SplitPoint, n: usize, v: i16, a: i16, impure: bool, reach: u64) bool {
        sp.mutex.lockUncancelable(sp.io);
        defer sp.mutex.unlock(sp.io);
        const mv = &sp.moves[n];
        if (mv.done) return true;
        const fail_high = v > a;
        if (fail_high and v < sp.beta) {
            if (v <= sp.alpha.load(.monotonic)) return false;
            sp.alpha.store(v, .monotonic);
        }
        mv.done = true;
        mv.stop.store(true, .monotonic);
        const idx = sp.pending[n];
        sp.vals[idx] = v;
        sp.fail_high[idx] = fail_high;
        if (impure) sp.impure = true;
        sp.max_reach = @max(sp.max_reach, reach + 1);
        // A cutoff: the node's other moves no longer matter.
        if (v >= sp.beta) sp.end();
        return true;
    }

    /// Takes a thread off a move another thread finished.
    fn leave(sp: *SplitPoint, n: usize) void {
        sp.mutex.lockUncancelable(sp.io);
        defer sp.mutex.unlock(sp.io);
        sp.moves[n].searchers -= 1;
    }

    fn fail(sp: *SplitPoint, err: Error) void {
        sp.mutex.lockUncancelable(sp.io);
        if (sp.err == null) sp.err = err;
        sp.mutex.unlock(sp.io);
        sp.end();
    }

    /// Ends the round for every thread.
    fn end(sp: *SplitPoint) void {
        sp.over.store(true, .monotonic);
        for (sp.moves) |*mv| mv.stop.store(true, .monotonic);
    }
};

const max_lanes = 2 * board_mod.max_dim;

/// The lanes (rows 0..dim, then columns dim..2*dim) whose plays a tile
/// play changes, and in each which squares (pathLists.moveHits): the
/// move's own lane from the square before it to the one after; for each
/// tile placed, its crossing lane around it and the lanes of the squares
/// past the crossing run through it; and the crossing lanes of the squares
/// at the two ends of the word.
pub fn moveHits(b: *const Board, m: SmallMove, lane_hits: *[max_lanes]u32) u64 {
    if (m.isPass()) return 0;
    lane_hits.* = @splat(0);
    const dim = b.dim;
    const v = m.vertical();
    const fixed: usize = if (v) m.col() else m.row();
    const start: usize = if (v) m.row() else m.col();
    const own_base: usize = if (v) dim else 0;
    const cross_base: usize = if (v) 0 else dim;
    const len: usize = m.len;
    const has = struct {
        fn f(bb: *const Board, vert: bool, lane: usize, pos: usize) bool {
            return if (vert) bb.hasLetter(pos, lane) else bb.hasLetter(lane, pos);
        }
    }.f;
    var hits = @as(u64, 1) << @intCast(own_base + fixed);
    const lo: usize = if (start == 0) 0 else start - 1;
    const hi: usize = @min(start + len, dim - 1);
    lane_hits[own_base + fixed] = ((@as(u32, 2) << @intCast(hi)) -% 1) & ~((@as(u32, 1) << @intCast(lo)) - 1);
    for (0..len) |i| {
        const pos = start + i;
        if (has(b, v, fixed, pos)) continue;
        hits |= @as(u64, 1) << @intCast(cross_base + pos);
        lane_hits[cross_base + pos] |= (@as(u32, 7) << @intCast(fixed)) >> 1;
        var above: isize = @as(isize, @intCast(fixed)) - 1;
        while (above >= 0 and has(b, v, @intCast(above), pos)) above -= 1;
        if (above >= 0) {
            hits |= @as(u64, 1) << @intCast(own_base + @as(usize, @intCast(above)));
            lane_hits[own_base + @as(usize, @intCast(above))] |= @as(u32, 1) << @intCast(pos);
        }
        var below = fixed + 1;
        while (below < dim and has(b, v, below, pos)) below += 1;
        if (below < dim) {
            hits |= @as(u64, 1) << @intCast(own_base + below);
            lane_hits[own_base + below] |= @as(u32, 1) << @intCast(pos);
        }
    }
    if (start > 0) {
        hits |= @as(u64, 1) << @intCast(cross_base + start - 1);
        lane_hits[cross_base + start - 1] |= @as(u32, 1) << @intCast(fixed);
    }
    if (start + len < dim) {
        hits |= @as(u64, 1) << @intCast(cross_base + start + len);
        lane_hits[cross_base + start + len] |= @as(u32, 1) << @intCast(fixed);
    }
    return hits;
}

/// The out-reply bound (outreply.go). A move that doesn't go out is worth,
/// to its player, at most what is left if the opponent goes out in reply:
/// its score, less the reply's score and twice the tiles it keeps. Any of
/// the opponent's going-out plays will do that the move leaves standing,
/// which it does when the play covers none of the squares the move changes
/// in the play's lane. A node keeps the opponent's best few.
const OutReply = struct {
    const max_plays = 8;
    ready: bool = false,
    count: usize = 0,
    score: [max_plays]i32 = undefined,
    lane: [max_plays]u8 = undefined,
    span: [max_plays]u32 = undefined,
    single: [max_plays]bool = undefined,
    base: i32 = 0,
    dim: usize = 0,
    /// MAGPIE's choice of replies (else Go's).
    magpie: bool = true,
    lane_hits: [max_lanes]u32 = undefined,

    fn init(o: *OutReply, s: *Solver, on: u1, our_spread: i32) Allocator.Error!void {
        o.ready = true;
        o.count = 0;
        o.base = our_spread - 2 * s.racks[on].scoreOn(s.dist);
        o.dim = s.board.dim;
        o.magpie = !s.opts.go_out_reply;
        const opp = &s.racks[1 - on];
        const opp_tiles = opp.num_tiles;
        if (opp_tiles == 0) return;
        if (o.magpie) return o.initMagpie(s, on, opp_tiles);
        if (opp_tiles == 1) {
            const lone = if (s.lists) |*l| try l.bestLoneTilePlay(s.frames.items.len, 1 - on, s) else s.bestLoneTilePlay(opp);
            if (lone == null) return;
        }
        if (s.lists) |*l| {
            for ((try l.get(s.frames.items.len, 1 - on)).moves.items) |mv| o.add(mv, opp_tiles);
        } else {
            try s.gen.genAll(opp, false);
            for (s.gen.plays.items) |*mv| o.add(.fromMove(mv), opp_tiles);
        }
    }

    /// MAGPIE's replies: with one tile, its three best drops, each on its
    /// own square; with more, the best play of the whole rack in each lane.
    fn initMagpie(o: *OutReply, s: *Solver, on: u1, opp_tiles: u8) Allocator.Error!void {
        const b = &s.board;
        const dim = b.dim;
        if (opp_tiles == 1) {
            const opp = &s.racks[1 - on];
            var tile: MachineLetter = 0;
            for (0..s.numLetters()) |ml| {
                if (opp.counts[ml] > 0) {
                    tile = @intCast(ml);
                    break;
                }
            }
            const face: i32 = if (tile == 0) 0 else s.dist.score_table[tile];
            const Drop = struct { score: i32, r: usize, c: usize };
            var drops: [3]Drop = undefined;
            var n: usize = 0;
            for (0..dim) |r| {
                for (0..dim) |c| {
                    if (b.hasLetter(r, c)) continue;
                    const score = pathlists.loneTileAt(b, tile, face, r, c) orelse continue;
                    // The best three, earlier squares first among equals.
                    var at = n;
                    while (at > 0 and drops[at - 1].score < score) at -= 1;
                    if (at == drops.len) continue;
                    var k = @min(n, drops.len - 1);
                    while (k > at) : (k -= 1) drops[k] = drops[k - 1];
                    drops[at] = .{ .score = score, .r = r, .c = c };
                    n = @min(n + 1, drops.len);
                }
            }
            for (drops[0..n]) |d| o.addRaw(d.score, d.r, @as(u32, 1) << @intCast(d.c), true);
            return;
        }
        var best_score: [max_lanes]i32 = @splat(-1);
        var best_span: [max_lanes]u32 = undefined;
        const Lane = struct {
            fn note(bs: *[max_lanes]i32, sp: *[max_lanes]u32, d: usize, mv: SmallMove, tiles_needed: u8) void {
                if (mv.isPass() or mv.tiles_played != tiles_needed) return;
                const lane: usize = if (mv.vertical()) d + mv.col() else mv.row();
                if (mv.score <= bs[lane]) return;
                const start: usize = if (mv.vertical()) mv.row() else mv.col();
                bs[lane] = mv.score;
                sp[lane] = ((@as(u32, 1) << @intCast(mv.len)) - 1) << @intCast(start);
            }
        };
        if (s.lists) |*l| {
            for ((try l.get(s.frames.items.len, 1 - on)).moves.items) |mv| Lane.note(&best_score, &best_span, dim, mv, opp_tiles);
        } else {
            try s.gen.genAll(&s.racks[1 - on], false);
            for (s.gen.plays.items) |*mv| Lane.note(&best_score, &best_span, dim, .fromMove(mv), opp_tiles);
        }
        for (0..2 * dim) |lane| {
            if (best_score[lane] >= 0) o.addRaw(best_score[lane], lane, best_span[lane], false);
        }
    }

    /// Keeps a reply among the best `max_plays`, best first (ties: the one
    /// kept earlier first).
    fn addRaw(o: *OutReply, score: i32, lane: usize, span: u32, single: bool) void {
        var slot = o.count;
        if (slot == max_plays) {
            if (score <= o.score[slot - 1]) return;
            slot -= 1;
        } else {
            o.count += 1;
        }
        while (slot > 0 and o.score[slot - 1] < score) : (slot -= 1) {
            o.score[slot] = o.score[slot - 1];
            o.lane[slot] = o.lane[slot - 1];
            o.span[slot] = o.span[slot - 1];
            o.single[slot] = o.single[slot - 1];
        }
        o.score[slot] = score;
        o.lane[slot] = @intCast(lane);
        o.span[slot] = span;
        o.single[slot] = single;
    }

    fn add(o: *OutReply, mv: SmallMove, opp_tiles: u8) void {
        {
            if (mv.isPass() or mv.tiles_played != opp_tiles) return;
            const score: i32 = mv.score;
            var slot = o.count;
            if (slot == max_plays) {
                if (score <= o.score[slot - 1]) return;
                slot -= 1;
            } else {
                o.count += 1;
            }
            while (slot > 0 and o.score[slot - 1] < score) : (slot -= 1) {
                o.score[slot] = o.score[slot - 1];
                o.lane[slot] = o.lane[slot - 1];
                o.span[slot] = o.span[slot - 1];
                o.single[slot] = o.single[slot - 1];
            }
            const lane: usize = if (mv.vertical()) o.dim + mv.col() else mv.row();
            const start: usize = if (mv.vertical()) mv.row() else mv.col();
            o.score[slot] = score;
            o.lane[slot] = @intCast(lane);
            o.span[slot] = ((@as(u32, 1) << @intCast(mv.len)) - 1) << @intCast(start);
            o.single[slot] = opp_tiles == 1;
        }
    }

    /// A bound on the value of `m` (which doesn't go out) to the player on
    /// turn, if a going-out reply is known to stand and the bound doesn't
    /// beat alpha.
    fn bound(o: *OutReply, s: *const Solver, m: SmallMove, alpha: i32) ?i32 {
        if (o.count == 0) return null;
        const after_move = o.base + m.score + 2 * s.playedTilesFace(m);
        if (after_move - o.score[0] > alpha) return null;
        const hits = moveHits(&s.board, m, &o.lane_hits);
        for (0..o.count) |i| {
            const value = after_move - o.score[i];
            if (value > alpha) break;
            const lane: usize = o.lane[i];
            const span = o.span[i];
            if (hits & (@as(u64, 1) << @intCast(lane)) != 0 and o.lane_hits[lane] & span != 0) continue;
            if (o.magpie) {
                // A drop on a row's square: its column must stand too.
                if (o.single[i] and lane < o.dim) {
                    const col: usize = @ctz(span);
                    if (hits & (@as(u64, 1) << @intCast(o.dim + col)) != 0 and o.lane_hits[o.dim + col] & (@as(u32, 1) << @intCast(lane)) != 0) continue;
                }
            } else if (o.single[i]) {
                // As Go has it, the square checked across is the last of
                // the reply's word, not necessarily the tile's own (MAGPIE
                // checks the tile's). Either way the check only passes over
                // replies: the lane check catches every change at the
                // tile's square, so the bound stays sound.
                const pos: usize = 31 - @clz(span);
                const cross: usize = if (lane < o.dim) o.dim + pos else pos;
                const cross_pos: usize = if (lane < o.dim) lane else lane - o.dim;
                if (hits & (@as(u64, 1) << @intCast(cross)) != 0 and o.lane_hits[cross] & (@as(u32, 1) << @intCast(cross_pos)) != 0) continue;
            }
            return value;
        }
        return null;
    }
};

/// The keep-one bound (keepone.go). A move that leaves one tile y on the
/// rack is worth, to its player, at most its score plus the most y can
/// score played alone afterwards plus the going-out bonus: the opponent
/// can always pass. Where a node has alpha to beat, a move whose bound
/// doesn't beat it needs no search. After the move y drops where it could
/// before, for the same score, or on a square the move changes, bounded
/// from the tiles the move puts next to it.
const KeepOne = struct {
    rack_sum: usize,
    rack_size: u8,
    base: i32,
    scanned: u64 = 0,
    best_drop: [tiles.max_alphabet_size + 1]i32 = undefined,

    fn init(s: *const Solver, on: u1, our_spread: i32) KeepOne {
        const rack = &s.racks[on];
        var sum: usize = 0;
        for (0..s.numLetters()) |ml| sum += ml * rack.counts[ml];
        return .{ .rack_sum = sum, .rack_size = rack.num_tiles, .base = our_spread + 2 * s.racks[1 - on].scoreOn(s.dist) };
    }

    fn bound(k: *KeepOne, s: *Solver, m: SmallMove, alpha: i32) Allocator.Error!i32 {
        const b = &s.board;
        const ld = s.dist;
        const n = m.tiles_played;
        var kept_sum = k.rack_sum;
        var placed: [movegen.rack_tile_limit]MachineLetter = undefined;
        for (0..@min(n, placed.len)) |i| {
            const ml = m.rackTile(i);
            kept_sum -= ml;
            placed[i] = ml;
        }
        const kept: MachineLetter = @intCast(kept_sum);
        if (k.scanned & (@as(u64, 1) << @intCast(kept)) == 0) {
            if (s.lists) |*lists| {
                const t = (try lists.get(s.frames.items.len, s.on_turn)).soloPlays(&s.board);
                k.best_drop[kept] = if (t.found & (@as(u64, 1) << @intCast(kept)) != 0) t.score[kept] else -1;
            } else {
                k.best_drop[kept] = if (s.scanLoneTile(kept)) |found| found.score else -1;
            }
            k.scanned |= @as(u64, 1) << @intCast(kept);
        }
        const move_bound = k.base + m.score;
        var next_drop = @max(k.best_drop[kept], 0);
        if (move_bound + next_drop > alpha) return move_bound + next_drop;

        const kept_face: i32 = if (kept == 0) 0 else ld.score_table[kept];
        const row = m.row();
        const col = m.col();
        const len: usize = m.len;
        const v = m.vertical();
        const dr: isize = if (v) 1 else 0;
        const dc: isize = if (v) 0 else 1;
        // Go's direction names: a horizontal play's cross words are
        // vertical.
        const main_dir: u1 = if (v) 1 else 0;
        const cross_dir: u1 = 1 - main_dir;
        var word_score: i32 = 0;
        var placed_face: [board_mod.max_dim]i32 = @splat(0);
        var is_placed: [board_mod.max_dim]bool = @splat(false);
        var t: usize = 0;
        for (0..len) |i| {
            const r: usize = @intCast(@as(isize, @intCast(row)) + dr * @as(isize, @intCast(i)));
            const c: usize = @intCast(@as(isize, @intCast(col)) + dc * @as(isize, @intCast(i)));
            const on_board = b.getLetter(r, c);
            if (on_board != 0) {
                word_score += ld.score_table[on_board];
                continue;
            }
            var face: i32 = 0;
            if (t < n and placed[t] != 0) face = ld.score_table[placed[t]];
            t += 1;
            placed_face[i] = face;
            is_placed[i] = true;
            word_score += face;
        }
        const r0: isize = @intCast(row);
        const c0: isize = @intCast(col);
        const ilen: isize = @intCast(len);
        next_drop = @max(next_drop, k.dropBeside(s, r0 - dr, c0 - dc, -dr, -dc, word_score, cross_dir, kept, kept_face), k.dropBeside(s, r0 + dr * ilen, c0 + dc * ilen, dr, dc, word_score, cross_dir, kept, kept_face));
        const dim: isize = @intCast(b.dim);
        for (0..len) |i| {
            if (!is_placed[i]) continue;
            const r = r0 + dr * @as(isize, @intCast(i));
            const c = c0 + dc * @as(isize, @intCast(i));
            const run = placed_face[i] + b.crossScore(@intCast(r), @intCast(c), @fromBackingInt(@intCast(cross_dir)));
            var sign: isize = -1;
            while (sign <= 1) : (sign += 2) {
                const sr = sign * dc;
                const sc = sign * dr;
                var er = r + sr;
                var ec = c + sc;
                while (er >= 0 and er < dim and ec >= 0 and ec < dim and b.hasLetter(@intCast(er), @intCast(ec))) {
                    er += sr;
                    ec += sc;
                }
                next_drop = @max(next_drop, k.dropBeside(s, er, ec, sr, sc, run, main_dir, kept, kept_face));
            }
        }
        return move_bound + next_drop;
    }

    /// Bounds what the kept tile scores alone on the empty square (r, c)
    /// at the end of a run of tiles worth `run_score` that the move makes
    /// or extends; (sr, sc) steps from the run through the square.
    fn dropBeside(k: *const KeepOne, s: *const Solver, r: isize, c: isize, sr: isize, sc: isize, run_score_in: i32, other_dir: u1, kept: MachineLetter, kept_face: i32) i32 {
        _ = k;
        const b = &s.board;
        const dim: isize = @intCast(b.dim);
        if (r < 0 or r >= dim or c < 0 or c >= dim) return 0;
        const other_set = b.crossSet(@intCast(r), @intCast(c), @fromBackingInt(@intCast(other_dir)));
        if (kept == 0) {
            if (other_set >> 1 == 0) return 0;
        } else if (other_set & (@as(u64, 1) << @intCast(kept)) == 0) return 0;
        var run_score = run_score_in;
        var fr = r + sr;
        var fc = c + sc;
        while (fr >= 0 and fr < dim and fc >= 0 and fc < dim and b.hasLetter(@intCast(fr), @intCast(fc))) {
            run_score += s.dist.score_table[b.getLetter(@intCast(fr), @intCast(fc))];
            fr += sr;
            fc += sc;
        }
        const tile = kept_face * b.letterMul(@intCast(r), @intCast(c));
        const wm: i32 = b.wordMul(@intCast(r), @intCast(c));
        var score = (run_score + tile) * wm;
        var orr = sc;
        var oc = sr;
        if (orr < 0 or oc < 0) {
            orr = -orr;
            oc = -oc;
        }
        const has_other = (r - orr >= 0 and c - oc >= 0 and b.hasLetter(@intCast(r - orr), @intCast(c - oc))) or
            (r + orr < dim and c + oc < dim and b.hasLetter(@intCast(r + orr), @intCast(c + oc)));
        if (has_other) score += (b.crossScore(@intCast(r), @intCast(c), @fromBackingInt(@intCast(other_dir))) + tile) * wm;
        return score;
    }
};

/// What a play or pass changed besides the board, to put back.
const Frame = struct {
    racks: [2]Rack,
    scores: [2]i32,
    on_turn: u1,
    scoreless: u32,
    playing: bool,
    tiles_played: u32,
    /// Where this move's record in the board's journal starts.
    saved_start: usize,
    lane_keys: [max_lanes]u64,
    occ: [board_mod.max_dim]u32,
};

pub const Result = struct {
    /// The change in the solving player's spread.
    value: i16,
    line: Line,
    /// The last depth searched to the end.
    depth: u32,
    /// Positions searched, without the moves of the greedy playouts.
    nodes: u64,
};

pub const Error = Allocator.Error || error{ BagNotEmpty, NoOpponentTiles, Stopped };

pub const LoneTile = struct { score: i32, r: usize, c: usize };

/// The best square for `tile` played alone, the first in row order of the
/// best score (scanLoneTile).
/// `occ` is the board's occupancy, a mask a row (bit c: column c holds a
/// tile).
pub fn scanLoneTileOn(b: *const Board, occ: *const [board_mod.max_dim]u32, ld: *const tiles.LetterDistribution, tile: MachineLetter) ?LoneTile {
    const face: i32 = if (tile == 0) 0 else ld.score_table[tile];
    const full = (@as(u32, 1) << @intCast(b.dim)) - 1;
    var found = false;
    var best_score: i32 = 0;
    var best_r: usize = 0;
    var best_c: usize = 0;
    for (0..b.dim) |r| {
        const h_nbr = (occ[r] << 1 | occ[r] >> 1) & full;
        var v_nbr: u32 = 0;
        if (r > 0) v_nbr |= occ[r - 1];
        if (r < b.dim - 1) v_nbr |= occ[r + 1];
        var cand = (h_nbr | v_nbr) & ~occ[r];
        while (cand != 0) {
            const c = @ctz(cand);
            cand &= cand - 1;
            const combined = b.loneTileSet(r, c);
            if (tile == 0) {
                if (combined >> 1 == 0) continue;
            } else if (combined & (@as(u64, 1) << @intCast(tile)) == 0) continue;
            const ls = face * b.letterMul(r, c);
            const wm: i32 = b.wordMul(r, c);
            var score: i32 = 0;
            if (h_nbr & (@as(u32, 1) << @intCast(c)) != 0) score += (ls + b.crossScore(r, c, .horizontal)) * wm;
            if (v_nbr & (@as(u32, 1) << @intCast(c)) != 0) score += (ls + b.crossScore(r, c, .vertical)) * wm;
            if (found and score <= best_score) continue;
            found = true;
            best_score = score;
            best_r = r;
            best_c = c;
        }
    }
    if (!found) return null;
    return .{ .score = best_score, .r = best_r, .c = best_c };
}

pub const Solver = struct {
    gpa: Allocator,
    opts: Options,
    /// The position solved, which helpers are made from.
    pos: *const position.Position,
    /// Where helpers run (Options.threads).
    io: ?std.Io = null,
    /// The helper threads' solvers, during a solve.
    helpers: []*Solver = &.{},
    /// Set when this solver's search of a depth ends, for its helpers.
    helpers_halt: std.atomic.Value(bool) = .init(false),
    /// A helper's: its solver's helpers_halt.
    halt: ?*const std.atomic.Value(bool) = null,
    /// The caches of solves to come (initIn).
    workspace: ?*Workspace = null,
    /// The moves the threads are searching, shared (Parallel.abdada).
    abdada: ?*abdada_mod.Table = null,
    /// The moves the nodes on the path have put off, by index.
    deferred: std.ArrayList(u16) = .empty,
    /// The moves from the root to the PV node this solver is at
    /// (Parallel.pv_split).
    split_path: std.ArrayList(SmallMove) = .empty,
    board: Board,
    dist: *const tiles.LetterDistribution,
    kwg: @import("../data/kwg.zig").Kwg,
    racks: [2]Rack,
    scores: [2]i32,
    on_turn: u1 = 0,
    scoreless: u32 = 0,
    last_scoreless: u32 = 0,
    playing: bool = true,

    gen: movegen.Generator,
    cross_cache: *cross_set.Cache,
    zobrist: *const zobrist_mod.Zobrist,
    tt: *ttable.Table,

    frames: std.ArrayList(Frame) = .empty,
    /// The board's undo record (attached while the solver lives).
    journal: board_mod.Journal = .{},
    /// The moves of the nodes on the search path, a block per node.
    arena: std.ArrayList(SmallMove) = .empty,
    stack_len: usize = 0,

    initial_spread: i32 = 0,
    root_moves: []SmallMove = &.{},
    current_depth: i32 = -1,
    /// The depth the solve searches to.
    target_depth: u32 = 0,
    impure: u64 = 0,
    reach: u64 = 0,
    nodes: u64 = 0,
    playout_nodes: u64 = 0,
    stop: ?*const std.atomic.Value(bool) = null,
    lists: ?pathlists.PathLists = null,
    pruned: ?wordprune.Pruned = null,
    best_cache: *bestsmall.Cache,
    /// Whether the solver made its caches (else a Workspace's).
    own_caches: bool,
    stuck_memo: [stuck_memo_size]StuckEntry = @splat(.{}),
    /// Set while the leaf under a first reply is searched.
    under_first_reply: bool = false,
    /// Each lane's key (bestsmall.laneKey), kept up as moves are played.
    lane_keys: [max_lanes]u64 = undefined,
    /// The board's occupancy, a mask a row, kept up as moves are played.
    occ: [board_mod.max_dim]u32 = @splat(0),
    sort_keys: std.ArrayList(u64) = .empty,
    bc_values: std.ArrayList(i32) = .empty,
    bc_order: std.ArrayList(usize) = .empty,
    sort_moves: std.ArrayList(SmallMove) = .empty,

    /// A solver for `p`'s position, the player on turn solving. `zobrist`
    /// and `tt` are shared with the caller, who sizes the table (Solve
    /// clears it).
    pub fn init(gpa: Allocator, p: *const position.Position, zobrist: *const zobrist_mod.Zobrist, tt: *ttable.Table, opts: Options) Error!*Solver {
        return initIn(gpa, p, zobrist, tt, opts, null, 0);
    }

    /// init with thread `index`'s caches from `ws` (if any): kept across
    /// solves, which salt them (see Workspace).
    pub fn initIn(gpa: Allocator, p: *const position.Position, zobrist: *const zobrist_mod.Zobrist, tt: *ttable.Table, opts: Options, ws: ?*Workspace, index: usize) Error!*Solver {
        const s = try gpa.create(Solver);
        errdefer gpa.destroy(s);
        s.* = .{
            .gpa = gpa,
            .opts = opts,
            .pos = p,
            .board = p.board,
            .dist = p.dist,
            .kwg = p.lexicon.kwg,
            .racks = p.racks,
            .scores = p.scores,
            .gen = undefined,
            .cross_cache = undefined,
            .best_cache = undefined,
            .own_caches = ws == null,
            .workspace = ws,
            .zobrist = zobrist,
            .tt = tt,
        };
        const caches = if (ws) |w| try w.get(index) else try Caches.create(gpa);
        errdefer if (ws == null) caches.destroy(gpa);
        s.cross_cache = &caches.cross;
        s.best_cache = &caches.best;
        s.gen = .init(gpa, &s.board, s.kwg, s.dist);
        s.gen.gen_pass = true;
        s.journal = try .init(gpa, s.board.dim * s.board.dim);
        errdefer s.journal.deinit(gpa);
        s.board.journal = &s.journal;
        if (opts.lists and s.dist.size + 1 <= 48) {
            s.lists = .init(gpa, .{ .board = &s.board, .occ = &s.occ, .racks = &s.racks, .on_turn = &s.on_turn, .gen = &s.gen, .dist = s.dist });
            s.lists.?.check = opts.check_lists;
        }

        // Endgame mode (Game.SetEndgameMode): two scoreless turns end the
        // game, so the count is put relative to that.
        s.scoreless = if (p.scoreless_turns + 1 >= p.max_scoreless_turns) 1 else 0;
        s.last_scoreless = 0;
        return s;
    }

    pub fn deinit(self: *Solver) void {
        const gpa = self.gpa;
        if (self.lists) |*l| l.deinit();
        if (self.pruned) |*pr| pr.deinit(gpa);
        if (self.own_caches) @as(*Caches, @fieldParentPtr("best", self.best_cache)).destroy(gpa);
        self.gen.deinit();
        self.frames.deinit(gpa);
        self.journal.deinit(gpa);
        self.arena.deinit(gpa);
        self.sort_keys.deinit(gpa);
        self.bc_values.deinit(gpa);
        self.bc_order.deinit(gpa);
        self.sort_moves.deinit(gpa);
        self.deferred.deinit(gpa);
        self.split_path.deinit(gpa);

        gpa.destroy(self);
    }

    inline fn spreadFor(self: *const Solver, p: u1) i32 {
        return self.scores[p] - self.scores[1 - p];
    }

    inline fn numLetters(self: *const Solver) usize {
        return self.dist.size;
    }

    // ---- playing and unplaying ----

    /// Plays `m` for the player on turn (Game.PlaySmallMove in endgame
    /// mode). Writes the tiles placed by position along the play into
    /// `placed` (0 where it played through).
    fn play(self: *Solver, m: SmallMove, placed: *[board_mod.max_dim]MachineLetter) Allocator.Error!void {
        if (self.lists) |*l| try l.onMove(self.frames.items.len, m);
        const p = self.on_turn;
        try self.journal.reserve(self.gpa);
        const saved_start = self.journal.mark();
        try self.frames.append(self.gpa, .{
            .racks = self.racks,
            .scores = self.scores,
            .on_turn = p,
            .scoreless = self.scoreless,
            .playing = self.playing,
            .tiles_played = self.board.tiles_played,
            .saved_start = saved_start,
            .lane_keys = self.lane_keys,
            .occ = self.occ,
        });
        if (m.isPass()) {
            self.last_scoreless = self.scoreless;
            self.scoreless += 1;
        } else {
            // Each square the move changes is in the journal as it was:
            // its two lanes' keys trade its old part for its new.
            defer for (self.journal.saved.items[saved_start..]) |*sq| bestsmall.updateKeys(&self.board, &self.lane_keys, sq);
            const b = &self.board;
            var mv: Move = .{ .kind = .play, .vertical = m.vertical(), .row = @intCast(m.row()), .col = @intCast(m.col()), .len = m.len, .tiles_played = m.tiles_played, .score = m.score };
            var t: usize = 0;
            const was_empty = b.tiles_played == 0;
            for (0..m.len) |i| {
                const r = if (mv.vertical) mv.row + i else mv.row;
                const c = if (mv.vertical) mv.col else mv.col + i;
                if (b.hasLetter(r, c)) {
                    placed[i] = 0;
                } else {
                    const ml = m.tile(t);
                    b.setLetter(r, c, ml);
                    self.occ[r] |= @as(u32, 1) << @intCast(c);
                    self.racks[p].take(m.rackTile(t));
                    placed[i] = ml;
                    t += 1;
                }
                mv.tiles[i] = placed[i];
            }
            b.tiles_played += m.tiles_played;
            cross_set.updateForPlay(b, &mv, self.kwg, self.dist, was_empty, self.cross_cache);
            self.last_scoreless = self.scoreless;
            self.scoreless = 0;
            self.scores[p] += m.score;
            if (self.racks[p].num_tiles == 0) {
                self.playing = false;
                self.scores[p] += 2 * self.racks[1 - p].scoreOn(self.dist);
            }
        }
        if (self.scoreless == max_scoreless) {
            self.playing = false;
            self.scores[p] -= self.racks[p].scoreOn(self.dist);
            self.scores[1 - p] -= self.racks[1 - p].scoreOn(self.dist);
        }
        self.on_turn = 1 - p;
    }

    /// Takes the last move back. Like Game.UnplayLastMove, it leaves the
    /// count of scoreless turns before that move as it was.
    fn unplay(self: *Solver) void {
        const f = self.frames.pop().?;
        const b = &self.board;
        self.journal.restore(b, f.saved_start);
        b.tiles_played = f.tiles_played;
        self.racks = f.racks;
        self.scores = f.scores;
        self.lane_keys = f.lane_keys;
        self.occ = f.occ;
        self.on_turn = f.on_turn;
        self.scoreless = f.scoreless;
        self.playing = f.playing;
    }

    fn stackRemaining(self: *const Solver) usize {
        return self.stack_len -| self.frames.items.len;
    }

    // ---- move generation and ordering ----

    /// Every move for the player on turn, a pass last, onto the arena.
    /// The slice lasts until the arena grows: a node keeps its moves' place
    /// in it rather than the slice.
    fn generate(self: *Solver) Allocator.Error![]SmallMove {
        if (self.current_depth >= 0) {
            if (self.lists) |*lists| {
                const l = try lists.get(self.frames.items.len, self.on_turn);
                const start = self.arena.items.len;
                try self.arena.ensureUnusedCapacity(self.gpa, l.moves.items.len + 1);
                self.arena.appendSliceAssumeCapacity(l.moves.items);
                self.arena.appendAssumeCapacity(.{});
                return self.arena.items[start..];
            }
        }
        try self.gen.genAll(&self.racks[self.on_turn], false);
        const start = self.arena.items.len;
        try self.arena.ensureUnusedCapacity(self.gpa, self.gen.plays.items.len);
        for (self.gen.plays.items) |*mv| self.arena.appendAssumeCapacity(.fromMove(mv));
        return self.arena.items[start..];
    }

    /// Values `moves` for the move ordering and sorts them, best first;
    /// with MAGPIE's lazy ordering (see Options.lazy_order) unless `sort`.
    fn assignEstimates(self: *Solver, moves: []SmallMove, depth: i32, tt_move: u64, node_key: u64, sort: bool) Allocator.Error!void {
        if (self.opts.heuristics and self.opts.magpie_ordering) return self.assignEstimatesMagpie(moves, tt_move, node_key, sort or !self.opts.lazy_order);
        const on = self.on_turn;
        const num_tiles = self.racks[on].num_tiles;
        const last_was_pass = self.scoreless > self.last_scoreless;
        const out_bonus = 2 * self.racks[1 - on].scoreOn(self.dist);
        for (moves) |*m| {
            if (m.tiles_played == num_tiles) {
                m.est = @as(i16, @truncate(@as(i32, m.score) + out_bonus)) +% going_out_bf;
            } else if (depth > 2) {
                m.est = @truncate(@as(i32, m.score) - 5 * @as(i32, m.tiles_played));
            } else {
                m.est = m.score;
            }
            if (m.tm == tt_move) m.est +%= hash_move_bf;
            if (last_was_pass and m.isPass()) m.est +%= early_pass_bf;
        }
        try self.sortByEstimate(moves);
    }

    const max_non_going_out_estimate: i32 = going_out_bf - 1;

    /// Sorts `moves` by estimate, best first, equal estimates in order.
    fn sortStable(self: *Solver, moves: []SmallMove) Allocator.Error!void {
        try self.sort_moves.resize(self.gpa, moves.len / 2 + 1);
        sortByEstimateStable(moves, self.sort_moves.items);
    }

    /// MAGPIE's move ordering, as Go's assignEstimatesHeuristic has it.
    fn assignEstimatesMagpie(self: *Solver, moves: []SmallMove, tt_move: u64, node_key: u64, sort: bool) Allocator.Error!void {
        const on = self.on_turn;
        const num_tiles = self.racks[on].num_tiles;
        const last_was_pass = self.scoreless > self.last_scoreless;
        const frac = try self.oppStuckFraction(node_key);
        const build = try self.buildChainValues(moves, frac);
        const other_score = self.racks[1 - on].scoreOn(self.dist);
        for (moves, 0..) |*m, i| {
            const tp = m.tiles_played;
            const partial = !m.isPass() and tp < num_tiles;
            var bonus: i32 = 0;
            if (frac > 0 and partial) bonus = self.conservationBonus(m.*, frac);
            if (tp == num_tiles) {
                m.est = @as(i16, @truncate(@as(i32, m.score) + 2 * other_score)) +% going_out_bf;
            } else {
                var estimate: i32 = m.score;
                if (build) |bv| {
                    if (bv[i] > estimate) estimate += @intFromFloat(frac * @as(f32, @floatFromInt(bv[i] - estimate)));
                    estimate -= bonus;
                } else {
                    estimate -= bonus;
                    if (partial) estimate -= self.playedTilesFace(m.*);
                }
                estimate = std.math.clamp(estimate, -max_non_going_out_estimate, max_non_going_out_estimate);
                m.est = @intCast(estimate);
            }
            if (m.tm == tt_move) m.est +%= hash_move_bf;
            if (last_was_pass and m.isPass()) m.est +%= early_pass_bf;
        }
        if (!sort) return;
        // MAGPIE sorts a list in full (the root's) stably.
        if (self.opts.lazy_order) return self.sortStable(moves);
        gosort.slice(SmallMove, moves, {}, struct {
            fn less(_: void, a: SmallMove, b: SmallMove) bool {
                return a.est > b.est;
            }
        }.less);
    }

    /// Rewards moves that build toward longer plays when the opponent is
    /// stuck (computeBuildChainValues): a play contained in a longer one
    /// of the same tiles is worth its score plus the best containing
    /// play's value. Null unless the opponent has stuck tiles.
    fn buildChainValues(self: *Solver, moves: []const SmallMove, frac: f32) Allocator.Error!?[]i32 {
        if (frac <= 0 or moves.len <= 1) return null;
        const n = moves.len;
        try self.bc_values.resize(self.gpa, n);
        try self.bc_order.resize(self.gpa, n);
        const values = self.bc_values.items;
        const order = self.bc_order.items;
        for (order, 0..) |*o, i| o.* = i;
        // By tiles played, most first (an insertion sort, as Go's).
        var i: usize = 1;
        while (i < n) : (i += 1) {
            const key = order[i];
            const tp = moves[key].tiles_played;
            var j = i;
            while (j > 0 and moves[order[j - 1]].tiles_played < tp) : (j -= 1) order[j] = order[j - 1];
            order[j] = key;
        }
        var tiles_a: [board_mod.max_dim]MachineLetter = undefined;
        var tiles_b: [board_mod.max_dim]MachineLetter = undefined;
        for (0..n) |oi| {
            const ia = order[oi];
            const a = moves[ia];
            values[ia] = a.score;
            if (a.isPass()) continue;
            var a_ready = false;
            var best_ext: i32 = 0;
            for (0..oi) |oj| {
                const jb = order[oj];
                const b = moves[jb];
                if (b.isPass() or b.tiles_played <= a.tiles_played) continue;
                if (b.vertical() != a.vertical()) continue;
                var contained: bool = undefined;
                var offset: usize = 0;
                if (a.vertical()) {
                    contained = a.col() == b.col() and a.row() >= b.row() and a.row() + a.len <= b.row() + b.len;
                    if (contained) offset = a.row() - b.row();
                } else {
                    contained = a.row() == b.row() and a.col() >= b.col() and a.col() + a.len <= b.col() + b.len;
                    if (contained) offset = a.col() - b.col();
                }
                if (!contained) continue;
                if (!a_ready) {
                    self.moveTiles(a, &tiles_a);
                    a_ready = true;
                }
                self.moveTiles(b, &tiles_b);
                var match = true;
                for (0..a.len) |ti| {
                    if (tiles_a[ti] == 0) continue;
                    const bi = ti + offset;
                    if (bi >= b.len or tiles_a[ti] != tiles_b[bi]) {
                        match = false;
                        break;
                    }
                }
                if (!match) continue;
                best_ext = @max(best_ext, values[jb]);
            }
            if (best_ext > 0) values[ia] += best_ext;
        }
        return values;
    }

    /// Best estimate first, ties in the order they came (Go sorts the
    /// estimate above the index): a stable sort, by insertion for a few
    /// moves and otherwise by two counting passes over the estimate's
    /// bytes.
    fn sortByEstimate(self: *Solver, moves: []SmallMove) Allocator.Error!void {
        const n = moves.len;
        if (n < 2) return;
        if (n <= 24) {
            var i: usize = 1;
            while (i < n) : (i += 1) {
                const m = moves[i];
                var j = i;
                while (j > 0 and moves[j - 1].est < m.est) : (j -= 1) moves[j] = moves[j - 1];
                moves[j] = m;
            }
            return;
        }
        try self.sort_moves.resize(self.gpa, n);
        const tmp = self.sort_moves.items;
        // The key: larger estimates first.
        const key = struct {
            inline fn f(m: SmallMove) u16 {
                return @intCast(32767 - @as(i32, m.est));
            }
        }.f;
        var counts: [2][257]u32 = @splat(@splat(0));
        for (moves) |m| {
            const k = key(m);
            counts[0][(k & 0xff) + 1] += 1;
            counts[1][(k >> 8) + 1] += 1;
        }
        for (0..2) |pass| {
            for (1..257) |b| counts[pass][b] += counts[pass][b - 1];
        }
        for (moves) |m| {
            const b = key(m) & 0xff;
            tmp[counts[0][b]] = m;
            counts[0][b] += 1;
        }
        for (tmp) |m| {
            const b = key(m) >> 8;
            moves[counts[1][b]] = m;
            counts[1][b] += 1;
        }
    }

    // ---- the greedy playout ----

    fn playedTilesFace(self: *const Solver, m: SmallMove) i32 {
        var face: i32 = 0;
        for (0..m.tiles_played) |i| face += self.dist.score_table[m.rackTile(i)];
        return face;
    }

    fn conservationBonus(self: *const Solver, m: SmallMove, frac: f32) i32 {
        const x: f32 = @floatFromInt(conservation_tile_weight * @as(i32, m.tiles_played) + conservation_value_weight * self.playedTilesFace(m));
        return @intFromFloat(x * frac);
    }

    fn stuckFractionFromBV(self: *const Solver, rack: *const Rack, bv: u64) f32 {
        var total: i32 = 0;
        var stuck: i32 = 0;
        for (0..self.numLetters()) |ml| {
            const count = rack.counts[ml];
            if (count == 0) continue;
            const s = @as(i32, count) * self.dist.score_table[ml];
            total += s;
            if (bv & (@as(u64, 1) << @intCast(ml)) == 0) stuck += s;
        }
        if (total == 0) return 0;
        return @as(f32, @floatFromInt(stuck)) / @as(f32, @floatFromInt(total));
    }

    fn occupancyFromBoard(self: *const Solver, occ: *[board_mod.max_dim]u32) void {
        const b = &self.board;
        for (0..b.dim) |r| {
            var mask: u32 = 0;
            for (0..b.dim) |c| {
                if (b.squares[r * b.dim + c] != 0) mask |= @as(u32, 1) << @intCast(c);
            }
            occ[r] = mask;
        }
    }

    /// Every letter with a legal play of that tile alone somewhere
    /// (boardPlayableTilesBV); stops once it has all of `target`.
    fn playableTiles(self: *const Solver, target: u64) u64 {
        const b = &self.board;
        if (b.tiles_played == 0) return board_mod.trivial_cross_set;
        const occ = &self.occ;
        const full = (@as(u32, 1) << @intCast(b.dim)) - 1;
        var playable: u64 = 0;
        for (0..b.dim) |r| {
            var cand = occ[r] << 1 | occ[r] >> 1;
            if (r > 0) cand |= occ[r - 1];
            if (r < b.dim - 1) cand |= occ[r + 1];
            cand &= ~occ[r];
            cand &= full;
            while (cand != 0) {
                const c = @ctz(cand);
                cand &= cand - 1;
                playable |= b.loneTileSet(r, c);
                if (target != 0 and playable & target == target) return playable;
            }
        }
        return playable;
    }

    /// The opponent's (the player not solving) share of rack score with no
    /// play anywhere (oppStuckFraction).
    /// oppStuckFraction remembered by position key (`key`; 0: not
    /// remembered), as Go's stuck-fraction memo: a node asks for it for
    /// its first reply, its move ordering and its leaves.
    /// The best play of `player` (on turn) by score plus, for one of their
    /// whole rack, `out_bonus`: bestsmall.best, or with `alone` false, the
    /// lists' best of the lanes the last move left alone and a search of
    /// the lanes it changed.
    fn bestPlay(self: *Solver, player: u1, out_bonus: i32, alone: bool) Allocator.Error!?SmallMove {
        const rack = &self.racks[player];
        if (self.opts.sibling) {
            if (self.lists) |*l| {
                const found = if (alone) try l.aloneBest(self.frames.items.len, player, out_bonus) else try l.siblingBest(self.frames.items.len, player, out_bonus);
                if (found) |sb| {
                    return bestsmall.bestIn(&self.gen, rack, out_bonus, self.best_cache, &self.lane_keys, sb.own, sb.best, sb.value);
                }
            }
        }
        return bestsmall.best(&self.gen, rack, out_bonus, self.best_cache, &self.lane_keys);
    }

    /// Whether leaves are played out one move only (Options.capped_first_depth).
    inline fn cappedLeaves(self: *const Solver) bool {
        return self.opts.capped_first_depth and self.opts.heuristics and self.current_depth <= 1 and self.target_depth > 1;
    }

    /// The table key of a leaf: its position's, salted where its value
    /// comes from a capped playout. 0 (no key) stays 0.
    inline fn leafKey(self: *const Solver, key: u64) u64 {
        return if (key != 0 and self.cappedLeaves()) key ^ capped_leaf_salt else key;
    }

    fn oppStuckFraction(self: *Solver, node_key: u64) Allocator.Error!f32 {
        if (node_key == 0) return self.oppStuckFractionFresh();
        // The fraction depends on the board and the opponent's rack only:
        // the node's key less the solving player's rack, the side to move
        // and the scoreless count (as MAGPIE keys its stuck-tiles cache).
        var key = node_key ^ self.zobrist.scoreless_turns[self.scoreless];
        if (self.on_turn != 0) key ^= self.zobrist.their_turn;
        for (0..self.numLetters()) |ml| key ^= self.zobrist.our_rack[ml][self.racks[0].counts[ml]];
        const slot = &self.stuck_memo[key & (stuck_memo_size - 1)];
        const types = if (slot.key == key) slot.types else fresh: {
            const t = try self.oppStuckTypesFresh();
            slot.* = .{ .key = key, .types = t };
            break :fresh t;
        };
        if (self.lists) |*lists| lists.noteStuckTypes(self.frames.items.len, types);
        return self.stuckFractionFromBV(&self.racks[1], types);
    }

    fn oppStuckFractionFresh(self: *Solver) Allocator.Error!f32 {
        return self.stuckFractionFromBV(&self.racks[1], try self.oppStuckTypesFresh());
    }

    /// The opponent's tile types that some play of theirs places (each
    /// type of their rack has a bit, the blank bit 0).
    fn oppStuckTypesFresh(self: *Solver) Allocator.Error!u64 {
        const opp = &self.racks[1];
        if (opp.num_tiles == 0) return 0;
        var rack_bv: u64 = 0;
        for (0..self.numLetters()) |ml| {
            if (opp.counts[ml] > 0) rack_bv |= @as(u64, 1) << @intCast(ml);
        }
        const n = self.frames.items.len;
        if (opp.num_tiles == 1) {
            if (self.lists) |*lists| {
                // One tile: stuck exactly when it has no play alone.
                if (try lists.bestLoneTilePlay(n, 1, self) == null) return 0;
                return rack_bv;
            }
        }
        const playable = self.playableTiles(rack_bv & ~@as(u64, 1));
        var tiles_bv = playable & rack_bv;
        if (rack_bv & 1 != 0 and playable >> 1 != 0) tiles_bv |= 1;
        if (tiles_bv == rack_bv) return tiles_bv;
        if (opp.num_tiles == 1) return tiles_bv;
        if (self.lists) |*lists| {
            const scanned = tiles_bv;
            const undecided = rack_bv & ~tiles_bv;
            // As MAGPIE does. The scan settles the plays of one tile, so an
            // undecided type is in a play of two or more, along its own
            // lane: if no play placed it before the last move (with the
            // rack they had, which held this one), a play that does now
            // covers a square that move changed.
            if (lists.stuckTypesBefore(n)) |before| {
                if (undecided & before == 0) {
                    const lanes = lists.lastMoveHits(n);
                    if (lanes != 0) tiles_bv |= try lists.tileTypesGenerated(1, lanes, undecided);
                    if (self.opts.check_lists) try self.checkStuckTypes(rack_bv, scanned, tiles_bv);
                    return tiles_bv;
                }
            }
            // Otherwise the types their plays that stand from their nearest
            // list place, and only if some are still wanted, a generation
            // of the plays the moves since may have changed.
            if (lists.standingTileTypes(n, 1, undecided)) |st| {
                tiles_bv |= st.found;
                if (tiles_bv != rack_bv and st.touched != 0) tiles_bv |= try lists.tileTypesGenerated(1, st.touched, rack_bv & ~tiles_bv);
                if (self.opts.check_lists) try self.checkStuckTypes(rack_bv, scanned, tiles_bv);
                return tiles_bv;
            }
            const l = try lists.get(n, 1);
            for (l.moves.items) |m| {
                for (0..m.tiles_played) |i| tiles_bv |= @as(u64, 1) << @intCast(m.rackTile(i));
                if (tiles_bv & rack_bv == rack_bv) break;
            }
            return tiles_bv & rack_bv;
        }
        try self.gen.genAll(opp, false);
        for (self.gen.plays.items) |*mv| {
            if (mv.kind != .play) continue;
            for (mv.tileSlice()) |ml| {
                if (ml == 0) continue;
                tiles_bv |= @as(u64, 1) << @intCast(if (tiles.isBlanked(ml)) 0 else ml);
            }
            if (tiles_bv & rack_bv == rack_bv) break;
        }
        return tiles_bv & rack_bv;
    }

    /// Checks the opponent's tile types against their full list (tests).
    fn checkStuckTypes(self: *Solver, rack_bv: u64, scanned: u64, types: u64) Allocator.Error!void {
        const l = try self.lists.?.get(self.frames.items.len, 1);
        var all = scanned;
        for (l.moves.items) |m| {
            for (0..m.tiles_played) |i| all |= @as(u64, 1) << @intCast(m.rackTile(i));
        }
        if (all & rack_bv != types) std.debug.panic("opponent's tile types {b}, their list's {b}", .{ types, all & rack_bv });
    }

    /// The best play of a one-tile rack, the first square in row order of
    /// the best score (bestSingleTilePlay), or null.
    fn bestLoneTilePlay(self: *const Solver, rack: *const Rack) ?SmallMove {
        var tile: MachineLetter = 0;
        for (0..self.numLetters()) |ml| {
            if (rack.counts[ml] > 0) {
                tile = @intCast(ml);
                break;
            }
        }
        const found = self.scanLoneTile(tile) orelse return null;
        return self.loneTileMove(tile, found.r, found.c, found.score);
    }

    fn scanLoneTile(self: *const Solver, tile: MachineLetter) ?LoneTile {
        return scanLoneTileOn(&self.board, &self.occ, self.dist, tile);
    }

    pub fn loneTileMove(self: *const Solver, tile: MachineLetter, r: usize, c: usize, score: i32) SmallMove {
        const b = &self.board;
        const has_h = (c > 0 and b.hasLetter(r, c - 1)) or (c + 1 < b.dim and b.hasLetter(r, c + 1));
        const vert = !has_h;
        var start: usize = undefined;
        var end: usize = undefined;
        if (!vert) {
            start = c;
            end = c;
            while (start > 0 and b.hasLetter(r, start - 1)) start -= 1;
            while (end + 1 < b.dim and b.hasLetter(r, end + 1)) end += 1;
        } else {
            start = r;
            end = r;
            while (start > 0 and b.hasLetter(start - 1, c)) start -= 1;
            while (end + 1 < b.dim and b.hasLetter(end + 1, c)) end += 1;
        }
        var chosen = tile;
        if (tile == 0) {
            const combined = b.loneTileSet(r, c);
            var ml: usize = 1;
            while (ml < self.numLetters()) : (ml += 1) {
                if (combined & (@as(u64, 1) << @intCast(ml)) != 0) {
                    chosen = @intCast(ml);
                    break;
                }
            }
        }
        var code: u64 = @as(u64, chosen) << 20;
        if (tile == 0) code |= 1 << 12;
        var mr = r;
        var mc = start;
        if (vert) {
            code |= 1;
            mr = start;
            mc = c;
        }
        code |= @as(u64, mc) << 1;
        code |= @as(u64, mr) << 6;
        return .{ .tm = code, .score = @intCast(score), .tiles_played = 1, .len = @intCast(end - start + 1) };
    }

    /// Checks a best play against the best of the full move list (tests).
    fn checkBestPlay(self: *Solver, found: ?SmallMove, rack: *Rack, out_bonus: i32) Allocator.Error!void {
        for (0..2 * self.board.dim) |lane| {
            if (self.lane_keys[lane] != bestsmall.laneKey(&self.board, lane)) std.debug.panic("lane {d}'s key is stale", .{lane});
        }
        try self.gen.genAll(rack, false);
        var want: ?SmallMove = null;
        var want_v: i32 = 0;
        for (self.gen.plays.items) |*mv| {
            if (mv.kind != .play) continue;
            const m: SmallMove = .fromMove(mv);
            const v: i32 = m.score + (if (m.tiles_played == rack.num_tiles) out_bonus else 0);
            if (want == null or v > want_v or (v == want_v and m.tm > want.?.tm)) {
                want = m;
                want_v = v;
            }
        }
        if ((want == null) != (found == null) or (want != null and want.?.tm != found.?.tm))
            std.debug.panic("best play search differs from the best of the full list", .{});
    }

    /// Values a leaf by playing the game out greedily (greedyPlayout): the
    /// spread for the player on turn there.
    fn greedyPlayout(self: *Solver, pv: *Line, node_key: u64) Allocator.Error!i16 {
        const leaf_player = self.on_turn;
        const budget = @min(if (self.cappedLeaves()) @as(usize, 1) else max_playout, self.stackRemaining());
        if (budget == 0) {
            pv.clear();
            return @intCast(self.spreadFor(leaf_player));
        }
        var moves: [max_playout]SmallMove = undefined;
        const frac = try self.oppStuckFraction(node_key);
        var rack_scores = [2]i32{ self.racks[0].scoreOn(self.dist), self.racks[1].scoreOn(self.dist) };
        var played: usize = 0;
        var on_board: usize = 0;
        var went_out = false;
        var out_gain: i32 = 0;
        var placed: [board_mod.max_dim]MachineLetter = undefined;
        if (self.lists) |*l| l.in_playout = true;
        defer if (self.lists) |*l| {
            l.in_playout = false;
        };
        while (played < budget and self.playing) {
            const on = self.on_turn;
            const rack = &self.racks[on];
            var buf: [2]SmallMove = undefined;
            var plays: []SmallMove = undefined;
            var picked_alone = false;
            const arena_mark = self.arena.items.len;
            if (rack.num_tiles == 1) {
                var n: usize = 0;
                const lone = if (self.lists) |*l| try l.bestLoneTilePlay(self.frames.items.len, on, self) else self.bestLoneTilePlay(rack);
                if (lone) |m| {
                    buf[n] = m;
                    n += 1;
                }
                // MAGPIE's pick goes out with the last tile wherever it
                // can, conserving or not; Go's weighs the pass.
                if (n == 0 or !self.opts.playout_out_bonus) {
                    buf[n] = .{};
                    n += 1;
                }
                plays = buf[0..n];
            } else if (self.opts.playout_out_bonus and !(frac > 0 and on == 0)) {
                // MAGPIE's pick wants only the best play: a best-first
                // search for it, with MAGPIE's caches of lane bounds and
                // anchors' best plays.
                var n: usize = 0;
                const found = if (played == 0)
                    try self.bestPlay(on, 2 * rack_scores[1 - on], self.under_first_reply)
                else
                    try bestsmall.best(&self.gen, rack, 2 * rack_scores[1 - on], self.best_cache, &self.lane_keys);
                if (self.opts.check_lists) try self.checkBestPlay(found, rack, 2 * rack_scores[1 - on]);
                if (found) |m| {
                    buf[n] = m;
                    n += 1;
                }
                buf[n] = .{};
                n += 1;
                plays = buf[0..n];
                picked_alone = true;
            } else {
                plays = try self.generate();
            }
            const conserve = frac > 0 and on == 0;
            var pass_penalty: i32 = 0;
            if (conserve) pass_penalty = @intFromFloat(@as(f32, @floatFromInt(rack_scores[0] + rack_scores[1])) * frac);
            var best_idx: usize = 0;
            var best_adj: i32 = std.math.minInt(i32);
            // MAGPIE's pick counts what going out earns (Go's, the score
            // alone); not under conservation, which has its own valuation.
            const out_bonus: i32 = if (self.opts.playout_out_bonus and !conserve) 2 * rack_scores[1 - on] else 0;
            for (plays, 0..) |m, idx| {
                var adj: i32 = m.score;
                if (out_bonus > 0 and !m.isPass() and m.tiles_played == rack.num_tiles) adj += out_bonus;
                if (conserve) {
                    if (m.isPass()) adj -= pass_penalty else adj -= self.conservationBonus(m, frac);
                }
                if (adj > best_adj) {
                    best_adj = adj;
                    best_idx = idx;
                }
            }
            const m = plays[best_idx];
            self.arena.shrinkRetainingCapacity(arena_mark);
            moves[played] = m;
            if (!m.isPass() and m.tiles_played == rack.num_tiles) {
                out_gain = @as(i32, m.score) + 2 * rack_scores[1 - on];
                if (on != leaf_player) out_gain = -out_gain;
                went_out = true;
                played += 1;
                break;
            }
            if (!picked_alone and rack.num_tiles >= 2 and m.tiles_played == rack.num_tiles - 1) {
                if (self.lists) |*l| try l.noteLonePlay(self.frames.items.len, on, pathlists.keptTile(rack, m, self.numLetters()));
            }
            if (!m.isPass()) rack_scores[on] -= self.playedTilesFace(m);
            try self.play(m, &placed);
            played += 1;
            on_board += 1;
        }
        pv.setPlayout(moves[0..played]);
        var spread = self.spreadFor(leaf_player) + out_gain;
        if (!went_out and self.playing) {
            spread -= 2 * self.racks[leaf_player].scoreOn(self.dist);
            spread += 2 * self.racks[1 - leaf_player].scoreOn(self.dist);
        }
        for (0..on_board) |_| self.unplay();
        self.playout_nodes += played;
        self.nodes += played;
        return @intCast(spread);
    }

    // ---- the search ----

    /// The tiles `m` puts down by position along it, 0 where it plays
    /// through (smallMoveTiles), without playing it.
    fn moveTiles(self: *const Solver, m: SmallMove, out: *[board_mod.max_dim]MachineLetter) void {
        var t: usize = 0;
        for (0..m.len) |i| {
            const r = if (m.vertical()) m.row() + i else m.row();
            const c = if (m.vertical()) m.col() else m.col() + i;
            if (self.board.hasLetter(r, c)) {
                out[i] = 0;
            } else {
                out[i] = m.tile(t);
                t += 1;
            }
        }
    }

    /// Values a child as searching it would, without playing it
    /// (valueChildUnplayed): a play of the whole rack ends the game, and a
    /// child whose node would answer from the table gets that answer.
    /// The value is from the child's side.
    fn valueChildUnplayed(self: *Solver, m: SmallMove, node_key: u64, depth: i32, alpha: i16, beta: i16, first_or_full: bool, pv_node: bool, our_spread: i32, on: u1, stm_tiles: u8, out_bonus: i32) ?i16 {
        if (!self.opts.child_shortcuts or m.isPass()) return null;
        if (m.tiles_played == stm_tiles) {
            self.reach = 0;
            return @intCast(-(our_spread + m.score + out_bonus));
        }
        if (node_key == 0) return null;
        var placed: [board_mod.max_dim]MachineLetter = undefined;
        self.moveTiles(m, &placed);
        const child_key = self.zobrist.addTilePlayUnplayed(node_key, m.row(), m.col(), m.vertical(), placed[0..m.len], &self.racks[on], on == 0, self.scoreless);
        const child_depth: u8 = @intCast(depth - 1);
        const e = self.tt.lookup(if (child_depth == 0) self.leafKey(child_key) else child_key);
        const ed = e.depth();
        if (!e.valid() or !(ed == child_depth or (e.pure() and ed <= child_depth))) return null;
        const score = e.score() -% @as(i16, @intCast(our_spread + m.score));
        var ca = -beta;
        var cb = -alpha;
        var cpv = pv_node;
        if (!first_or_full) {
            ca = -alpha - 1;
            cb = -alpha;
            cpv = false;
        }
        switch (e.flag()) {
            ttable.exact => if (cpv) return null,
            ttable.lower => if (@max(ca, score) < cb) return null,
            ttable.upper => if (ca < @min(cb, score)) return null,
            else => {},
        }
        if (!first_or_full and alpha < -score and -score < beta) return null;
        if (e.pure()) {
            self.reach = ed;
        } else {
            self.impure += 1;
            self.reach = 0;
        }
        return score;
    }

    fn negamax(self: *Solver, node_key: u64, depth: i32, alpha_in: i16, beta_in: i16, pv: *Line, pv_node: bool) Error!i16 {
        if (self.stop) |st| {
            if (st.load(.monotonic)) return error.Stopped;
        }
        if (self.halt) |h| {
            if (h.load(.monotonic)) return error.Stopped;
        }
        if (self.opts.node_limit != 0 and self.nodes >= self.opts.node_limit) return error.Stopped;
        var alpha = alpha_in;
        var beta = beta_in;
        const on = self.on_turn;
        const our_spread = self.spreadFor(on);
        const alpha_orig = alpha;
        var tt_move: u64 = ttable.invalid_move;
        const impure_at_entry = self.impure;
        var max_reach: u64 = 0;
        {
            const e = self.tt.lookup(if (depth == 0) self.leafKey(node_key) else node_key);
            const ed = e.depth();
            const d: u8 = @intCast(depth);
            const usable = e.valid() and (ed == d or (e.pure() and ed <= d));
            if (!usable and e.valid() and ed > d) {
                tt_move = e.play();
            } else if (usable) {
                if (e.pure()) max_reach = @max(max_reach, ed) else self.impure += 1;
                const score = e.score() +% @as(i16, @intCast(our_spread));
                const flag = e.flag();
                if (!pv_node) {
                    if (flag == ttable.exact) {
                        self.reach = max_reach;
                        return score;
                    }
                    if (flag == ttable.lower) alpha = @max(alpha, score) else if (flag == ttable.upper) beta = @min(beta, score);
                    if (alpha >= beta) {
                        self.reach = max_reach;
                        return score;
                    }
                }
                tt_move = e.play();
            }
        }

        if (depth == 0 or !self.playing) {
            if (self.playing) self.impure += 1;
            if (self.opts.heuristics and self.playing) {
                const val = try self.greedyPlayout(pv, node_key);
                if (node_key != 0) {
                    const leaf_key = self.leafKey(node_key);
                    const existing = self.tt.lookup(leaf_key);
                    if (!existing.valid() or existing.depth() == 0) {
                        self.tt.store(leaf_key, val -% @as(i16, @intCast(our_spread)), ttable.exact << 6, ttable.invalid_move, false);
                    }
                }
                self.reach = 0;
                return val;
            }
            pv.clear();
            self.reach = 0;
            return @intCast(self.spreadFor(self.on_turn));
        }

        var child_pv: Line = .{};
        const arena_mark = self.arena.items.len;
        defer self.arena.shrinkRetainingCapacity(arena_mark);
        const at_root = self.current_depth == depth;
        var best_value: i16 = -huge;
        var best_move: SmallMove = .{};
        var placed: [board_mod.max_dim]MachineLetter = undefined;

        // One ply above the leaves a node nearly always cuts off on its
        // first move, yet making its list costs far more than searching
        // that move. So, as MAGPIE does, its best play by score (going out
        // first) is searched first, and the list made only if it does not
        // cut off. Not where the opponent has stuck tiles (the ordering
        // then weighs conservation), nor after a pass.
        var first_reply: ?SmallMove = null;
        if (self.opts.first_reply and self.opts.heuristics and depth == 1 and !at_root and tt_move == ttable.invalid_move and
            self.scoreless == 0 and self.racks[on].num_tiles >= 2 and try self.oppStuckFraction(node_key) == 0)
        {
            const found = try self.bestPlay(on, first_reply_out_priority, false);
            if (self.opts.check_lists) try self.checkBestPlay(found, &self.racks[on], first_reply_out_priority);
            if (found) |fr| {
                try self.play(fr, &placed);
                self.nodes += 1;
                const child_key = self.zobrist.addMove(node_key, false, fr.row(), fr.col(), fr.vertical(), placed[0..fr.len], &self.racks[on], self.numLetters(), on == 0, self.scoreless, self.last_scoreless);
                // Its leaf has no siblings to share lane bests with.
                self.under_first_reply = true;
                const value = self.negamax(child_key, depth - 1, -beta, -alpha, &child_pv, pv_node) catch |err| {
                    self.under_first_reply = false;
                    self.unplay();
                    return err;
                };
                self.under_first_reply = false;
                self.unplay();
                max_reach = @max(max_reach, self.reach + 1);
                first_reply = fr;
                best_value = -value;
                best_move = fr;
                pv.update(fr, &child_pv, best_value -% @as(i16, @intCast(self.initial_spread)));
                if (best_value >= beta) return self.storeNode(node_key, depth, best_value, alpha_orig, beta, our_spread, best_move, impure_at_entry, max_reach);
                alpha = @max(alpha, best_value);
                child_pv.clear();
            }
        }
        // Whether the first move of the list gets the full window.
        const first_full = first_reply == null;

        const num_children = if (at_root) self.root_moves.len else blk: {
            const children = try self.generate();
            try self.assignEstimates(children, depth, tt_move, node_key, false);
            break :blk children.len;
        };
        const stm_tiles = self.racks[on].num_tiles;
        const out_bonus = 2 * self.racks[1 - on].scoreOn(self.dist);
        const keep_one_active = self.opts.keep_one and depth >= 2 and !at_root and stm_tiles >= 2;
        var keep_one: KeepOne = if (keep_one_active) .init(self, on, our_spread) else undefined;
        // Go's only with the move lists (it reads them); MAGPIE's also one
        // ply above the leaves when the opponent holds one tile, which the
        // leaf's playout then plays out with its best drop. The opponent's
        // playout never conserves tiles; the solving player's goes out with
        // its last tile only with MAGPIE's picks (see greedyPlayout).
        const out_reply_active = self.opts.out_reply and !at_root and (if (self.opts.go_out_reply)
            self.lists != null and depth >= 2
        else
            depth >= 2 or (depth == 1 and self.opts.heuristics and self.racks[1 - on].num_tiles == 1 and
                (on == 0 or self.opts.playout_out_bonus)));
        var leaves_played: usize = 0;
        var out_reply: OutReply = .{};
        // ABDADA (threads > 1): moves another thread is searching are put
        // off to a second pass.
        const abdada_here = self.abdada != null and depth >= abdada_mod.min_depth;
        const deferred_mark = self.deferred.items.len;
        defer self.deferred.shrinkRetainingCapacity(deferred_mark);
        // MAGPIE's lazy ordering: the moves before `ordered` are in place.
        var tail_sorted = at_root or !(self.opts.heuristics and self.opts.magpie_ordering and self.opts.lazy_order);
        var ordered: usize = 0;
        for (0..num_children) |idx| {
            // (Deeper nodes may move the arena.)
            if (!tail_sorted) {
                if (ordered == lazy_selection_limit) try self.sort_moves.resize(self.gpa, num_children / 2 + 1);
                tail_sorted = orderThrough(self.arena.items[arena_mark..][0..num_children], &ordered, idx, self.sort_moves.items);
            }
            // (Deeper nodes may move the arena.)
            const m = if (at_root) self.root_moves[idx] else self.arena.items[arena_mark + idx];
            if (first_reply) |fr| if (m.tm == fr.tm) continue;
            if (out_reply_active and (idx > 0 or !first_full)) {
                if (!out_reply.ready) try out_reply.init(self, on, our_spread);
                if (m.tiles_played != stm_tiles and (!m.isPass() or self.scoreless == 0)) {
                    if (out_reply.bound(self, m, alpha)) |bound| {
                        if (@as(i16, @truncate(bound)) > best_value) {
                            best_value = @truncate(bound);
                            best_move = m;
                        }
                        self.nodes += 1;
                        continue;
                    }
                }
            }
            if (keep_one_active and !m.isPass() and m.tiles_played == keep_one.rack_size - 1) {
                const bound = try keep_one.bound(self, m, alpha);
                if (bound <= alpha) {
                    if (@as(i16, @truncate(bound)) > best_value) {
                        best_value = @truncate(bound);
                        best_move = m;
                    }
                    self.nodes += 1;
                    continue;
                }
            }
            var value: i16 = undefined;
            if (self.valueChildUnplayed(m, node_key, depth, alpha, beta, idx == 0 and first_full, pv_node, our_spread, on, stm_tiles, out_bonus)) |v| {
                value = v;
                self.nodes += 1;
                child_pv.clear();
                max_reach = @max(max_reach, self.reach + 1);
                if (-value > best_value) {
                    best_value = -value;
                    best_move = m;
                    pv.update(best_move, &child_pv, best_value -% @as(i16, @intCast(self.initial_spread)));
                }
                if (at_root) self.root_moves[idx].est = -value;
                alpha = @max(alpha, best_value);
                if (best_value >= beta) break;
                child_pv.clear();
                continue;
            }
            if (abdada_here) {
                // Another thread is searching it: after the others, when
                // its value may be in the table.
                if (self.abdada.?.busy(node_key, m.tm, @intCast(depth))) {
                    try self.deferred.append(self.gpa, @intCast(idx));
                    continue;
                }
            }
            if (depth == 1 and self.opts.heuristics) {
                // The children are leaves, each played out starting with
                // the other player's best play: once a second needs it,
                // build that player's list here for the others to derive
                // theirs from.
                leaves_played += 1;
                if (leaves_played == 2 and self.racks[1 - on].num_tiles > 1) {
                    if (self.lists) |*l| _ = try l.get(self.frames.items.len, 1 - on);
                }
            }
            value = try self.searchChild(m, node_key, depth, alpha, beta, idx == 0 and first_full, pv_node, &child_pv, abdada_here);
            max_reach = @max(max_reach, self.reach + 1);
            if (-value > best_value) {
                best_value = -value;
                best_move = m;
                pv.update(best_move, &child_pv, best_value -% @as(i16, @intCast(self.initial_spread)));
            }
            if (at_root) self.root_moves[idx].est = -value;
            alpha = @max(alpha, best_value);
            if (best_value >= beta) break;
            child_pv.clear();
        }
        // The moves put off, now searched (ABDADA's second pass).
        var d = deferred_mark;
        while (d < self.deferred.items.len and best_value < beta) : (d += 1) {
            const idx = self.deferred.items[d];
            const m = if (at_root) self.root_moves[idx] else self.arena.items[arena_mark + idx];
            child_pv.clear();
            const value = try self.searchChild(m, node_key, depth, alpha, beta, idx == 0 and first_full, pv_node, &child_pv, true);
            max_reach = @max(max_reach, self.reach + 1);
            if (-value > best_value) {
                best_value = -value;
                best_move = m;
                pv.update(best_move, &child_pv, best_value -% @as(i16, @intCast(self.initial_spread)));
            }
            if (at_root) self.root_moves[idx].est = -value;
            alpha = @max(alpha, best_value);
        }

        return self.storeNode(node_key, depth, best_value, alpha_orig, beta, our_spread, best_move, impure_at_entry, max_reach);
    }

    /// Plays child `m` of the node `node_key` (the player on turn's move),
    /// searches it, and takes it back; returns its value, to its own
    /// player. `full` searches the window (alpha, beta) at once, else a
    /// null window at alpha first, again in full only if it falls inside;
    /// `shared` notes the search in the threads' table (ABDADA).
    fn searchChild(self: *Solver, m: SmallMove, node_key: u64, depth: i32, alpha: i16, beta: i16, full: bool, pv_node: bool, child_pv: *Line, shared: bool) Error!i16 {
        const on = self.on_turn;
        var placed: [board_mod.max_dim]MachineLetter = undefined;
        if (shared) self.abdada.?.start(node_key, m.tm, @intCast(depth));
        defer if (shared) self.abdada.?.finish(node_key, m.tm);
        try self.play(m, &placed);
        defer self.unplay();
        self.nodes += 1;
        const leave = &self.racks[on];
        const child_key = self.zobrist.addMove(node_key, m.isPass(), m.row(), m.col(), m.vertical(), placed[0..m.len], leave, self.numLetters(), on == 0, self.scoreless, self.last_scoreless);
        if (full) return self.negamax(child_key, depth - 1, -beta, -alpha, child_pv, pv_node);
        const value = try self.negamax(child_key, depth - 1, -alpha - 1, -alpha, child_pv, false);
        if (alpha < -value and -value < beta) return self.negamax(child_key, depth - 1, -beta, -alpha, child_pv, pv_node);
        return value;
    }

    /// Stores a node's result in the table and returns its value.
    fn storeNode(self: *Solver, node_key: u64, depth: i32, best_value: i16, alpha_orig: i16, beta: i16, our_spread: i32, best_move: SmallMove, impure_at_entry: u64, max_reach: u64) i16 {
        const score = best_value -% @as(i16, @intCast(our_spread));
        const flag: u8 = if (best_value <= alpha_orig) ttable.upper else if (best_value >= beta) ttable.lower else ttable.exact;
        const pure = self.impure == impure_at_entry;
        const stored_depth: u8 = if (pure) @intCast(max_reach) else @intCast(@min(depth, ttable.depth_mask));
        self.tt.store(node_key, score, (flag << 6) +% stored_depth, best_move.tm, pure);
        self.reach = max_reach;
        return best_value;
    }

    /// Readies the solver to search to `plies`: its memos and lane keys,
    /// the pruned word graph (`main`'s, for a helper), and both root lists.
    fn prepare(self: *Solver, plies: u32, main: ?*const Solver) Error!void {
        self.stuck_memo = @splat(.{});
        self.target_depth = plies;
        for (0..2 * self.board.dim) |lane| self.lane_keys[lane] = bestsmall.laneKey(&self.board, lane);
        self.occupancyFromBoard(&self.occ);
        if (main) |m| {
            self.kwg = m.kwg;
            self.gen.kwg = self.kwg;
        } else if (self.opts.pruned_kwg and self.pruned == null) {
            if (try wordprune.generate(self.gpa, &self.board, &self.racks[0], &self.racks[1], self.numLetters(), self.kwg)) |pr| {
                self.pruned = pr;
                self.kwg = self.pruned.?.kwg();
                self.gen.kwg = self.kwg;
            }
        }
        if (self.lists) |*l| {
            l.reset(self.frames.items.len);
            // Both sides' root lists, which the lists one move down derive
            // from (MAGPIE seeds the opponent's): without the opponent's,
            // each of its lists one move down is generated in full.
            _ = try l.get(self.frames.items.len, self.on_turn);
            if (self.racks[1 - self.on_turn].num_tiles >= 2) _ = try l.get(self.frames.items.len, 1 - self.on_turn);
        }
        self.initial_spread = self.spreadFor(0);
        self.stack_len = plies + 2 + max_playout;
        self.nodes = 0;
        self.playout_nodes = 0;
        self.impure = 0;
    }

    /// Makes the helper threads' solvers (Options.threads), each readied
    /// to search to `plies` from the root, with the root's moves.
    fn startHelpers(self: *Solver, plies: u32) Error!void {
        const n = self.opts.threads - 1;
        const table = try self.gpa.create(abdada_mod.Table);
        table.* = .{};
        self.abdada = table;
        errdefer self.stopHelpers();
        self.helpers = try self.gpa.alloc(*Solver, n);
        var made: usize = 0;
        errdefer self.helpers = self.helpers[0..made];
        var opts = self.opts;
        opts.threads = 1;
        opts.clear_tt = false;
        for (self.helpers, 1..) |*h, i| {
            const helper = try Solver.initIn(self.gpa, self.pos, self.zobrist, self.tt, opts, self.workspace, i);
            errdefer helper.deinit();
            helper.halt = &self.helpers_halt;
            helper.stop = self.stop;
            helper.abdada = table;
            try helper.prepare(plies, self);
            helper.root_moves = try self.gpa.dupe(SmallMove, self.root_moves);
            h.* = helper;
            made += 1;
        }
    }

    fn stopHelpers(self: *Solver) void {
        for (self.helpers) |h| {
            self.gpa.free(h.root_moves);
            h.deinit();
        }
        self.gpa.free(self.helpers);
        self.helpers = &.{};
        if (self.abdada) |t| self.gpa.destroy(t);
        self.abdada = null;
    }

    /// Searches the root to `depth` in the window (alpha, beta): with the
    /// helpers too from depth 2 (Parallel.abdada: every thread searches
    /// the root, in this solver's order of its moves, until this one is
    /// done).
    fn searchRoot(self: *Solver, key: u64, depth: u32, alpha: i16, beta: i16, pv: *Line) Error!i16 {
        if (self.helpers.len == 0 or depth < 2) return self.negamax(key, @intCast(depth), alpha, beta, pv, true);
        const io = self.io.?;
        if (self.opts.parallel == .pv_split) {
            for (self.helpers) |h| h.current_depth = @intCast(depth);
            self.split_path.clearRetainingCapacity();
            return self.pvSplit(key, @intCast(depth), alpha, beta, pv);
        }
        self.helpers_halt.store(false, .monotonic);
        for (self.helpers) |h| {
            h.current_depth = @intCast(depth);
            @memcpy(h.root_moves, self.root_moves);
        }
        const Helper = struct {
            fn run(h: *Solver, k: u64, d: u32, a: i16, b: i16) void {
                var line: Line = .{};
                // A helper's result is in the table; its errors are its
                // being stopped, or memory, which the solver's own search
                // meets too.
                _ = h.negamax(k, @intCast(d), a, b, &line, true) catch {};
            }
        };
        var group: std.Io.Group = .init;
        for (self.helpers) |h| group.concurrent(io, Helper.run, .{ h, key, depth, alpha, beta }) catch break;
        const value = self.negamax(key, @intCast(depth), alpha, beta, pv, true);
        self.helpers_halt.store(true, .monotonic);
        group.await(io) catch {};
        return value;
    }

    /// negamax for a node of the principal variation, splitting its moves
    /// among the threads (Parallel.pv_split); returns what negamax would.
    fn pvSplit(self: *Solver, node_key: u64, depth: i32, alpha_in: i16, beta: i16, pv: *Line) Error!i16 {
        if (depth < pv_split_min_depth or !self.playing) return self.negamax(node_key, depth, alpha_in, beta, pv, true);
        if (self.stop) |st| {
            if (st.load(.monotonic)) return error.Stopped;
        }
        const io = self.io.?;
        var alpha = alpha_in;
        const on = self.on_turn;
        const our_spread = self.spreadFor(on);
        const at_root = self.current_depth == depth;
        const alpha_orig = alpha;
        const impure_at_entry = self.impure;
        var max_reach: u64 = 0;

        // As negamax at a PV node: the table's entry only orders the moves.
        var tt_move: u64 = ttable.invalid_move;
        const e = self.tt.lookup(node_key);
        if (e.valid()) tt_move = e.play();

        // The node's moves, best first, in memory of their own: the threads
        // read them while deeper nodes grow the arena.
        const children = blk: {
            if (at_root) break :blk try self.gpa.dupe(SmallMove, self.root_moves);
            const arena_mark = self.arena.items.len;
            defer self.arena.shrinkRetainingCapacity(arena_mark);
            const list = try self.generate();
            try self.assignEstimates(list, depth, tt_move, node_key, true);
            break :blk try self.gpa.dupe(SmallMove, list);
        };
        defer self.gpa.free(children);
        const vals = try self.gpa.alloc(i16, children.len);
        defer self.gpa.free(vals);
        const fail_high = try self.gpa.alloc(bool, children.len);
        defer self.gpa.free(fail_high);
        var pending: std.ArrayList(u16) = try .initCapacity(self.gpa, children.len);
        defer pending.deinit(self.gpa);

        var best_value: i16 = -huge;
        var best_move: SmallMove = .{};
        var child_pv: Line = .{};

        // Child idx searched as the principal variation, splitting along it.
        const searchPv = struct {
            fn f(s: *Solver, key: u64, d: i32, idx: usize, kids: []const SmallMove, a: *i16, b: i16, best_v: *i16, best_m: *SmallMove, line: *Line, cpv: *Line, reach: *u64, root: bool) Error!void {
                const m = kids[idx];
                cpv.clear();
                var placed: [board_mod.max_dim]MachineLetter = undefined;
                const player = s.on_turn;
                try s.play(m, &placed);
                defer s.unplay();
                s.nodes += 1;
                const child_key = s.zobrist.addMove(key, m.isPass(), m.row(), m.col(), m.vertical(), placed[0..m.len], &s.racks[player], s.numLetters(), player == 0, s.scoreless, s.last_scoreless);
                try s.split_path.append(s.gpa, m);
                defer _ = s.split_path.pop();
                const value = -(try s.pvSplit(child_key, d - 1, -b, -a.*, cpv));
                reach.* = @max(reach.*, s.reach + 1);
                if (value > best_v.*) {
                    best_v.* = value;
                    best_m.* = m;
                    line.update(m, cpv, value -% @as(i16, @intCast(s.initial_spread)));
                }
                if (root) s.root_moves[idx].est = value;
                a.* = @max(a.*, best_v.*);
            }
        }.f;

        if (children.len > 0) try searchPv(self, node_key, depth, 0, children, &alpha, beta, &best_value, &best_move, pv, &child_pv, &max_reach, at_root);
        for (1..children.len) |i| pending.appendAssumeCapacity(@intCast(i));

        while (best_value < beta and pending.items.len > 0) {
            const moves = try self.gpa.alloc(SplitPoint.Claim, pending.items.len);
            defer self.gpa.free(moves);
            @memset(moves, .{});
            var sp: SplitPoint = .{
                .io = io,
                .path = self.split_path.items,
                .node_key = node_key,
                .depth = depth,
                .children = children,
                .pending = pending.items,
                .moves = moves,
                .vals = vals,
                .fail_high = fail_high,
                .alpha = .init(alpha),
                .beta = beta,
            };
            const Helper = struct {
                fn run(h: *Solver, point: *SplitPoint) void {
                    var placed: [board_mod.max_dim]MachineLetter = undefined;
                    var played: usize = 0;
                    defer for (0..played) |_| h.unplay();
                    for (point.path) |m| {
                        h.play(m, &placed) catch |err| return point.fail(err);
                        played += 1;
                    }
                    h.splitWork(point);
                }
            };
            var group: std.Io.Group = .init;
            for (self.helpers) |h| group.concurrent(io, Helper.run, .{ h, &sp }) catch break;
            self.splitWork(&sp);
            group.await(io) catch {};
            if (self.stop) |st| {
                if (st.load(.monotonic)) return error.Stopped;
            }
            if (sp.err) |err| return err;
            if (sp.impure) self.impure += 1;
            max_reach = @max(max_reach, sp.max_reach);

            // The move with the highest lower bound is the one to search as
            // the new principal variation; one at or past beta ends the node.
            var cand: ?u16 = null;
            for (pending.items) |idx| {
                if (fail_high[idx] and (cand == null or vals[idx] > vals[cand.?])) cand = idx;
            }
            if (cand) |c| {
                if (vals[c] >= beta) {
                    best_value = vals[c];
                    best_move = children[c];
                    child_pv.clear();
                    pv.update(best_move, &child_pv, best_value -% @as(i16, @intCast(self.initial_spread)));
                    if (at_root) self.root_moves[c].est = best_value;
                    break;
                }
            } else {
                // Every move failed low against alpha, which did not move:
                // their values are upper bounds, as a search alone has them.
                for (pending.items) |idx| {
                    if (vals[idx] > best_value) {
                        best_value = vals[idx];
                        best_move = children[idx];
                        child_pv.clear();
                        pv.update(best_move, &child_pv, best_value -% @as(i16, @intCast(self.initial_spread)));
                    }
                    if (at_root) self.root_moves[idx].est = vals[idx];
                }
                break;
            }
            try searchPv(self, node_key, depth, cand.?, children, &alpha, beta, &best_value, &best_move, pv, &child_pv, &max_reach, at_root);
            // Left to search against the new alpha: the other moves that
            // failed high (they beat only an older bound), and any whose
            // upper bound is above alpha.
            var kept: usize = 0;
            for (pending.items) |idx| {
                if (idx == cand.?) continue;
                if (fail_high[idx] or vals[idx] > alpha) {
                    pending.items[kept] = idx;
                    kept += 1;
                } else if (at_root) self.root_moves[idx].est = vals[idx];
            }
            pending.shrinkRetainingCapacity(kept);
        }
        _ = our_spread;
        return self.storeNode(node_key, depth, best_value, alpha_orig, beta, self.spreadFor(on), best_move, impure_at_entry, max_reach);
    }

    /// Takes moves from a split point and searches each with a null window
    /// at its best bound, until none is left. The solver must be at the
    /// split point's node.
    fn splitWork(self: *Solver, sp: *SplitPoint) void {
        const on = self.on_turn;
        var child_pv: Line = .{};
        var placed: [board_mod.max_dim]MachineLetter = undefined;
        const halt = self.halt;
        defer self.halt = halt;
        while (!sp.over.load(.monotonic)) {
            const n = sp.claim() orelse return;
            const m = sp.children[sp.pending[n]];
            self.halt = &sp.moves[n].stop;
            self.play(m, &placed) catch |err| return sp.fail(err);
            self.nodes += 1;
            const child_key = self.zobrist.addMove(sp.node_key, m.isPass(), m.row(), m.col(), m.vertical(), placed[0..m.len], &self.racks[on], self.numLetters(), on == 0, self.scoreless, self.last_scoreless);
            const impure_before = self.impure;
            var reach: u64 = 0;
            const failed: ?Error = while (true) {
                const a: i16 = @intCast(sp.alpha.load(.monotonic));
                const value = self.negamax(child_key, sp.depth - 1, -a - 1, -a, &child_pv, false) catch |err| break err;
                reach = @max(reach, self.reach);
                if (sp.finish(n, -value, a, self.impure != impure_before, reach)) break null;
            };
            self.unplay();
            if (failed) |err| {
                if (sp.over.load(.monotonic)) return;
                if (err == error.Stopped and sp.moves[n].stop.load(.monotonic)) {
                    // Another thread finished this move.
                    sp.leave(n);
                    continue;
                }
                return sp.fail(err);
            }
        }
    }

    /// Positions searched by every thread, playouts left out.
    fn searchedNodes(self: *const Solver) u64 {
        var n = self.nodes - self.playout_nodes;
        for (self.helpers) |h| n += h.nodes - h.playout_nodes;
        return n;
    }

    /// The root position's key.
    fn rootKey(self: *const Solver) u64 {
        return self.zobrist.hash(self.board.squares[0 .. self.board.dim * self.board.dim], &self.racks[0], &self.racks[1], self.numLetters(), false, self.scoreless);
    }

    /// Solves to `plies` (iterativelyDeepen), calling `progress` with each
    /// depth's result.
    pub fn solve(self: *Solver, plies: u32, progress: ?*const fn (ctx: ?*anyopaque, r: *const Result) void, ctx: ?*anyopaque) Error!Result {
        if (self.racks[1].num_tiles == 0) return error.NoOpponentTiles;
        if (self.opts.clear_tt) try self.tt.reset(self.gpa, self.opts.tt_bytes);
        if (self.workspace) |w| w.newSolve();
        try self.prepare(plies, null);
        const key = self.rootKey();
        var alpha: i16 = -huge;
        var beta: i16 = huge;
        if (self.opts.first_win) {
            alpha = -1;
            beta = 1;
        }
        self.current_depth = -1;
        const root = try self.generate();
        self.root_moves = try self.gpa.dupe(SmallMove, root);
        defer self.gpa.free(self.root_moves);
        self.arena.clearRetainingCapacity();
        try self.assignEstimates(self.root_moves, 0, ttable.invalid_move, key, true);
        if (self.opts.threads > 1 and self.io != null and plies >= 2) try self.startHelpers(plies);
        defer if (self.helpers.len > 0) self.stopHelpers();

        var result: Result = .{ .value = 0, .line = .{}, .depth = 0, .nodes = 0 };
        var reported = true;
        const start: u32 = if (self.opts.iterative_deepening) 1 else plies;
        var p = start;
        var prev_value: i32 = 0;
        while (p <= plies) : (p = nextDepth(p, plies)) {
            self.current_depth = @intCast(p);
            var pv: Line = .{};
            var impure_before = self.impure;
            var val: i16 = undefined;
            if (self.opts.aspiration and p > start and !self.opts.first_win) {
                // MAGPIE's: 25 points either side, doubled each failure.
                var window: i32 = aspiration_window;
                var a: i32 = prev_value - window;
                var b: i32 = prev_value + window;
                while (true) {
                    pv = .{};
                    impure_before = self.impure;
                    val = self.searchRoot(key, p, @intCast(std.math.clamp(a, -huge, huge)), @intCast(std.math.clamp(b, -huge, huge)), &pv) catch |err| switch (err) {
                        error.Stopped => break,
                        else => |e| return e,
                    };
                    if (val <= a) {
                        a = if (window >= huge / 2) -huge else prev_value - 2 * window;
                        window *= 2;
                    } else if (val >= b) {
                        b = if (window >= huge / 2) huge else prev_value + 2 * window;
                        window *= 2;
                    } else break;
                }
                if (self.stop) |st| if (st.load(.monotonic)) break;
            } else {
                val = self.searchRoot(key, p, alpha, beta, &pv) catch |err| switch (err) {
                    error.Stopped => break,
                    else => |e| return e,
                };
            }
            prev_value = val;
            const root_pure = self.impure == impure_before;
            // By this depth's values; MAGPIE's ordering keeps equal values
            // in their last order (a stable sort).
            if (self.opts.heuristics and self.opts.magpie_ordering and self.opts.lazy_order) {
                try self.sortStable(self.root_moves);
            } else {
                gosort.slice(SmallMove, self.root_moves, {}, struct {
                    fn less(_: void, a: SmallMove, b: SmallMove) bool {
                        return a.est > b.est;
                    }
                }.less);
            }
            result = .{ .value = val -% @as(i16, @intCast(self.initial_spread)), .line = pv, .depth = p, .nodes = self.searchedNodes() };
            if (root_pure) result.depth = plies;
            reported = false;
            // The last depth is reported once its line is complete.
            if (root_pure or nextDepth(p, plies) > plies) break;
            if (progress) |f| f(ctx, &result);
            reported = true;
        }
        if (result.depth > 0 and self.opts.complete_line) {
            self.stop = null;
            try self.completeLine(&result.line, result.depth, plies);
            result.nodes = self.searchedNodes();
        }
        if (!reported) {
            if (progress) |f| f(ctx, &result);
        }
        return result;
    }

    /// The key of the position the solver is at, from scratch.
    fn positionKey(self: *const Solver) u64 {
        return self.zobrist.hash(self.board.squares[0 .. self.board.dim * self.board.dim], &self.racks[0], &self.racks[1], self.numLetters(), self.on_turn != 0, self.scoreless);
    }

    /// Extends a line that stops before the game is over (completeLine): a
    /// table hit at a node off the principal variation returns its value
    /// without the moves behind it. From where the line stops, search again
    /// with the depth the search had left there, until the line reaches the
    /// horizon, then play it out greedily as a leaf would.
    fn completeLine(self: *Solver, line: *Line, last_depth: u32, plies: u32) Error!void {
        if (line.len == 0 or line.searched < line.len) return;
        var placed: [board_mod.max_dim]MachineLetter = undefined;
        var played: usize = 0;
        defer for (0..played) |_| self.unplay();
        for (0..line.len) |i| {
            if (!self.playing or self.stackRemaining() == 0) return;
            try self.play(line.moves[i], &placed);
            played += 1;
        }
        const horizon: usize = if (last_depth != 0) last_depth else plies;
        const saved_depth = self.current_depth;
        self.current_depth = max_variant + 1;
        defer self.current_depth = saved_depth;
        while (self.playing and line.len < max_variant and line.searched == line.len and (line.len < horizon or self.opts.heuristics)) {
            const depth: usize = if (horizon > line.len) horizon - line.len else 0;
            var sub: Line = .{};
            _ = try self.negamax(self.positionKey(), @intCast(depth), -huge, huge, &sub, true);
            if (sub.len == 0) break;
            const start = line.len;
            const n = @min(sub.len, max_variant - start);
            @memcpy(line.moves[start..][0..n], sub.moves[0..n]);
            line.len += n;
            line.searched += @min(sub.searched, n);
            for (start..line.len) |i| {
                if (!self.playing or self.stackRemaining() == 0) break;
                try self.play(line.moves[i], &placed);
                played += 1;
            }
        }
    }

    fn nextDepth(p: u32, plies: u32) u32 {
        if (p + 1 >= plies or (plies - (p + 1)) % 2 == 0) return p + 1;
        return p + 2;
    }

    /// The moves of `line` as moves on the root position's board, for
    /// display: each is played to find the squares it plays through.
    pub fn lineMoves(self: *Solver, line: *const Line, out: []Move) Allocator.Error!usize {
        var placed: [board_mod.max_dim]MachineLetter = undefined;
        var n: usize = 0;
        while (n < line.len and n < out.len and self.playing) : (n += 1) {
            const m = line.moves[n];
            if (m.isPass()) {
                out[n] = .pass();
            } else {
                out[n] = .{ .kind = .play, .vertical = m.vertical(), .row = @intCast(m.row()), .col = @intCast(m.col()), .len = m.len, .tiles_played = m.tiles_played, .score = m.score };
            }
            try self.play(m, &placed);
            if (!m.isPass()) @memcpy(out[n].tiles[0..m.len], placed[0..m.len]);
        }
        for (0..n) |_| self.unplay();
        return n;
    }
};

const testing = std.testing;

/// Go's TestSolveStandard (VsCanik, 2019 Nationals), also MAGPIE's
/// test_solve_standard.
const standard_cgp = "9A1PIXY/9S1L3/2ToWNLETS1O3/9U1DA1R/3GERANIAL1U1I/9g2T1C/8WE2OBI/6EMU4ON/6AID3GO1/5HUN4ET1/4ZA1T4ME1/1Q1FAKEY3JOES/FIVE1E5IT1C/5SPORRAN2A/6ORE2N2D BGIV/DEHILOR 384/389 0 lex NWL20;";

fn sameSquares(a: *const Board, b: *const Board) bool {
    const n = a.dim * a.dim;
    if (!std.mem.eql(MachineLetter, a.squares[0..n], b.squares[0..n])) return false;
    for (0..2) |d| {
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(a.views[d][0..n]), std.mem.sliceAsBytes(b.views[d][0..n]))) return false;
    }
    return a.tiles_played == b.tiles_played;
}

test "playing and unplaying moves two deep puts the board back" {
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const p = try testdata.position(&store, standard_cgp);
    var z = try zobrist_mod.Zobrist.init(testing.allocator, p.board.dim, @splat(1));
    defer z.deinit(testing.allocator);
    var tt: ttable.Table = .{};
    defer tt.deinit(testing.allocator);
    const s = try Solver.init(testing.allocator, &p, &z, &tt, .{});
    defer s.deinit();
    const original = s.board;
    var placed: [board_mod.max_dim]MachineLetter = undefined;
    const firsts = try testing.allocator.dupe(SmallMove, try s.generate());
    defer testing.allocator.free(firsts);
    s.arena.clearRetainingCapacity();
    try testing.expect(firsts.len > 10);
    for (firsts) |m| {
        try s.play(m, &placed);
        const after = s.board;
        const replies = try testing.allocator.dupe(SmallMove, try s.generate());
        defer testing.allocator.free(replies);
        s.arena.clearRetainingCapacity();
        for (replies) |r| {
            try s.play(r, &placed);
            s.unplay();
            try testing.expect(sameSquares(&after, &s.board));
        }
        s.unplay();
        try testing.expect(sameSquares(&original, &s.board));
        try testing.expectEqual(p.racks, s.racks);
        try testing.expectEqual(p.scores, s.scores);
        try testing.expectEqual(0, s.on_turn);
    }
}

test "the standard endgame is worth 11 at 4 plies" {
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const p = try testdata.position(&store, standard_cgp);
    var z = try zobrist_mod.Zobrist.init(testing.allocator, p.board.dim, @splat(1));
    defer z.deinit(testing.allocator);
    var tt: ttable.Table = .{};
    defer tt.deinit(testing.allocator);
    const s = try Solver.init(testing.allocator, &p, &z, &tt, .{ .tt_bytes = 0 });
    defer s.deinit();
    const r = try s.solve(4, null, null);
    try testing.expectEqual(11, r.value);
    try testing.expectEqual(4, r.depth);
}

test "the lazy move ordering sorts by estimate, ties in MAGPIE's order" {
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    var scratch: [40]SmallMove = undefined;
    for (0..500) |_| {
        var list: [40]SmallMove = undefined;
        const n = r.uintAtMost(usize, list.len);
        // Few estimates, so many ties; each move's code tells them apart.
        for (list[0..n], 0..) |*m, i| m.* = .{ .tm = i, .est = r.intRangeAtMost(i16, -3, 3) };
        var ordered: usize = 0;
        var sorted = false;
        var seen: u64 = 0;
        for (0..n) |idx| {
            if (!sorted) sorted = orderThrough(list[0..n], &ordered, idx, &scratch);
            // Each move in its place once taken: none after it is better.
            for (list[idx + 1 .. n]) |m| try testing.expect(m.est <= list[idx].est);
            seen |= @as(u64, 1) << @intCast(list[idx].tm);
        }
        try testing.expectEqual(if (n == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(n)) - 1, seen);
    }
    // MAGPIE's selection swaps the best into place: estimates 1, 1, 2 give
    // the 2, then the second 1, then the first.
    var three = [_]SmallMove{ .{ .tm = 0, .est = 1 }, .{ .tm = 1, .est = 1 }, .{ .tm = 2, .est = 2 } };
    var ordered: usize = 0;
    _ = orderThrough(&three, &ordered, 2, &scratch);
    try testing.expectEqual(2, three[0].tm);
    try testing.expectEqual(1, three[1].tm);
    try testing.expectEqual(0, three[2].tm);
}

test "the tail sort is stable" {
    var prng: std.Random.DefaultPrng = .init(9);
    const r = prng.random();
    var scratch: [100]SmallMove = undefined;
    for (0..300) |_| {
        var list: [200]SmallMove = undefined;
        const n = r.uintAtMost(usize, list.len);
        for (list[0..n], 0..) |*m, i| m.* = .{ .tm = i, .est = r.intRangeAtMost(i16, -4, 4) };
        var want = list;
        std.sort.block(SmallMove, want[0..n], {}, struct {
            fn f(_: void, a: SmallMove, b: SmallMove) bool {
                return a.est > b.est;
            }
        }.f);
        sortByEstimateStable(list[0..n], &scratch);
        for (want[0..n], list[0..n]) |w, g| try testing.expectEqual(w.tm, g.tm);
    }
}

test "the parallel searches give the values one thread does" {
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const cgps = [_][]const u8{
        "CATENOID7/O14/IT1S11/FERN11/1GOY2Q3V4/1UTE1BI1TEIID2/1LI2U1WAWL4/TA1VERSO2LAZAR/AR3R4O4/X3BY1AMUSE3/I3A1PRETENDS1/N1FUCK9/GEE1HOG8/1N3HI8/1D1JuLEP7 AIMOO/?EN 354/408 0 lex NWL23;",
        "1OAK3D7/OHIA3U7/M6C7/I5LI7/C3JEU8/RHINO1V4P3/O1NUB1V4R3/N4OYEZ1TE3/5B1FA1AS3/ALTERERS2XI1W1/5Y4AD1I1/3UnSWEAR1I1P1/1QAT4GEMOTE1/FILED8R1/EST1EaNLING4 ADGT/ENOO 358/440 0 lex NWL23;",
        "3SECONDO1C1FA/5I3WARNER/5L4BEAD1/3MOI3N1T3/3AHA3AHI3/V2US4GIN3/A2Z2ROB1L4/U2YOKED2LAMIA/L1Q1WIFEY5N/TOUTER2E5O/1PEED3G5i/JEST4G5N/E1O11t/U1SPIV8E/X13R IIT/RT 437/391 0 lex NWL23;",
        standard_cgp,
    };
    for (cgps) |text| {
        const p = try testdata.position(&store, text);
        var z = try zobrist_mod.Zobrist.init(testing.allocator, p.board.dim, @splat(1));
        defer z.deinit(testing.allocator);
        var tt: ttable.Table = .{};
        defer tt.deinit(testing.allocator);
        for ([_]u32{ 2, 4 }) |plies| {
            const want = blk: {
                const s = try Solver.init(testing.allocator, &p, &z, &tt, .{ .tt_bytes = 0 });
                defer s.deinit();
                break :blk (try s.solve(plies, null, null)).value;
            };
            for ([_]Parallel{ .abdada, .pv_split }) |parallel| {
                const s = try Solver.init(testing.allocator, &p, &z, &tt, .{ .tt_bytes = 0, .threads = 3, .parallel = parallel });
                defer s.deinit();
                s.io = testing.io;
                try testing.expectEqual(want, (try s.solve(plies, null, null)).value);
            }
        }
    }
}

test "the speed options leave endgame values as they were" {
    // Endgames in which, one ply above the leaves of a 4-ply search, the
    // opponent often moves while the solving player holds one tile (the
    // out-reply bound's case there), and the standard one.
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const cgps = [_][]const u8{
        "CATENOID7/O14/IT1S11/FERN11/1GOY2Q3V4/1UTE1BI1TEIID2/1LI2U1WAWL4/TA1VERSO2LAZAR/AR3R4O4/X3BY1AMUSE3/I3A1PRETENDS1/N1FUCK9/GEE1HOG8/1N3HI8/1D1JuLEP7 AIMOO/?EN 354/408 0 lex NWL23;",
        "1OAK3D7/OHIA3U7/M6C7/I5LI7/C3JEU8/RHINO1V4P3/O1NUB1V4R3/N4OYEZ1TE3/5B1FA1AS3/ALTERERS2XI1W1/5Y4AD1I1/3UnSWEAR1I1P1/1QAT4GEMOTE1/FILED8R1/EST1EaNLING4 ADGT/ENOO 358/440 0 lex NWL23;",
        "3SECONDO1C1FA/5I3WARNER/5L4BEAD1/3MOI3N1T3/3AHA3AHI3/V2US4GIN3/A2Z2ROB1L4/U2YOKED2LAMIA/L1Q1WIFEY5N/TOUTER2E5O/1PEED3G5i/JEST4G5N/E1O11t/U1SPIV8E/X13R IIT/RT 437/391 0 lex NWL23;",
        standard_cgp,
    };
    const variants = [_]Options{
        .{ .tt_bytes = 0, .out_reply = false },
        .{ .tt_bytes = 0, .lazy_order = false },
        .{ .tt_bytes = 0, .capped_first_depth = false },
        .{ .tt_bytes = 0, .sibling = false, .first_reply = false },
    };
    for (cgps) |text| {
        const p = try testdata.position(&store, text);
        var z = try zobrist_mod.Zobrist.init(testing.allocator, p.board.dim, @splat(1));
        defer z.deinit(testing.allocator);
        var tt: ttable.Table = .{};
        defer tt.deinit(testing.allocator);
        for ([_]u32{ 1, 3, 4 }) |plies| {
            const want = blk: {
                const s = try Solver.init(testing.allocator, &p, &z, &tt, .{ .tt_bytes = 0 });
                defer s.deinit();
                break :blk (try s.solve(plies, null, null)).value;
            };
            for (variants) |opts| {
                const s = try Solver.init(testing.allocator, &p, &z, &tt, opts);
                defer s.deinit();
                try testing.expectEqual(want, (try s.solve(plies, null, null)).value);
            }
        }
    }
}
