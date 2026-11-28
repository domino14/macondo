//! The pre-endgame: with one to six tiles in the bag, each play of the
//! player on turn is valued by the endgames it leads to, for every ordering
//! of the tiles they can't see. Port of Macondo's preendgame package, its
//! pessimistic model:
//!
//! - The opponent knows what is in the bag: for each ordering, every reply
//!   is searched, and one that beats us makes the ordering a loss (a draw a
//!   draw, if none beats us).
//! - We don't: when it is our turn again with tiles in the bag, we choose a
//!   play across every ordering we can't tell apart (a nested pre-endgame,
//!   up to `nested_depth_limit` levels; past it, our best play by score
//!   plus leave), and the verdict for the actual ordering is the worst
//!   result among the plays tied for best.
//! - A play's points are its wins plus half its draws, each ordering
//!   weighted by how many orderings of the unseen tiles it stands for.
//!
//! The endgames are Macondo's quick result-only searches to a fixed depth;
//! the plays tied for first that empty the bag are told apart by their
//! spread over full-window searches.
//!
//! No I/O: the position comes in set up, and results go out as values.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("../data/tiles.zig");
const position = @import("../board/position.zig");
const move_mod = @import("../board/move.zig");
const movegen = @import("../movegen/movegen.zig");
const equity = @import("../equity/equity.zig");
const endgame = @import("../endgame/endgame.zig");
const ttable = @import("../endgame/ttable.zig");
const zobrist_mod = @import("../endgame/zobrist.zig");
const wordprune = @import("../endgame/wordprune.zig");
const Kwg = @import("../data/kwg.zig").Kwg;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const Move = move_mod.Move;
const Position = position.Position;

/// The most tiles in the bag a pre-endgame is solved with.
pub const max_bag = 6;
const rack_tile_limit = 7;
const max_alpha = tiles.max_alphabet_size;
/// The plays tied for first that the spread tiebreak solves (by score).
pub const tiebreak_plays = 20;
/// Moves played in one line of the search, nested pre-endgames included:
/// every two plies draw a tile or end the game (two passes in a row), so a
/// line is short.
const max_levels = 48;

pub const Outcome = enum(u8) {
    none,
    win,
    draw,
    loss,

    pub fn name(o: Outcome) []const u8 {
        return switch (o) {
            .none => "none",
            .win => "win",
            .draw => "draw",
            .loss => "loss",
        };
    }
};

/// Tiles in the bag, in the order they are drawn.
pub const Draw = struct {
    len: u8 = 0,
    tiles: [max_bag]MachineLetter = undefined,

    pub fn slice(self: *const Draw) []const MachineLetter {
        return self.tiles[0..self.len];
    }

    fn eql(a: *const Draw, b: *const Draw) bool {
        return std.mem.eql(MachineLetter, a.slice(), b.slice());
    }
};

/// An ordering of the bag's tiles, and how many orderings of the unseen
/// tiles (told apart by tile, not letter) it stands for.
pub const Perm = struct {
    draw: Draw,
    count: u32,
};

/// Every ordering of `k` tiles drawn from `counts` (by letter), in order of
/// the first tile drawn, then the second, and so on, each with its count:
/// the product of how many of each letter were left when it was drawn.
/// Port of generatePermutations.
pub fn permutations(gpa: Allocator, counts: *const [max_alpha]u8, k: usize, out: *std.ArrayList(Perm)) Allocator.Error!void {
    var left = counts.*;
    var cur: Draw = .{};
    try permute(gpa, &left, k, &cur, 1, out);
}

fn permute(gpa: Allocator, left: *[max_alpha]u8, k: usize, cur: *Draw, count: u32, out: *std.ArrayList(Perm)) Allocator.Error!void {
    if (cur.len == k) return out.append(gpa, .{ .draw = cur.*, .count = count });
    for (left, 0..) |*n, ml| {
        if (n.* == 0) continue;
        const c = count * n.*;
        n.* -= 1;
        cur.tiles[cur.len] = @intCast(ml);
        cur.len += 1;
        try permute(gpa, left, k, cur, c, out);
        cur.len -= 1;
        n.* += 1;
    }
}

pub const Options = struct {
    /// Endgame depth, in plies.
    plies: u32 = 3,
    /// Solve at 1, 2, ... `plies` plies, each depth trying the plays in the
    /// order the last one ranked them (only the last depth's results count).
    iterative_deepening: bool = true,
    /// Only plays leaving at most this many tiles in the bag are solved
    /// (-1: all), unless `only` names the plays.
    max_tiles_left: i32 = 1,
    /// Nested pre-endgames: how many levels of our turns with tiles still
    /// in the bag are solved (-1: all). Past it our play is the best by
    /// score plus leave.
    nested_depth_limit: i32 = 1,
    /// Tell the plays tied for first that empty the bag apart by spread.
    tiebreak: bool = true,
    /// Stop solving a play once it has more losses than the best play that
    /// empties the bag can still have (its results are then partial).
    early_cutoff: bool = false,
    /// Stop solving a play at its first loss.
    skip_loss: bool = false,
    /// Solve only these plays (whatever they leave in the bag).
    only: []const Move = &.{},
    /// Plays the cutoffs never stop, and that the tiebreak always solves if
    /// they are tied for first and empty the bag.
    avoid_prune: []const Move = &.{},
    /// Tiles known to be on the opponent's rack (some or all of it).
    known_opp: []const MachineLetter = &.{},
};

pub const PlayOutcome = struct {
    draw: Draw,
    outcome: Outcome,
    count: u32,
};

pub const PlayResult = struct {
    move: Move,
    /// Wins plus half the draws, doubled (an integer).
    points2: u64,
    /// Losses plus half the draws, doubled.
    losses2: u64,
    /// The total spread over the orderings, if the tiebreak worked it out.
    spread: ?i64,
    /// Stopped by a cutoff: the results are partial.
    cut: bool,
    /// One for each ordering of the bag the play was solved for.
    outcomes: []PlayOutcome,
};

pub const Result = struct {
    /// Best first.
    plays: []PlayResult,
    /// Endgames searched.
    endgames: u64,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        for (self.plays) |p| gpa.free(p.outcomes);
        gpa.free(self.plays);
        self.* = undefined;
    }
};

pub const Error = Allocator.Error || error{
    /// The player on turn must hold a full rack.
    RackNotFull,
    /// The bag must hold 1 to `max_bag` tiles.
    BagSize,
    /// A known opponent tile isn't among the unseen tiles.
    BadKnownRack,
} || endgame.Error;

/// A play's results as they are gathered (Go's PreEndgamePlay).
const Row = struct {
    outcome: Outcome = .none,
    count: u32 = 0,
    finalized: bool = false,
};

const Acc = struct {
    rows: []Row,
    points2: u64 = 0,
    losses2: u64 = 0,
    spread: i64 = 0,
    spread_set: bool = false,

    fn credit(self: *Acc, o: Outcome, ct: u32) void {
        switch (o) {
            .win => self.points2 += 2 * @as(u64, ct),
            .draw => {
                self.points2 += ct;
                self.losses2 += ct;
            },
            .loss => self.losses2 += 2 * @as(u64, ct),
            .none => {},
        }
    }

    /// A final outcome for ordering `idx` (addWinPctStat).
    fn add(self: *Acc, o: Outcome, idx: usize, ct: u32) void {
        const r = &self.rows[idx];
        r.outcome = o;
        r.count += ct;
        self.credit(o, ct);
        r.finalized = true;
    }

    /// One line's outcome for ordering `idx`: the ordering is a loss if
    /// any line loses, a draw if any draws, a win only if all win
    /// (setUnfinalizedWinPctStat).
    fn setUnfinalized(self: *Acc, o: Outcome, idx: usize, ct: u32) void {
        const r = &self.rows[idx];
        if (r.outcome == .none) r.count = ct;
        switch (o) {
            .win => if (r.outcome != .draw and r.outcome != .loss) {
                r.outcome = .win;
            },
            .draw => if (r.outcome != .loss) {
                r.outcome = .draw;
            },
            .loss => r.outcome = .loss,
            .none => {},
        }
    }

    fn finalize(self: *Acc) void {
        for (self.rows) |*r| {
            if (r.finalized) continue;
            self.credit(r.outcome, r.count);
            r.finalized = true;
        }
    }

    fn hasLoss(self: *const Acc, idx: usize) bool {
        return self.rows[idx].outcome == .loss;
    }

    fn record(self: *Acc, o: Outcome, idx: usize, ct: u32, final: bool) void {
        if (final) self.add(o, idx, ct) else self.setUnfinalized(o, idx, ct);
    }
};

/// A position in the search: the board and racks with the player on turn
/// first (as Position keeps them), the bag in draw order, and whose turn it
/// is. Two passes in a row end the game (Macondo's endgame mode).
const Node = struct {
    pos: Position,
    bag: Draw,
    /// The solving player is on turn.
    us: bool,
    scoreless: u8,
    over: bool = false,

    fn spreadForUs(self: *const Node) i32 {
        const s = self.pos.scores[0] - self.pos.scores[1];
        return if (self.us) s else -s;
    }
};

/// Nested pre-endgames are solved once for each information state: the
/// board, our rack, the unseen tiles and the scoreless turns (Go's
/// nestedCacheKey). The verdict for each ordering of the bag is kept.
const CacheKey = struct {
    board: u64,
    ours: [max_alpha]u8,
    unseen: [max_alpha]u8,
    scoreless: u8,
};

const Verdict = struct { draw: Draw, outcome: Outcome };

/// One thread's search state.
const Thread = struct {
    gpa: Allocator,
    index: usize,
    gen: movegen.Generator,
    ws: endgame.Workspace,
    nodes: []Node,
    lists: []std.ArrayList(Move),
    endgames: u64 = 0,
    /// The endgame solver, made once and reset for each endgame, and the
    /// position it searches.
    eg: ?*endgame.Solver = null,
    eg_pos: Position = undefined,

    fn init(gpa: Allocator, index: usize, root: *const Position, kwg: Kwg) Allocator.Error!*Thread {
        const t = try gpa.create(Thread);
        errdefer gpa.destroy(t);
        const nodes = try gpa.alloc(Node, max_levels);
        errdefer gpa.free(nodes);
        const lists = try gpa.alloc(std.ArrayList(Move), max_levels);
        for (lists) |*l| l.* = .empty;
        t.* = .{
            .gpa = gpa,
            .index = index,
            .gen = undefined,
            .ws = .init(gpa),
            .nodes = nodes,
            .lists = lists,
        };
        nodes[0].pos = root.*;
        t.gen = .init(gpa, &nodes[0].pos.board, kwg, root.dist);
        t.gen.gen_pass = true;
        return t;
    }

    fn deinit(self: *Thread) void {
        const gpa = self.gpa;
        if (self.eg) |eg| eg.deinit();
        for (self.lists) |*l| l.deinit(gpa);
        gpa.free(self.lists);
        gpa.free(self.nodes);
        self.gen.deinit();
        self.ws.deinit();
        gpa.destroy(self);
    }

    /// Every move of the player on turn at `n` (tile plays and a pass)
    /// into the list of `level`.
    fn genAll(self: *Thread, n: *Node, level: usize) Allocator.Error![]Move {
        self.gen.board = &n.pos.board;
        var rack = n.pos.racks[0];
        try self.gen.genAll(&rack, false);
        const l = &self.lists[level];
        l.clearRetainingCapacity();
        try l.appendSlice(self.gpa, self.gen.plays.items);
        return l.items;
    }
};

/// `parent` after the player on turn makes `m` and draws.
fn play(parent: *const Node, child: *Node, m: *const Move) void {
    child.* = parent.*;
    child.pos.last_move = null;
    if (m.kind == .pass) {
        child.scoreless += 1;
        const sc = child.pos.scores;
        const rk = child.pos.racks;
        child.pos.racks = .{ rk[1], rk[0] };
        child.pos.scores = .{ sc[1], sc[0] };
        if (child.scoreless >= 2) {
            // Each player loses the value of their rack.
            child.over = true;
            child.pos.scores[0] -= child.pos.racks[0].scoreOn(child.pos.dist);
            child.pos.scores[1] -= child.pos.racks[1].scoreOn(child.pos.dist);
        }
    } else {
        const parsed: move_mod.Parsed = .{ .move = m.* };
        _ = child.pos.playMove(&parsed, null) orelse unreachable;
        const n = @min(m.tiles_played, child.bag.len);
        for (child.bag.tiles[0..n]) |ml| child.pos.racks[1].add(ml);
        std.mem.copyForwards(MachineLetter, child.bag.tiles[0 .. child.bag.len - n], child.bag.tiles[n..child.bag.len]);
        child.bag.len -= n;
        child.scoreless = 0;
        // The bag held tiles, so the mover drew some: no one is out.
        std.debug.assert(child.pos.racks[1].num_tiles > 0);
    }
    child.pos.last_move = null;
    child.us = !parent.us;
}

fn sameMove(a: *const Move, b: *const Move) bool {
    if (a.kind != b.kind) return false;
    if (a.kind == .pass) return true;
    return a.row == b.row and a.col == b.col and a.vertical == b.vertical and
        std.mem.eql(MachineLetter, a.tileSlice(), b.tileSlice());
}

/// A candidate play being solved.
const Play = struct {
    move: Move,
    equity: f64,
    acc: Acc,
    cut: bool = false,
};

pub const Solver = struct {
    gpa: Allocator,
    root: Position,
    opts: Options,
    zobrist: *const zobrist_mod.Zobrist,
    tt: *ttable.Table,
    /// The tiles we can't see (the bag and the opponent's rack), less the
    /// known opponent tiles: what the bag can hold.
    maybe_in_bag: [max_alpha]u8 = @splat(0),
    num_in_bag: u8 = 0,
    perms: std.ArrayList(Perm) = .empty,
    num_combos: u64 = 0,
    plies: u32 = 0,
    /// Losses (doubled) that the best play emptying the bag can still
    /// have (the early cutoff's bound).
    min_potential2: u64 = std.math.maxInt(u64),
    cache: std.AutoHashMapUnmanaged(CacheKey, []Verdict) = .empty,
    /// The words the tiles in play can make on the board (Go builds the
    /// same graph): moves and endgames are generated over it.
    pruned: ?wordprune.Pruned = null,
    kwg: Kwg,
    thread: *Thread,

    /// `p` has the player to solve for on turn, with a full rack. The
    /// opponent's rack in `p` counts as unseen (`known_opp` gives what is
    /// known of it). `zobrist` and `tt` are the endgames', which the caller
    /// sizes; the solver doesn't clear the table.
    pub fn init(gpa: Allocator, p: *const Position, zobrist: *const zobrist_mod.Zobrist, tt: *ttable.Table, opts: Options) Error!Solver {
        if (p.racks[0].num_tiles != rack_tile_limit) return error.RackNotFull;
        var s: Solver = .{
            .gpa = gpa,
            .root = p.*,
            .opts = opts,
            .zobrist = zobrist,
            .tt = tt,
            .kwg = p.lexicon.kwg,
            .thread = undefined,
        };
        s.root.last_move = null;
        // Unseen: every tile less the board's and our rack's.
        const ld = p.dist;
        for (0..ld.size) |ml| s.maybe_in_bag[ml] = @intCast(ld.counts[ml]);
        for (p.board.squares[0 .. p.board.dim * p.board.dim]) |ml| {
            if (ml != 0) s.maybe_in_bag[tiles.intrinsicIndex(ml)] -= 1;
        }
        for (0..ld.size) |ml| s.maybe_in_bag[ml] -= p.racks[0].counts[ml];
        var unseen: u32 = 0;
        for (s.maybe_in_bag) |c| unseen += c;
        if (unseen <= rack_tile_limit or unseen > rack_tile_limit + max_bag) return error.BagSize;
        s.num_in_bag = @intCast(unseen - rack_tile_limit);
        for (opts.known_opp) |ml| {
            if (s.maybe_in_bag[ml] == 0) return error.BadKnownRack;
            s.maybe_in_bag[ml] -= 1;
        }
        // Go counts the orderings of the bag drawn from the bag and a full
        // rack, known tiles or not.
        s.num_combos = 1;
        for (0..s.num_in_bag) |i| s.num_combos *= rack_tile_limit + s.num_in_bag - i;
        try permutations(gpa, &s.maybe_in_bag, s.num_in_bag, &s.perms);
        errdefer s.perms.deinit(gpa);
        const pool: Rack = .{ .counts = s.unseenCounts(), .num_tiles = @intCast(unseen) };
        s.pruned = try wordprune.generate(gpa, &p.board, &p.racks[0], &pool, ld.size, p.lexicon.kwg);
        errdefer if (s.pruned) |*pr| pr.deinit(gpa);
        if (s.pruned) |*pr| s.kwg = pr.kwg();
        s.thread = try Thread.init(gpa, 0, &s.root, s.kwg);
        return s;
    }

    pub fn deinit(self: *Solver) void {
        self.clearCache();
        self.cache.deinit(self.gpa);
        self.perms.deinit(self.gpa);
        self.thread.deinit();
        if (self.pruned) |*pr| pr.deinit(self.gpa);
        self.* = undefined;
    }

    fn clearCache(self: *Solver) void {
        var it = self.cache.valueIterator();
        while (it.next()) |v| self.gpa.free(v.*);
        self.cache.clearRetainingCapacity();
    }

    fn avoidPrune(self: *const Solver, m: *const Move) bool {
        for (self.opts.avoid_prune) |*a| {
            if (sameMove(a, m)) return true;
        }
        return false;
    }

    /// The root position with the bag holding `perm` and the opponent the
    /// rest of the unseen tiles.
    fn rootNode(self: *const Solver, perm: *const Perm, n: *Node) void {
        n.pos = self.root;
        var opp = self.unseenCounts();
        for (perm.draw.slice()) |ml| opp[ml] -= 1;
        n.pos.racks[1] = .{};
        for (opp, 0..) |c, ml| {
            for (0..c) |_| n.pos.racks[1].add(@intCast(ml));
        }
        n.bag = perm.draw;
        n.us = true;
        n.scoreless = if (self.root.scoreless_turns + 1 >= self.root.max_scoreless_turns) 1 else 0;
        n.over = false;
    }

    /// The unseen tiles with the known opponent tiles back in.
    fn unseenCounts(self: *const Solver) [max_alpha]u8 {
        var c = self.maybe_in_bag;
        for (self.opts.known_opp) |ml| c[ml] += 1;
        return c;
    }

    pub fn solve(self: *Solver) Error!Result {
        const gpa = self.gpa;
        const t = self.thread;
        var plays: std.ArrayList(Play) = .empty;
        defer {
            for (plays.items) |*p| gpa.free(p.acc.rows);
            plays.deinit(gpa);
        }
        // The candidates, best static equity first.
        const ctx = self.root.equityContext();
        var moves: std.ArrayList(Move) = .empty;
        defer moves.deinit(gpa);
        if (self.opts.only.len > 0) {
            try moves.appendSlice(gpa, self.opts.only);
        } else {
            t.nodes[0].pos = self.root;
            try moves.appendSlice(gpa, try t.genAll(&t.nodes[0], 0));
        }
        const eqs = try gpa.alloc(f64, moves.items.len);
        defer gpa.free(eqs);
        for (moves.items, eqs) |*m, *e| {
            const leave = m.leave(&self.root.racks[0]);
            e.* = ctx.equity(m, &leave);
        }
        {
            const order = try gpa.alloc(usize, moves.items.len);
            defer gpa.free(order);
            for (order, 0..) |*o, i| o.* = i;
            std.sort.pdq(usize, order, SortCtx{ .moves = moves.items, .eqs = eqs }, SortCtx.lessThan);
            for (order) |i| {
                const rows = try gpa.alloc(Row, self.perms.items.len);
                @memset(rows, .{});
                try plays.append(gpa, .{ .move = moves.items[i], .equity = eqs[i], .acc = .{ .rows = rows } });
            }
        }

        self.plies = if (self.opts.iterative_deepening) 1 else self.opts.plies;
        while (self.plies <= self.opts.plies) : (self.plies += 1) {
            for (plays.items) |*p| {
                @memset(p.acc.rows, .{});
                p.acc = .{ .rows = p.acc.rows };
                p.cut = false;
            }
            self.min_potential2 = std.math.maxInt(u64);
            self.clearCache();
            for (plays.items) |*p| try self.solvePlay(p);
            std.sort.insertion(Play, plays.items, {}, byPoints);
            if (self.opts.tiebreak) try self.tiebreak(plays.items);
        }

        const out = try gpa.alloc(PlayResult, plays.items.len);
        var done: usize = 0;
        errdefer {
            for (out[0..done]) |r| gpa.free(r.outcomes);
            gpa.free(out);
        }
        for (plays.items, out) |*p, *r| {
            var n: usize = 0;
            for (p.acc.rows) |row| {
                if (row.outcome != .none) n += 1;
            }
            const outcomes = try gpa.alloc(PlayOutcome, n);
            n = 0;
            for (p.acc.rows, self.perms.items) |row, perm| {
                if (row.outcome == .none) continue;
                outcomes[n] = .{ .draw = perm.draw, .outcome = row.outcome, .count = row.count };
                n += 1;
            }
            r.* = .{
                .move = p.move,
                .points2 = p.acc.points2,
                .losses2 = p.acc.losses2,
                .spread = if (p.acc.spread_set) p.acc.spread else null,
                .cut = p.cut,
                .outcomes = outcomes,
            };
            done += 1;
        }
        return .{ .plays = out, .endgames = t.endgames };
    }

    const SortCtx = struct {
        moves: []const Move,
        eqs: []const f64,
        fn lessThan(c: SortCtx, a: usize, b: usize) bool {
            return equity.better(&c.moves[a], c.eqs[a], &c.moves[b], c.eqs[b]);
        }
    };

    fn byPoints(_: void, a: Play, b: Play) bool {
        return a.acc.points2 > b.acc.points2;
    }

    /// Whether the cutoffs stop play `p` now.
    fn cutNow(self: *const Solver, p: *const Play) bool {
        if (self.avoidPrune(&p.move)) return false;
        if (self.opts.skip_loss and p.acc.losses2 > 0) return true;
        return self.opts.early_cutoff and p.acc.losses2 > self.min_potential2;
    }

    fn solvePlay(self: *Solver, p: *Play) Error!void {
        if (self.cutNow(p)) {
            p.cut = true;
            return;
        }
        const tp = if (p.move.kind == .play) p.move.tiles_played else 0;
        const left = @as(i32, self.num_in_bag) - @as(i32, tp);
        if (self.opts.max_tiles_left >= 0 and left > self.opts.max_tiles_left and self.opts.only.len == 0) return;
        const empties = tp >= self.num_in_bag;
        const t = self.thread;
        for (self.perms.items, 0..) |*perm, idx| {
            if (self.opts.early_cutoff and !self.avoidPrune(&p.move) and p.acc.losses2 > self.min_potential2) {
                p.cut = true;
                return;
            }
            self.rootNode(perm, &t.nodes[0]);
            try self.recurse(t, &p.acc, 0, &p.move, idx, perm.count, empties, false, 0, true);
        }
        p.acc.finalize();
    }

    /// Plays `m` at the node of `level` and values what follows for
    /// ordering `idx` (count `ct`) into `acc` (recursiveSolve). `empties`:
    /// the play being solved empties the bag (its outcomes are final at
    /// once); `full`: the tiebreak's full-window search, which adds up
    /// spread; `top`: a play of the root (whose emptying endgames feed the
    /// early cutoff).
    fn recurse(self: *Solver, t: *Thread, acc: *Acc, level: usize, m: *const Move, idx: usize, ct: u32, empties: bool, full: bool, nested: u32, top: bool) Error!void {
        if (!full and acc.hasLoss(idx)) return;
        const child = &t.nodes[level + 1];
        play(&t.nodes[level], child, m);
        if (child.over or child.bag.len == 0) {
            const final = try self.evaluate(t, child, full);
            if (full and top) {
                acc.spread += @as(i64, final) * ct;
                acc.spread_set = true;
                return;
            }
            const o: Outcome = if (final > 0) .win else if (final < 0) .loss else .draw;
            acc.record(o, idx, ct, empties);
            if (empties and top) {
                const bound = 2 * self.num_combos -| acc.points2;
                self.min_potential2 = @min(self.min_potential2, bound);
            }
            return;
        }
        if (child.us) {
            try self.ourTurn(t, acc, level + 1, idx, ct, empties, nested);
        } else {
            const replies = try t.genAll(child, level + 1);
            for (replies) |*r| {
                try self.recurse(t, acc, level + 1, r, idx, ct, empties, full, nested, top);
                if (!full and acc.hasLoss(idx)) break;
            }
        }
    }

    /// The final spread for us at `n`, whose bag is empty or whose game is
    /// over: the endgame's search for the player on turn.
    fn evaluate(self: *Solver, t: *Thread, n: *const Node, full: bool) Error!i32 {
        if (n.over) return n.spreadForUs();
        const pos = &t.eg_pos;
        pos.* = n.pos;
        pos.scoreless_turns = n.scoreless;
        pos.max_scoreless_turns = 2;
        pos.last_move = null;
        // Macondo's quick search: a fixed depth, the window (-1, 1) unless
        // the spread is wanted, its playouts' picks, a pass taking a ply and
        // no sure bounds.
        const opts: endgame.Options = .{
            .first_win = !full,
            .forced_pass = false,
            .sure_out = false,
            .precise_pure = false,
            .playout_out_bonus = false,
            .clear_tt = false,
            .pruned_kwg = false,
            .kwg = self.kwg,
            .complete_line = false,
        };
        const solver = if (t.eg) |eg| blk: {
            eg.reset(pos, opts);
            break :blk eg;
        } else blk: {
            t.eg = try endgame.Solver.initIn(self.gpa, pos, self.zobrist, self.tt, opts, &t.ws, t.index);
            break :blk t.eg.?;
        };
        const res = try solver.solve(self.plies, null, null);
        t.endgames += 1;
        const final = @as(i32, res.value) + pos.scores[0] - pos.scores[1];
        return if (n.us) final else -final;
    }

    /// Our turn at the node of `level`, with tiles in the bag
    /// (iterateOurReplies).
    fn ourTurn(self: *Solver, t: *Thread, acc: *Acc, level: usize, idx: usize, ct: u32, empties: bool, nested: u32) Error!void {
        const limit = self.opts.nested_depth_limit;
        if (limit >= 0 and nested + 1 > limit) {
            const plays = try t.genAll(&t.nodes[level], level);
            const best = self.bestPlay(&t.nodes[level], plays);
            return self.recurse(t, acc, level, &best, idx, ct, empties, false, nested, false);
        }
        const o = try self.nestedSolve(t, level, nested + 1);
        acc.record(o, idx, ct, empties);
    }

    /// Our best play at `n` by score plus leave value, ties going by
    /// equity.tiebreaksBetter (bestNestedPlay).
    fn bestPlay(self: *const Solver, n: *const Node, plays: []const Move) Move {
        _ = self;
        var best: usize = 0;
        var best_est: f64 = 0;
        for (plays, 0..) |*m, i| {
            const est = estimate(n, m);
            if (i == 0 or est > best_est or (est == best_est and equity.tiebreaksBetter(m, &plays[best]))) {
                best = i;
                best_est = est;
            }
        }
        return plays[best];
    }

    /// The nested pre-endgame at the node of `level`: our turn with tiles
    /// in the bag, solved across every ordering of what we can't see
    /// (nestedOurTurnSolve). Returns the verdict for the node's own bag.
    fn nestedSolve(self: *Solver, t: *Thread, level: usize, depth: u32) Error!Outcome {
        const gpa = self.gpa;
        const n = &t.nodes[level];
        const dim = n.pos.board.dim;
        var key: CacheKey = .{
            .board = self.zobrist.boardHash(n.pos.board.squares[0 .. dim * dim]),
            .ours = n.pos.racks[0].counts,
            .unseen = n.pos.racks[1].counts,
            .scoreless = n.scoreless,
        };
        for (n.bag.slice()) |ml| key.unseen[ml] += 1;
        if (self.cache.get(key)) |vs| {
            for (vs) |v| {
                if (v.draw.eql(&n.bag)) return v.outcome;
            }
        }

        var sub: std.ArrayList(Perm) = .empty;
        defer sub.deinit(gpa);
        try permutations(gpa, &key.unseen, n.bag.len, &sub);

        const ours = try gpa.dupe(Move, try t.genAll(n, level));
        defer gpa.free(ours);
        // Strongest first, so the leader's losses cut the others short.
        std.sort.pdq(Move, ours, n, struct {
            fn lt(node: *const Node, a: Move, b: Move) bool {
                return estimateTrunc(node, &a) > estimateTrunc(node, &b);
            }
        }.lt);

        const accs = try gpa.alloc(Acc, ours.len);
        var made: usize = 0;
        defer {
            for (accs[0..made]) |a| gpa.free(a.rows);
            gpa.free(accs);
        }
        var min_losses: u64 = std.math.maxInt(u64);
        const saved = n.*;
        defer n.* = saved;
        var early_win = false;
        for (ours, 0..) |*m, mi| {
            const rows = try gpa.alloc(Row, sub.items.len);
            @memset(rows, .{});
            accs[mi] = .{ .rows = rows };
            made += 1;
            const a = &accs[mi];
            const sub_empties = m.kind == .play and m.tiles_played >= saved.bag.len;
            for (sub.items, 0..) |*sp, pi| {
                n.* = saved;
                n.pos.racks[1] = .{};
                var opp = key.unseen;
                for (sp.draw.slice()) |ml| opp[ml] -= 1;
                for (opp, 0..) |c, ml| {
                    for (0..c) |_| n.pos.racks[1].add(@intCast(ml));
                }
                n.bag = sp.draw;
                try self.recurse(t, a, level, m, pi, sp.count, sub_empties, false, depth, false);
                if (a.losses2 > min_losses) break;
            }
            a.finalize();
            if (a.losses2 < min_losses) {
                min_losses = a.losses2;
                if (min_losses == 0) {
                    early_win = true;
                    break;
                }
            }
        }

        const vs = try gpa.alloc(Verdict, sub.items.len);
        errdefer gpa.free(vs);
        for (sub.items, vs, 0..) |*sp, *v, pi| {
            var worst: Outcome = .none;
            if (early_win) {
                worst = .win;
            } else for (accs[0..made]) |*a| {
                if (a.losses2 > min_losses) continue;
                worst = @fromBackingInt(@intCast(@max(@backingInt(worst), @backingInt(a.rows[pi].outcome))));
            }
            v.* = .{ .draw = sp.draw, .outcome = worst };
        }
        var verdict: Outcome = .none;
        for (vs) |v| {
            if (v.draw.eql(&saved.bag)) verdict = v.outcome;
        }
        const gop = try self.cache.getOrPut(gpa, key);
        if (gop.found_existing) gpa.free(gop.value_ptr.*);
        gop.value_ptr.* = vs;
        return verdict;
    }

    /// Breaks the tie among the plays level with the first on points that
    /// empty the bag by their spread (maybeTiebreak).
    fn tiebreak(self: *Solver, plays: []Play) Error!void {
        var i: usize = 0;
        while (i + 1 < plays.len and plays[i].acc.points2 == plays[i + 1].acc.points2) i += 1;
        if (i == 0) return;
        const num_winners = i + 1;
        var top: std.ArrayList(usize) = .empty;
        defer top.deinit(self.gpa);
        for (0..num_winners) |k| {
            if (self.emptiesBag(&plays[k].move)) try top.append(self.gpa, k);
        }
        if (top.items.len == 1) {
            std.mem.swap(Play, &plays[top.items[0]], &plays[0]);
            return;
        }
        if (top.items.len == 0) return;
        // The best by score first.
        std.sort.pdq(usize, top.items, plays, struct {
            fn lt(ps: []Play, a: usize, b: usize) bool {
                return equity.tiebreaksBetter(&ps[a].move, &ps[b].move);
            }
        }.lt);
        top.shrinkRetainingCapacity(@min(top.items.len, tiebreak_plays));
        for (0..num_winners) |k| {
            if (self.avoidPrune(&plays[k].move) and self.emptiesBag(&plays[k].move) and
                std.mem.indexOfScalar(usize, top.items, k) == null) try top.append(self.gpa, k);
        }
        const t = self.thread;
        for (top.items) |k| {
            const p = &plays[k];
            for (self.perms.items, 0..) |*perm, idx| {
                self.rootNode(perm, &t.nodes[0]);
                try self.recurse(t, &p.acc, 0, &p.move, idx, perm.count, true, true, 0, true);
            }
        }
        // By points, then those with a spread by spread (sortPlaysAfterTiebreak).
        std.sort.insertion(Play, plays, {}, struct {
            fn lt(_: void, a: Play, b: Play) bool {
                if (a.acc.points2 != b.acc.points2) return a.acc.points2 > b.acc.points2;
                if (a.acc.spread_set != b.acc.spread_set) return a.acc.spread_set;
                return a.acc.spread_set and a.acc.spread > b.acc.spread;
            }
        }.lt);
    }

    fn emptiesBag(self: *const Solver, m: *const Move) bool {
        return m.kind == .play and m.tiles_played >= self.num_in_bag;
    }
};

/// Score plus the value of the leave of the player on turn at `n`.
fn estimate(n: *const Node, m: *const Move) f64 {
    const leave = m.leave(&n.pos.racks[0]);
    var buf: [rack_tile_limit]MachineLetter = undefined;
    const k = leave.tiles(&buf);
    const lv = if (n.pos.leaves) |klv| klv.leaveValue(buf[0..k]) else 0;
    return @as(f64, @floatFromInt(m.score)) + lv;
}

/// The estimate as Go orders nested plays by it (truncated to an integer).
fn estimateTrunc(n: *const Node, m: *const Move) i32 {
    return @intFromFloat(estimate(n, m));
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

test "the orderings of two tiles from AAB" {
    var counts: [max_alpha]u8 = @splat(0);
    counts[1] = 2;
    counts[2] = 1;
    var out: std.ArrayList(Perm) = .empty;
    defer out.deinit(testing.allocator);
    try permutations(testing.allocator, &counts, 2, &out);
    // AA (2x1), AB (2x1), BA (1x2).
    try testing.expectEqual(3, out.items.len);
    try testing.expectEqualSlices(MachineLetter, &.{ 1, 1 }, out.items[0].draw.slice());
    try testing.expectEqual(2, out.items[0].count);
    try testing.expectEqualSlices(MachineLetter, &.{ 1, 2 }, out.items[1].draw.slice());
    try testing.expectEqual(2, out.items[1].count);
    try testing.expectEqualSlices(MachineLetter, &.{ 2, 1 }, out.items[2].draw.slice());
    try testing.expectEqual(2, out.items[2].count);
}

/// The plays of `moves` (as UCGI text) among every move of the player on
/// turn, with their scores.
fn findMoves(gpa: Allocator, p: *const Position, texts: []const []const u8) ![]Move {
    var pos = p.*;
    var gen: movegen.Generator = .init(gpa, &pos.board, p.lexicon.kwg, p.dist);
    defer gen.deinit();
    gen.gen_pass = true;
    var rack = pos.racks[0];
    try gen.genAll(&rack, false);
    const out = try gpa.alloc(Move, texts.len);
    errdefer gpa.free(out);
    for (texts, out) |text, *m| {
        const parsed = try move_mod.parse(text, p.dist, p.board.dim);
        for (gen.plays.items) |*g| {
            if (sameMove(g, &parsed.move)) {
                m.* = g.*;
                break;
            }
        } else return error.MoveNotFound;
    }
    return out;
}

const GoldenPlay = struct {
    move: Move,
    cut: bool = false,
    points2: u64 = 0,
    spread: ?i64 = null,
    outcomes: std.ArrayList(PlayOutcome) = .empty,
};

/// Solves a golden file's pre-endgame (see testdata/README.md) and checks
/// each play's results against the Go engine's. With `sample` (and without
/// `-Dpeg-full`), only every sample-th of Go's plays is solved: a play's
/// points and outcomes don't depend on the others solved, but which plays
/// the tiebreak takes does, so spreads and the best play are then left
/// out.
fn checkGolden(text: []const u8, sample_in: ?usize) !void {
    const sample = if (@import("build_options").peg_full) null else sample_in;
    const gpa = testing.allocator;
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();

    var lines = std.mem.splitScalar(u8, text, '\n');
    var p: Position = undefined;
    var opts: Options = .{};
    var only_texts: std.ArrayList([]const u8) = .empty;
    defer only_texts.deinit(gpa);
    var known: [rack_tile_limit]MachineLetter = undefined;
    var golden: std.ArrayList(GoldenPlay) = .empty;
    defer {
        for (golden.items) |*g| g.outcomes.deinit(gpa);
        golden.deinit(gpa);
    }
    var best_text: []const u8 = "";
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const kw = f.next().?;
        if (std.mem.eql(u8, kw, "cgp")) {
            p = try testdata.position(&store, line[4..]);
        } else if (std.mem.eql(u8, kw, "options")) {
            while (f.next()) |k| {
                const v = try std.fmt.parseInt(i32, f.next().?, 10);
                if (std.mem.eql(u8, k, "plies")) opts.plies = @intCast(v) else if (std.mem.eql(u8, k, "id")) opts.iterative_deepening = v == 1 else if (std.mem.eql(u8, k, "maxtilesleft")) opts.max_tiles_left = v else if (std.mem.eql(u8, k, "nested")) opts.nested_depth_limit = v else if (std.mem.eql(u8, k, "tiebreak")) opts.tiebreak = v == 1 else if (std.mem.eql(u8, k, "cutoff")) opts.early_cutoff = v == 1 else return error.BadGolden;
            }
        } else if (std.mem.eql(u8, kw, "only")) {
            while (f.next()) |m| try only_texts.append(gpa, m);
        } else if (std.mem.eql(u8, kw, "opprack")) {
            const n = try p.dist.toMachineLetters(f.next().?, &known);
            opts.known_opp = known[0..n];
        } else if (std.mem.eql(u8, kw, "play")) {
            const parsed = try move_mod.parse(f.next().?, p.dist, p.board.dim);
            var g: GoldenPlay = .{ .move = parsed.move };
            while (f.next()) |k| {
                if (std.mem.eql(u8, k, "cut")) {
                    g.cut = true;
                } else if (std.mem.eql(u8, k, "points")) {
                    const pts = try std.fmt.parseFloat(f64, f.next().?);
                    g.points2 = @intFromFloat(pts * 2);
                } else if (std.mem.eql(u8, k, "spread")) {
                    g.spread = try std.fmt.parseInt(i64, f.next().?, 10);
                } else return error.BadGolden;
            }
            try golden.append(gpa, g);
        } else if (std.mem.eql(u8, kw, "outcome")) {
            var d: Draw = .{};
            d.len = @intCast(try p.dist.toMachineLetters(f.next().?, &d.tiles));
            const o_text = f.next().?;
            const o: Outcome = if (std.mem.eql(u8, o_text, "win")) .win else if (std.mem.eql(u8, o_text, "draw")) .draw else if (std.mem.eql(u8, o_text, "loss")) .loss else return error.BadGolden;
            const ct = try std.fmt.parseInt(u32, f.next().?, 10);
            try golden.items[golden.items.len - 1].outcomes.append(gpa, .{ .draw = d, .outcome = o, .count = ct });
        } else if (std.mem.eql(u8, kw, "best")) {
            best_text = f.next().?;
        } else return error.BadGolden;
    }
    if (sample) |k| {
        std.debug.assert(only_texts.items.len == 0);
        var i: usize = 0;
        while (i < golden.items.len) : (i += k) {
            var w: std.Io.Writer = .fixed(try gpa.alloc(u8, 64));
            try golden.items[i].move.format(p.dist, &w);
            try only_texts.append(gpa, w.buffered());
        }
    }
    defer if (sample != null) for (only_texts.items) |t| gpa.free(t.ptr[0..64]);
    const only = try findMoves(gpa, &p, only_texts.items);
    defer gpa.free(only);
    opts.only = only;

    var z = try zobrist_mod.Zobrist.init(gpa, p.board.dim, @splat(1));
    defer z.deinit(gpa);
    var tt: ttable.Table = .{};
    defer tt.deinit(gpa);
    try tt.reset(gpa, 0);
    var s = try Solver.init(gpa, &p, &z, &tt, opts);
    defer s.deinit();
    var res = try s.solve();
    defer res.deinit(gpa);

    try testing.expectEqual(if (sample != null) only.len else golden.items.len, res.plays.len);
    var buf: [64]u8 = undefined;
    for (res.plays) |*r| {
        const g = for (golden.items) |*g| {
            if (sameMove(&g.move, &r.move)) break g;
        } else {
            std.debug.print("Zig solved a play Go didn't: {any}\n", .{r.move});
            return error.TestUnexpectedResult;
        };
        if (g.cut or r.cut) continue;
        var w: std.Io.Writer = .fixed(&buf);
        try r.move.format(p.dist, &w);
        const name = w.buffered();
        if (g.points2 != r.points2) {
            std.debug.print("{s}: points {d} in Go, {d} in Zig\n", .{ name, @as(f64, @floatFromInt(g.points2)) / 2, @as(f64, @floatFromInt(r.points2)) / 2 });
            return error.TestUnexpectedResult;
        }
        if (g.outcomes.items.len != r.outcomes.len) {
            std.debug.print("{s}: {d} outcomes in Go, {d} in Zig\n", .{ name, g.outcomes.items.len, r.outcomes.len });
            return error.TestUnexpectedResult;
        }
        for (g.outcomes.items) |go| {
            for (r.outcomes) |zo| {
                if (!zo.draw.eql(&go.draw)) continue;
                if (zo.outcome != go.outcome or zo.count != go.count) {
                    std.debug.print("{s}, bag {any}: {s} x{d} in Go, {s} x{d} in Zig\n", .{ name, go.draw.slice(), go.outcome.name(), go.count, zo.outcome.name(), zo.count });
                    return error.TestUnexpectedResult;
                }
                break;
            } else {
                std.debug.print("{s}: no outcome in Zig for bag {any}\n", .{ name, go.draw.slice() });
                return error.TestUnexpectedResult;
            }
        }
        if (sample == null and g.spread != r.spread) {
            std.debug.print("{s}: spread {?d} in Go, {?d} in Zig\n", .{ name, g.spread, r.spread });
            return error.TestUnexpectedResult;
        }
    }
    // The best play, unless Go's first two are level.
    const g0 = &golden.items[0];
    const level = golden.items.len > 1 and golden.items[1].points2 == g0.points2 and
        (golden.items[1].spread orelse std.math.minInt(i64)) == (g0.spread orelse std.math.minInt(i64));
    if (sample == null and !level and !g0.cut and !res.plays[0].cut) {
        const want = try move_mod.parse(best_text, p.dist, p.board.dim);
        try testing.expect(sameMove(&want.move, &res.plays[0].move));
    }
}

test "golden: two in the bag, one play" {
    try checkGolden(@embedFile("testdata/two-in-bag-x.txt"), null);
}

test "golden: two in the bag, past the nested limit" {
    try checkGolden(@embedFile("testdata/two-in-bag-capped.txt"), null);
}

test "golden: a known opponent rack" {
    try checkGolden(@embedFile("testdata/known-opp-rack.txt"), null);
}

test "golden: the early cutoff" {
    try checkGolden(@embedFile("testdata/polish-cutoff.txt"), null);
}

test "golden: a pass wins" {
    try checkGolden(@embedFile("testdata/pass-fra20.txt"), 4);
}

test "golden: one in the bag" {
    try checkGolden(@embedFile("testdata/straightforward.txt"), 20);
}

test "golden: one in the bag, iterative deepening to 3 plies" {
    try checkGolden(@embedFile("testdata/straightforward-id3.txt"), 20);
}
