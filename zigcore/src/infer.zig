//! Inference: what the opponent kept after their last play. Port of
//! Macondo's rangefinder as the elite bot runs it: for each candidate
//! leave, a mini-sim of the position before the play (the opponent's top
//! plays by static equity plus the one they made) gives the likelihood of
//! their play under a softmax over the plays' log-odds of winning; the
//! leaves measured (all of them when there are few, a sample from the
//! prior otherwise) give a complete posterior over every leave, the
//! unmeasured ones imputed from how sub-leaves lifted the likelihood; then
//! rounds of measurement drawn from that posterior refine it. An exchange
//! is read from the leave values alone.
//!
//! The leaves are dealt as Go's single-threaded rangefinder deals them
//! (the same game copy, the same random source), in batches whose
//! mini-sims run on as many threads as there are, and are recorded in the
//! order dealt: the posterior is the one Go's rangefinder reads on one
//! thread, on any number of threads. Only the budget mode (a fixed number
//! of mini-sims) is deterministic, and only it is ported.

const std = @import("std");
const tiles = @import("tiles.zig");
const position_mod = @import("position.zig");
const game_mod = @import("game.zig");
const move_mod = @import("move.zig");
const sim_mod = @import("sim.zig");
const classic = @import("classic.zig");
const gomath = @import("gomath.zig");
const gosort = @import("gosort.zig");
const gorand = @import("gorand.zig");
const ucgi = @import("ucgi.zig");
const Allocator = std.mem.Allocator;
const MachineLetter = tiles.MachineLetter;
const Move = move_mod.Move;
const Game = game_mod.Game;
const Position = position_mod.Position;

const rack_size = game_mod.rack_tile_limit;
const empty4 = [4][]f64{ &.{}, &.{}, &.{}, &.{} };

pub const Error = error{
    /// The position has no last move (CGP lm).
    NoLastMove,
    BagEmpty,
    /// A pass or a phony says nothing about the rack.
    MoveNotSupported,
    /// A bingo leaves nothing to infer.
    NoInformation,
} || Allocator.Error;

pub const Options = struct {
    /// Mini-sims in all (the elite bot's paired budget).
    budget: u32 = 200,
    /// A mini-sim's iteration cap, plays and plies.
    sim_iters: u64 = 200,
    mini_plays: usize = 10,
    mini_plies: u32 = 2,
    /// A fixed temperature; 0 for the fitted schedule by bag size.
    tau: f64 = 0,
    /// The most the temperature can be (the elite bot's 0.1); 0 for none.
    tau_cap: f64 = 0.1,
    /// Refinement rounds after the first measurements.
    max_rounds: u32 = 6,
    /// Leaves above which the posterior is sampled rather than every leave
    /// measured; 0 for the budget.
    max_enumerated: u32 = 0,
    /// Mini-sims run at once.
    threads: usize = 1,
};

/// A leave and its posterior weight (relative to the heaviest, 1).
pub const Leave = struct {
    tiles: [rack_size]MachineLetter = undefined,
    len: u8 = 0,
    weight: f64 = 0,

    pub fn slice(self: *const Leave) []const MachineLetter {
        return self.tiles[0..self.len];
    }
};

/// One leave a mini-sim measured, in the order measured.
pub const Draw = struct {
    round: u32,
    leave: Leave,
    /// P(their play | this leave).
    likelihood: f64,
};

pub const Result = struct {
    /// Every leave the opponent could have kept, with its weight, in
    /// Macondo's order (by first tile, then the rest).
    leaves: []Leave,
    /// The measurements (the rangefinder's trace).
    draws: []Draw,
    /// Tiles kept.
    rack_length: usize,
    /// The temperature used.
    tau: f64,
    /// Mini-sims run.
    sims: u64,
    /// Leaves measured by a mini-sim (the rest imputed).
    measured: usize,
    /// Whether every leave was listed exhaustively (few enough to measure
    /// them all).
    enumerated: bool,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.leaves);
        gpa.free(self.draws);
        self.* = undefined;
    }
};

/// Reads the opponent's leave from `p` (the position after their move,
/// CGP lm) with the player on turn holding `p.racks[0]`. `live_seed` and
/// `live_turn` seed it as the game's seed and turn number seed Macondo's.
/// The mini-sims run on `io`'s threads (on this one without it).
pub fn infer(gpa: Allocator, io: ?std.Io, p: *const Position, live_seed: [32]u8, live_turn: u64, win_table: *const sim_mod.WinTable, opts: Options) Error!Result {
    var f: Finder = undefined;
    try f.init(gpa, p, live_seed, live_turn, win_table, opts);
    f.io = io;
    defer f.deinit();
    try f.run();
    return try f.result();
}

// The rangefinder's constants.
const max_alpha = tiles.max_alphabet_size;
const logit_eps = 1e-6;
const default_tau = 0.05;
const one_tile_tau_scale = 2.0;
const one_tile_tau_min_bag = 21;
const default_max_enumerated = 750;
const enumerate_always_full = default_max_enumerated;
const enumerate_early_exit_eps = 0.01;
const inference_seed_tag: u64 = 0x494e464552; // "INFER"
const refine_exploration_lambda = 1.0;
const refine_stage0_frac = 0.4;
const refine_converged_tol = 0.10;
const refine_mass_covered = 0.02;
const refine_min_batch = 16;
const imputation_lambda = 10.0;
const value_first_lambda = 30;
const value_term_min_leave = 6;
const max_abs_log_lift = 13.8;
const max_abs_interaction = 2.302585;
const negligible_weight_fraction = 1e-15;
const calibration_folds = 5;
const exchange_value_beta = 0.072;

const fitted_tau_knots = [_]struct { bag: i32, tau: f64 }{
    .{ .bag = 2, .tau = 2.5 },    .{ .bag = 7, .tau = 0.45 },   .{ .bag = 14, .tau = 0.14 },
    .{ .bag = 20, .tau = 0.075 }, .{ .bag = 30, .tau = 0.047 }, .{ .bag = 45, .tau = 0.034 },
    .{ .bag = 60, .tau = 0.029 }, .{ .bag = 86, .tau = 0.021 },
};

fn tauFittedForBag(bag: i32) f64 {
    const k = fitted_tau_knots;
    if (bag <= k[0].bag) return k[0].tau;
    for (1..k.len) |i| {
        if (bag <= k[i].bag) {
            const w = @as(f64, @floatFromInt(bag - k[i - 1].bag)) / @as(f64, @floatFromInt(k[i].bag - k[i - 1].bag));
            return gomath.exp((1 - w) * gomath.log(k[i - 1].tau) + w * gomath.log(k[i].tau));
        }
    }
    return k[k.len - 1].tau;
}

/// A run of equal tiles in a sorted leave.
const Run = struct { t: MachineLetter, c: u8 };

const RunBuf = struct {
    runs: [rack_size]Run = undefined,
    n: usize = 0,

    fn of(sorted: []const MachineLetter) RunBuf {
        var b: RunBuf = .{};
        for (sorted) |t| {
            if (b.n > 0 and b.runs[b.n - 1].t == t) {
                b.runs[b.n - 1].c += 1;
            } else {
                b.runs[b.n] = .{ .t = t, .c = 1 };
                b.n += 1;
            }
        }
        return b;
    }

    fn slice(self: *const RunBuf) []const Run {
        return self.runs[0..self.n];
    }
};

/// The sub-leaves of a leave, up to order 3, in the rangefinder's order:
/// calls `f.visit(order, index)` for each.
fn walkSubleaves(runs: []const Run, a: usize, max_order: u8, f: anytype) void {
    for (runs) |r| f.visit(1, r.t);
    if (max_order >= 2) {
        for (runs, 0..) |r, i| {
            if (r.c >= 2) f.visit(2, @as(usize, r.t) * a + r.t);
            for (runs[i + 1 ..]) |r2| f.visit(2, @as(usize, r.t) * a + r2.t);
        }
    }
    if (max_order >= 3) {
        for (runs, 0..) |r, i| {
            const t: usize = r.t;
            if (r.c >= 3) f.visit(3, (t * a + t) * a + t);
            for (i + 1..runs.len) |j| {
                const r2 = runs[j];
                const u: usize = r2.t;
                if (r.c >= 2) f.visit(3, (t * a + t) * a + u);
                if (r2.c >= 2) f.visit(3, (t * a + u) * a + u);
                for (runs[j + 1 ..]) |r3| f.visit(3, (t * a + u) * a + r3.t);
            }
        }
    }
}

/// Containment marginals of the measured likelihoods by sub-leave: for
/// each sub-leave, the importance weight of the draws containing it, its
/// square, and the weighted likelihood (subleaveAccumulator).
const Accumulator = struct {
    a: usize,
    max_order: u8,
    n: usize = 0,
    wt_total: f64 = 0,
    wtsq_total: f64 = 0,
    lik_total: f64 = 0,
    /// By order (index 1..3): wt, wtsq, lik.
    wt: [4][]f64,
    wtsq: [4][]f64,
    lik: [4][]f64,

    fn init(gpa: Allocator, a: usize, max_order: u8) Allocator.Error!Accumulator {
        std.debug.assert(max_order >= 1 and max_order <= 3);
        var acc: Accumulator = .{ .a = a, .max_order = max_order, .wt = empty4, .wtsq = empty4, .lik = empty4 };
        errdefer acc.deinit(gpa);
        var size: usize = 1;
        for (1..@as(usize, max_order) + 1) |o| {
            size *= a;
            acc.wt[o] = try gpa.alloc(f64, size);
            @memset(acc.wt[o], 0);
            acc.wtsq[o] = try gpa.alloc(f64, size);
            @memset(acc.wtsq[o], 0);
            acc.lik[o] = try gpa.alloc(f64, size);
            @memset(acc.lik[o], 0);
        }
        return acc;
    }

    fn deinit(self: *Accumulator, gpa: Allocator) void {
        for (1..4) |o| {
            if (self.wt[o].len > 0) gpa.free(self.wt[o]);
            if (self.wtsq[o].len > 0) gpa.free(self.wtsq[o]);
            if (self.lik[o].len > 0) gpa.free(self.lik[o]);
        }
    }

    fn record(self: *Accumulator, sorted: []const MachineLetter, w: f64, u: f64) void {
        self.n += 1;
        self.wt_total += u;
        self.wtsq_total += u * u;
        self.lik_total += u * w;
        const rb = RunBuf.of(sorted);
        const V = struct {
            acc: *Accumulator,
            u: f64,
            uu: f64,
            uw: f64,
            fn visit(v: @This(), order: u8, j: usize) void {
                v.acc.wt[order][j] += v.u;
                v.acc.wtsq[order][j] += v.uu;
                v.acc.lik[order][j] += v.uw;
            }
        };
        walkSubleaves(rb.slice(), self.a, self.max_order, V{ .acc = self, .u = u, .uu = u * u, .uw = u * w });
    }

    /// `self` without `sub`'s draws, clamped at zero.
    fn minus(self: *const Accumulator, gpa: Allocator, sub: *const Accumulator) Allocator.Error!Accumulator {
        var out = try Accumulator.init(gpa, self.a, self.max_order);
        out.n = self.n - sub.n;
        out.wt_total = goMax0(self.wt_total - sub.wt_total);
        out.wtsq_total = goMax0(self.wtsq_total - sub.wtsq_total);
        out.lik_total = goMax0(self.lik_total - sub.lik_total);
        for (1..@as(usize, self.max_order) + 1) |o| {
            subInto(out.wt[o], self.wt[o], sub.wt[o]);
            subInto(out.wtsq[o], self.wtsq[o], sub.wtsq[o]);
            subInto(out.lik[o], self.lik[o], sub.lik[o]);
        }
        return out;
    }

    fn subInto(dst: []f64, a: []const f64, b: []const f64) void {
        for (dst, a, b) |*d, x, y| {
            const v = x - y;
            if (v > 0) d.* = v;
        }
    }
};

fn goMax0(x: f64) f64 {
    return if (x > 0) x else 0;
}

fn ess(wt: f64, wtsq: f64) f64 {
    if (wtsq <= 0) return 0;
    return wt * wt / wtsq;
}

fn clampAbs(x: f64, bound: f64) f64 {
    if (x > bound) return bound;
    if (x < -bound) return -bound;
    return x;
}

/// The leave value term's inputs: the leave values of the static player.
const Valuer = struct {
    ctx: @import("equity.zig").Context,

    fn value(self: *const Valuer, runs: []const Run) f64 {
        var buf: [8]MachineLetter = undefined;
        var n: usize = 0;
        for (runs) |r| {
            var k: usize = 0;
            while (k < r.c and n < buf.len) : (k += 1) {
                buf[n] = r.t;
                n += 1;
            }
        }
        return self.ctx.leaveValue(buf[0..n]);
    }
};

/// Log-lifts of the likelihood by sub-leave (imputationModel), and the
/// leave-value slope when that term is on.
const Model = struct {
    a: usize,
    max_order: u8,
    phi: [4][]f64,
    unc: [4][]f64,
    valuer: ?*const Valuer = null,
    beta: f64 = 0,
    vbar: f64 = 0,
    marginals: bool = true,

    fn deinit(self: *Model, gpa: Allocator) void {
        for (1..4) |o| {
            if (self.phi[o].len > 0) gpa.free(self.phi[o]);
            if (self.unc[o].len > 0) gpa.free(self.unc[o]);
        }
    }

    fn logImputed(self: *const Model, runs: []const Run) f64 {
        const V = struct {
            m: *const Model,
            s: *f64,
            fn visit(v: @This(), order: u8, j: usize) void {
                v.s.* += v.m.phi[order][j];
            }
        };
        var s: f64 = 0;
        if (self.marginals) walkSubleaves(runs, self.a, self.max_order, V{ .m = self, .s = &s });
        if (self.valuer) |val| {
            if (self.beta != 0) s += self.beta * (val.value(runs) - self.vbar);
        }
        return s;
    }

    fn uncertainty(self: *const Model, runs: []const Run) f64 {
        const V = struct {
            m: *const Model,
            s: *f64,
            fn visit(v: @This(), order: u8, j: usize) void {
                const u = v.m.unc[order][j];
                v.s.* += u * u;
            }
        };
        var s: f64 = 0;
        walkSubleaves(runs, self.a, self.max_order, V{ .m = self, .s = &s });
        return @sqrt(s);
    }
};

/// buildImputationModel.
fn buildModel(gpa: Allocator, acc: *const Accumulator, lambda: f64, clamp_lift: f64, clamp_inter: f64) Allocator.Error!Model {
    const a = acc.a;
    var mod: Model = .{ .a = a, .max_order = acc.max_order, .phi = empty4, .unc = empty4 };
    errdefer mod.deinit(gpa);
    var size: usize = 1;
    for (1..@as(usize, acc.max_order) + 1) |o| {
        size *= a;
        mod.phi[o] = try gpa.alloc(f64, size);
        @memset(mod.phi[o], 0);
        mod.unc[o] = try gpa.alloc(f64, size);
        @memset(mod.unc[o], 0);
    }
    if (acc.n == 0 or acc.lik_total <= 0) return mod;

    const w_mean = acc.lik_total / acc.wt_total;
    const S = struct {
        lambda: f64,
        w_mean: f64,
        clamp_lift: f64,
        sig_sum: [5]f64 = @splat(0),
        sig_cnt: [5]f64 = @splat(0),
        fn shrink(s: *const @This(), c: f64) f64 {
            return c / (c + s.lambda);
        }
        fn rawLogLift(s: *const @This(), lik: f64, wt: f64) f64 {
            if (lik <= 0) return -s.clamp_lift;
            return clampAbs(gomath.log(lik / wt / s.w_mean), s.clamp_lift);
        }
        fn uncOf(s: *@This(), order: usize, raw: f64, e: f64) f64 {
            s.sig_sum[order] += raw * raw;
            s.sig_cnt[order] += 1;
            return (1 - s.shrink(e)) * @abs(raw);
        }
    };
    var st: S = .{ .lambda = lambda, .w_mean = w_mean, .clamp_lift = clamp_lift };

    for (0..a) |t| {
        if (acc.wt[1][t] > 0) {
            const e = ess(acc.wt[1][t], acc.wtsq[1][t]);
            const raw = st.rawLogLift(acc.lik[1][t], acc.wt[1][t]);
            mod.phi[1][t] = st.shrink(e) * raw;
            mod.unc[1][t] = st.uncOf(1, raw, e);
        }
    }
    const phi1 = mod.phi[1];
    if (acc.max_order >= 2) {
        const phi2 = mod.phi[2];
        for (0..a) |t| {
            for (t..a) |u| {
                const j = t * a + u;
                if (acc.wt[2][j] == 0) continue;
                const e = ess(acc.wt[2][j], acc.wtsq[2][j]);
                const raw = st.rawLogLift(acc.lik[2][j], acc.wt[2][j]);
                var inter = raw - phi1[t];
                if (u != t) inter -= phi1[u];
                inter = clampAbs(inter, clamp_inter);
                phi2[j] = st.shrink(e) * inter;
                mod.unc[2][j] = st.uncOf(2, inter, e);
            }
        }
    }
    if (acc.max_order >= 3) {
        const phi2 = mod.phi[2];
        for (0..a) |t| {
            for (t..a) |u| {
                for (u..a) |v| {
                    const j = (t * a + u) * a + v;
                    if (acc.wt[3][j] == 0) continue;
                    const e = ess(acc.wt[3][j], acc.wtsq[3][j]);
                    const raw = st.rawLogLift(acc.lik[3][j], acc.wt[3][j]);
                    var inter: f64 = undefined;
                    if (t == u and u == v) {
                        inter = raw - phi1[t] - phi2[t * a + t];
                    } else if (t == u) {
                        inter = raw - phi1[t] - phi1[v] - phi2[t * a + t] - phi2[t * a + v];
                    } else if (u == v) {
                        inter = raw - phi1[t] - phi1[u] - phi2[u * a + u] - phi2[t * a + u];
                    } else {
                        inter = raw - phi1[t] - phi1[u] - phi1[v] - phi2[t * a + u] - phi2[t * a + v] - phi2[u * a + v];
                    }
                    inter = clampAbs(inter, clamp_inter);
                    mod.phi[3][j] = st.shrink(e) * inter;
                    mod.unc[3][j] = st.uncOf(3, inter, e);
                }
            }
        }
    }
    for (1..@as(usize, mod.max_order) + 1) |order| {
        var sigma0: f64 = 0.5;
        if (st.sig_cnt[order] > 0) sigma0 = @sqrt(st.sig_sum[order] / st.sig_cnt[order]);
        for (mod.unc[order], acc.wt[order]) |*un, w| {
            if (w == 0) un.* = sigma0;
        }
    }
    return mod;
}

/// What the mini-sims measured for one leave.
const Measured = struct {
    sum_w: f64 = 0,
    count: u32 = 0,
    sum_u: f64 = 0,
    round: u32 = 0,
    predicted: f64 = 0,

    fn mean(self: *const Measured) f64 {
        if (self.count == 0) return 0;
        return self.sum_w / @as(f64, @floatFromInt(self.count));
    }
};

/// A leave as a map key (its tiles, sorted).
const Key = struct {
    b: [rack_size]u8 = @splat(0),
    n: u8 = 0,

    fn of(sorted: []const MachineLetter) Key {
        var k: Key = .{ .n = @intCast(sorted.len) };
        @memcpy(k.b[0..sorted.len], sorted);
        return k;
    }

    fn bytes(self: *const Key) []const u8 {
        return self.b[0..self.n];
    }

    /// Go's string order.
    fn less(_: void, a: Key, b: Key) bool {
        return std.mem.order(u8, a.bytes(), b.bytes()) == .lt;
    }
};

fn foldForKey(key: []const u8, n_folds: u32) u32 {
    var h: u32 = 2166136261;
    for (key) |c| {
        h ^= c;
        h *%= 16777619;
    }
    return h % n_folds;
}

const MeasuredMap = std.AutoArrayHashMapUnmanaged(Key, Measured);

/// The measured leaves' keys in Go's string order.
fn sortedKeys(gpa: Allocator, measured: *const MeasuredMap) Allocator.Error![]Key {
    const keys = try gpa.dupe(Key, measured.keys());
    std.mem.sort(Key, keys, {}, Key.less);
    return keys;
}

const ValueMode = enum { off, first };

const Imputed = struct {
    leaves: []Leave = &.{},
    model: ?Model = null,
    log_calib: f64 = 0,
    max_log_w: f64 = -std.math.inf(f64),
    measured_leaves: usize = 0,

    fn deinit(self: *Imputed, gpa: Allocator) void {
        gpa.free(self.leaves);
        if (self.model) |*m| m.deinit(gpa);
        self.* = .{};
    }
};

/// One leave to measure: the game as the dealer left it (the opponent's
/// hypothetical rack dealt), and then its likelihood.
const Job = struct {
    game: Game,
    leave: Leave,
    likelihood: f64 = 0,
};

/// Leaves dealt and measured at a time.
const batch_size = 64;

const Finder = struct {
    gpa: Allocator,
    io: ?std.Io = null,
    opts: Options,
    win_table: *const sim_mod.WinTable,
    /// The position before the opponent's move, them on turn.
    before: Position,
    /// The game the leaves are dealt on (Go's thread 0 game copy).
    dealer: Game,
    seed: [32]u8,
    target: Move,
    is_exchange: bool,
    /// The tiles the opponent played (blanks as blanks), sorted.
    played: [rack_size]MachineLetter = undefined,
    n_played: usize = 0,
    k: usize,
    bag_map: [max_alpha]u8 = @splat(0),
    alpha: usize,
    tau: f64,
    valuer: Valuer,
    budget: u32,

    acc: ?Accumulator = null,
    fold_accs: [calibration_folds]Accumulator = undefined,
    measured: MeasuredMap = .empty,
    imputed: Imputed = .{},
    leaves: []Leave = &.{},
    complete: bool = false,
    enumerated: bool = false,
    sims: u64 = 0,
    current_round: u32 = 0,
    draws: std.ArrayList(Draw) = .empty,

    fn init(self: *Finder, gpa: Allocator, p: *const Position, live_seed: [32]u8, live_turn: u64, win_table: *const sim_mod.WinTable, opts: Options) Error!void {
        const lm = p.last_move orelse return error.NoLastMove;
        const unseen = p.unseenBy(0);
        const live_bag = if (p.racks[1].num_tiles > 0) unseen - @as(i32, p.racks[1].num_tiles) else @max(0, unseen - rack_size);
        if (live_bag <= 0) return error.BagEmpty;
        if (lm.phony or lm.move.kind == .pass) return error.MoveNotSupported;
        if (lm.move.tiles_played == rack_size) return error.NoInformation;

        const before = p.beforeLastMove().?;
        var target = lm.move;
        target.score = lm.score;
        self.* = .{
            .gpa = gpa,
            .opts = opts,
            .win_table = win_table,
            .before = before,
            .dealer = undefined,
            .seed = undefined,
            .target = target,
            .is_exchange = target.kind == .exchange,
            .k = 0,
            .alpha = p.dist.size,
            .tau = 0,
            .valuer = .{ .ctx = before.equityContext() },
            .budget = opts.budget,
        };

        // The seed: the game's, its turn and a tag (deriveSeed).
        var buf: [48]u8 = undefined;
        buf[0..32].* = live_seed;
        std.mem.writeInt(u64, buf[32..40], live_turn, .big);
        std.mem.writeInt(u64, buf[40..48], inference_seed_tag, .big);
        std.crypto.hash.sha2.Sha256.hash(&buf, &self.seed, .{});

        // The game copy: the position before the move, the opponent on turn
        // with their rack thrown in, ours kept, the bag in letter order and
        // shuffled from the seed.
        var g = Game.fromPosition(&before, self.seed) catch return error.NoInformation;
        g.bag.shuffle(&g.rng);
        for (g.bag.tiles[0..g.bag.len]) |ml| self.bag_map[ml] += 1;
        if (target.kind == .play) {
            for (target.tileSlice()) |ml| {
                if (ml == 0) continue;
                self.played[self.n_played] = if (tiles.isBlanked(ml)) 0 else ml;
                self.n_played += 1;
            }
            std.mem.sort(MachineLetter, self.played[0..self.n_played], {}, std.sort.asc(MachineLetter));
            var extra: [rack_size]MachineLetter = undefined;
            _ = g.setRandomRack(0, self.played[0..self.n_played], &extra) catch return error.NoInformation;
            for (self.played[0..self.n_played]) |ml| self.bag_map[ml] -= 1;
        } else {
            self.n_played = target.len;
            var extra: [rack_size]MachineLetter = undefined;
            _ = g.setRandomRack(0, &.{}, &extra) catch unreachable;
        }
        self.k = rack_size - self.n_played;
        // Each thread's copy draws from the seed afresh (SeedBag).
        g.rng = .init(self.seed);
        self.dealer = g;

        // The temperature: the fitted schedule by bag size (or a fixed one),
        // doubled for a one-tile play early, capped.
        const phase = tauFittedForBag(live_bag);
        var scale: f64 = 1;
        if (opts.tau == 0 and self.k == rack_size - 1 and !self.is_exchange and live_bag >= one_tile_tau_min_bag) scale = one_tile_tau_scale;
        var tau = scale * (if (opts.tau != 0) opts.tau else if (phase != 0) phase else default_tau);
        if (opts.tau_cap > 0 and tau > opts.tau_cap) tau = opts.tau_cap;
        self.tau = tau;
    }

    fn deinit(self: *Finder) void {
        const gpa = self.gpa;
        if (self.acc) |*a| {
            a.deinit(gpa);
            for (&self.fold_accs) |*f| f.deinit(gpa);
        }
        self.measured.deinit(gpa);
        self.imputed.deinit(gpa);
        gpa.free(self.leaves);
        self.draws.deinit(gpa);
    }

    fn result(self: *Finder) Allocator.Error!Result {
        const draws = try self.draws.toOwnedSlice(self.gpa);
        const leaves = self.leaves;
        self.leaves = &.{};
        return .{
            .leaves = leaves,
            .draws = draws,
            .rack_length = self.k,
            .tau = self.tau,
            .sims = self.sims,
            .measured = self.imputed.measured_leaves,
            .enumerated = self.enumerated,
        };
    }

    fn run(self: *Finder) Error!void {
        if (self.is_exchange) return self.exchangeByValue();
        if (self.k == 0) return;
        const limit: u64 = if (self.opts.max_enumerated > 0) self.opts.max_enumerated else if (self.budget > 0) self.budget else default_max_enumerated;
        if (countMultisets(&self.bag_map, self.alpha, self.k) <= limit) return self.inferEnumerated();
        try self.initImputation();
        // Round 0: leaves drawn from the prior.
        const jobs = try self.gpa.alloc(Job, batch_size);
        defer self.gpa.free(jobs);
        var left = self.stage0Budget();
        while (left > 0) {
            const batch = jobs[0..@min(left, batch_size)];
            for (batch) |*j| {
                var extra: [rack_size]MachineLetter = undefined;
                const n = self.dealer.setRandomRack(0, self.played[0..self.n_played], &extra) catch unreachable;
                std.mem.sort(MachineLetter, extra[0..n], {}, std.sort.asc(MachineLetter));
                j.* = .{ .game = self.dealer, .leave = .{ .len = @intCast(n) } };
                @memcpy(j.leave.tiles[0..n], extra[0..n]);
            }
            try self.measureAll(batch);
            for (batch) |*j| try self.recordSample(j.leave.slice(), j.likelihood, 1);
            left -= @intCast(batch.len);
        }
        try self.finalize();
        try self.refineRounds();
    }

    fn stage0Budget(self: *const Finder) u32 {
        if (self.budget == 0) return 0;
        if (self.acc == null or self.opts.max_rounds == 0) return self.budget;
        const n: u32 = @intFromFloat(@as(f64, @floatFromInt(self.budget)) * refine_stage0_frac);
        return @max(n, 1);
    }

    fn initImputation(self: *Finder) Allocator.Error!void {
        var order: u8 = @intCast((self.k + 1) / 2);
        order = std.math.clamp(order, 1, 3);
        self.acc = try .init(self.gpa, self.alpha, order);
        for (&self.fold_accs) |*f| f.* = try .init(self.gpa, self.alpha, order);
    }

    /// The likelihood of the opponent's play with the rack dealt in `g`: a
    /// mini-sim of their top plays and the one they made, then a softmax
    /// over the log-odds of winning (GenAndSim, softmaxLikelihood).
    fn measure(self: *const Finder, g: *const Game) Allocator.Error!f64 {
        const gpa = self.gpa;
        const ranked = try ucgi.rankedMoves(gpa, &g.pos);
        defer gpa.free(ranked);
        const n_top = @min(self.opts.mini_plays, ranked.len);
        var moves = try gpa.alloc(Move, n_top + 1);
        defer gpa.free(moves);
        for (moves[0..n_top], ranked[0..n_top]) |*m, r| m.* = r.m;
        var n = n_top;
        const transpose = g.pos.board.tiles_played == 0;
        var present = false;
        for (moves[0..n_top]) |*m| {
            if (equals(m, &self.target, transpose)) {
                present = true;
                break;
            }
        }
        if (!present) {
            moves[n] = self.target;
            n += 1;
        }
        const sim = try sim_mod.Sim.create(gpa, g, self.seed, 0, moves[0..n], self.opts.mini_plies, self.win_table, 1);
        defer sim.destroy();
        try classic.run(sim, self.opts.sim_iters, .{});
        const order = try gpa.alloc(*const sim_mod.Play, sim.plays.len);
        defer gpa.free(order);
        sim.ranked(order);
        return softmaxLikelihood(order, &self.target, transpose, self.tau);
    }

    fn recordSample(self: *Finder, sorted: []const MachineLetter, w: f64, u: f64) Allocator.Error!void {
        var l: Leave = .{ .len = @intCast(sorted.len) };
        @memcpy(l.tiles[0..sorted.len], sorted);
        try self.draws.append(self.gpa, .{ .round = self.current_round, .leave = l, .likelihood = w });
        const acc = &self.acc.?;
        acc.record(sorted, w, u);
        const key = Key.of(sorted);
        self.fold_accs[foldForKey(key.bytes(), calibration_folds)].record(sorted, w, u);
        const gop = try self.measured.getOrPut(self.gpa, key);
        if (!gop.found_existing) gop.value_ptr.* = .{ .round = self.current_round };
        gop.value_ptr.sum_w += w;
        gop.value_ptr.count += 1;
        gop.value_ptr.sum_u += u;
    }

    /// Deals `job.leave` (sorted) with the tiles played and keeps the game
    /// for its mini-sim.
    fn dealLeave(self: *Finder, job: *Job) void {
        const leave = job.leave.slice();
        var full: [rack_size]MachineLetter = undefined;
        @memcpy(full[0..self.n_played], self.played[0..self.n_played]);
        @memcpy(full[self.n_played..][0..leave.len], leave);
        var extra: [rack_size]MachineLetter = undefined;
        _ = self.dealer.setRandomRack(0, full[0 .. self.n_played + leave.len], &extra) catch unreachable;
        job.game = self.dealer;
    }

    /// Runs the jobs' mini-sims, on the io's threads when there are several.
    fn measureAll(self: *Finder, jobs: []Job) Allocator.Error!void {
        self.sims += jobs.len;
        const threads = @min(self.opts.threads, jobs.len);
        const io = self.io orelse {
            for (jobs) |*j| j.likelihood = try self.measure(&j.game);
            return;
        };
        if (threads <= 1) {
            for (jobs) |*j| j.likelihood = try self.measure(&j.game);
            return;
        }
        var next: std.atomic.Value(usize) = .init(0);
        var failed: std.atomic.Value(bool) = .init(false);
        const W = struct {
            fn go(f: *const Finder, js: []Job, nx: *std.atomic.Value(usize), fl: *std.atomic.Value(bool)) void {
                while (!fl.load(.monotonic)) {
                    const i = nx.fetchAdd(1, .monotonic);
                    if (i >= js.len) return;
                    js[i].likelihood = f.measure(&js[i].game) catch {
                        fl.store(true, .monotonic);
                        return;
                    };
                }
            }
        };
        var group: std.Io.Group = .init;
        for (1..threads) |_| group.concurrent(io, W.go, .{ self, jobs, &next, &failed }) catch break;
        W.go(self, jobs, &next, &failed);
        group.await(io) catch {};
        if (failed.load(.monotonic)) return error.OutOfMemory;
    }

    fn inferEnumerated(self: *Finder) Error!void {
        const gpa = self.gpa;
        self.enumerated = true;
        var list: std.ArrayList(EnumLeaf) = .empty;
        defer list.deinit(gpa);
        try enumerateLeaves(gpa, &self.bag_map, self.alpha, self.k, &list);
        if (list.items.len == 0) return;
        var leaves = list.items;
        gosort.slice(EnumLeaf, leaves, {}, struct {
            fn higher(_: void, a: EnumLeaf, b: EnumLeaf) bool {
                return a.prior > b.prior;
            }
        }.higher);
        if (leaves.len > enumerate_always_full) {
            var total: f64 = 0;
            for (leaves) |l| total += l.prior;
            const keep_threshold = total * (1 - enumerate_early_exit_eps);
            var cum: f64 = 0;
            var keep = leaves.len;
            for (leaves, 0..) |l, i| {
                cum += l.prior;
                if (cum >= keep_threshold) {
                    keep = i + 1;
                    break;
                }
            }
            leaves = leaves[0..keep];
        }
        if (self.budget > 0 and leaves.len > self.budget) leaves = leaves[0..self.budget];
        try self.initImputation();
        const jobs = try gpa.alloc(Job, batch_size);
        defer gpa.free(jobs);
        var at: usize = 0;
        while (at < leaves.len) {
            const batch = jobs[0..@min(leaves.len - at, batch_size)];
            for (batch, leaves[at..][0..batch.len]) |*j, *l| {
                j.leave = l.leave;
                self.dealLeave(j);
            }
            try self.measureAll(batch);
            for (batch) |*j| try self.recordSample(j.leave.slice(), j.likelihood, 1);
            at += batch.len;
        }
        try self.finalize();
        try self.refineRounds();
    }

    /// The complete posterior from what was measured (finalizePlacementPosterior).
    fn finalize(self: *Finder) Allocator.Error!void {
        const acc = &(self.acc orelse return);
        if (acc.n == 0) return;
        const imp = try self.impute();
        if (imp.leaves.len == 0) {
            // Nothing heavy enough: keep the last posterior's leaves.
            self.imputed.deinit(self.gpa);
            self.imputed = imp;
            return;
        }
        self.imputed.deinit(self.gpa);
        self.gpa.free(self.leaves);
        self.leaves = try self.gpa.dupe(Leave, imp.leaves);
        self.imputed = imp;
        self.complete = true;
    }

    fn valueMode(self: *const Finder) ValueMode {
        return if (self.k >= value_term_min_leave) .first else .off;
    }

    /// fitValueTerm with only the value (no marginals): the slope of the
    /// measured log-likelihoods on the leave values.
    fn fitValue(self: *Finder, keys: []const Key, excl: ?u32) struct { beta: f64, vbar: f64 } {
        var sum_u: f64 = 0;
        var sum_uv: f64 = 0;
        var sum_ur: f64 = 0;
        const Pt = struct { u: f64, v: f64, r: f64 };
        var pts: std.ArrayList(Pt) = .empty;
        defer pts.deinit(self.gpa);
        for (keys) |key| {
            if (excl) |e| if (foldForKey(key.bytes(), calibration_folds) == e) continue;
            const ml = self.measured.getPtr(key).?;
            const w = ml.mean();
            if (w <= 0 or ml.sum_u <= 0) continue;
            const rb = RunBuf.of(key.bytes());
            const v = self.valuer.value(rb.slice());
            const r = gomath.log(w) - 0;
            pts.append(self.gpa, .{ .u = ml.sum_u, .v = v, .r = r }) catch return .{ .beta = 0, .vbar = 0 };
            sum_u += ml.sum_u;
            sum_uv += ml.sum_u * v;
            sum_ur += ml.sum_u * r;
        }
        if (pts.items.len < 3 or sum_u <= 0) return .{ .beta = 0, .vbar = 0 };
        const vbar = sum_uv / sum_u;
        const rbar = sum_ur / sum_u;
        var sxy: f64 = 0;
        var sxx: f64 = 0;
        for (pts.items) |p| {
            sxy += p.u * (p.v - vbar) * (p.r - rbar);
            sxx += p.u * (p.v - vbar) * (p.v - vbar);
        }
        if (sxx <= 0) return .{ .beta = 0, .vbar = 0 };
        return .{ .beta = sxy / sxx, .vbar = vbar };
    }

    /// valueFirstModel: the value slope first, then the marginals of what
    /// it leaves unexplained.
    fn valueFirstModel(self: *Finder, keys: []const Key, lambda: f64, excl: ?u32) Allocator.Error!Model {
        const fit = self.fitValue(keys, excl);
        var acc = try Accumulator.init(self.gpa, self.alpha, self.acc.?.max_order);
        defer acc.deinit(self.gpa);
        for (keys) |key| {
            if (excl) |e| if (foldForKey(key.bytes(), calibration_folds) == e) continue;
            const ml = self.measured.getPtr(key).?;
            if (ml.sum_u <= 0) continue;
            const rb = RunBuf.of(key.bytes());
            const adjust = gomath.exp(fit.beta * (self.valuer.value(rb.slice()) - fit.vbar));
            acc.record(key.bytes(), ml.mean() / adjust, ml.sum_u);
        }
        var mod = try buildModel(self.gpa, &acc, lambda, max_abs_log_lift, max_abs_interaction);
        mod.valuer = &self.valuer;
        mod.beta = fit.beta;
        mod.vbar = fit.vbar;
        mod.marginals = true;
        return mod;
    }

    /// imputeFullPosteriorTuned with the elite bot's tuning.
    fn impute(self: *Finder) Allocator.Error!Imputed {
        const gpa = self.gpa;
        const acc = &self.acc.?;
        const mode = self.valueMode();
        const lambda: f64 = if (mode == .first) value_first_lambda else imputation_lambda;
        const keys = try sortedKeys(gpa, &self.measured);
        defer gpa.free(keys);

        var mod = if (mode == .first)
            try self.valueFirstModel(keys, lambda, null)
        else
            try buildModel(gpa, acc, lambda, max_abs_log_lift, max_abs_interaction);
        errdefer mod.deinit(gpa);
        var folds: [calibration_folds]Model = undefined;
        var made: usize = 0;
        defer for (folds[0..made]) |*fm| fm.deinit(gpa);
        for (&self.fold_accs, 0..) |*fa, i| {
            if (mode == .first) {
                folds[i] = try self.valueFirstModel(keys, lambda, @intCast(i));
            } else {
                var rest = try acc.minus(gpa, fa);
                defer rest.deinit(gpa);
                folds[i] = try buildModel(gpa, &rest, lambda, max_abs_log_lift, max_abs_interaction);
            }
            made += 1;
        }
        const log_calib = self.calibrate(keys, &mod, folds[0..made]);

        var n_total: usize = 0;
        for (self.bag_map[0..self.alpha]) |c| n_total += c;
        var res: Imputed = .{ .model = mod, .log_calib = log_calib };
        if (self.k == 0 or n_total < self.k) return res;
        const log_choose_nk = logBinomial(n_total, self.k);

        const Scored = struct { leave: Leave, log_w: f64, measured: bool };
        var scored: std.ArrayList(Scored) = .empty;
        defer scored.deinit(gpa);
        const Visitor = struct {
            f: *Finder,
            mod: *const Model,
            log_calib: f64,
            log_choose_nk: f64,
            out: *std.ArrayList(Scored),
            fn visit(v: *const @This(), buf: []const MachineLetter, log_num: f64) Allocator.Error!void {
                const log_prior = log_num - v.log_choose_nk;
                var log_w: f64 = undefined;
                var is_measured = false;
                if (v.f.measured.getPtr(Key.of(buf))) |ml| {
                    is_measured = true;
                    const w = ml.mean();
                    if (w <= 0) return;
                    log_w = log_prior + gomath.log(w);
                } else {
                    const rb = RunBuf.of(buf);
                    log_w = log_prior + v.log_calib + v.mod.logImputed(rb.slice());
                }
                var l: Leave = .{ .len = @intCast(buf.len) };
                @memcpy(l.tiles[0..buf.len], buf);
                try v.out.append(v.f.gpa, .{ .leave = l, .log_w = log_w, .measured = is_measured });
            }
        };
        const vis: Visitor = .{ .f = self, .mod = &res.model.?, .log_calib = log_calib, .log_choose_nk = log_choose_nk, .out = &scored };
        var buf: [rack_size]MachineLetter = undefined;
        try self.recurseLeaves(&vis, &buf, 0, 0, self.k, 0);

        var max_log_w = -std.math.inf(f64);
        for (scored.items) |s| {
            if (s.log_w > max_log_w) max_log_w = s.log_w;
        }
        res.max_log_w = max_log_w;
        if (scored.items.len == 0 or std.math.isInf(max_log_w)) return res;
        var out: std.ArrayList(Leave) = .empty;
        errdefer out.deinit(gpa);
        const log_threshold = gomath.log(negligible_weight_fraction);
        for (scored.items) |s| {
            const rel = s.log_w - max_log_w;
            if (rel < log_threshold) continue;
            var l = s.leave;
            l.weight = gomath.exp(rel);
            try out.append(gpa, l);
            if (s.measured) res.measured_leaves += 1;
        }
        res.leaves = try out.toOwnedSlice(gpa);
        return res;
    }

    /// Every leave of `remaining` more tiles from the bag, tiles from
    /// `min_idx` on, with the log of its count of draws (the posterior's
    /// enumeration, by first tile).
    fn recurseLeaves(self: *Finder, vis: anytype, buf: *[rack_size]MachineLetter, n: usize, min_idx: usize, remaining: usize, log_num: f64) Allocator.Error!void {
        if (remaining == 0) return vis.visit(buf[0..n], log_num);
        for (min_idx..self.alpha) |i| {
            if (self.bag_map[i] == 0) continue;
            const max_copies = @min(self.bag_map[i], remaining);
            var ln = log_num;
            const kk: f64 = @floatFromInt(self.bag_map[i]);
            var cnt: usize = 1;
            while (cnt <= max_copies) : (cnt += 1) {
                buf[n + cnt - 1] = @intCast(i);
                const c: f64 = @floatFromInt(cnt);
                ln += gomath.log((kk - c + 1) / c);
                try self.recurseLeaves(vis, buf, n + cnt, i + 1, remaining - cnt, ln);
            }
        }
    }

    /// calibrateLogConstant: the cross-fitted constant that scales imputed
    /// likelihoods to the measured ones.
    fn calibrate(self: *Finder, keys: []const Key, full: *const Model, folds: []const Model) f64 {
        const gpa = self.gpa;
        var sum_cw: f64 = 0;
        var lp_full: std.ArrayList(f64) = .empty;
        defer lp_full.deinit(gpa);
        var lp_fold: std.ArrayList(f64) = .empty;
        defer lp_fold.deinit(gpa);
        for (keys) |key| {
            const ml = self.measured.getPtr(key).?;
            const w = ml.mean();
            if (w <= 0 or ml.sum_u <= 0) continue;
            const rb = RunBuf.of(key.bytes());
            const log_c = gomath.log(ml.sum_u);
            sum_cw += ml.sum_u * w;
            lp_full.append(gpa, log_c + full.logImputed(rb.slice())) catch return 0;
            if (folds.len > 0) lp_fold.append(gpa, log_c + folds[foldForKey(key.bytes(), @intCast(folds.len))].logImputed(rb.slice())) catch return 0;
        }
        if (sum_cw <= 0 or lp_full.items.len == 0) return 0;
        const log_sum_cw = gomath.log(sum_cw);
        const in_sample = log_sum_cw - logSumExp(lp_full.items);
        if (lp_fold.items.len > 0) return log_sum_cw - logSumExp(lp_fold.items);
        return in_sample;
    }

    const Candidate = struct {
        leave: Leave,
        weight: f64,
        q: f64,
        u: f64 = 0,
        lhat: f64,
        unc: f64,
        prior: f64,
    };

    /// The next round's proposal over the unmeasured leaves: posterior
    /// weight, widened where the model is unsure (buildProposal).
    fn buildProposal(self: *Finder, cands: *std.ArrayList(Candidate), lambda_ex: f64) Allocator.Error!f64 {
        cands.clearRetainingCapacity();
        const model = &(self.imputed.model orelse return 0);
        const max_w = gomath.exp(self.imputed.max_log_w);
        var total_w: f64 = 0;
        var unmeasured_w: f64 = 0;
        var q_total: f64 = 0;
        for (self.leaves) |*l| {
            total_w += l.weight;
            if (self.measured.getPtr(Key.of(l.slice()))) |ml| {
                if (ml.count > 0) continue;
            }
            unmeasured_w += l.weight;
            const rb = RunBuf.of(l.slice());
            const unc = model.uncertainty(rb.slice());
            const lhat = gomath.exp(self.imputed.log_calib + model.logImputed(rb.slice()));
            var prior: f64 = 0;
            if (lhat > 0) prior = l.weight * max_w / lhat;
            const q = l.weight * (1 + lambda_ex * unc);
            if (q <= 0) continue;
            q_total += q;
            try cands.append(self.gpa, .{ .leave = l.*, .weight = l.weight, .q = q, .unc = unc, .lhat = lhat, .prior = prior });
        }
        const unmeasured_mass = if (total_w > 0) unmeasured_w / total_w else 0;
        if (q_total <= 0) {
            cands.clearRetainingCapacity();
            return unmeasured_mass;
        }
        for (cands.items) |*c| {
            const q = c.q / q_total;
            c.q = q;
            if (q > 0) c.u = c.prior / q;
        }
        return unmeasured_mass;
    }

    fn roundBatchSize(self: *const Finder, round: u32, candidates: usize) usize {
        if (self.budget == 0) return 0;
        const spent: i64 = @intCast(self.sims);
        const left = @as(i64, self.budget) - spent;
        if (left <= 0) return 0;
        const rounds_left: i64 = self.opts.max_rounds - round + 1;
        var b = @divTrunc(left, rounds_left);
        if (b < refine_min_batch) b = @min(refine_min_batch, left);
        return @min(@as(usize, @intCast(b)), candidates);
    }

    fn refineRounds(self: *Finder) Error!void {
        if (self.opts.max_rounds == 0 or self.acc == null) return;
        const gpa = self.gpa;
        var rng = gorand.Rand.init(@bitCast(std.mem.readInt(u64, self.seed[0..8], .big)));
        var cands: std.ArrayList(Candidate) = .empty;
        defer cands.deinit(gpa);
        var round: u32 = 1;
        while (round <= self.opts.max_rounds) : (round += 1) {
            self.current_round = round;
            const unmeasured_mass = try self.buildProposal(&cands, refine_exploration_lambda);
            if (cands.items.len == 0) return;
            if (unmeasured_mass < refine_mass_covered) return;
            const batch = self.roundBatchSize(round, cands.items.len);
            if (batch == 0) return;
            truncateWeights(cands.items);

            const qs = try gpa.alloc(f64, cands.items.len);
            defer gpa.free(qs);
            for (qs, cands.items) |*q, c| q.* = c.q;
            const picks = try systematicSample(gpa, qs, batch, &rng);
            defer gpa.free(picks);

            // Distinct picks in the order first drawn, with multiplicities.
            const Entry = struct { cand: usize, mult: u32, w: f64 = 0 };
            var entries: std.ArrayList(Entry) = .empty;
            defer entries.deinit(gpa);
            var entry_of: std.AutoHashMapUnmanaged(usize, usize) = .empty;
            defer entry_of.deinit(gpa);
            for (picks) |pk| {
                const gop = try entry_of.getOrPut(gpa, pk);
                if (gop.found_existing) {
                    entries.items[gop.value_ptr.*].mult += 1;
                    continue;
                }
                gop.value_ptr.* = entries.items.len;
                try entries.append(gpa, .{ .cand = pk, .mult = 1 });
            }
            const jobs = try gpa.alloc(Job, @min(entries.items.len, batch_size));
            defer gpa.free(jobs);
            var at: usize = 0;
            while (at < entries.items.len) {
                const es = entries.items[at..][0..@min(entries.items.len - at, batch_size)];
                const chunk = jobs[0..es.len];
                for (chunk, es) |*j, e| {
                    j.leave = cands.items[e.cand].leave;
                    self.dealLeave(j);
                }
                try self.measureAll(chunk);
                for (chunk, es) |j, *e| e.w = j.likelihood;
                at += es.len;
            }

            var num: f64 = 0;
            var den: f64 = 0;
            const us = try gpa.alloc(f64, entries.items.len);
            defer gpa.free(us);
            for (entries.items, us) |e, *uo| {
                const c = &cands.items[e.cand];
                const u = c.u * @as(f64, @floatFromInt(e.mult));
                try self.recordSample(c.leave.slice(), e.w, u);
                self.measured.getPtr(Key.of(c.leave.slice())).?.predicted = c.lhat;
                uo.* = u;
                num += u * e.w;
                den += u * c.lhat;
            }
            const evaluated = entries.items.len;
            var converged = false;
            if (den > 0) {
                const ratio = num / den;
                var v: f64 = 0;
                for (entries.items, us) |e, u| {
                    const resid = e.w - ratio * cands.items[e.cand].lhat;
                    v += u * u * resid * resid;
                }
                const se = @sqrt(v) / den;
                if (ratio > 0) {
                    const log_ratio = gomath.log(ratio);
                    const se_log = se / ratio;
                    converged = evaluated >= refine_min_batch and @abs(log_ratio) + 2 * se_log < refine_converged_tol;
                }
            }
            try self.finalize();
            if (converged) return;
        }
    }

    /// An exchange keeps the leave of most value more often: the prior
    /// times exp(beta (value - best value)), no mini-sims
    /// (inferExchangeByValue).
    fn exchangeByValue(self: *Finder) Error!void {
        if (self.k < 1) return error.NoInformation;
        const gpa = self.gpa;
        var list: std.ArrayList(EnumLeaf) = .empty;
        defer list.deinit(gpa);
        try enumerateLeaves(gpa, &self.bag_map, self.alpha, self.k, &list);
        if (list.items.len == 0) return;
        const values = try gpa.alloc(f64, list.items.len);
        defer gpa.free(values);
        var vmax = -std.math.inf(f64);
        for (list.items, values) |*l, *v| {
            const rb = RunBuf.of(l.leave.slice());
            v.* = self.valuer.value(rb.slice());
            vmax = @max(vmax, v.*);
        }
        const out = try gpa.alloc(Leave, list.items.len);
        for (list.items, values, out) |*l, v, *o| {
            o.* = l.leave;
            o.weight = l.prior * gomath.exp(exchange_value_beta * (v - vmax));
        }
        gpa.free(self.leaves);
        self.leaves = out;
        self.complete = true;
        self.enumerated = true;
    }
};

fn truncateWeights(cands: []Finder.Candidate) void {
    if (cands.len == 0) return;
    var sum: f64 = 0;
    for (cands) |c| sum += c.u;
    const n: f64 = @floatFromInt(cands.len);
    const limit = @sqrt(n) * sum / n;
    for (cands) |*c| {
        if (c.u > limit) c.u = limit;
    }
}

fn systematicSample(gpa: Allocator, q: []const f64, m: usize, rng: *gorand.Rand) Allocator.Error![]usize {
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    if (m == 0 or q.len == 0) return out.toOwnedSlice(gpa);
    const step = 1.0 / @as(f64, @floatFromInt(m));
    var pos = rng.float64() * step;
    var cum: f64 = 0;
    var i: usize = 0;
    while (out.items.len < m and i < q.len) : (i += 1) {
        cum += q[i];
        while (out.items.len < m and pos <= cum) {
            try out.append(gpa, i);
            pos += step;
        }
    }
    while (out.items.len < m) {
        var last = q.len - 1;
        while (last > 0 and q[last] <= 0) last -= 1;
        try out.append(gpa, last);
    }
    return out.toOwnedSlice(gpa);
}

fn logSumExp(xs: []const f64) f64 {
    var max_x = -std.math.inf(f64);
    for (xs) |x| {
        if (x > max_x) max_x = x;
    }
    if (std.math.isNegativeInf(max_x)) return max_x;
    var sum: f64 = 0;
    for (xs) |x| sum += gomath.exp(x - max_x);
    return max_x + gomath.log(sum);
}

fn logBinomial(n: usize, k: usize) f64 {
    if (k > n) return -std.math.inf(f64);
    const lgn = gomath.lgamma(@floatFromInt(n + 1));
    const lgk = gomath.lgamma(@floatFromInt(k + 1));
    const lgnk = gomath.lgamma(@floatFromInt(n - k + 1));
    return lgn - lgk - lgnk;
}

/// P(leave) when drawing its size from the bag: the multivariate
/// hypergeometric (combinatorialPrior).
fn combinatorialPrior(leave: []const MachineLetter, bag_map: []const u8) f64 {
    if (leave.len == 0) return 0;
    var counts: [256]usize = @splat(0);
    for (leave) |t| counts[t] += 1;
    var n_total: usize = 0;
    for (bag_map, 0..) |c, i| {
        n_total += c;
        if (counts[i] > c) return 0;
    }
    for (bag_map.len..256) |t| {
        if (counts[t] > 0) return 0;
    }
    if (n_total == 0) return 0;
    var log_p = -logBinomial(n_total, leave.len);
    for (counts[0..bag_map.len], 0..) |lc, t| {
        if (lc > 0) log_p += logBinomial(bag_map[t], lc);
    }
    return gomath.exp(log_p);
}

fn countMultisets(bag_map: *const [max_alpha]u8, alpha: usize, k: usize) u64 {
    const R = struct {
        fn go(bm: *const [max_alpha]u8, a: usize, min_idx: usize, remaining: usize) u64 {
            if (remaining == 0) return 1;
            var count: u64 = 0;
            for (min_idx..a) |i| {
                if (bm[i] == 0) continue;
                const max_copies = @min(bm[i], remaining);
                for (1..max_copies + 1) |cnt| count += go(bm, a, i + 1, remaining - cnt);
            }
            return count;
        }
    };
    return R.go(bag_map, alpha, 0, k);
}

const EnumLeaf = struct { leave: Leave, prior: f64 };

fn enumerateLeaves(gpa: Allocator, bag_map: *const [max_alpha]u8, alpha: usize, k: usize, out: *std.ArrayList(EnumLeaf)) Allocator.Error!void {
    const R = struct {
        fn go(g: Allocator, bm: *const [max_alpha]u8, a: usize, buf: *[rack_size]MachineLetter, n: usize, min_idx: usize, remaining: usize, o: *std.ArrayList(EnumLeaf)) Allocator.Error!void {
            if (remaining == 0) {
                var l: Leave = .{ .len = @intCast(n) };
                @memcpy(l.tiles[0..n], buf[0..n]);
                try o.append(g, .{ .leave = l, .prior = combinatorialPrior(buf[0..n], bm[0..a]) });
                return;
            }
            for (min_idx..a) |i| {
                if (bm[i] == 0) continue;
                const max_copies = @min(bm[i], remaining);
                for (1..max_copies + 1) |cnt| {
                    buf[n + cnt - 1] = @intCast(i);
                    try go(g, bm, a, buf, n + cnt, i + 1, remaining - cnt, o);
                }
            }
        }
    };
    var buf: [rack_size]MachineLetter = undefined;
    try R.go(gpa, bag_map, alpha, &buf, 0, 0, k, out);
}

/// Move.Equals with the leave ignored: the score, kind, tiles placed,
/// squares (or their transposition on an empty board) and tiles.
fn equals(a: *const Move, b: *const Move, transpose: bool) bool {
    if (a.score != b.score or a.kind != b.kind or a.tiles_played != b.tiles_played) return false;
    const same = a.row == b.row and a.col == b.col and a.vertical == b.vertical;
    const transposed = transpose and a.row == b.col and a.col == b.row and a.vertical != b.vertical;
    if (!same and !transposed) return false;
    if (a.len != b.len) return false;
    if (a.kind == .exchange) {
        var x = a.tiles;
        var y = b.tiles;
        std.mem.sort(MachineLetter, x[0..a.len], {}, std.sort.asc(MachineLetter));
        std.mem.sort(MachineLetter, y[0..b.len], {}, std.sort.asc(MachineLetter));
        return std.mem.eql(MachineLetter, x[0..a.len], y[0..b.len]);
    }
    return std.mem.eql(MachineLetter, a.tileSlice(), b.tileSlice());
}

/// The rangefinder's key for a one-tile play: where the tile went and
/// which tile (uniqueSingleTileKey). An exchange of one tile keys as a play
/// of it at the corner, as in Go.
fn singleTileKey(m: *const Move) usize {
    var idx: usize = 0;
    var tile: MachineLetter = 0;
    for (m.tileSlice(), 0..) |t, i| {
        idx = i;
        tile = t;
        if (t != 0) break;
    }
    var row: usize = m.row;
    var col: usize = m.col;
    if (m.vertical) row += idx else col += idx;
    const max_alphabet = 62;
    return row + max_alphabet * col + max_alphabet * max_alphabet * @as(usize, tile);
}

fn sameMove(a: *const Move, b: *const Move, transpose: bool) bool {
    if (equals(a, b, transpose)) return true;
    return a.tiles_played == 1 and b.tiles_played == 1 and singleTileKey(a) == singleTileKey(b);
}

fn logit(p_in: f64) f64 {
    var p = p_in;
    if (p < logit_eps) p = logit_eps else if (p > 1 - logit_eps) p = 1 - logit_eps;
    return gomath.log(p / (1 - p));
}

/// P(target | leave): a softmax with temperature `tau` over the log-odds of
/// the plays' win rates; 0 if the target wasn't simmed.
fn softmaxLikelihood(plays: []const *const sim_mod.Play, target: *const Move, transpose: bool, tau: f64) f64 {
    if (plays.len == 0) return 0;
    var log_odds: [classic.max_plays]f64 = undefined;
    var target_idx: ?usize = null;
    for (plays, 0..) |p, i| {
        log_odds[i] = logit(p.stats.win.mean());
        if (sameMove(&p.move, target, transpose)) target_idx = i;
    }
    const ti = target_idx orelse return 0;
    const lo = log_odds[0..plays.len];
    var max_lo = -std.math.inf(f64);
    for (lo) |x| {
        if (x > max_lo) max_lo = x;
    }
    var sum: f64 = 0;
    for (lo) |x| sum += gomath.exp((x - max_lo) / tau);
    return gomath.exp((lo[ti] - max_lo) / tau) / sum;
}

test "tau schedule and priors" {
    try std.testing.expectEqual(2.5, tauFittedForBag(1));
    try std.testing.expectEqual(0.021, tauFittedForBag(90));
    var bm: [max_alpha]u8 = @splat(0);
    bm[1] = 2;
    bm[2] = 1;
    // Two of {A, A, B}: AA 1/3, AB 2/3.
    const aa = combinatorialPrior(&.{ 1, 1 }, bm[0..3]);
    const ab = combinatorialPrior(&.{ 1, 2 }, bm[0..3]);
    try std.testing.expectApproxEqAbs(1.0 / 3.0, aa, 1e-12);
    try std.testing.expectApproxEqAbs(2.0 / 3.0, ab, 1e-12);
    try std.testing.expectEqual(2, countMultisets(&bm, 3, 2));
}
