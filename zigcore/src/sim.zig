//! Monte Carlo simulation: each candidate play is played, then the game is
//! rolled out a few plies with best static plays against a random opponent
//! rack, and the plays are ranked by how often the rollouts win. Port of
//! Macondo's montecarlo.Simmer, drawing the same tiles from the same seed
//! so that a single-threaded sim gives exactly the Go engine's statistics.
//!
//! Two ways to run: plain, every play rolled out once an iteration until
//! stopped (or a number of iterations), and best-arm identification (BAI),
//! which picks the plays to roll out and stops when the best is told apart.
//! Both run on any number of workers, each with its own game copies,
//! generator and random source, run as a std.Io group. A best-arm run gives
//! the same result on any number of workers; a plain run on one worker
//! gives exactly Go's single-threaded statistics, and on several workers
//! Go's statistics in distribution.
//!
//! Not yet ported: Go's other stopping rules, inference and the endgame
//! leaves (Go searches the endgames that rollouts reach with at most two
//! tiles in the bag; here they are played out statically).

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const game_mod = @import("game.zig");
const movegen = @import("movegen.zig");
const chacha8 = @import("chacha8.zig");
const carry_mod = @import("carry.zig");
const cross_set = @import("cross_set.zig");
const bai = @import("bai.zig");
const gomath = @import("gomath.zig");
const Move = @import("move.zig").Move;
const Game = game_mod.Game;
const MachineLetter = tiles.MachineLetter;

/// Running mean and variance by Welford's algorithm, updated in the same
/// order of operations as Go's stats.Statistic so that the results are
/// the same to the last bit.
pub const Stat = struct {
    n: u64 = 0,
    old_m: f64 = 0,
    new_m: f64 = 0,
    old_s: f64 = 0,
    new_s: f64 = 0,

    pub fn push(self: *Stat, v: f64) void {
        self.n += 1;
        if (self.n == 1) {
            self.old_m = v;
            self.new_m = v;
            self.old_s = 0;
        } else {
            self.new_m = self.old_m + (v - self.old_m) / @as(f64, @floatFromInt(self.n));
            self.new_s = self.old_s + (v - self.old_m) * (v - self.new_m);
            self.old_m = self.new_m;
            self.old_s = self.new_s;
        }
    }

    pub fn mean(self: *const Stat) f64 {
        return if (self.n > 0) self.new_m else 0;
    }

    pub fn variance(self: *const Stat) f64 {
        if (self.n <= 1) return 0;
        return self.new_s / @as(f64, @floatFromInt(self.n - 1));
    }

    pub fn stdErr(self: *const Stat) f64 {
        return @sqrt(self.variance() / @as(f64, @floatFromInt(self.n)));
    }

    /// Adds the values `o` was made of (Chan et al.'s pairwise update).
    /// Into an empty statistic it is a copy, so a merge of one is exact.
    pub fn merge(self: *Stat, o: *const Stat) void {
        if (o.n == 0) return;
        if (self.n == 0) {
            self.* = o.*;
            return;
        }
        const na: f64 = @floatFromInt(self.n);
        const nb: f64 = @floatFromInt(o.n);
        const n = na + nb;
        const delta = o.new_m - self.new_m;
        const m = self.new_m + delta * nb / n;
        const s2 = self.new_s + o.new_s + delta * delta * na * nb / n;
        self.n += o.n;
        self.old_m = m;
        self.new_m = m;
        self.old_s = s2;
        self.new_s = s2;
    }
};

/// Spreads beyond this are looked up as this (equity.MaxRepresentedWinSpread).
pub const max_win_spread = 300;
/// Tiles unseen beyond this are looked up as this.
pub const max_win_unseen = 93;

/// The chance of winning by spread (from +300 down to -300) and tiles
/// unseen (0 to 93), for the player who just moved: winpct.csv.
pub const WinTable = struct {
    rows: [2 * max_win_spread + 1][max_win_unseen + 1]f32,

    pub const Error = error{InvalidData};

    /// Parses winpct.csv: a header row, then one row per spread with the
    /// spread first. Rows are taken in order (the spread column is not
    /// read), as Go's loader does.
    pub fn parse(self: *WinTable, text: []const u8) Error!void {
        self.rows = @splat(@splat(0));
        var lines = std.mem.splitScalar(u8, text, '\n');
        _ = lines.next() orelse return error.InvalidData;
        var r: usize = 0;
        while (lines.next()) |line_raw| {
            const line = std.mem.trimEnd(u8, line_raw, "\r");
            if (line.len == 0) continue;
            if (r >= self.rows.len) return error.InvalidData;
            var fields = std.mem.splitScalar(u8, line, ',');
            _ = fields.next();
            var c: usize = 0;
            while (fields.next()) |f| : (c += 1) {
                if (c >= self.rows[r].len) return error.InvalidData;
                self.rows[r][c] = std.fmt.parseFloat(f32, f) catch return error.InvalidData;
            }
            r += 1;
        }
        if (r != self.rows.len) return error.InvalidData;
    }

    /// The win probability of a rollout for the player whose sim it is,
    /// `spread` ahead with `leftover` equity on the racks. With an even
    /// number of plies the opponent moves next, so the table is read from
    /// their side. Port of rolloutWinPct.
    pub fn rolloutWin(self: *const WinTable, spread: i32, leftover: f64, game_over: bool, tiles_unseen_in: i32, plies_even: bool) f64 {
        if (game_over or tiles_unseen_in == 0) {
            if (spread == 0) return 0.5;
            return if (spread > 0) 1.0 else 0.0;
        }
        const tiles_unseen = @min(tiles_unseen_in, max_win_unseen);
        var s = spread + @as(i32, @intFromFloat(@round(leftover)));
        if (plies_even) s = -s;
        s = std.math.clamp(s, -max_win_spread, max_win_spread);
        var pct = self.rows[@intCast(max_win_spread - s)][@intCast(tiles_unseen)];
        if (plies_even) pct = 1 - pct;
        return pct;
    }
};

/// The most plies a rollout can have.
pub const max_plies = 32;

/// What one rollout came to.
pub const Rollout = struct {
    /// The play rolled out (its index in Sim.plays).
    play: u32 = 0,
    win: f64 = 0,
    /// Spread gained over the rollout plus the leftover leave values.
    equity: f64 = 0,
    leftover: f64 = 0,
    /// Plies played, and by ply the score of the play made and whether it
    /// was a bingo (bit `ply`).
    plies: u8 = 0,
    scores: [max_plies]i32 = undefined,
    bingos: u32 = 0,
};

/// What a play's rollouts came to.
pub const PlayStats = struct {
    equity: Stat = .{},
    leftover: Stat = .{},
    win: Stat = .{},
    /// By ply: the score of the play made and whether it was a bingo.
    scores: []Stat,
    bingos: []Stat,

    fn add(self: *PlayStats, r: *const Rollout) void {
        for (0..r.plies) |ply| {
            self.scores[ply].push(@floatFromInt(r.scores[ply]));
            self.bingos[ply].push(@floatFromInt((r.bingos >> @intCast(ply)) & 1));
        }
        self.equity.push(r.equity);
        self.leftover.push(r.leftover);
        self.win.push(r.win);
    }

    fn reset(self: *PlayStats) void {
        self.equity = .{};
        self.leftover = .{};
        self.win = .{};
        @memset(self.scores, .{});
        @memset(self.bingos, .{});
    }

    fn merge(self: *PlayStats, o: *const PlayStats) void {
        self.equity.merge(&o.equity);
        self.leftover.merge(&o.leftover);
        self.win.merge(&o.win);
        for (self.scores, o.scores) |*a, *b| a.merge(b);
        for (self.bingos, o.bingos) |*a, *b| a.merge(b);
    }
};

/// One candidate play and what its rollouts came to.
pub const Play = struct {
    move: Move,
    stats: PlayStats,
    ignored: bool = false,
    /// The classic stopping rule never drops it (AvoidPruningMoves).
    unignorable: bool = false,
};

/// `n` plays' statistics, `plies` plies each, in one allocation.
fn allocStats(gpa: Allocator, n: usize, plies: u32) Allocator.Error!struct { stats: []PlayStats, buf: []Stat } {
    const stats = try gpa.alloc(PlayStats, n);
    errdefer gpa.free(stats);
    const buf = try gpa.alloc(Stat, 2 * n * plies);
    @memset(buf, .{});
    for (stats, 0..) |*ps, i| {
        const b = buf[2 * i * plies ..][0 .. 2 * plies];
        ps.* = .{ .scores = b[0..plies], .bingos = b[plies..] };
    }
    return .{ .stats = stats, .buf = buf };
}

/// Win probabilities this close (or both all but certain) rank the plays
/// by equity instead (winProbsIndistinguishable).
const win_sort_epsilon = 1e-9;
const min_reasonable_win = 0.005;

fn winsIndistinguishable(a: f64, b: f64) bool {
    if (@abs(a - b) <= win_sort_epsilon) return true;
    const both_won = a > 1 - min_reasonable_win and b > 1 - min_reasonable_win;
    const both_lost = a < min_reasonable_win and b < min_reasonable_win;
    return both_won or both_lost;
}

/// Whether `a` ranks above `b`: by win probability, then equity
/// (sortPlaysByWinRate).
pub fn ranksAbove(a: *const Play, b: *const Play) bool {
    const wa = a.stats.win.mean();
    const wb = b.stats.win.mean();
    if (!winsIndistinguishable(wa, wb)) return wa > wb;
    return a.stats.equity.mean() > b.stats.equity.mean();
}

pub const BaiOptions = struct {
    /// The error probability the stopping rule allows (0.01 for 99%).
    delta: f64,
    /// Rollouts every play gets before any is preferred.
    min_samples: u32 = default_bai_min_samples,
    /// The sim's budget, in rollouts per play on average.
    cap_per_play: u64 = default_bai_cap_per_play,
    /// Stops once the leading play has this many rollouts; 0 for never.
    best_play_cap: u64 = 0,
    /// Set to stop the run (a stop command, a time limit).
    stop: ?*const std.atomic.Value(bool) = null,
    /// Each sample a cycle of rollouts (MAGPIE's bag cycling, the elite
    /// bot's setting): one shuffle of the unseen tiles, rotated a rack
    /// at a time so that every unseen tile starts on the opponent's rack
    /// once. A sample's reward is its rollouts' mean; budgets and
    /// minimums still count rollouts.
    bag_cycle: bool = false,
};

/// The most rollouts a bag-cycled sample makes: rotations of 7 tiles over
/// the unseen tiles of the largest distribution.
pub const max_cycle = 32;

/// What a best-arm sample holds back until it folds: its rollouts.
pub const Sample = struct {
    n: u8 = 0,
    rollouts: [max_cycle]Rollout = undefined,
};

pub const RunOptions = struct {
    /// Iterations to run (counting those run before); 0 for no limit.
    iterations: u64 = 0,
    /// Set to stop the run (a stop command, a time limit).
    stop: ?*const std.atomic.Value(bool) = null,
};

pub const default_bai_min_samples = 128;
pub const default_bai_cap_per_play = 1500;
/// The weight of the spread term in a best-arm sim's reward (the win
/// probability's is 1), and the spread its sigmoid is scaled by.
const default_bai_spread_weight = 0.1;
const bai_spread_scale = 100.0;
/// A best-arm sim stops once the best play's reward is this close to 0 or
/// 1: a decided game.
const bai_cutoff = 0.005 / 100.0;
/// The fewest samples a play gets in a bag-cycled sim, so that its
/// variance is never estimated from one.
const bai_min_cycles = 2;

/// The delta of a stopping condition given as a confidence percentage:
/// 90, 95, 98, 99 or 999 (99.9).
pub fn stopDelta(condition: u32) ?f64 {
    return switch (condition) {
        90 => 0.10,
        95 => 0.05,
        98 => 0.02,
        99 => 0.01,
        999 => 0.001,
        else => null,
    };
}

/// What a rollout is worth to a best-arm sim: its win probability blended
/// with a sigmoid of the final spread, which ranks the plays of a decided
/// game and breaks near-ties toward spread (baiReward).
pub fn baiReward(win: f64, final_spread: f64, spread_weight: f64) f64 {
    if (spread_weight <= 0) return win;
    const x = final_spread / bai_spread_scale;
    const sig = if (x >= 0) 1 / (1 + gomath.exp(-x)) else blk: {
        const e = gomath.exp(x);
        break :blk e / (1 + e);
    };
    return (win + spread_weight * sig) / (1 + spread_weight);
}

/// Whether two plays count as the same to the stopping rules: from the
/// same square in the same direction with the same tiles (similarPlayKey).
pub fn similarPlays(a: *const Move, b: *const Move) bool {
    if (a.kind != b.kind or a.row != b.row or a.col != b.col or a.vertical != b.vertical or
        a.tiles_played != b.tiles_played or a.len != b.len) return false;
    var x: [@import("board.zig").max_dim]MachineLetter = undefined;
    var y: [@import("board.zig").max_dim]MachineLetter = undefined;
    @memcpy(x[0..a.len], a.tileSlice());
    @memcpy(y[0..b.len], b.tileSlice());
    std.mem.sort(MachineLetter, x[0..a.len], {}, std.sort.asc(MachineLetter));
    std.mem.sort(MachineLetter, y[0..b.len], {}, std.sort.asc(MachineLetter));
    return std.mem.eql(MachineLetter, x[0..a.len], y[0..b.len]);
}

/// The random source of sim thread `thread` for a game seeded with `seed`
/// at turn `turn`: SHA-256 of the three (makeRNGs).
pub fn threadSeed(seed: [32]u8, turn: u64, thread: u64) [32]u8 {
    var buf: [48]u8 = undefined;
    buf[0..32].* = seed;
    std.mem.writeInt(u64, buf[32..40], turn, .big);
    std.mem.writeInt(u64, buf[40..48], thread, .big);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&buf, &out, .{});
    return out;
}

pub const Sim = struct {
    gpa: Allocator,
    plays: []Play,
    stat_buf: []Stat,
    plies: u32,
    win_table: *const WinTable,
    initial_player: u1,
    initial_spread: i32,
    workers: []Worker,
    /// Plain runs: iterations claimed by workers, and finished.
    claimed: std.atomic.Value(u64) = .init(0),
    finished: std.atomic.Value(u64) = .init(0),
    /// Iterations: of a plain run, finished; of a best-arm run, rollouts
    /// over plays.
    iterations: u64 = 0,
    /// What best-arm sims draw their samples from: SHA-256 of the game's
    /// seed and turn (startBAI).
    bai_seed: [32]u8,
    /// The play a best-arm run picked, which ranks first.
    winner: ?usize = null,
    /// The best-arm run in progress, for progress reports.
    bai_runner: ?*bai.Runner(Sample) = null,
    /// Rollouts a best-arm sample makes (more than one when bag-cycled).
    rotations: u32 = 1,
    /// Keeps a progress report and a plain run's last merge apart.
    stats_mutex: std.Io.Mutex = .init,
    bai_stop: ?*const std.atomic.Value(bool) = null,

    /// A sim of `moves` from `g` (whose seed is `seed` at turn `turn`),
    /// rolling each out `plies` plies (at most max_plies), on `threads`
    /// workers. Worker i is Go's sim thread i.
    pub fn create(gpa: Allocator, g: *const Game, seed: [32]u8, turn: u64, moves: []const Move, plies: u32, win_table: *const WinTable, threads: usize) Allocator.Error!*Sim {
        std.debug.assert(plies >= 1 and plies <= max_plies);
        const sim = try gpa.create(Sim);
        errdefer gpa.destroy(sim);
        const a = try allocStats(gpa, moves.len, plies);
        errdefer {
            gpa.free(a.buf);
            gpa.free(a.stats);
        }
        const plays = try gpa.alloc(Play, moves.len);
        errdefer gpa.free(plays);
        for (plays, moves, a.stats) |*p, m, ps| p.* = .{ .move = m, .stats = ps };
        gpa.free(a.stats);
        const workers = try gpa.alloc(Worker, @max(threads, 1));
        errdefer gpa.free(workers);
        sim.* = .{
            .gpa = gpa,
            .plays = plays,
            .stat_buf = a.buf,
            .plies = plies,
            .win_table = win_table,
            .initial_player = g.on_turn,
            .initial_spread = g.pos.scores[g.on_turn] - g.pos.scores[1 - g.on_turn],
            .workers = workers,
            .bai_seed = undefined,
        };
        var buf: [40]u8 = undefined;
        buf[0..32].* = seed;
        std.mem.writeInt(u64, buf[32..40], turn, .big);
        std.crypto.hash.sha2.Sha256.hash(&buf, &sim.bai_seed, .{});
        var made: usize = 0;
        errdefer for (workers[0..made]) |*w| w.deinit();
        for (workers, 0..) |*w, i| {
            try w.init(sim, g, threadSeed(seed, turn, i));
            made += 1;
            if (i == 0) {
                // Every copy draws the bag as it lies, after one shuffle
                // from thread 0's source (makeGameCopies).
                w.start.bag.shuffle(&w.rng);
            } else {
                w.start.bag = workers[0].start.bag;
            }
        }
        return sim;
    }

    pub fn destroy(self: *Sim) void {
        const gpa = self.gpa;
        for (self.workers) |*w| w.deinit();
        gpa.free(self.workers);
        gpa.free(self.stat_buf);
        gpa.free(self.plays);
        gpa.destroy(self);
    }

    /// Moves made (candidate plays and rollout plies), as Go counts nodes.
    /// Safe to call while the sim runs.
    pub fn nodes(self: *const Sim) u64 {
        var n: u64 = 0;
        for (self.workers) |*w| n += w.nodes.load(.monotonic);
        return n;
    }

    /// One iteration on worker 0, on the calling thread (a plain run).
    pub fn iterate(self: *Sim) Allocator.Error!void {
        try self.workers[0].iterate(null);
        self.iterations += 1;
        self.collect(null);
    }

    /// A plain run: each worker takes iterations (a new opponent rack,
    /// then a rollout of every play not ignored) until `opts` says to
    /// stop. The plays' statistics are then the workers' merged.
    pub fn run(self: *Sim, io: std.Io, opts: RunOptions) Allocator.Error!void {
        self.claimed.store(self.iterations, .monotonic);
        self.finished.store(self.iterations, .monotonic);
        var err: ?Allocator.Error = null;
        if (self.workers.len == 1) {
            self.workers[0].loop(io, opts, &err);
        } else {
            var group: std.Io.Group = .init;
            for (self.workers[1..]) |*w| group.concurrent(io, Worker.loop, .{ w, io, opts, &err }) catch break;
            self.workers[0].loop(io, opts, &err);
            group.await(io) catch {};
        }
        self.iterations = self.finished.load(.monotonic);
        self.collect(io);
        if (err) |e| return e;
    }

    /// Makes the plays' statistics those of every worker merged. With
    /// `io` it locks what it reads and writes, so that it can run during a
    /// plain run, for a progress report.
    pub fn collect(self: *Sim, io: ?std.Io) void {
        if (io) |i| self.stats_mutex.lockUncancelable(i);
        defer if (io) |i| self.stats_mutex.unlock(i);
        for (self.plays) |*p| p.stats.reset();
        for (self.workers) |*w| {
            if (io) |i| w.mutex.lockUncancelable(i);
            for (self.plays, w.stats) |*p, *ws| p.stats.merge(ws);
            if (io) |i| w.mutex.unlock(i);
        }
    }

    /// What a progress report says: rollouts so far, and the best play by
    /// the statistics as they stand. Safe to call while the sim runs.
    pub const Progress = struct { rollouts: u64, best: ?Move, win: f64, equity: f64 };

    pub fn progress(self: *Sim, io: std.Io, order: []*const Play) Progress {
        var res: Progress = .{ .rollouts = 0, .best = null, .win = 0, .equity = 0 };
        if (self.bai_runner) |r| {
            // The plays' statistics change at folds, under the runner's
            // lock.
            r.mutex.lockUncancelable(io);
            defer r.mutex.unlock(io);
            res.rollouts = r.total;
            self.fillProgress(&res, order);
        } else {
            self.collect(io);
            self.stats_mutex.lockUncancelable(io);
            defer self.stats_mutex.unlock(io);
            for (self.plays) |*p| res.rollouts += p.stats.win.n;
            self.fillProgress(&res, order);
        }
        return res;
    }

    fn fillProgress(self: *const Sim, res: *Progress, order: []*const Play) void {
        self.ranked(order);
        if (order.len == 0 or order[0].stats.win.n == 0) return;
        res.best = order[0].move;
        res.win = order[0].stats.win.mean();
        res.equity = order[0].stats.equity.mean();
    }

    /// A best-arm run: samples are single rollouts of one play, and the
    /// runner picks which plays to roll out (mostly the leader and its
    /// closest challenger) and when the leader is told apart from the rest
    /// (Go's simBAI). Sample n of every play deals the opponent the same
    /// rack and the bag the same draws, on whichever worker it is drawn.
    pub fn runBai(self: *Sim, io: std.Io, opts: BaiOptions) !bai.Result {
        var runner = try bai.Runner(Sample).init(self.gpa, self.plays.len);
        defer runner.deinit();
        self.bai_runner = &runner;
        defer self.bai_runner = null;
        self.bai_stop = opts.stop;
        self.rotations = 1;
        if (opts.bag_cycle) {
            const g = &self.workers[0].start;
            const unseen = g.bag.len + g.pos.racks[1 - self.initial_player].num_tiles;
            if (unseen > game_mod.rack_tile_limit) self.rotations = @intCast((unseen + game_mod.rack_tile_limit - 1) / game_mod.rack_tile_limit);
            std.debug.assert(self.rotations <= max_cycle);
        }
        // Minimums, budgets and round sizes count rollouts; a sample is
        // `rot` of them.
        const rot: u64 = self.rotations;
        const n: u64 = self.plays.len;
        var min_samples: u64 = (opts.min_samples + rot - 1) / rot;
        if (rot > 1 and min_samples < bai_min_cycles) min_samples = bai_min_cycles;
        const cb: bai.Callbacks(*Sim, Sample) = .{
            .ctx = self,
            .sample = baiSample,
            .commit = baiCommit,
            .similar = baiSimilar,
            .stop = if (opts.stop != null) baiStop else null,
        };
        const res = try runner.run(io, self.workers.len, .{
            .delta = opts.delta,
            .sample_minimum = @intCast(min_samples),
            .cutoff = bai_cutoff,
            .sample_limit = (opts.cap_per_play * n + rot - 1) / rot,
            .best_arm_limit = if (opts.best_play_cap > 0) (opts.best_play_cap + rot - 1) / rot else 0,
            .round_size = @intCast(@max((bai.default_round_size + rot / 2) / rot, 1)),
        }, cb);
        self.winner = res.best;
        self.iterations = res.samples * rot / @max(n, 1);
        return res;
    }

    fn baiSample(self: *Sim, worker: usize, arm: usize, ordinal: u64, held: *Sample) anyerror!f64 {
        const w = &self.workers[worker];
        var buf: [40]u8 = undefined;
        buf[0..32].* = self.bai_seed;
        std.mem.writeInt(u64, buf[32..40], ordinal, .big);
        var seed: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(&buf, &seed, .{});
        // A cycle: the same shuffle of the unseen tiles, moved round by a
        // rack for each rollout. The sample is the rollouts' mean.
        var sum: f64 = 0;
        held.n = @intCast(self.rotations);
        for (0..self.rotations) |rotation| {
            w.dealSeeded(.init(seed), rotation);
            const r = try w.rollout(arm);
            held.rollouts[rotation] = r;
            sum += baiReward(r.win, r.equity + @as(f64, @floatFromInt(self.initial_spread)), default_bai_spread_weight);
        }
        return if (self.rotations == 1) sum else sum / @as(f64, @floatFromInt(self.rotations));
    }

    fn baiCommit(self: *Sim, held: *const Sample) void {
        for (held.rollouts[0..held.n]) |*r| self.plays[r.play].stats.add(r);
    }

    fn baiSimilar(self: *Sim, a: usize, b: usize) bool {
        return similarPlays(&self.plays[a].move, &self.plays[b].move);
    }

    fn baiStop(self: *Sim) bai.Status {
        const s = self.bai_stop orelse return .none;
        return if (s.load(.monotonic)) .canceled else .none;
    }

    /// The plays best first (sortPlaysByWinRate with the ignored plays
    /// last; plays the comparison can't separate keep their order).
    pub fn ranked(self: *const Sim, out: []*const Play) void {
        std.debug.assert(out.len == self.plays.len);
        var n: usize = 0;
        for (self.plays) |*p| {
            if (!p.ignored) {
                out[n] = p;
                n += 1;
            }
        }
        const active = n;
        for (self.plays) |*p| {
            if (p.ignored) {
                out[n] = p;
                n += 1;
            }
        }
        const S = struct {
            fn lt(_: void, a: *const Play, b: *const Play) bool {
                return ranksAbove(a, b);
            }
        };
        std.sort.insertion(*const Play, out[0..active], {}, S.lt);
        std.sort.insertion(*const Play, out[active..], {}, S.lt);
        // A best-arm run's pick goes first: the best by the reward it ran
        // on, which can differ from the order above between plays it
        // could not tell apart.
        const w = &self.plays[self.winner orelse return];
        if (w.ignored) return;
        const i = std.mem.indexOfScalar(*const Play, out, w).?;
        std.mem.copyBackwards(*const Play, out[1 .. i + 1], out[0..i]);
        out[0] = w;
    }
};

/// One of a sim's workers: its game copies, generator and random source.
pub const Worker = struct {
    sim: *Sim,
    /// The position after this iteration's opponent rack was dealt: where
    /// every rollout starts.
    start: Game,
    /// The game a rollout is played on.
    work: Game,
    /// The worker's random source. It carries on from rollout to rollout
    /// (the games are reset, it isn't).
    rng: chacha8.Rng,
    gen: *movegen.Generator,
    /// The rollout generator's memory of best plays by rack.
    carry: *carry_mod.Carry,
    /// The rollout games' memory of cross-sets by the words beside a
    /// square.
    cross_cache: *cross_set.Cache,
    /// Tests: a generator without the memory that checks every rollout
    /// move the memory helped find.
    check: ?*movegen.Generator = null,
    /// Plain runs: this worker's statistics, merged into the plays' when
    /// they are read, and the lock that keeps a reader out of them.
    stats: []PlayStats,
    stat_buf: []Stat,
    mutex: std.Io.Mutex = .init,
    nodes: std.atomic.Value(u64) = .init(0),

    fn init(self: *Worker, sim: *Sim, g: *const Game, seed: [32]u8) Allocator.Error!void {
        const gpa = sim.gpa;
        const a = try allocStats(gpa, sim.plays.len, sim.plies);
        errdefer {
            gpa.free(a.buf);
            gpa.free(a.stats);
        }
        const gen = try gpa.create(movegen.Generator);
        errdefer gpa.destroy(gen);
        // Each rollout points the generator at its own board first.
        gen.* = .init(gpa, @constCast(&g.pos.board), g.pos.lexicon.kwg, g.pos.dist);
        gen.word_map = g.pos.word_map;
        gen.rit = g.pos.rit;
        errdefer gen.deinit();
        const carry = try gpa.create(carry_mod.Carry);
        errdefer gpa.destroy(carry);
        carry.* = .init(gpa);
        gen.carry = carry;
        errdefer carry.deinit();
        const cross_cache = try gpa.create(cross_set.Cache);
        errdefer gpa.destroy(cross_cache);
        cross_cache.* = try .init(gpa, cross_set.Cache.default_index_bits);
        self.* = .{
            .sim = sim,
            .start = g.*,
            .work = undefined,
            .rng = .init(seed),
            .gen = gen,
            .carry = carry,
            .cross_cache = cross_cache,
            .stats = a.stats,
            .stat_buf = a.buf,
        };
        self.start.paired = false;
        self.start.cross_cache = cross_cache;
        self.start.bag.fixed_order = true;
    }

    fn deinit(self: *Worker) void {
        const gpa = self.sim.gpa;
        self.cross_cache.deinit(gpa);
        gpa.destroy(self.cross_cache);
        self.carry.deinit();
        gpa.destroy(self.carry);
        self.gen.deinit();
        gpa.destroy(self.gen);
        gpa.free(self.stat_buf);
        gpa.free(self.stats);
        self.* = undefined;
    }

    /// Plain runs: iterations until the options say to stop.
    fn loop(self: *Worker, io: ?std.Io, opts: RunOptions, err: *?Allocator.Error) void {
        const sim = self.sim;
        while (true) {
            if (opts.stop) |s| if (s.load(.monotonic)) return;
            if (err.* != null) return;
            const n = sim.claimed.fetchAdd(1, .monotonic);
            if (opts.iterations > 0 and n >= opts.iterations) return;
            self.iterate(io) catch |e| {
                err.* = e;
                return;
            };
            _ = sim.finished.fetchAdd(1, .monotonic);
        }
    }

    /// One iteration into this worker's statistics (locked as each
    /// rollout's result goes in, when there is `io`).
    pub fn iterate(self: *Worker, io: ?std.Io) Allocator.Error!void {
        self.dealOpponent();
        for (self.sim.plays, self.stats, 0..) |*p, *ws, i| {
            if (p.ignored) continue;
            const r = try self.rollout(i);
            if (io) |x| self.mutex.lockUncancelable(x);
            ws.add(&r);
            if (io) |x| self.mutex.unlock(x);
        }
    }

    /// Throws the opponent's rack in and deals a new one (SetRandomRack
    /// with nothing known: Bag.Redraw).
    fn dealOpponent(self: *Worker) void {
        const g = &self.start;
        const opp = 1 - self.sim.initial_player;
        var old: [tiles.max_alphabet_size]MachineLetter = undefined;
        const n = g.pos.racks[opp].tiles(&old);
        g.rng = self.rng;
        g.putBack(old[0..n]);
        var drawn: [game_mod.rack_tile_limit]MachineLetter = undefined;
        const k = g.drawAtMost(&drawn);
        g.pos.racks[opp] = .fromLetters(drawn[0..k]);
        self.rng = g.rng;
    }

    /// Deals the opponent a rack from the unseen tiles put in order and
    /// shuffled with `rng`, which the bag then draws with
    /// (SetRandomRackSeeded with nothing known; with a rotation,
    /// SetRandomRackCycled: the order moved round to the right by that many
    /// racks first).
    fn dealSeeded(self: *Worker, rng: chacha8.Rng, rotation: usize) void {
        const g = &self.start;
        const opp = 1 - self.sim.initial_player;
        var old: [tiles.max_alphabet_size]MachineLetter = undefined;
        const n = g.pos.racks[opp].tiles(&old);
        // Go puts them back with the last sample's source, but sorts the
        // bag next, which undoes any order that gave.
        g.bag.putBack(old[0..n]);
        std.mem.sort(MachineLetter, g.bag.tiles[0..g.bag.len], {}, std.sort.asc(MachineLetter));
        g.rng = rng;
        g.bag.shuffle(&g.rng);
        if (rotation > 0) {
            const t = g.bag.tiles[0..g.bag.len];
            const k = rotation * game_mod.rack_tile_limit % @max(t.len, 1);
            if (k > 0) {
                std.mem.reverse(MachineLetter, t);
                std.mem.reverse(MachineLetter, t[0..k]);
                std.mem.reverse(MachineLetter, t[k..]);
            }
        }
        var drawn: [game_mod.rack_tile_limit]MachineLetter = undefined;
        const k = g.drawAtMost(&drawn);
        g.pos.racks[opp] = .fromLetters(drawn[0..k]);
        self.rng = g.rng;
    }

    /// Plays play `index` from the start position and rolls the game out.
    fn rollout(self: *Worker, index: usize) Allocator.Error!Rollout {
        const sim = self.sim;
        const g = &self.work;
        g.* = self.start;
        g.rng = self.rng;
        defer self.rng = g.rng;
        self.gen.board = &g.pos.board;

        var r: Rollout = .{ .play = @intCast(index) };
        g.play(&sim.plays[index].move);
        _ = self.nodes.fetchAdd(1, .monotonic);
        for (0..sim.plies) |ply| {
            const on_turn = g.on_turn;
            if (!g.playing) break;
            const rack = g.pos.racks[on_turn];
            const best = try self.bestStaticTurn(g);
            if (ply + 1 == sim.plies) g.playLast(&best.move) else g.play(&best.move);
            _ = self.nodes.fetchAdd(1, .monotonic);
            if (ply + 2 >= sim.plies) {
                // Our last turn or the opponent's: count what they keep.
                const leave = best.move.leave(&rack);
                var buf: [tiles.max_alphabet_size]MachineLetter = undefined;
                const v = if (g.pos.leaves) |klv| klv.leaveValue(buf[0..leave.tiles(&buf)]) else 0;
                if (on_turn == sim.initial_player) r.leftover += v else r.leftover -= v;
            }
            r.scores[ply] = best.move.score;
            if (best.move.kind == .play and best.move.tiles_played == game_mod.rack_tile_limit) r.bingos |= @as(u32, 1) << @intCast(ply);
            r.plies += 1;
        }
        const ip = sim.initial_player;
        const spread = g.pos.scores[ip] - g.pos.scores[1 - ip];
        r.equity = @as(f64, @floatFromInt(spread - sim.initial_spread)) + r.leftover;
        const unseen = g.tilesInBag() + g.pos.racks[1 - ip].num_tiles;
        r.win = sim.win_table.rolloutWin(spread, r.leftover, !g.playing, unseen, sim.plies % 2 == 0);
        return r;
    }

    /// The best static move for the player on turn (GenBestStaticTurn).
    fn bestStaticTurn(self: *Worker, g: *Game) Allocator.Error!movegen.Generator.Best {
        const ex = g.exchangeRules();
        self.gen.max_can_exchange = ex.max;
        const ctx = g.equityContext();
        var rack = g.pos.racks[g.on_turn];
        const best = try self.gen.genBest(&rack, ex.allowed, &ctx);
        if (self.check) |check| {
            check.board = &g.pos.board;
            check.max_can_exchange = ex.max;
            var r = g.pos.racks[g.on_turn];
            const want = try check.genBest(&r, ex.allowed, &ctx);
            const eq = @import("equity.zig");
            if (want.equity != best.equity or eq.tiebreaksBetter(&want.move, &best.move) or eq.tiebreaksBetter(&best.move, &want.move)) {
                std.debug.print("carry found {any} ({d}), plain generation {any} ({d})\n", .{ best.move, best.equity, want.move, want.equity });
                @panic("best-move memory changed the best move");
            }
        }
        return best;
    }
};

// Tests ----------------------------------------------------------------------

const testing = std.testing;
const testdata = @import("testdata.zig");

test "welford statistics" {
    var s: Stat = .{};
    try testing.expectEqual(0, s.mean());
    try testing.expectEqual(0, s.variance());
    for ([_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 }) |v| s.push(v);
    try testing.expectEqual(5, s.mean());
    try testing.expectApproxEqAbs(32.0 / 7.0, s.variance(), 1e-12);
    try testing.expectApproxEqAbs(@sqrt(32.0 / 7.0 / 8.0), s.stdErr(), 1e-12);
}

test "rollout win lookup" {
    const t = try testing.allocator.create(WinTable);
    defer testing.allocator.destroy(t);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(testing.allocator);
    try text.appendSlice(testing.allocator, "header\n");
    var s: i32 = max_win_spread;
    while (s >= -max_win_spread) : (s -= 1) {
        try text.print(testing.allocator, "{d}", .{s});
        // Win chance by spread only, plus a hint of the unseen count.
        for (0..max_win_unseen + 1) |u| {
            const v = 0.5 + @as(f64, @floatFromInt(s)) / 1000.0 + @as(f64, @floatFromInt(u)) / 1e6;
            try text.print(testing.allocator, ",{d:.6}", .{v});
        }
        try text.append(testing.allocator, '\n');
    }
    try t.parse(text.items);
    // Game over: decided by the spread.
    try testing.expectEqual(1.0, t.rolloutWin(3, -10, true, 20, false));
    try testing.expectEqual(0.5, t.rolloutWin(0, 10, false, 0, false));
    try testing.expectEqual(0.0, t.rolloutWin(-1, 0, false, 0, true));
    // Odd plies: our side of the table; the leftover is rounded in.
    const f32v = struct {
        fn v(spread: i32, u: u32) f64 {
            return @as(f32, @floatCast(0.5 + @as(f64, @floatFromInt(spread)) / 1000.0 + @as(f64, @floatFromInt(u)) / 1e6));
        }
    }.v;
    try testing.expectApproxEqAbs(f32v(23, 10), t.rolloutWin(20, 2.6, false, 10, false), 1e-6);
    // Even plies: the opponent's side, flipped.
    try testing.expectApproxEqAbs(1 - f32v(-20, 10), t.rolloutWin(20, 0, false, 10, true), 1e-6);
    // Clamped spread and unseen count.
    try testing.expectApproxEqAbs(f32v(300, 93), t.rolloutWin(900, 0, false, 100, false), 1e-6);
    try testing.expectError(error.InvalidData, t.parse("header\n1,0.5\n"));
}

test "thread seeds" {
    // sha256 of 32 zero bytes, turn 0 and thread 0, as Go's makeRNGs
    // derives it.
    var buf: [48]u8 = @splat(0);
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&buf, &want, .{});
    try testing.expectEqual(want, threadSeed(@splat(0), 0, 0));
    buf[47] = 1;
    std.crypto.hash.sha2.Sha256.hash(&buf, &want, .{});
    try testing.expectEqual(want, threadSeed(@splat(0), 0, 1));
}

test "sim ranks a clearly better play first and repeats with a seed" {
    var store = testdata.store();
    defer store.deinit();
    var pos = try testdata.position(&store,
        \\15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AEINRST/ 0/0 0 lex NWL20;
    );
    const wt = try testing.allocator.create(WinTable);
    defer testing.allocator.destroy(wt);
    const csv = try testdata.loadFile("strategy/default/winpct.csv");
    defer testing.allocator.free(csv);
    try wt.parse(csv);

    // A bingo against a pass.
    var g = try Game.fromPosition(&pos, @splat(3));
    var gen = movegen.Generator.init(testing.allocator, &pos.board, pos.lexicon.kwg, pos.dist);
    defer gen.deinit();
    var rack = pos.racks[0];
    try gen.genAll(&rack, false);
    var bingo: ?Move = null;
    for (gen.plays.items) |m| {
        if (m.kind == .play and m.tiles_played == 7) bingo = m;
    }
    const moves = [_]Move{ .pass(), bingo.? };

    var results: [2][2]f64 = undefined;
    for (&results) |*r| {
        var sim = try Sim.create(testing.allocator, &g, @splat(3), 0, &moves, 2, wt, 1);
        defer sim.destroy();
        for (0..40) |_| try sim.iterate();
        var order: [2]*const Play = undefined;
        sim.ranked(&order);
        try testing.expectEqual(.play, order[0].move.kind);
        try testing.expectEqual(40, sim.plays[1].stats.win.n);
        try testing.expectEqual(2 * 40 * 3, sim.nodes());
        r.* = .{ sim.plays[0].stats.equity.mean(), sim.plays[1].stats.win.mean() };
    }
    try testing.expectEqual(results[0], results[1]);
}

test "the best-move memory finds the moves a plain generation finds" {
    var store = testdata.store();
    defer store.deinit();
    const wt = try testing.allocator.create(WinTable);
    defer testing.allocator.destroy(wt);
    const csv = try testdata.loadFile("strategy/default/winpct.csv");
    defer testing.allocator.free(csv);
    try wt.parse(csv);
    // Positions from a static game: early, middle and late.
    var pos = try testdata.position(&store,
        \\15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;
    );
    for ([_]bool{ false, true }) |word_maps| {
        pos.word_map = if (word_maps) try store.wordMap(@import("build_options").data_path, "NWL20", pos.lexicon, pos.dist) else null;
        var g: Game = .start(&pos, @splat(9), false);
        var gen = movegen.Generator.init(testing.allocator, &g.pos.board, pos.lexicon.kwg, pos.dist);
        defer gen.deinit();
        gen.word_map = pos.word_map;
        var check = movegen.Generator.init(testing.allocator, &g.pos.board, pos.lexicon.kwg, pos.dist);
        defer check.deinit();
        check.word_map = pos.word_map;
        var hits: u64 = 0;
        var skipped: u64 = 0;
        var seeded: u64 = 0;
        var turn: usize = 0;
        while (g.playing and g.bag.len > 0) : (turn += 1) {
            if (turn % 4 == 1) {
                var rack = g.pos.racks[g.on_turn];
                try gen.genAll(&rack, false);
                const ctx = g.equityContext();
                const ranked = try testing.allocator.alloc(Move, @min(8, gen.plays.items.len));
                defer testing.allocator.free(ranked);
                // Any eight plays will do.
                @memcpy(ranked, gen.plays.items[0..ranked.len]);
                _ = ctx;
                var sim = try Sim.create(testing.allocator, &g, @splat(5), turn, ranked, 3, wt, 1);
                defer sim.destroy();
                sim.workers[0].check = &check;
                for (0..12) |_| try sim.iterate();
                hits += sim.workers[0].carry.rack_hits;
                skipped += sim.workers[0].carry.lanes_skipped;
                seeded += sim.workers[0].carry.seeded;
            }
            gen.board = &g.pos.board;
            _ = try game_mod.playStaticTurn(&g, &gen);
        }
        try testing.expect(hits > 100);
        try testing.expect(skipped > 100);
        try testing.expect(seeded > 10);
    }
}

/// A midgame NWL20 position, its 8 best plays by static equity and the
/// win table, for the multi-worker tests.
const MidgameFixture = struct {
    store: @import("data.zig").Store,
    pos: @import("position.zig").Position,
    game: Game,
    moves: [8]Move,
    wt: *WinTable,

    fn init(self: *MidgameFixture) !void {
        self.store = testdata.store();
        errdefer self.store.deinit();
        self.pos = try testdata.position(&self.store, "15/15/15/15/15/15/15/5TURnERY3/15/15/15/15/15/15/15 BEIMOQS/ 0/76 0 lex NWL20;");
        self.pos.word_map = try self.store.wordMap(@import("build_options").data_path, "NWL20", self.pos.lexicon, self.pos.dist);
        self.wt = try testing.allocator.create(WinTable);
        errdefer testing.allocator.destroy(self.wt);
        const csv = try testdata.loadFile("strategy/default/winpct.csv");
        defer testing.allocator.free(csv);
        try self.wt.parse(csv);
        self.game = try Game.fromPosition(&self.pos, @splat(7));
        var gen = movegen.Generator.init(testing.allocator, &self.pos.board, self.pos.lexicon.kwg, self.pos.dist);
        defer gen.deinit();
        var rack = self.pos.racks[0];
        try gen.genAll(&rack, false);
        const ctx = self.pos.equityContext();
        const Scored = struct { m: Move, eq: f64 };
        const all = try testing.allocator.alloc(Scored, gen.plays.items.len);
        defer testing.allocator.free(all);
        for (gen.plays.items, all) |m, *x| {
            const lv = m.leave(&rack);
            x.* = .{ .m = m, .eq = ctx.equity(&m, &lv) };
        }
        std.mem.sort(Scored, all, {}, struct {
            fn lt(_: void, a: Scored, b: Scored) bool {
                return @import("equity.zig").better(&a.m, a.eq, &b.m, b.eq);
            }
        }.lt);
        for (&self.moves, all[0..8]) |*d, x| d.* = x.m;
    }

    fn deinit(self: *MidgameFixture) void {
        testing.allocator.destroy(self.wt);
        self.store.deinit();
    }
};

test "a best-arm sim is the same on any number of workers" {
    var f: MidgameFixture = undefined;
    try f.init();
    defer f.deinit();
    var want: ?[8][3]f64 = null;
    var want_res: bai.Result = undefined;
    for ([_]usize{ 1, 2, 4 }) |threads| {
        var sim = try Sim.create(testing.allocator, &f.game, @splat(7), 0, &f.moves, 2, f.wt, threads);
        defer sim.destroy();
        const res = try sim.runBai(testing.io, .{ .delta = 0.01, .cap_per_play = 300, .min_samples = 32 });
        var got: [8][3]f64 = undefined;
        for (&got, sim.plays) |*g, *p| g.* = .{ @floatFromInt(p.stats.win.n), p.stats.win.mean(), p.stats.equity.mean() };
        if (want) |w| {
            try testing.expectEqual(want_res, res);
            try testing.expectEqual(w, got);
        } else {
            want = got;
            want_res = res;
            try testing.expect(res.samples >= 8 * 32);
        }
    }
}

test "a plain sim on several workers agrees with one, and stops when told" {
    var f: MidgameFixture = undefined;
    try f.init();
    defer f.deinit();
    var means: [2][8]f64 = undefined;
    for ([_]usize{ 1, 4 }, 0..) |threads, k| {
        var sim = try Sim.create(testing.allocator, &f.game, @splat(7), 0, &f.moves, 2, f.wt, threads);
        defer sim.destroy();
        try sim.run(testing.io, .{ .iterations = 400 });
        try testing.expectEqual(400, sim.iterations);
        for (sim.plays, 0..) |*p, i| {
            try testing.expectEqual(400, p.stats.win.n);
            means[k][i] = p.stats.equity.mean();
        }
        // Several workers shared the iterations (the Io may run fewer
        // than asked for at once, but not one).
        if (threads > 1) {
            var busy: usize = 0;
            for (sim.workers) |*w| busy += @intFromBool(w.stats[0].win.n > 0);
            try testing.expect(busy >= 2);
        }
    }
    // The same plays, measured on different draws: equal within a few
    // standard errors (equities spread by ~25 points: 400 rollouts give
    // a standard error of about 1.3 a play).
    for (means[0], means[1]) |a, b| try testing.expect(@abs(a - b) < 10);

    // No limit but a stop flag.
    var sim = try Sim.create(testing.allocator, &f.game, @splat(7), 0, &f.moves, 2, f.wt, 3);
    defer sim.destroy();
    var stop: std.atomic.Value(bool) = .init(false);
    const Stopper = struct {
        fn go(s: *std.atomic.Value(bool), sm: *Sim) void {
            while (sm.finished.load(.monotonic) < 50) std.Thread.yield() catch {};
            s.store(true, .monotonic);
        }
    };
    var t = try std.Thread.spawn(.{}, Stopper.go, .{ &stop, sim });
    try sim.run(testing.io, .{ .stop = &stop });
    t.join();
    try testing.expect(sim.iterations >= 50);
}
