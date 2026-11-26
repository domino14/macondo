//! Best-arm identification for sims: which of several plays has the
//! highest mean reward, and when the sample is big enough to tell. Port of
//! Macondo's montecarlo/bai package (itself a port of MAGPIE's bai.h: the
//! top-two IDS sampling rule with the GK16 threshold, with paired
//! statistics for arms whose samples share their luck).
//!
//! Samples are scheduled as Go schedules them: in rounds, each laid out
//! from the statistics as they stood when an earlier round folded, and
//! folded in slot order, so that the result is a function of the samples
//! alone. This port draws on one thread, in slot order: every sample drawn
//! is folded (the run only stops on a fold, or before drawing past the
//! sample limit), so a sample's other effects (the sim's statistics) can
//! be made as it is drawn. Go's speculative draws for idle threads change
//! nothing a single thread does.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gomath = @import("gomath.zig");

pub const Status = enum {
    none,
    /// Every other arm is similar to the best or provably worse.
    threshold,
    /// The best arm's mean is at an extreme (a decided game).
    cutoff,
    /// The sample budget is spent.
    sample_limit,
    /// The best arm has all the samples it may have.
    best_arm_limit,
    /// The time limit passed.
    timeout,
    /// The run was stopped from outside.
    canceled,

    pub fn name(self: Status) []const u8 {
        return switch (self) {
            .none => "none",
            .threshold => "threshold",
            .cutoff => "cutoff",
            .sample_limit => "sample-limit",
            .best_arm_limit => "best-arm-limit",
            .timeout => "timeout",
            .canceled => "canceled",
        };
    }
};

/// Sample values are scaled into i64 accumulators, so that sums don't
/// depend on the order samples are added in.
const fixed_point_shift = 30;
const fixed_point_scale: i64 = 1 << fixed_point_shift;
const max_sample_magnitude = 1 << 10;
const minimum_variance = 1e-10;
/// How many ordinals the paired statistics keep.
const max_paired_ordinals = 1024;
pub const default_round_size = 32;
/// Round j is laid out when round j - lag - 1 folds: far ahead in the
/// initial (round-robin) phase, two rounds in the adaptive one.
const round_robin_lag = 32;
const adaptive_lag = 2;

pub const Options = struct {
    /// The error probability the stopping threshold allows (0.01 for 99%);
    /// 0 for no threshold.
    delta: f64 = 0,
    /// Samples every arm gets, round-robin, before any adaptive sampling
    /// or stopping. Fixed by a runner's first run.
    sample_minimum: u32 = 1,
    /// The most samples over all arms; 0 for no limit.
    sample_limit: u64 = 0,
    /// Stops once the best arm has this many samples; 0 for no limit.
    best_arm_limit: u64 = 0,
    /// Stops once the best mean is within this of 0 or 1; 0 for never.
    cutoff: f64 = 0,
    /// Samples per round. Fixed by a runner's first run.
    round_size: u32 = default_round_size,
};

/// What a run calls: `sample` draws arm's ordinal-th sample, `similar`
/// says whether two arms are interchangeable (an arm similar to the best
/// never needs beating), `stop` is checked between folds (a time limit, a
/// stop command) and says why to stop, or .none.
pub fn Callbacks(comptime Ctx: type) type {
    return struct {
        ctx: Ctx,
        sample: *const fn (ctx: Ctx, arm: usize, ordinal: u64) anyerror!f64,
        similar: ?*const fn (ctx: Ctx, a: usize, b: usize) bool = null,
        stop: ?*const fn (ctx: Ctx) Status = null,
    };
}

pub const Result = struct {
    status: Status,
    /// The arm with the highest mean (lowest index on a tie).
    best: usize,
    /// Samples folded, over all arms.
    samples: u64,
};

const ArmDatum = struct {
    num_samples: u64 = 0,
    num_scheduled: u64 = 0,
    sum: i64 = 0,
    sum_squares: i64 = 0,
    mean: f64 = 0,
    variance: f64 = 0,

    fn add(self: *ArmDatum, v: f64) error{SampleOutOfRange}!void {
        if (std.math.isNan(v) or @abs(v) > max_sample_magnitude) return error.SampleOutOfRange;
        self.num_samples += 1;
        self.sum += quantize(v);
        self.sum_squares += quantize(v * v);
        const inv = 1.0 / @as(f64, @floatFromInt(fixed_point_scale));
        const n: f64 = @floatFromInt(self.num_samples);
        self.mean = @as(f64, @floatFromInt(self.sum)) * inv / n;
        self.variance = @as(f64, @floatFromInt(self.sum_squares)) * inv / n - self.mean * self.mean;
        if (self.variance < minimum_variance) self.variance = minimum_variance;
    }
};

fn quantize(v: f64) i64 {
    return @intFromFloat(@round(v * @as(f64, @floatFromInt(fixed_point_scale))));
}

/// The mean that best explains both arms if they are equal, weighting each
/// by its sample count.
fn altLambda(mu1: f64, sigma21: f64, w1: f64, mua: f64, sigma2a: f64, wa: f64) f64 {
    if (w1 == 0) return mua;
    if (wa == 0 or mu1 == mua) return mu1;
    const x = wa / w1;
    return (sigma2a * mu1 + x * sigma21 * mua) / (sigma2a + x * sigma21);
}

fn klGaussian(mu: f64, sigma2: f64, lambda: f64) f64 {
    const d = mu - lambda;
    return 0.5 * d * d / sigma2;
}

/// The GLR statistic for "astar beats a", weighting the arms by the given
/// counts.
fn zWithCounts(astar: *const ArmDatum, na: f64, a: *const ArmDatum, nb: f64) f64 {
    const l = altLambda(astar.mean, astar.variance, na, a.mean, a.variance, nb);
    return na * klGaussian(astar.mean, astar.variance, l) + nb * klGaussian(a.mean, a.variance, l);
}

const Slot = struct {
    arm: usize,
    ordinal: u64,
    value: f64 = 0,
};

const Round = struct {
    slots: std.ArrayList(Slot) = .empty,
    /// Samples laid out before this round.
    start: u64 = 0,
    /// Slots drawn, and folded (fewer than all only when a sample limit
    /// ended a run partway).
    drawn: usize = 0,
    folded: usize = 0,
    last_initial: bool = false,
};

pub const Runner = struct {
    gpa: Allocator,
    arms: []ArmDatum,
    total: u64 = 0,
    astar: ?usize = null,
    // Paired statistics over each arm's first max_paired_ordinals samples.
    paired_values: []f64,
    prefix: []i64,
    prefix_squares: []i64,
    cross: []i64,

    opts: Options = .{},
    started: bool = false,
    round_size: u32 = default_round_size,
    initial_limit: u64 = 0,
    /// True until the round holding the initial phase's last sample folds.
    initial_phase: bool = true,
    /// Samples laid out in rounds.
    requested: u64 = 0,
    /// Rounds laid out and not folded, oldest first.
    live: std.ArrayList(Round) = .empty,
    spare: std.ArrayList(Round) = .empty,
    rounds_scheduled: u64 = 0,
    rounds_folded: u64 = 0,

    pub fn init(gpa: Allocator, num_arms: usize) Allocator.Error!Runner {
        const arms = try gpa.alloc(ArmDatum, num_arms);
        errdefer gpa.free(arms);
        @memset(arms, .{});
        const pv = try gpa.alloc(f64, num_arms * max_paired_ordinals);
        errdefer gpa.free(pv);
        const prefix = try gpa.alloc(i64, num_arms * (max_paired_ordinals + 1));
        errdefer gpa.free(prefix);
        const prefix_squares = try gpa.alloc(i64, num_arms * (max_paired_ordinals + 1));
        errdefer gpa.free(prefix_squares);
        const cross = try gpa.alloc(i64, num_arms * num_arms);
        @memset(pv, 0);
        @memset(prefix, 0);
        @memset(prefix_squares, 0);
        @memset(cross, 0);
        return .{ .gpa = gpa, .arms = arms, .paired_values = pv, .prefix = prefix, .prefix_squares = prefix_squares, .cross = cross };
    }

    pub fn deinit(self: *Runner) void {
        for (self.live.items) |*r| r.slots.deinit(self.gpa);
        for (self.spare.items) |*r| r.slots.deinit(self.gpa);
        self.live.deinit(self.gpa);
        self.spare.deinit(self.gpa);
        self.gpa.free(self.arms);
        self.gpa.free(self.paired_values);
        self.gpa.free(self.prefix);
        self.gpa.free(self.prefix_squares);
        self.gpa.free(self.cross);
        self.* = undefined;
    }

    pub fn mean(self: *const Runner, arm: usize) f64 {
        return self.arms[arm].mean;
    }

    pub fn count(self: *const Runner, arm: usize) u64 {
        return self.arms[arm].num_samples;
    }

    /// Folds arm's sample at ordinal into the paired statistics. Each arm's
    /// samples arrive in ordinal order, so each pair's product at an
    /// ordinal is added once, by whichever arm folds it second.
    fn record(self: *Runner, arm: usize, ordinal: u64, v: f64) void {
        if (ordinal >= max_paired_ordinals) return;
        const n = self.arms.len;
        const o: usize = @intCast(ordinal);
        self.paired_values[arm * max_paired_ordinals + o] = v;
        const base = arm * (max_paired_ordinals + 1);
        self.prefix[base + o + 1] = self.prefix[base + o] + quantize(v);
        self.prefix_squares[base + o + 1] = self.prefix_squares[base + o] + quantize(v * v);
        for (self.arms, 0..) |*other, j| {
            if (j == arm or other.num_samples <= ordinal) continue;
            const lo = @min(arm, j);
            const hi = @max(arm, j);
            self.cross[lo * n + hi] += quantize(v * self.paired_values[j * max_paired_ordinals + o]);
        }
    }

    fn prefixSum(p: []const i64, arm: usize, pairs: u64) f64 {
        return @as(f64, @floatFromInt(p[arm * (max_paired_ordinals + 1) + @as(usize, @intCast(pairs))])) / @as(f64, @floatFromInt(fixed_point_scale));
    }

    /// The Z statistic for "astar beats arm" when the arms' samples at an
    /// ordinal share their luck:
    ///   Var(mean_a - mean_b) = v_a/n_a + v_b/n_b - 2 c m / (n_a n_b)
    /// with the variances and covariance from the first m common ordinals;
    /// zWithCounts, which ignores the shared luck, when that isn't
    /// positive.
    fn pairedZ(self: *const Runner, astar_idx: usize, arm: usize) f64 {
        const astar = &self.arms[astar_idx];
        const a = &self.arms[arm];
        const na: f64 = @floatFromInt(astar.num_samples);
        const nb: f64 = @floatFromInt(a.num_samples);
        const common = @min(astar.num_samples, a.num_samples);
        const pairs = @min(common, max_paired_ordinals);
        var variance: f64 = 0;
        if (pairs > 1) {
            const p: f64 = @floatFromInt(pairs);
            const ma = prefixSum(self.prefix, astar_idx, pairs) / p;
            const mb = prefixSum(self.prefix, arm, pairs) / p;
            const va = prefixSum(self.prefix_squares, astar_idx, pairs) / p - ma * ma;
            const vb = prefixSum(self.prefix_squares, arm, pairs) / p - mb * mb;
            const lo = @min(astar_idx, arm);
            const hi = @max(astar_idx, arm);
            const cross_sum = @as(f64, @floatFromInt(self.cross[lo * self.arms.len + hi])) / @as(f64, @floatFromInt(fixed_point_scale));
            const cov = cross_sum / p - ma * mb;
            variance = va / na + vb / nb - 2 * cov * @as(f64, @floatFromInt(common)) / (na * nb);
        }
        if (!(variance > 0)) return zWithCounts(astar, na, a, nb);
        const gap = astar.mean - a.mean;
        return gap * gap / (2 * variance);
    }

    fn updateAstar(self: *Runner) f64 {
        self.astar = null;
        var best: f64 = 0;
        for (self.arms, 0..) |*a, i| {
            if (a.num_samples == 0) continue;
            if (self.astar == null or a.mean > best) {
                self.astar = i;
                best = a.mean;
            }
        }
        return best;
    }

    fn thresholdReached(self: *const Runner, cb: anytype) bool {
        const beta = gomath.log((gomath.log(@floatFromInt(self.total)) + 1) / self.opts.delta);
        const astar = self.astar.?;
        for (0..self.arms.len) |i| {
            if (i == astar or similar(cb, astar, i)) continue;
            if (self.opts.delta <= 0) return false;
            const z = self.pairedZ(astar, i);
            if (!(z > beta)) return false;
        }
        return true;
    }

    /// The top-two challenger: the arm minimizing Z + log(N) against
    /// astar, weighting arms by the samples scheduled to them; astar if
    /// every other arm is similar to it.
    fn challenger(self: *const Runner, cb: anytype) usize {
        const astar_idx = self.astar.?;
        const astar = &self.arms[astar_idx];
        const na: f64 = @floatFromInt(astar.num_scheduled);
        var best: ?usize = null;
        var best_value: f64 = 0;
        for (self.arms, 0..) |*a, i| {
            if (i == astar_idx or similar(cb, astar_idx, i)) continue;
            const n: f64 = @floatFromInt(a.num_scheduled);
            const v = zWithCounts(astar, na, a, n) + gomath.log(n);
            if (best == null or v < best_value) {
                best = i;
                best_value = v;
            }
        }
        return best orelse astar_idx;
    }

    /// Splits n samples between astar and the challenger by the top-two
    /// IDS proportion, rounded.
    fn scheduleTopTwo(self: *Runner, cb: anytype, n: usize, out: *std.ArrayList(Slot)) Allocator.Error!void {
        const c = self.challenger(cb);
        const astar = self.astar.?;
        const a = &self.arms[astar];
        const b = &self.arms[c];
        const na: f64 = @floatFromInt(a.num_scheduled);
        const nb: f64 = @floatFromInt(b.num_scheduled);
        const theta = (na * a.mean + nb * b.mean) / (na + nb);
        const num = na * klGaussian(a.mean, a.variance, theta);
        const den = num + nb * klGaussian(b.mean, b.variance, theta);
        var share: f64 = 0;
        if (den > 0) share = @min(@max(num / den, 0), 1);
        const rounded: i64 = @intFromFloat(@round(share * @as(f64, @floatFromInt(n))));
        const astar_count: usize = @intCast(std.math.clamp(rounded, 0, @as(i64, @intCast(n))));
        for (0..n) |i| {
            const arm = if (i < astar_count) astar else c;
            try out.append(self.gpa, .{ .arm = arm, .ordinal = self.arms[arm].num_scheduled });
            self.arms[arm].num_scheduled += 1;
        }
    }

    /// Lays out every round the folds so far allow.
    fn scheduleRounds(self: *Runner, cb: anytype) Allocator.Error!void {
        while (true) {
            const lag: u64 = if (self.requested < self.initial_limit) round_robin_lag else adaptive_lag;
            if (self.rounds_scheduled > self.rounds_folded + lag) return;
            try self.scheduleRound(cb);
        }
    }

    /// Lays out the next round: round-robin over the arms until the
    /// initial phase has folded, by the top-two rule after.
    fn scheduleRound(self: *Runner, cb: anytype) Allocator.Error!void {
        var n: usize = self.round_size;
        // A round never straddles the end of the initial phase.
        if (self.requested < self.initial_limit and self.initial_limit - self.requested < n) {
            n = @intCast(self.initial_limit - self.requested);
        }
        var rd: Round = if (self.spare.pop()) |r| r else .{};
        rd.slots.clearRetainingCapacity();
        rd.start = self.requested;
        rd.drawn = 0;
        rd.folded = 0;
        rd.last_initial = false;
        errdefer rd.slots.deinit(self.gpa);
        if (self.initial_phase) {
            // Iteration-major: consecutive slots are the same ordinal
            // across consecutive arms.
            for (0..n) |i| {
                const arm: usize = @intCast((self.requested + i) % self.arms.len);
                try rd.slots.append(self.gpa, .{ .arm = arm, .ordinal = self.arms[arm].num_scheduled });
                self.arms[arm].num_scheduled += 1;
            }
        } else {
            try self.scheduleTopTwo(cb, n, &rd.slots);
        }
        if (self.requested < self.initial_limit and self.requested + n >= self.initial_limit) rd.last_initial = true;
        self.requested += n;
        self.rounds_scheduled += 1;
        try self.live.append(self.gpa, rd);
    }

    /// Folds the oldest round if it is complete (up to the sample limit),
    /// and checks the stop conditions. Returns why to stop, or .none.
    fn fold(self: *Runner, cb: anytype, limit: u64) !Status {
        while (self.live.items.len > 0) {
            const rd = &self.live.items[0];
            var n = rd.slots.items.len - rd.folded;
            var partial = false;
            if (limit > 0 and limit - self.total < n) {
                n = @intCast(limit - self.total);
                partial = true;
            }
            if (rd.drawn < rd.folded + n) return .none;
            // In slot order, so the statistics are a function of the
            // samples, not of the order they finished in.
            for (rd.slots.items[rd.folded..][0..n]) |sl| {
                const ordinal = self.arms[sl.arm].num_samples;
                try self.arms[sl.arm].add(sl.value);
                self.record(sl.arm, ordinal, sl.value);
            }
            self.total += n;
            rd.folded += n;
            const best = self.updateAstar();
            // The rest of the round is for a later run.
            if (partial) return .sample_limit;
            const last_initial = rd.last_initial;
            try self.spare.append(self.gpa, self.live.orderedRemove(0));
            self.rounds_folded += 1;
            if (last_initial) self.initial_phase = false;
            const status: Status = blk: {
                if (cb.stop) |stop| {
                    const s = stop(cb.ctx);
                    if (s != .none) break :blk s;
                }
                if (self.initial_phase) break :blk .none;
                if (self.opts.cutoff > 0 and (best <= self.opts.cutoff or best >= 1 - self.opts.cutoff)) break :blk .cutoff;
                if (self.thresholdReached(cb)) break :blk .threshold;
                if (self.bestArmLimitReached(self.opts.best_arm_limit)) break :blk .best_arm_limit;
                if (limit > 0 and self.total >= limit) break :blk .sample_limit;
                break :blk .none;
            };
            if (status != .none) return status;
            try self.scheduleRounds(cb);
        }
        return .none;
    }

    fn bestArmLimitReached(self: *const Runner, limit: u64) bool {
        const a = self.astar orelse return false;
        return limit > 0 and self.arms[a].num_samples >= limit;
    }

    /// Samples until a stop condition, continuing from earlier runs (the
    /// sample limit counts their samples too).
    pub fn run(self: *Runner, opts_in: Options, cb: anytype) !Result {
        var opts = opts_in;
        if (opts.round_size < 1) opts.round_size = default_round_size;
        if (opts.sample_minimum < 1) opts.sample_minimum = 1;
        self.opts = opts;
        if (!self.started) {
            self.started = true;
            self.round_size = opts.round_size;
            self.initial_limit = @as(u64, self.arms.len) * opts.sample_minimum;
        }
        const limit = opts.sample_limit;
        var status: Status = .none;
        if (limit > 0 and self.total >= limit) {
            status = .sample_limit;
        } else if (!self.initial_phase and self.bestArmLimitReached(opts.best_arm_limit)) {
            status = .best_arm_limit;
        } else {
            try self.scheduleRounds(cb);
            status = try self.fold(cb, limit);
        }
        while (status == .none) {
            // The oldest round with a slot left to draw, within the limit.
            const rd: ?*Round = for (self.live.items) |*r| {
                if (r.drawn < r.slots.items.len) break r;
            } else null;
            const r = rd orelse break;
            if (limit > 0 and r.start + r.drawn >= limit) break;
            const sl = &r.slots.items[r.drawn];
            sl.value = try cb.sample(cb.ctx, sl.arm, sl.ordinal);
            r.drawn += 1;
            status = try self.fold(cb, limit);
            if (status == .none) {
                if (cb.stop) |stop| status = stop(cb.ctx);
            }
        }
        // Nothing left to draw within the sample limit.
        if (status == .none) status = .sample_limit;
        return .{ .status = status, .best = self.astar orelse 0, .samples = self.total };
    }
};

fn similar(cb: anytype, a: usize, b: usize) bool {
    const f = cb.similar orelse return false;
    return f(cb.ctx, a, b);
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;
const chacha8 = @import("chacha8.zig");

const Bernoulli = struct {
    probs: []const f64,
    seed: u64,

    fn sample(self: *const Bernoulli, arm: usize, ordinal: u64) anyerror!f64 {
        var seed: [32]u8 = @splat(0);
        std.mem.writeInt(u64, seed[0..8], self.seed, .little);
        std.mem.writeInt(u64, seed[8..16], ordinal, .little);
        var rng = chacha8.Rng.init(seed);
        // Common random numbers: the same uniform for every arm at an
        // ordinal.
        return if (rng.float64() < self.probs[arm]) 1 else 0;
    }
};

test "finds the best arm and stops on the threshold" {
    var r = try Runner.init(testing.allocator, 4);
    defer r.deinit();
    const b: Bernoulli = .{ .probs = &.{ 0.3, 0.6, 0.45, 0.2 }, .seed = 7 };
    const cb: Callbacks(*const Bernoulli) = .{ .ctx = &b, .sample = Bernoulli.sample };
    const res = try r.run(.{ .delta = 0.01, .sample_minimum = 16, .sample_limit = 100_000 }, cb);
    try testing.expectEqual(Status.threshold, res.status);
    try testing.expectEqual(1, res.best);
    // Sampling went mostly to the leader and its challenger.
    try testing.expect(r.count(1) + r.count(2) > r.count(0) + r.count(3));
    try testing.expectEqual(res.samples, r.count(0) + r.count(1) + r.count(2) + r.count(3));
}

test "sample limit, continued runs and similar arms" {
    const b: Bernoulli = .{ .probs = &.{ 0.5, 0.5, 0.5 }, .seed = 3 };
    // Equal arms: no threshold; the limit stops it mid-round.
    var r = try Runner.init(testing.allocator, 3);
    defer r.deinit();
    const cb: Callbacks(*const Bernoulli) = .{ .ctx = &b, .sample = Bernoulli.sample };
    var res = try r.run(.{ .delta = 0.01, .sample_minimum = 8, .sample_limit = 100 }, cb);
    try testing.expectEqual(Status.sample_limit, res.status);
    try testing.expectEqual(100, res.samples);
    // Continued: the same samples as one run to the larger limit.
    res = try r.run(.{ .delta = 0.01, .sample_minimum = 8, .sample_limit = 250 }, cb);
    try testing.expectEqual(250, res.samples);
    var one = try Runner.init(testing.allocator, 3);
    defer one.deinit();
    const res1 = try one.run(.{ .delta = 0.01, .sample_minimum = 8, .sample_limit = 250 }, cb);
    try testing.expectEqual(res1.samples, res.samples);
    for (0..3) |i| {
        try testing.expectEqual(one.count(i), r.count(i));
        try testing.expectEqual(one.mean(i), r.mean(i));
    }

    // All arms similar: stops on the threshold as soon as it may.
    const All = struct {
        fn sim(_: *const Bernoulli, _: usize, _: usize) bool {
            return true;
        }
    };
    var s = try Runner.init(testing.allocator, 3);
    defer s.deinit();
    const cbs: Callbacks(*const Bernoulli) = .{ .ctx = &b, .sample = Bernoulli.sample, .similar = All.sim };
    const res2 = try s.run(.{ .delta = 0.01, .sample_minimum = 8 }, cbs);
    try testing.expectEqual(Status.threshold, res2.status);
    try testing.expectEqual(24, res2.samples);
}

test "cutoff and best-arm limit" {
    var r = try Runner.init(testing.allocator, 2);
    defer r.deinit();
    const won: Bernoulli = .{ .probs = &.{ 1, 1 }, .seed = 1 };
    const cb: Callbacks(*const Bernoulli) = .{ .ctx = &won, .sample = Bernoulli.sample };
    const res = try r.run(.{ .delta = 0.01, .sample_minimum = 4, .cutoff = 0.005 }, cb);
    try testing.expectEqual(Status.cutoff, res.status);
    try testing.expectEqual(8, res.samples);

    var s = try Runner.init(testing.allocator, 2);
    defer s.deinit();
    const close: Bernoulli = .{ .probs = &.{ 0.5, 0.51 }, .seed = 2 };
    const cb2: Callbacks(*const Bernoulli) = .{ .ctx = &close, .sample = Bernoulli.sample };
    const res2 = try s.run(.{ .delta = 0.01, .sample_minimum = 4, .best_arm_limit = 200 }, cb2);
    try testing.expectEqual(Status.best_arm_limit, res2.status);
    try testing.expect(s.count(res2.best) >= 200);
}
