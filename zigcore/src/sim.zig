//! Monte Carlo simulation: each candidate play is played, then the game is
//! rolled out a few plies with best static plays against a random opponent
//! rack, and the plays are ranked by how often the rollouts win. Port of
//! Macondo's montecarlo.Simmer, drawing the same tiles from the same seed
//! so that a single-threaded sim gives exactly the Go engine's statistics.
//!
//! Not yet ported: the stopping conditions, best-arm identification, paired
//! stops, inference and the endgame leaves (Go searches the endgames that
//! rollouts reach with at most two tiles in the bag; here they are played
//! out statically).

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const game_mod = @import("game.zig");
const movegen = @import("movegen.zig");
const chacha8 = @import("chacha8.zig");
const carry_mod = @import("carry.zig");
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

/// One candidate play and what its rollouts came to.
pub const Play = struct {
    move: Move,
    /// Spread gained over the rollout plus the leftover leave values.
    equity: Stat = .{},
    leftover: Stat = .{},
    win: Stat = .{},
    /// By ply: the score of the play made and whether it was a bingo.
    scores: []Stat,
    bingos: []Stat,
    ignored: bool = false,
};

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
    const wa = a.win.mean();
    const wb = b.win.mean();
    if (!winsIndistinguishable(wa, wb)) return wa > wb;
    return a.equity.mean() > b.equity.mean();
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
    /// Checked between rounds: why to stop now (time, a stop command), or
    /// .none.
    stop: ?*const fn (*Sim) bai.Status = null,
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
    plies: u32,
    win_table: *const WinTable,
    /// The position after this iteration's opponent rack was dealt: where
    /// every rollout starts.
    start: Game,
    /// The game a rollout is played on.
    work: Game,
    /// The thread's random source. It carries on from rollout to rollout
    /// (the games are reset, it isn't).
    rng: chacha8.Rng,
    gen: *movegen.Generator,
    /// The rollout generator's memory of best plays by rack.
    carry: *carry_mod.Carry,
    /// Tests: a generator without the memory that checks every rollout
    /// move the memory helped find.
    check: ?*movegen.Generator = null,
    initial_player: u1,
    initial_spread: i32,
    iterations: u64 = 0,
    /// Moves made (candidate plays and rollout plies), as Go counts nodes.
    nodes: u64 = 0,
    stat_buf: []Stat,
    /// What best-arm sims draw their samples from: SHA-256 of the game's
    /// seed and turn (startBAI).
    bai_seed: [32]u8,
    /// The play a best-arm run picked, which ranks first.
    winner: ?usize = null,

    /// A sim of `moves` from `g` (whose seed is `seed` at turn `turn`),
    /// rolling each out `plies` plies. Single-threaded: thread 0 of Go's
    /// sim.
    pub fn init(gpa: Allocator, g: *const Game, seed: [32]u8, turn: u64, moves: []const Move, plies: u32, win_table: *const WinTable) Allocator.Error!Sim {
        const plays = try gpa.alloc(Play, moves.len);
        errdefer gpa.free(plays);
        const stat_buf = try gpa.alloc(Stat, 2 * moves.len * plies);
        errdefer gpa.free(stat_buf);
        @memset(stat_buf, .{});
        for (plays, moves, 0..) |*p, m, i| {
            const s = stat_buf[2 * i * plies ..][0 .. 2 * plies];
            p.* = .{ .move = m, .scores = s[0..plies], .bingos = s[plies..] };
        }
        const gen = try gpa.create(movegen.Generator);
        errdefer gpa.destroy(gen);
        // Each rollout points the generator at its own board first.
        gen.* = .init(gpa, @constCast(&g.pos.board), g.pos.lexicon.kwg, g.pos.dist);
        gen.word_map = g.pos.word_map;
        errdefer gen.deinit();
        const carry = try gpa.create(carry_mod.Carry);
        errdefer gpa.destroy(carry);
        carry.* = .init(gpa);
        gen.carry = carry;

        var sim: Sim = .{
            .gpa = gpa,
            .plays = plays,
            .plies = plies,
            .win_table = win_table,
            .start = g.*,
            .work = undefined,
            .rng = .init(threadSeed(seed, turn, 0)),
            .gen = gen,
            .carry = carry,
            .initial_player = g.on_turn,
            .initial_spread = g.pos.scores[g.on_turn] - g.pos.scores[1 - g.on_turn],
            .stat_buf = stat_buf,
            .bai_seed = undefined,
        };
        var buf: [40]u8 = undefined;
        buf[0..32].* = seed;
        std.mem.writeInt(u64, buf[32..40], turn, .big);
        std.crypto.hash.sha2.Sha256.hash(&buf, &sim.bai_seed, .{});
        // The copies draw the bag as it lies, after one shuffle from the
        // thread's source (makeGameCopies).
        sim.start.paired = false;
        sim.start.bag.fixed_order = true;
        sim.start.bag.shuffle(&sim.rng);
        return sim;
    }

    pub fn deinit(self: *Sim) void {
        self.carry.deinit();
        self.gpa.destroy(self.carry);
        self.gen.deinit();
        self.gpa.destroy(self.gen);
        self.gpa.free(self.stat_buf);
        self.gpa.free(self.plays);
        self.* = undefined;
    }

    /// One iteration: a new opponent rack, then a rollout of every play
    /// not ignored.
    pub fn iterate(self: *Sim) Allocator.Error!void {
        self.dealOpponent();
        for (self.plays) |*p| {
            if (p.ignored) continue;
            _ = try self.rollout(p);
        }
        self.iterations += 1;
    }

    /// Throws the opponent's rack in and deals a new one (SetRandomRack
    /// with nothing known: Bag.Redraw).
    fn dealOpponent(self: *Sim) void {
        const g = &self.start;
        const opp = 1 - self.initial_player;
        var old: [tiles.max_alphabet_size]MachineLetter = undefined;
        const n = g.pos.racks[opp].tiles(&old);
        g.rng = self.rng;
        g.putBack(old[0..n]);
        var drawn: [game_mod.rack_tile_limit]MachineLetter = undefined;
        const k = g.drawAtMost(&drawn);
        g.pos.racks[opp] = .fromLetters(drawn[0..k]);
        self.rng = g.rng;
    }

    /// A best-arm sim: samples are single rollouts of one play, and the
    /// runner picks which plays to roll out (mostly the leader and its
    /// closest challenger) and when the leader is told apart from the rest
    /// (Go's simBAI with one thread). Sample n of every play deals the
    /// opponent the same rack and the bag the same draws.
    pub fn runBai(self: *Sim, opts: BaiOptions) !bai.Result {
        var runner = try bai.Runner.init(self.gpa, self.plays.len);
        defer runner.deinit();
        const n: u64 = self.plays.len;
        const cb: bai.Callbacks(*Sim) = .{ .ctx = self, .sample = baiSample, .similar = baiSimilar, .stop = opts.stop };
        const res = try runner.run(.{
            .delta = opts.delta,
            .sample_minimum = opts.min_samples,
            .cutoff = bai_cutoff,
            .sample_limit = opts.cap_per_play * n,
            .best_arm_limit = opts.best_play_cap,
        }, cb);
        self.winner = res.best;
        self.iterations = res.samples / @max(n, 1);
        return res;
    }

    fn baiSample(self: *Sim, arm: usize, ordinal: u64) anyerror!f64 {
        var buf: [40]u8 = undefined;
        buf[0..32].* = self.bai_seed;
        std.mem.writeInt(u64, buf[32..40], ordinal, .big);
        var seed: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(&buf, &seed, .{});
        self.dealSeeded(.init(seed));
        const r = try self.rollout(&self.plays[arm]);
        return baiReward(r.win, r.equity + @as(f64, @floatFromInt(self.initial_spread)), default_bai_spread_weight);
    }

    /// Deals the opponent a rack from the unseen tiles put in order and
    /// shuffled with `rng`, which the bag then draws with
    /// (SetRandomRackSeeded with nothing known).
    fn dealSeeded(self: *Sim, rng: chacha8.Rng) void {
        const g = &self.start;
        const opp = 1 - self.initial_player;
        var old: [tiles.max_alphabet_size]MachineLetter = undefined;
        const n = g.pos.racks[opp].tiles(&old);
        // Go puts them back with the last sample's source, but sorts the
        // bag next, which undoes any order that gave.
        g.bag.putBack(old[0..n]);
        std.mem.sort(MachineLetter, g.bag.tiles[0..g.bag.len], {}, std.sort.asc(MachineLetter));
        g.rng = rng;
        g.bag.shuffle(&g.rng);
        var drawn: [game_mod.rack_tile_limit]MachineLetter = undefined;
        const k = g.drawAtMost(&drawn);
        g.pos.racks[opp] = .fromLetters(drawn[0..k]);
        self.rng = g.rng;
    }

    fn baiSimilar(self: *Sim, a: usize, b: usize) bool {
        return similarPlays(&self.plays[a].move, &self.plays[b].move);
    }

    fn rollout(self: *Sim, p: *Play) Allocator.Error!struct { win: f64, equity: f64 } {
        const g = &self.work;
        g.* = self.start;
        g.rng = self.rng;
        defer self.rng = g.rng;
        self.gen.board = &g.pos.board;

        g.play(&p.move);
        self.nodes += 1;
        var leftover: f64 = 0;
        for (0..self.plies) |ply| {
            const on_turn = g.on_turn;
            if (!g.playing) break;
            const rack = g.pos.racks[on_turn];
            const best = try self.bestStaticTurn(g);
            if (ply + 1 == self.plies) g.playLast(&best.move) else g.play(&best.move);
            self.nodes += 1;
            if (ply + 2 >= self.plies) {
                // Our last turn or the opponent's: count what they keep.
                const leave = best.move.leave(&rack);
                var buf: [tiles.max_alphabet_size]MachineLetter = undefined;
                const v = if (g.pos.leaves) |klv| klv.leaveValue(buf[0..leave.tiles(&buf)]) else 0;
                if (on_turn == self.initial_player) leftover += v else leftover -= v;
            }
            p.scores[ply].push(@floatFromInt(best.move.score));
            const bingo = best.move.kind == .play and best.move.tiles_played == game_mod.rack_tile_limit;
            p.bingos[ply].push(if (bingo) 1 else 0);
        }
        const ip = self.initial_player;
        const spread = g.pos.scores[ip] - g.pos.scores[1 - ip];
        const eq = @as(f64, @floatFromInt(spread - self.initial_spread)) + leftover;
        p.equity.push(eq);
        p.leftover.push(leftover);
        const unseen = g.tilesInBag() + g.pos.racks[1 - ip].num_tiles;
        const win = self.win_table.rolloutWin(spread, leftover, !g.playing, unseen, self.plies % 2 == 0);
        p.win.push(win);
        return .{ .win = win, .equity = eq };
    }

    /// The best static move for the player on turn (GenBestStaticTurn).
    fn bestStaticTurn(self: *Sim, g: *Game) Allocator.Error!movegen.Generator.Best {
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
        var sim = try Sim.init(testing.allocator, &g, @splat(3), 0, &moves, 2, wt);
        defer sim.deinit();
        for (0..40) |_| try sim.iterate();
        var order: [2]*const Play = undefined;
        sim.ranked(&order);
        try testing.expectEqual(.play, order[0].move.kind);
        try testing.expectEqual(40, sim.plays[1].win.n);
        try testing.expectEqual(2 * 40 * 3, sim.nodes);
        r.* = .{ sim.plays[0].equity.mean(), sim.plays[1].win.mean() };
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
                var sim = try Sim.init(testing.allocator, &g, @splat(5), turn, ranked, 3, wt);
                defer sim.deinit();
                sim.check = &check;
                for (0..12) |_| try sim.iterate();
                hits += sim.carry.rack_hits;
                skipped += sim.carry.lanes_skipped;
                seeded += sim.carry.seeded;
            }
            gen.board = &g.pos.board;
            _ = try game_mod.playStaticTurn(&g, &gen);
        }
        try testing.expect(hits > 100);
        try testing.expect(skipped > 100);
        try testing.expect(seeded > 10);
    }
}
