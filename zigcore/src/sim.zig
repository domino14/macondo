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
    initial_player: u1,
    initial_spread: i32,
    iterations: u64 = 0,
    /// Moves made (candidate plays and rollout plies), as Go counts nodes.
    nodes: u64 = 0,
    stat_buf: []Stat,

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

        var sim: Sim = .{
            .gpa = gpa,
            .plays = plays,
            .plies = plies,
            .win_table = win_table,
            .start = g.*,
            .work = undefined,
            .rng = .init(threadSeed(seed, turn, 0)),
            .gen = gen,
            .initial_player = g.on_turn,
            .initial_spread = g.pos.scores[g.on_turn] - g.pos.scores[1 - g.on_turn],
            .stat_buf = stat_buf,
        };
        // The copies draw the bag as it lies, after one shuffle from the
        // thread's source (makeGameCopies).
        sim.start.paired = false;
        sim.start.bag.fixed_order = true;
        sim.start.bag.shuffle(&sim.rng);
        return sim;
    }

    pub fn deinit(self: *Sim) void {
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
            try self.rollout(p);
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

    fn rollout(self: *Sim, p: *Play) Allocator.Error!void {
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
            const best = try game_mod.playStaticTurn(g, self.gen);
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
        p.equity.push(@as(f64, @floatFromInt(spread - self.initial_spread)) + leftover);
        p.leftover.push(leftover);
        const unseen = g.tilesInBag() + g.pos.racks[1 - ip].num_tiles;
        p.win.push(self.win_table.rolloutWin(spread, leftover, !g.playing, unseen, self.plies % 2 == 0));
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
