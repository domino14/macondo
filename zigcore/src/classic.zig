//! Macondo's classic stopping rule (montecarlo's AutoStopper), for the
//! inference's mini-sims only: every `check_interval` iterations, rank the
//! plays, drop those whose upper confidence bound falls below the leader's
//! lower one (and, late, those that place the same tiles on the same
//! squares as the leader), and stop once one play is left, the leader is
//! confidently ahead of every play that can't be dropped, or the
//! iteration cutoff is reached. Searches use the best-arm rule instead.

const std = @import("std");
const sim_mod = @import("sim.zig");
const gomath = @import("gomath.zig");
const move_mod = @import("move.zig");
const Sim = sim_mod.Sim;
const Play = sim_mod.Play;
const Stat = sim_mod.Stat;

pub const Options = struct {
    /// Stop99.
    delta: f64 = 0.01,
    z: f64 = 2.576,
    check_interval: u64 = 64,
    iterations_cutoff: u64 = 2000,
    per_ply_scaling: u64 = 625,
    similar_plays_cutoff: u64 = 750,
    min_reasonable_win: f64 = 0.005,
};

/// The most plays the rule ranks: Go sorts them with sort.Slice, which is
/// an insertion sort up to 12 elements; past that its order between plays
/// the comparison can't separate would differ.
pub const max_plays = 12;

/// Plays iterations on the sim's first worker until `max_iters` (0: no
/// cap) or the rule says to stop (SimSingleThread with Stop99).
pub fn run(sim: *Sim, max_iters: u64, opts: Options) std.mem.Allocator.Error!void {
    std.debug.assert(sim.plays.len <= max_plays);
    var ran: u64 = 0;
    while (max_iters == 0 or ran < max_iters) : (ran += 1) {
        try sim.workers[0].iterate(null);
        sim.iterations += 1;
        if (sim.iterations % opts.check_interval == 0) {
            sim.collect(null);
            if (shouldStop(sim, sim.iterations, opts)) break;
        }
    }
    sim.collect(null);
}

/// A play as the rule saw it at the start of a check.
const Seen = struct {
    play: *Play,
    ignore: bool,
};

/// One check of the rule after `iteration_count` iterations. It may mark
/// plays ignored; true means stop.
pub fn shouldStop(sim: *Sim, iteration_count: u64, opts: Options) bool {
    const plays = sim.plays;
    if (plays.len < 2) return true;
    if (iteration_count > opts.iterations_cutoff + @as(u64, sim.plies) * opts.per_ply_scaling) return true;

    var buf: [max_plays]Seen = undefined;
    const seen = buf[0..plays.len];
    var ignored: usize = 0;
    for (seen, plays) |*s, *p| {
        s.* = .{ .play = p, .ignore = p.ignored };
        if (p.ignored) ignored += 1;
    }
    if (ignored >= seen.len - 1) return true;

    std.sort.insertion(Seen, seen, {}, ranksAbove);

    var bottom_win: f64 = 0;
    var bottom_err: f64 = 0;
    var i = seen.len;
    while (i > 0) {
        i -= 1;
        if (!seen[i].ignore) {
            bottom_win = seen[i].play.stats.win.mean();
            bottom_err = seen[i].play.stats.win.stdErr();
            break;
        }
    }

    var by_equity = false;
    var leader = seen[0];
    var mu = leader.play.stats.win.mean();
    var e = leader.play.stats.win.stdErr();
    if (zTest(opts.min_reasonable_win, mu, e, -opts.z, true)) {
        by_equity = true;
    } else if (zTest(1 - opts.min_reasonable_win, mu, e, opts.z, false) and
        zTest(1 - opts.min_reasonable_win, bottom_win, bottom_err, opts.z, false))
    {
        by_equity = true;
    }
    if (by_equity) {
        // The decided case: rank by equity. (Ignored plays count here, as
        // in Go.)
        var highest: f64 = -1000000.0;
        var at: usize = 0;
        var found = false;
        for (seen, 0..) |s, k| {
            const eq = s.play.stats.equity.mean();
            if (eq > highest) {
                highest = eq;
                at = k;
                found = true;
            }
        }
        if (found and at != 0) {
            std.mem.swap(Seen, &seen[0], &seen[at]);
            leader = seen[0];
        }
        mu = leader.play.stats.equity.mean();
        e = leader.play.stats.equity.stdErr();
    }

    var k: f64 = @floatFromInt(seen.len - ignored);
    if (k < 2) k = 2;
    const c = @sqrt(2 * gomath.log(k / opts.delta));
    const lcb_leader = mu - c * e;
    const min_iterations_for_pruning = 128;

    var new_ignored: usize = 0;
    for (seen[1..]) |s| {
        if (s.ignore or s.play.unignorable) continue;
        const st = statOf(s.play, by_equity);
        if (st.n < min_iterations_for_pruning) continue;
        const ucb = st.mean() + c * st.stdErr();
        if (lcb_leader > ucb) {
            s.play.ignored = true;
            new_ignored += 1;
        } else if (iteration_count > opts.similar_plays_cutoff and materiallySimilar(&leader.play.move, &s.play.move)) {
            s.play.ignored = true;
            new_ignored += 1;
        }
    }
    if (ignored + new_ignored >= seen.len - 1) return true;

    // Plays just dropped still count below, as they do in Go (it reads
    // the flags it copied at the start of the check).
    var remaining_ignorable: usize = 0;
    var beats_all = true;
    for (seen[1..]) |s| {
        if (s.ignore) continue;
        if (!s.play.unignorable) {
            remaining_ignorable += 1;
            continue;
        }
        const st = statOf(s.play, by_equity);
        if (st.n < min_iterations_for_pruning) {
            beats_all = false;
            continue;
        }
        if (lcb_leader <= st.mean() + c * st.stdErr()) beats_all = false;
    }
    return remaining_ignorable == 0 and beats_all;
}

fn statOf(p: *const Play, by_equity: bool) *const Stat {
    return if (by_equity) &p.stats.equity else &p.stats.win;
}

/// The rule's ranking: plays not ignored first, then by win rate (within
/// 1e-9 by equity).
fn ranksAbove(_: void, a: Seen, b: Seen) bool {
    if (a.ignore) return false;
    if (b.ignore) return true;
    const wa = a.play.stats.win.mean();
    const wb = b.play.stats.win.mean();
    if (@abs(wa - wb) <= 1e-9) return a.play.stats.equity.mean() > b.play.stats.equity.mean();
    return wa > wb;
}

fn zTest(mu: f64, m: f64, e: f64, z: f64, flip: bool) bool {
    const zc = (m - mu) / e;
    return if (flip) z > zc else zc > z;
}

/// The same tiles on the same squares in a different order (or with the
/// blank standing for something else).
fn materiallySimilar(a: *const move_mod.Move, b: *const move_mod.Move) bool {
    if (a.row != b.row or a.col != b.col or a.vertical != b.vertical) return false;
    if (a.tiles_played != b.tiles_played or a.len != b.len) return false;
    var x = a.tiles;
    var y = b.tiles;
    std.mem.sort(u8, x[0..a.len], {}, std.sort.asc(u8));
    std.mem.sort(u8, y[0..b.len], {}, std.sort.asc(u8));
    return std.mem.eql(u8, x[0..a.len], y[0..b.len]);
}
