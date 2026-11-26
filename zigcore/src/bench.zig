//! Times the best-move search (`go static`) on a file of positions, one CGP
//! per line, to compare with the Go engine's on the same file (see
//! crosscheck/bench_test.go). Usage:
//!
//!     zig build bench -Doptimize=ReleaseFast -- <positions> [rounds] [data path]
//!
//! or times making a lexicon's word map from its KWG:
//!
//!     zig build bench -Doptimize=ReleaseFast -- wmp <lexicon> [data path]

const std = @import("std");
const Io = std.Io;
const macondo = @import("macondo");
const Files = @import("files.zig").Files;

/// A sampling profiler for the bench: with BENCH_PROFILE=<file>, a
/// profiling timer interrupts the run every 50 us of CPU time and the
/// interrupted instruction's address is kept; at the end they go to the
/// file, one hex address per line, for llvm-symbolizer --inlining.
const Profiler = struct {
    var samples: [1 << 22]u64 = undefined;
    var count: usize = 0;

    fn handler(_: std.os.linux.SIG, _: *const std.os.linux.siginfo_t, ctx: ?*anyopaque) callconv(.c) void {
        if (count == samples.len) return;
        // x86_64 Linux: the ucontext's flags, link and stack (40 bytes), then
        // the saved r8-r15, rdi, rsi, rbp, rbx, rdx, rax, rcx, rsp, rip.
        const words: [*]const u64 = @ptrCast(@alignCast(ctx.?));
        samples[count] = words[(40 / 8) + 16];
        count += 1;
    }

    fn start() void {
        const linux = std.os.linux;
        var act: linux.Sigaction = .{
            .handler = .{ .sigaction = handler },
            .mask = linux.sigemptyset(),
            .flags = linux.SA.SIGINFO | linux.SA.RESTART,
        };
        _ = linux.sigaction(.PROF, &act, null);
        // The kernel reads a timeval: the second field is microseconds.
        const t: linux.itimerspec = .{ .it_interval = .{ .sec = 0, .nsec = 50 }, .it_value = .{ .sec = 0, .nsec = 50 } };
        _ = linux.setitimer(@backingInt(linux.ITIMER.PROF), &t, null);
    }

    fn stop(io: Io, path: []const u8) !void {
        const linux = std.os.linux;
        const t: linux.itimerspec = .{ .it_interval = .{ .sec = 0, .nsec = 0 }, .it_value = .{ .sec = 0, .nsec = 0 } };
        _ = linux.setitimer(@backingInt(linux.ITIMER.PROF), &t, null);
        const file = try Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var w = file.writer(io, &buf);
        for (samples[0..count]) |a| try w.interface.print("0x{x}\n", .{a});
        try w.interface.flush();
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.log.err("usage: bench <positions file> [rounds] [data path]", .{});
        return error.Usage;
    }
    if (std.mem.eql(u8, args[1], "wmp")) return benchWmp(init, args);
    const rounds = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 5;
    const data_path = if (args.len > 3) args[3] else "../data";

    var files: Files = .{ .io = io };
    var store: macondo.data.Store = .init(gpa, files.source());
    defer store.deinit();

    const text = try Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .unlimited);
    defer gpa.free(text);
    var positions: std.ArrayList(macondo.position.Position) = .empty;
    defer positions.deinit(gpa);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var p: macondo.position.Position = undefined;
        p.fromCgpWith(line, &store, data_path, true) catch |err| {
            std.log.err("{t} for {s} ({s})", .{ err, line, store.lastPath() });
            return err;
        };
        try positions.append(gpa, p);
    }

    var out_buf: [512]u8 = undefined;
    var w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    if (std.mem.eql(u8, init.environ_map.get("BENCH_MODE") orelse "", "phases")) return benchPhases(io, gpa, positions.items, rounds, init.environ_map.get("BENCH_NO_WMP") != null);
    if (std.mem.eql(u8, init.environ_map.get("BENCH_MODE") orelse "", "sim")) {
        const profile = init.environ_map.get("BENCH_PROFILE");
        if (profile != null) Profiler.start();
        defer if (profile) |path| Profiler.stop(io, path) catch {};
        return benchSim(io, gpa, &store, data_path, positions.items, rounds, &w.interface);
    }
    // BENCH_MODE=wmp or gaddag times one only (for profiling).
    const only = init.environ_map.get("BENCH_MODE");
    const profile = init.environ_map.get("BENCH_PROFILE");
    if (profile != null) Profiler.start();
    defer if (profile) |path| Profiler.stop(io, path) catch {};
    for ([_]bool{ false, true }) |use_wmp| {
        if (only) |m| if (std.mem.eql(u8, m, "wmp") != use_wmp) continue;
        var checksum: f64 = 0;
        var best_ns: i96 = std.math.maxInt(i96);
        // One generator for all, as a game or a sim uses one.
        const gen = try gpa.create(macondo.movegen.Generator);
        defer gpa.destroy(gen);
        gen.* = .init(gpa, &positions.items[0].board, positions.items[0].lexicon.kwg, positions.items[0].dist);
        defer gen.deinit();
        for (0..rounds) |round| {
            const start = Io.Timestamp.now(io, .awake);
            var sum: f64 = 0;
            for (positions.items) |*p| {
                gen.board = &p.board;
                gen.kwg = p.lexicon.kwg;
                gen.ld = p.dist;
                gen.word_map = if (use_wmp) p.word_map else null;
                const ex = macondo.movegen.exchangeRules(p.unseenBy(0), p.lexiconName());
                gen.max_can_exchange = ex.max;
                const ctx = p.equityContext();
                var rack = p.racks[0];
                const best = try gen.genBest(&rack, ex.allowed, &ctx);
                sum += best.equity;
            }
            const ns = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
            best_ns = @min(best_ns, ns);
            if (round == 0) {
                checksum = sum;
                try w.interface.print("{s} round 1: {d:.2} us/position\n", .{ if (use_wmp) "word maps" else "GADDAG", @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(positions.items.len)) / 1000 });
            }
        }
        const n = positions.items.len;
        const per: f64 = @as(f64, @floatFromInt(best_ns)) / @as(f64, @floatFromInt(n));
        try w.interface.print("{s}: {d} positions, best of {d} rounds: {d:.2} us/position, {d:.0} positions/s, equity sum {d:.6}\n", .{ if (use_wmp) "word maps" else "GADDAG", n, rounds, per / 1000, 1e9 / per, checksum });
    }
    try w.interface.flush();
}

/// BENCH_MODE=sim: a single-threaded sim of each position's 15 best plays
/// by static equity, two plies, `iterations` iterations (`go sim plies 2
/// iterations <n> plays 15`, without the opponent's rack).
fn benchSim(io: Io, gpa: std.mem.Allocator, store: *macondo.data.Store, data_path: []const u8, positions: []macondo.position.Position, iterations: u32, w: *Io.Writer) !void {
    const win_table = try store.winTable(data_path);
    var rollouts: u64 = 0;
    var nodes: u64 = 0;
    var total_ns: i96 = 0;
    var gens: u64 = 0;
    var hits: u64 = 0;
    for (positions) |*p| {
        p.racks[1] = .{};
        const ranked = try macondo.ucgi.rankedMoves(gpa, p);
        defer gpa.free(ranked);
        const moves = try gpa.alloc(macondo.move.Move, @min(15, ranked.len));
        defer gpa.free(moves);
        for (moves, ranked[0..moves.len]) |*m, r| m.* = r.m;
        const seed = macondo.game.deriveSeed(1, 0);
        const g = try gpa.create(macondo.game.Game);
        defer gpa.destroy(g);
        g.* = try .fromPosition(p, seed);
        const sim = try gpa.create(macondo.sim.Sim);
        defer gpa.destroy(sim);
        sim.* = try .init(gpa, g, seed, 0, moves, 2, win_table);
        defer sim.deinit();
        const start = Io.Timestamp.now(io, .awake);
        for (0..iterations) |_| try sim.iterate();
        total_ns += start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
        rollouts += iterations * moves.len;
        nodes += sim.nodes;
        gens += sim.carry.generations;
        hits += sim.carry.rack_hits;
    }
    try w.print("sim: {d} positions, {d} rollouts, {d:.2} us/rollout, {d:.0} nodes/s\n", .{
        positions.len,                                                                rollouts,
        @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(rollouts)) / 1000, @as(f64, @floatFromInt(nodes)) * 1e9 / @as(f64, @floatFromInt(total_ns)),
    });
    try w.print("best-move memory: {d} generations, {d:.1}% found their rack\n", .{ gens, 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(gens, 1))) });
    try w.flush();
}

fn benchWmp(init: std.process.Init, args: []const []const u8) !void {
    const io = init.io;
    const gpa = init.gpa;
    if (args.len < 3) return error.Usage;
    const data_path = if (args.len > 3) args[3] else "../data";
    var files: Files = .{ .io = io };
    var store: macondo.data.Store = .init(gpa, files.source());
    defer store.deinit();
    const lex = try store.lexicon(data_path, args[2]);
    const ld = try store.distribution(data_path, "english");
    const start = Io.Timestamp.now(io, .awake);
    const bytes = try macondo.wmp_maker.makeFromKwg(gpa, lex.kwg, ld, 15);
    defer gpa.free(bytes);
    const ns = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
    var out_buf: [256]u8 = undefined;
    var w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    try w.interface.print("{s}: {d} bytes in {d:.3} s (one thread)\n", .{ args[2], bytes.len, @as(f64, @floatFromInt(ns)) / 1e9 });
    try w.interface.flush();
}

/// The best-play search's phases timed apart (word maps on), as genBest
/// runs them.
fn benchPhases(io: Io, gpa: std.mem.Allocator, positions: []macondo.position.Position, rounds: u32, no_wmp: bool) !void {
    const names = [_][]const u8{ "leaves and exchanges", "word-map subracks", "shadow", "anchors" };
    var best: [names.len]i96 = @splat(std.math.maxInt(i96));
    const gen = try gpa.create(macondo.movegen.Generator);
    defer gpa.destroy(gen);
    gen.* = .init(gpa, &positions[0].board, positions[0].lexicon.kwg, positions[0].dist);
    defer gen.deinit();
    for (0..rounds) |_| {
        var total: [names.len]i96 = @splat(0);
        for (positions) |*p| {
            gen.board = &p.board;
            gen.kwg = p.lexicon.kwg;
            gen.ld = p.dist;
            gen.word_map = if (no_wmp) null else p.word_map;
            const ex = macondo.movegen.exchangeRules(p.unseenBy(0), p.lexiconName());
            gen.max_can_exchange = ex.max;
            const ctx = p.equityContext();
            var rack = p.racks[0];
            var t = Io.Timestamp.now(io, .awake);
            gen.bestStart(&rack, ex.allowed, &ctx);
            t = lap(io, t, &total[0]);
            gen.bestSubracks(&rack);
            t = lap(io, t, &total[1]);
            macondo.shadow.genShadow(gen, &rack);
            t = lap(io, t, &total[2]);
            macondo.shadow.recordFromAnchors(gen, &rack);
            t = lap(io, t, &total[3]);
            _ = try gen.bestFinish(&rack);
        }
        for (&best, total) |*b, x| b.* = @min(b.*, x);
    }
    var out_buf: [512]u8 = undefined;
    var w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    for (names, best) |name, ns| {
        try w.interface.print("{s}: {d:.2} us/position\n", .{ name, @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(positions.len)) / 1000 });
    }
    try w.interface.flush();
}

fn lap(io: Io, t: Io.Timestamp, total: *i96) Io.Timestamp {
    const now = Io.Timestamp.now(io, .awake);
    total.* += t.durationTo(now).nanoseconds;
    return now;
}
