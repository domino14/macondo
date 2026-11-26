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
    if (std.mem.eql(u8, args[1], "rit")) return benchRit(init, args);
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
        p.fromCgpWith(line, &store, data_path, .{ .word_maps = true, .rack_tables = if (init.environ_map.get("BENCH_RIT") != null) .auto else .never }) catch |err| {
            std.log.err("{t} for {s} ({s})", .{ err, line, store.lastPath() });
            return err;
        };
        try positions.append(gpa, p);
    }

    var out_buf: [512]u8 = undefined;
    var w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    if (std.mem.eql(u8, init.environ_map.get("BENCH_MODE") orelse "", "phases")) return benchPhases(io, gpa, positions.items, rounds, init.environ_map.get("BENCH_NO_WMP") != null);
    if (std.mem.eql(u8, init.environ_map.get("BENCH_MODE") orelse "", "endgame")) {
        const profile = init.environ_map.get("BENCH_PROFILE");
        if (profile != null) Profiler.start();
        defer if (profile) |path| Profiler.stop(io, path) catch {};
        return benchEndgame(io, gpa, positions.items, rounds, init.environ_map.get("BENCH_ROWS") != null, init.environ_map.get("BENCH_GO_ORDER") != null, &w.interface);
    }
    if (std.mem.eql(u8, init.environ_map.get("BENCH_MODE") orelse "", "sim")) {
        const profile = init.environ_map.get("BENCH_PROFILE");
        if (profile != null) Profiler.start();
        defer if (profile) |path| Profiler.stop(io, path) catch {};
        const plies: u32 = if (init.environ_map.get("BENCH_PLIES")) |v| try std.fmt.parseInt(u32, v, 10) else 2;
        const threads: usize = if (init.environ_map.get("BENCH_THREADS")) |v| try std.fmt.parseInt(usize, v, 10) else 1;
        return benchSim(io, gpa, &store, data_path, positions.items, rounds, plies, threads, init.environ_map.get("BENCH_BAI") != null, &w.interface);
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
                gen.rit = if (use_wmp) p.rit else null;
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
/// by static equity, two plies (BENCH_PLIES), on BENCH_THREADS workers,
/// `iterations` iterations (with BENCH_BAI, a best-arm run to 99% with a
/// budget of `iterations` rollouts a play) (`go sim plies 2
/// iterations <n> plays 15`, without the opponent's rack).
fn benchSim(io: Io, gpa: std.mem.Allocator, store: *macondo.data.Store, data_path: []const u8, positions: []macondo.position.Position, iterations: u32, plies: u32, threads: usize, bai: bool, w: *Io.Writer) !void {
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
        // Timed from the sim's creation (its workers' set-up counts).
        const start = Io.Timestamp.now(io, .awake);
        const sim = try macondo.sim.Sim.create(gpa, g, seed, 0, moves, plies, win_table, threads);
        defer sim.destroy();
        if (bai) {
            // A best-arm run to 99% with a budget of `iterations` rollouts
            // a play.
            const res = try sim.runBai(io, .{ .delta = 0.01, .cap_per_play = iterations });
            rollouts += res.samples;
        } else {
            try sim.run(io, .{ .iterations = iterations });
            rollouts += iterations * moves.len;
        }
        total_ns += start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
        nodes += sim.nodes();
        for (sim.workers) |*wk| {
            gens += wk.carry.generations;
            hits += wk.carry.rack_hits;
        }
    }
    try w.print("sim ({d} threads): {d} positions, {d} rollouts, {d:.2} us/rollout, {d:.0} nodes/s\n", .{
        threads,
        positions.len,
        rollouts,
        @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(rollouts)) / 1000,
        @as(f64, @floatFromInt(nodes)) * 1e9 / @as(f64, @floatFromInt(total_ns)),
    });
    try w.print("best-move memory: {d} generations, {d:.1}% found their rack\n", .{ gens, 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(gens, 1))) });
    try w.flush();
}

/// BENCH_MODE=endgame: solves each position's endgame to `plies` plies
/// (the second argument) on one thread, with a table of 2^24 entries
/// cleared before each, outside the timing (`go endgame plies <n>`). With BENCH_ROWS, a line
/// per position; with BENCH_GO_ORDER, Go's move ordering.
fn benchEndgame(io: Io, gpa: std.mem.Allocator, positions: []macondo.position.Position, plies: u32, rows: bool, go_order: bool, w: *Io.Writer) !void {
    var tt: macondo.ttable.Table = .{};
    defer tt.deinit(gpa);
    var zobrist: ?macondo.zobrist.Zobrist = null;
    defer if (zobrist) |*z| z.deinit(gpa);
    var nodes: u64 = 0;
    var value_sum: i64 = 0;
    var total_ns: i96 = 0;
    for (positions, 0..) |*p, i| {
        if (zobrist == null or zobrist.?.dim != p.board.dim) {
            if (zobrist) |*z| z.deinit(gpa);
            zobrist = try .init(gpa, p.board.dim, @splat(0x5a));
        }
        // As MAGPIE's benchmark: the table is cleared before the clock starts.
        try tt.reset(gpa, 256 << 20);
        const start = Io.Timestamp.now(io, .awake);
        const solver = try macondo.endgame.Solver.init(gpa, p, &zobrist.?, &tt, .{ .clear_tt = false, .magpie_ordering = !go_order });
        defer solver.deinit();
        const r = try solver.solve(plies, null, null);
        const ns = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
        total_ns += ns;
        nodes += r.nodes;
        value_sum += r.value;
        if (rows) try w.print("{d} value {d} depth {d} nodes {d} time {d:.4}\n", .{ i, r.value, r.depth, r.nodes, @as(f64, @floatFromInt(ns)) / 1e9 });
    }
    try w.print("endgame ({d} plies, 1 thread): {d} positions in {d:.3} s, {d} nodes, {d:.0} nodes/s, value sum {d}\n", .{
        plies,
        positions.len,
        @as(f64, @floatFromInt(total_ns)) / 1e9,
        nodes,
        @as(f64, @floatFromInt(nodes)) * 1e9 / @as(f64, @floatFromInt(@max(total_ns, 1))),
        value_sum,
    });
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

/// `rit <lexicon> [data] [out|-] [MAGPIE .rit]`: makes the lexicon's rack
/// info table (with its own leaves and word map) on every core, times it,
/// writes it to `out` if given, and compares the word-list fields with a
/// MAGPIE table made from the same files if given.
fn benchRit(init: std.process.Init, args: []const []const u8) !void {
    const io = init.io;
    const gpa = init.gpa;
    if (args.len < 3) return error.Usage;
    const data_path = if (args.len > 3) args[3] else "../data";
    var files: Files = .{ .io = io };
    var store: macondo.data.Store = .init(gpa, files.source());
    defer store.deinit();
    const lex = try store.lexicon(data_path, args[2]);
    const ld = try store.distribution(data_path, "english");
    const leaves = (try store.leavesFor(data_path, args[2], false)).?;
    const word_map = (try store.wordMap(data_path, args[2], lex, ld)).?;
    const threads = try std.Thread.getCpuCount();
    var out_buf: [512]u8 = undefined;
    var w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    const start = Io.Timestamp.now(io, .awake);
    const bytes = try macondo.rit.make(gpa, .{ .ld = ld, .kwg_bytes = lex.bytes, .klv = &leaves.klv, .klv_bytes = leaves.bytes, .word_map = word_map }, threads);
    defer gpa.free(bytes);
    const ns = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
    const rit = try macondo.rit.Rit.init(bytes);
    try w.interface.print("{s}: {d} racks, {d} bytes in {d:.1} s ({d} threads)\n", .{ args[2], rit.entries.len, bytes.len, @as(f64, @floatFromInt(ns)) / 1e9, threads });
    try w.interface.flush();
    if (args.len > 4 and !std.mem.eql(u8, args[4], "-")) {
        const f = try Io.Dir.cwd().createFile(io, args[4], .{});
        defer f.close(io);
        try f.writePositionalAll(io, bytes, 0);
    }
    if (args.len > 5) try compareMagpieRit(io, gpa, &rit, args[5], &w.interface);
}

/// Compares a table with MAGPIE's (version 13 layout): the word-list
/// fields must be equal, the leave values equal once rounded to MAGPIE's
/// thousandths, and the best exchanges equal but where MAGPIE's rounding
/// ties two.
fn compareMagpieRit(io: Io, gpa: std.mem.Allocator, rit: *const macondo.rit.Rit, path: []const u8, w: *Io.Writer) !void {
    const m = try Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .unlimited, .@"16", null);
    defer gpa.free(m);
    const num_buckets = std.mem.readInt(u32, m[4..8], .little);
    const num_entries = std.mem.readInt(u32, m[8..12], .little);
    const entries_at = 12 + (@as(usize, num_buckets) + 1) * 4;
    const esize = (m.len - entries_at - 16) / num_entries;
    try w.print("MAGPIE table: version {d}, {d} entries of {d} bytes\n", .{ m[0], num_entries, esize });
    if (num_entries != rit.entries.len) return error.EntryCountDiffers;
    var diffs: [6]u64 = @splat(0);
    const names = [_][]const u8{ "playthrough union", "tp7 masks", "tp6 masks", "np word lengths", "leaves (rounded)", "best exchange (beyond rounding ties)" };
    for (0..num_entries) |i| {
        const e = m[entries_at + i * esize ..][0..esize];
        const br = std.mem.readInt(u128, e[esize - 16 ..][0..16], .little);
        const z = rit.lookup(br) orelse return error.RackMissing;
        var off: usize = 384;
        const u32At = struct {
            fn f(b: []const u8, o: usize) u32 {
                return std.mem.readInt(u32, b[o..][0..4], .little);
            }
        }.f;
        for (0..8) |k| {
            if (u32At(e, off + 4 * k) != z.playthrough_union[k]) {
                diffs[0] += 1;
                break;
            }
        }
        off += 32 + 32 + 32; // union, np best leaves, best leaves
        for (0..8) |k| {
            if (u32At(e, off + 4 * k) != z.multi_pt_tp7[k]) {
                diffs[1] += 1;
                break;
            }
        }
        off += 32;
        for (0..9) |k| {
            if (u32At(e, off + 4 * k) != z.multi_pt_tp6[k]) {
                diffs[2] += 1;
                break;
            }
        }
        off += 36;
        if (e[off] != z.np_has_word) diffs[3] += 1;
        off += 1 + 1 + 7; // np words, bingo count, bingo words
        const strip = e[off..][0..7];
        const n_ex = e[off + 7];
        if (n_ex != z.exchange_len or !std.mem.eql(u8, strip[0..n_ex], z.exchange[0..z.exchange_len])) {
            // Fine if MAGPIE's thousandths tie the two: compare the exact
            // values of both exchanges' leaves.
            const theirs_v = exchangeLeaveValue(z, strip[0..n_ex]);
            const mine_v = exchangeLeaveValue(z, z.exchange[0..z.exchange_len]);
            if (@round(theirs_v * 1000) != @round(mine_v * 1000) or theirs_v > mine_v) diffs[5] += 1;
        }
        // Leaves: MAGPIE's leave map index runs over the played tiles
        // where Zig's runs over those kept, so compare as sets.
        var mine: [128]i32 = undefined;
        var theirs: [128]i32 = undefined;
        for (0..128) |k| {
            mine[k] = @intFromFloat(@round(@as(f64, z.leaves[k]) * 1000));
            const raw: i32 = @as(i32, e[3 * k]) | (@as(i32, e[3 * k + 1]) << 8) | (@as(i32, e[3 * k + 2]) << 16);
            theirs[k] = (raw ^ (1 << 23)) - (1 << 23);
        }
        std.mem.sort(i32, &mine, {}, std.sort.asc(i32));
        std.mem.sort(i32, &theirs, {}, std.sort.asc(i32));
        if (!std.mem.eql(i32, &mine, &theirs)) diffs[4] += 1;
    }
    for (names, diffs) |name, d| try w.print("  {s}: {d} racks differ\n", .{ name, d });
    try w.flush();
}

/// The exact leave value of exchanging `strip` from entry `e`'s rack.
fn exchangeLeaveValue(e: *const macondo.rit.Entry, strip: []const u8) f64 {
    var rack: macondo.tiles.Rack = .{};
    for (0..32) |ml| {
        for (0..macondo.wmp.count(e.rack, @intCast(ml))) |_| rack.add(@intCast(ml));
    }
    var lm: macondo.leavemap.LeaveMap = .{};
    lm.init(&rack);
    var leave = rack;
    for (strip) |ml| leave.take(ml);
    var idx: u32 = 0;
    for (leave.counts, 0..) |n, ml| {
        if (n > 0) idx |= ((@as(u32, 1) << @intCast(n)) - 1) << @intCast(lm.base[ml]);
    }
    return e.leaves[idx];
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
