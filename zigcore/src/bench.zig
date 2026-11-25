//! Times the best-move search (`go static`) on a file of positions, one CGP
//! per line, to compare with the Go engine's on the same file (see
//! crosscheck/bench_test.go). Usage:
//!
//!     zig build bench -Doptimize=ReleaseFast -- <positions> [rounds] [data path]

const std = @import("std");
const Io = std.Io;
const macondo = @import("macondo");
const Files = @import("files.zig").Files;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.log.err("usage: bench <positions file> [rounds] [data path]", .{});
        return error.Usage;
    }
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
        p.fromCgp(line, &store, data_path) catch |err| {
            std.log.err("{t} for {s} ({s})", .{ err, line, store.lastPath() });
            return err;
        };
        try positions.append(gpa, p);
    }

    var checksum: f64 = 0;
    var best_ns: i96 = std.math.maxInt(i96);
    for (0..rounds) |round| {
        const start = Io.Timestamp.now(io, .awake);
        var sum: f64 = 0;
        for (positions.items) |*p| {
            var gen = macondo.movegen.Generator.init(gpa, &p.board, p.lexicon.kwg, p.dist);
            defer gen.deinit();
            const ex = macondo.movegen.exchangeRules(p.unseenBy(0), p.lexiconName());
            gen.max_can_exchange = ex.max;
            const ctx = p.equityContext();
            var rack = p.racks[0];
            const best = try gen.genBest(&rack, ex.allowed, &ctx);
            sum += best.equity;
        }
        const ns = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
        best_ns = @min(best_ns, ns);
        if (round == 0) checksum = sum;
    }
    const n = positions.items.len;
    const per: f64 = @as(f64, @floatFromInt(best_ns)) / @as(f64, @floatFromInt(n));
    var out_buf: [256]u8 = undefined;
    var w: Io.File.Writer = .init(.stdout(), io, &out_buf);
    try w.interface.print("{d} positions, best of {d} rounds: {d:.2} us/position, {d:.0} positions/s, equity sum {d:.6}\n", .{ n, rounds, per / 1000, 1e9 / per, checksum });
    try w.interface.flush();
}
