//! UCGI command handling (see zigcore/UCGI.md). The engine takes one input
//! line at a time and writes its response lines to the writer it is given;
//! it never reads or writes anything else, so every adapter (stdio, WASM,
//! sockets) drives it the same way.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const data = @import("data.zig");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const Position = @import("position.zig").Position;
const movegen = @import("movegen.zig");
const Move = @import("move.zig").Move;
const equity = @import("equity.zig");
const game_mod = @import("game.zig");
const sim_mod = @import("sim.zig");

pub const engine_name = "MacondoZig";
pub const engine_author = "the Macondo authors";

const threads_max = 512;
const hash_mb_max = 65536;

pub const Options = struct {
    threads: u32 = 1,
    hash_mb: u32 = 256,
    /// Owned by the engine's allocator.
    data_path: []u8,
    /// Use word maps in the best-play search (making them if missing).
    word_maps: bool = true,
};

/// What the adapter should do after a command.
pub const Status = enum { @"continue", quit };

pub const Engine = struct {
    gpa: Allocator,
    version: []const u8,
    options: Options,
    store: data.Store,
    position: ?Position = null,

    /// `source` is how the engine reads data files; see data.Source.
    pub fn init(gpa: Allocator, version: []const u8, source: data.Source) Allocator.Error!Engine {
        return .{
            .gpa = gpa,
            .version = version,
            .options = .{ .data_path = try gpa.dupe(u8, "./data") },
            .store = .init(gpa, source),
        };
    }

    pub fn deinit(self: *Engine) void {
        self.store.deinit();
        self.gpa.free(self.options.data_path);
        self.* = undefined;
    }

    /// Handles one input line (without its trailing newline) and writes the
    /// response lines to `out`. The caller flushes `out`.
    pub fn handle(self: *Engine, line: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!Status {
        var tokens = std.mem.tokenizeAny(u8, line, " \t\r");
        const cmd = tokens.next() orelse return .@"continue";

        if (std.mem.eql(u8, cmd, "ucgi")) {
            try self.identify(out);
        } else if (std.mem.eql(u8, cmd, "isready")) {
            // Commands run synchronously for now, so everything sent before
            // this one has finished.
            try out.writeAll("readyok\n");
        } else if (std.mem.eql(u8, cmd, "setoption")) {
            try self.setOption(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "position")) {
            try self.setPosition(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "load")) {
            try self.load(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "autoplay")) {
            try self.autoplay(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "go")) {
            try self.go(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "gen")) {
            try self.printMoves(out);
        } else if (std.mem.eql(u8, cmd, "crosssets")) {
            try self.printCrossSets(out);
        } else if (std.mem.eql(u8, cmd, "leave")) {
            try self.printLeave(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "quit")) {
            return .quit;
        } else {
            try out.print("error unknown-command {s}\n", .{cmd});
        }
        return .@"continue";
    }

    fn identify(self: *Engine, out: *Writer) Writer.Error!void {
        try out.print("id name {s} {s}\n", .{ engine_name, self.version });
        try out.print("id author {s}\n", .{engine_author});
        try out.print("option name threads type spin default 1 min 1 max {d}\n", .{threads_max});
        try out.print("option name hash-mb type spin default 256 min 1 max {d}\n", .{hash_mb_max});
        try out.writeAll("option name data-path type string default ./data\n");
        try out.writeAll("option name word-maps type check default true\n");
        try out.writeAll("ucgiok\n");
    }

    /// `args` is the rest of the line after `setoption`:
    /// `name <id> value <value...>`. A string value is the rest of the line.
    fn setOption(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        const name_kw = tokens.next() orelse return out.writeAll("error bad-setoption\n");
        if (!std.mem.eql(u8, name_kw, "name")) return out.writeAll("error bad-setoption\n");
        const name = tokens.next() orelse return out.writeAll("error bad-setoption\n");
        const value_kw = tokens.next() orelse return out.writeAll("error bad-setoption\n");
        if (!std.mem.eql(u8, value_kw, "value")) return out.writeAll("error bad-setoption\n");
        const value = std.mem.trim(u8, tokens.rest(), " \t\r");
        if (value.len == 0) return out.writeAll("error bad-setoption\n");

        if (std.mem.eql(u8, name, "threads")) {
            self.options.threads = parseSpin(value, 1, threads_max) orelse
                return out.print("error bad-value {s}\n", .{name});
        } else if (std.mem.eql(u8, name, "hash-mb")) {
            self.options.hash_mb = parseSpin(value, 1, hash_mb_max) orelse
                return out.print("error bad-value {s}\n", .{name});
        } else if (std.mem.eql(u8, name, "word-maps")) {
            if (std.mem.eql(u8, value, "true")) {
                self.options.word_maps = true;
            } else if (std.mem.eql(u8, value, "false")) {
                self.options.word_maps = false;
            } else return out.print("error bad-value {s}\n", .{name});
        } else if (std.mem.eql(u8, name, "data-path")) {
            const copy = try self.gpa.dupe(u8, value);
            self.gpa.free(self.options.data_path);
            self.options.data_path = copy;
        } else {
            try out.print("error unknown-option {s}\n", .{name});
        }
    }

    /// Writes the error line for a failed data load.
    fn dataError(self: *Engine, err: data.Error, out: *Writer) (Writer.Error || Allocator.Error)!void {
        switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.FileNotFound => try out.print("error data-not-found {s}\n", .{self.store.lastPath()}),
            error.ReadFailed => try out.print("error data-read-failed {s}\n", .{self.store.lastPath()}),
            error.InvalidData => try out.print("error data-invalid {s}\n", .{self.store.lastPath()}),
            error.InvalidName => try out.writeAll("error bad-name\n"),
        }
    }

    /// `position cgp <CGP>`
    fn setPosition(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const trimmed = std.mem.trim(u8, args, " \t\r");
        if (!std.mem.startsWith(u8, trimmed, "cgp ")) return out.writeAll("error bad-position\n");
        const text = trimmed["cgp ".len..];
        var p: Position = undefined;
        p.fromCgpWith(text, &self.store, self.options.data_path, self.options.word_maps) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.FileNotFound, error.ReadFailed, error.InvalidData, error.InvalidName => |e| return self.dataError(e, out),
            else => |e| {
                try out.writeAll("error bad-cgp ");
                try writeKebab(out, @errorName(e));
                return out.writeAll("\n");
            },
        };
        self.position = p;
    }

    /// `load lex <name> [dist <name>]`: loads data ahead of time.
    fn load(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            const name = tokens.next() orelse return out.writeAll("error bad-load\n");
            if (std.mem.eql(u8, key, "lex")) {
                _ = self.store.lexicon(self.options.data_path, name) catch |err| return self.dataError(err, out);
            } else if (std.mem.eql(u8, key, "dist")) {
                _ = self.store.distribution(self.options.data_path, name) catch |err| return self.dataError(err, out);
            } else {
                return out.writeAll("error bad-load\n");
            }
        }
    }

    /// `autoplay games <n> seed <master> [pairs] [moves]`: plays n games
    /// between static players (HastyBot) on the position's layout and
    /// lexicon, game i seeded with automatic.DeriveSeed(master, i) as the
    /// Go engine's autoplay seeds it (with `pairs`, pair i's two games
    /// share that seed in paired-bag mode). Writes `game <i> score <a> <b>
    /// turns <t>` per game (with `moves`, each move first as `move <i>
    /// <move> score <s> equity <e>`), then `autoplayok games <n>`.
    fn autoplay(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        var games: u64 = 1;
        var master: u64 = 1;
        var pairs = false;
        var show_moves = false;
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            if (std.mem.eql(u8, key, "pairs")) {
                pairs = true;
            } else if (std.mem.eql(u8, key, "moves")) {
                show_moves = true;
            } else if (std.mem.eql(u8, key, "games") or std.mem.eql(u8, key, "seed")) {
                const v = tokens.next() orelse return out.writeAll("error bad-autoplay\n");
                const n = std.fmt.parseInt(u64, v, 10) catch return out.writeAll("error bad-autoplay\n");
                if (key[0] == 'g') games = n else master = n;
            } else return out.print("error bad-autoplay {s}\n", .{key});
        }

        const gen = try self.gpa.create(movegen.Generator);
        defer self.gpa.destroy(gen);
        gen.* = .init(self.gpa, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        gen.word_map = p.word_map;
        for (0..games) |gi| {
            const seed = game_mod.deriveSeed(master, if (pairs) gi / 2 else gi);
            var g: game_mod.Game = .start(p, seed, pairs);
            gen.board = &g.pos.board;
            while (g.playing) {
                const best = try game_mod.playStaticTurn(&g, gen);
                if (show_moves) {
                    try out.print("move {d} ", .{gi});
                    try best.move.format(p.dist, out);
                    try out.print(" score {d} equity {d}\n", .{ best.move.score, best.equity });
                }
            }
            try out.print("game {d} score {d} {d} turns {d}\n", .{ gi, g.pos.scores[0], g.pos.scores[1], g.turn });
        }
        try out.print("autoplayok games {d}\n", .{games});
    }

    /// `go static`: the best move by static equity. Writes
    /// `info move <move> score <n> equity <e>` and `bestmove <move>`.
    fn go(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        const kind = tokens.next() orelse return out.writeAll("error bad-go\n");
        const is_sim = std.mem.eql(u8, kind, "sim");
        if (!is_sim and !std.mem.eql(u8, kind, "static")) return out.print("error bad-go {s}\n", .{kind});
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        if (is_sim) return self.goSim(p, tokens.rest(), out);
        var rack = p.racks[0];
        if (rack.num_tiles > movegen.rack_tile_limit) return out.writeAll("error rack-too-big\n");

        var gen = movegen.Generator.init(self.gpa, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        gen.word_map = p.word_map;
        const ex = movegen.exchangeRules(p.unseenBy(0), p.lexiconName());
        gen.max_can_exchange = ex.max;
        const ctx = p.equityContext();
        const best = try gen.genBest(&rack, ex.allowed, &ctx);
        try out.writeAll("info move ");
        try best.move.format(p.dist, out);
        try out.print(" score {d} equity {d}\n", .{ best.move.score, best.equity });
        try out.writeAll("bestmove ");
        try best.move.format(p.dist, out);
        try out.writeAll("\n");
    }

    /// Debug: every move for the player on turn with its score and static
    /// equity (a pass gets the static pass penalty, as in the Go static
    /// player's move list), best first, then `genok count <n>`.
    fn printMoves(self: *Engine, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        const ranked = try rankedMoves(self.gpa, p);
        defer self.gpa.free(ranked);
        for (ranked) |*r| {
            try out.writeAll("move ");
            try r.m.format(p.dist, out);
            try out.print(" score {d} equity {d}\n", .{ r.m.score, r.eq });
        }
        try out.print("genok count {d}\n", .{ranked.len});
    }

    /// `go sim`: a Monte Carlo sim of the best plays by static equity,
    /// single-threaded, run to a fixed number of iterations. Writes one
    /// `info sim` line per play, best first, then `bestmove`.
    fn goSim(self: *Engine, p: *const Position, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var plies: u32 = 2;
        var iterations: u64 = 1000;
        var num_plays: usize = 15;
        var seed: u64 = 1;
        var stop: ?u32 = null;
        var cap: u64 = default_sim_cap;
        var best_cap: u64 = default_sim_best_cap;
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            const v = tokens.next() orelse return out.writeAll("error bad-go\n");
            const n = std.fmt.parseInt(u64, v, 10) catch return out.writeAll("error bad-go\n");
            if (std.mem.eql(u8, key, "plies") and n >= 1 and n <= 32) {
                plies = @intCast(n);
            } else if (std.mem.eql(u8, key, "iterations")) {
                iterations = n;
            } else if (std.mem.eql(u8, key, "plays") and n >= 1 and n <= 1000) {
                num_plays = @intCast(n);
            } else if (std.mem.eql(u8, key, "seed")) {
                seed = n;
            } else if (std.mem.eql(u8, key, "stop") and sim_mod.stopDelta(@intCast(@min(n, 1000))) != null) {
                stop = @intCast(n);
            } else if (std.mem.eql(u8, key, "cap") and n >= 1) {
                cap = n;
            } else if (std.mem.eql(u8, key, "bestcap")) {
                best_cap = n;
            } else return out.print("error bad-go {s}\n", .{key});
        }
        if (p.racks[0].num_tiles > movegen.rack_tile_limit) return out.writeAll("error rack-too-big\n");
        const win_table = self.store.winTable(self.options.data_path) catch |err| return self.dataError(err, out);

        const ranked = try rankedMoves(self.gpa, p);
        defer self.gpa.free(ranked);
        const moves = try self.gpa.alloc(Move, @min(num_plays, ranked.len));
        defer self.gpa.free(moves);
        for (moves, ranked[0..moves.len]) |*m, r| m.* = r.m;

        const game_seed = game_mod.deriveSeed(seed, 0);
        const g = try self.gpa.create(game_mod.Game);
        defer self.gpa.destroy(g);
        g.* = game_mod.Game.fromPosition(p, game_seed) catch return out.writeAll("error bad-position\n");
        const sim = try self.gpa.create(sim_mod.Sim);
        defer self.gpa.destroy(sim);
        sim.* = try .init(self.gpa, g, game_seed, 0, moves, plies, win_table);
        defer sim.deinit();
        if (stop) |condition| {
            const res = sim.runBai(.{
                .delta = sim_mod.stopDelta(condition).?,
                .cap_per_play = cap,
                .best_play_cap = best_cap,
            }) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return out.print("error sim-failed {t}\n", .{err}),
            };
            try out.print("info sim status {s} rollouts {d}\n", .{ res.status.name(), res.samples });
        } else if (g.playing) {
            for (0..iterations) |_| try sim.iterate();
        }

        const order = try self.gpa.alloc(*const sim_mod.Play, moves.len);
        defer self.gpa.free(order);
        sim.ranked(order);
        for (order) |play| {
            try out.print("info sim iterations {d} play ", .{play.win.n});
            try play.move.format(p.dist, out);
            try out.print(" win {d} equity {d} stderr {d}\n", .{ play.win.mean(), play.equity.mean(), play.equity.stdErr() });
        }
        try out.writeAll("bestmove ");
        if (order.len > 0) try order[0].move.format(p.dist, out) else try out.writeAll("pass");
        try out.writeAll("\n");
    }

    /// Debug: every square's cross-set, cross-score, anchor and extension
    /// sets, in both directions, then `crosssetsok`.
    fn printCrossSets(self: *Engine, out: *Writer) Writer.Error!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        const b = &p.board;
        for (0..b.dim) |r| {
            for (0..b.dim) |c| {
                for ([_]board_mod.Direction{ .horizontal, .vertical }) |d| {
                    try out.print("cs {d} {d} {c} set {x} score {d} anchor {d} lext {x} rext {x}\n", .{
                        r,                     c,                      @as(u8, if (d == .horizontal) 'h' else 'v'),
                        b.crossSet(r, c, d),   b.crossScore(r, c, d),  @intFromBool(b.isAnchor(r, c, d)),
                        b.leftExtSet(r, c, d), b.rightExtSet(r, c, d),
                    });
                }
            }
        }
        try out.writeAll("crosssetsok\n");
    }

    /// Debug: `leave <tiles>` prints the value of keeping the tiles, from
    /// the leave values the position's lexicon is played with (0 if none).
    fn printLeave(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        const text = std.mem.trim(u8, args, " \t\r");
        var mls: [32]tiles.MachineLetter = undefined;
        const n = p.dist.toMachineLetters(text, &mls) catch return out.writeAll("error bad-tiles\n");
        for (mls[0..n]) |ml| if (tiles.isBlanked(ml)) return out.writeAll("error bad-tiles\n");
        std.mem.sort(tiles.MachineLetter, mls[0..n], {}, std.sort.asc(tiles.MachineLetter));
        const klv = p.leaves orelse return out.writeAll("leave 0\n");
        // The leave file holds f32s; print the f32 exactly.
        const v: f32 = @floatCast(klv.leaveValue(mls[0..n]));
        try out.print("leave {d}\n", .{v});
    }
};

/// Writes "TooManyTiles" as "too-many-tiles", for error tokens.
/// A best-arm sim's budget in rollouts per play, and the rollouts after
/// which it settles for the leading play: the elite bot's.
const default_sim_cap = 1000;
const default_sim_best_cap = 10000;

pub const Ranked = struct { m: Move, eq: f64 };

/// Every move for the player on turn with its static equity (a pass gets
/// the static pass penalty), best first by equity.better: the static
/// player's move list.
pub fn rankedMoves(gpa: Allocator, p: *const Position) Allocator.Error![]Ranked {
    var gen = movegen.Generator.init(gpa, @constCast(&p.board), p.lexicon.kwg, p.dist);
    defer gen.deinit();
    const ex = movegen.exchangeRules(p.unseenBy(0), p.lexiconName());
    gen.max_can_exchange = ex.max;
    var rack = p.racks[0];
    try gen.genAll(&rack, ex.allowed);

    const ctx = p.equityContext();
    const ranked = try gpa.alloc(Ranked, gen.plays.items.len);
    for (gen.plays.items, ranked) |*m, *r| {
        const lv = m.leave(&rack);
        r.* = .{ .m = m.*, .eq = if (m.kind == .pass) equity.static_pass_penalty else ctx.equity(m, &lv) };
    }
    std.mem.sort(Ranked, ranked, {}, struct {
        fn before(_: void, a: Ranked, b: Ranked) bool {
            return equity.better(&a.m, a.eq, &b.m, b.eq);
        }
    }.before);
    return ranked;
}

fn writeKebab(out: *Writer, name: []const u8) Writer.Error!void {
    for (name, 0..) |ch, i| {
        if (std.ascii.isUpper(ch)) {
            if (i > 0) try out.writeByte('-');
            try out.writeByte(std.ascii.toLower(ch));
        } else try out.writeByte(ch);
    }
}

fn parseSpin(value: []const u8, min: u32, max: u32) ?u32 {
    const v = std.fmt.parseInt(u32, value, 10) catch return null;
    if (v < min or v > max) return null;
    return v;
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

/// Runs `lines` through a fresh engine and returns everything it wrote.
fn transcript(lines: []const []const u8) ![]u8 {
    var engine = try Engine.init(testing.allocator, "0.0.0", .empty);
    defer engine.deinit();
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    for (lines) |line| {
        if (try engine.handle(line, &out.writer) == .quit) break;
    }
    return out.toOwnedSlice();
}

test "ucgi handshake" {
    const got = try transcript(&.{"ucgi"});
    defer testing.allocator.free(got);
    try testing.expectEqualStrings(
        \\id name MacondoZig 0.0.0
        \\id author the Macondo authors
        \\option name threads type spin default 1 min 1 max 512
        \\option name hash-mb type spin default 256 min 1 max 65536
        \\option name data-path type string default ./data
        \\option name word-maps type check default true
        \\ucgiok
        \\
    , got);
}

test "isready, blank lines, unknown commands, quit" {
    const got = try transcript(&.{ "", "  \r", "isready", "frobnicate x y", "quit", "isready" });
    defer testing.allocator.free(got);
    try testing.expectEqualStrings("readyok\nerror unknown-command frobnicate\n", got);
}

test "setoption" {
    var engine = try Engine.init(testing.allocator, "0.0.0", .empty);
    defer engine.deinit();
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    _ = try engine.handle("setoption name threads value 8", &out.writer);
    _ = try engine.handle("setoption name hash-mb value 1024", &out.writer);
    _ = try engine.handle("setoption name data-path value /opt/my data\r", &out.writer);
    try testing.expectEqualStrings("", out.written());
    try testing.expectEqual(8, engine.options.threads);
    try testing.expectEqual(1024, engine.options.hash_mb);
    try testing.expectEqualStrings("/opt/my data", engine.options.data_path);

    _ = try engine.handle("setoption name threads value 0", &out.writer);
    _ = try engine.handle("setoption name threads value 513", &out.writer);
    _ = try engine.handle("setoption name hash-mb value lots", &out.writer);
    _ = try engine.handle("setoption name colour value blue", &out.writer);
    _ = try engine.handle("setoption threads 8", &out.writer);
    _ = try engine.handle("setoption name threads value", &out.writer);
    try testing.expectEqualStrings(
        \\error bad-value threads
        \\error bad-value threads
        \\error bad-value hash-mb
        \\error unknown-option colour
        \\error bad-setoption
        \\error bad-setoption
        \\
    , out.written());
    try testing.expectEqual(8, engine.options.threads);
}

test "go static and gen with equities" {
    const testdata = @import("testdata.zig");
    const build_options = @import("build_options");
    var engine = try Engine.init(testing.allocator, "0.0.0", testdata.file_source);
    defer engine.deinit();
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var buf: [256]u8 = undefined;
    _ = try engine.handle(try std.fmt.bufPrint(&buf, "setoption name data-path value {s}", .{build_options.data_path}), &out.writer);

    _ = try engine.handle("go static", &out.writer);
    _ = try engine.handle("go infer", &out.writer);
    // The opening: FERAL and FARLE score 24, but FERAL puts a vowel next
    // to a double-letter square.
    _ = try engine.handle("position cgp 15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AEFLR/ 0/0 0 lex NWL23;", &out.writer);
    if (std.mem.startsWith(u8, out.written(), "error no-position\nerror bad-go infer\nerror data-not-found")) return error.SkipZigTest;
    _ = try engine.handle("go static", &out.writer);
    _ = try engine.handle("gen", &out.writer);
    var lines = std.mem.splitScalar(u8, out.written(), '\n');
    try testing.expectEqualStrings("error no-position", lines.next().?);
    try testing.expectEqualStrings("error bad-go infer", lines.next().?);
    try testing.expectEqualStrings("info move 8D.FARLE score 24 equity 24", lines.next().?);
    try testing.expectEqualStrings("bestmove 8D.FARLE", lines.next().?);
    // gen: best first, FARLE before FLARE on the tiebreak (same score,
    // same square, F-A before F-L).
    try testing.expectEqualStrings("move 8D.FARLE score 24 equity 24", lines.next().?);
    try testing.expectEqualStrings("move 8D.FLARE score 24 equity 24", lines.next().?);
    try testing.expectEqualStrings("move 8D.FERAL score 24 equity 23.3", lines.next().?);
}
