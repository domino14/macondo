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
const cross_set = @import("cross_set.zig");
const move_mod = @import("move.zig");
const Position = @import("position.zig").Position;
const movegen = @import("movegen.zig");
const Move = @import("move.zig").Move;
const equity = @import("equity.zig");
const game_mod = @import("game.zig");
const sim_mod = @import("sim.zig");
const classic = @import("classic.zig");
const infer_mod = @import("infer.zig");
const endgame = @import("endgame.zig");
const zobrist_mod = @import("zobrist.zig");
const ttable = @import("ttable.zig");
const bai_mod = @import("bai.zig");

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
    /// Use rack info tables (see data.RackTables): auto, true (make one
    /// if missing), false.
    rack_tables: data.RackTables = .auto,
};

/// What the adapter should do after a command.
pub const Status = enum { @"continue", quit };

pub const Engine = struct {
    gpa: Allocator,
    /// For running searches in the background and their workers: the
    /// engine does no I/O through it.
    io: std.Io,
    version: []const u8,
    options: Options,
    store: data.Store,
    position: ?Position = null,
    /// Where responses go: from the command loop and from a search running
    /// in the background, one at a time.
    out: *Writer,
    out_mutex: std.Io.Mutex = .init,
    search: ?*Search = null,
    /// The cross-set cache `play` updates the board with (made on first
    /// use; its keys include the word graph, so positions can share it).
    cross_cache: ?cross_set.Cache = null,
    /// The endgame's position keys (made for the board's size on first
    /// use, or loaded with the zobrist-file option) and table.
    zobrist: ?zobrist_mod.Zobrist = null,
    tt: ttable.Table = .{},

    /// `source` is how the engine reads data files; see data.Source.
    /// Responses are written to `out`, and flushed after each command and
    /// each line a background search writes.
    pub fn init(gpa: Allocator, io: std.Io, version: []const u8, source: data.Source, out: *Writer) Allocator.Error!Engine {
        return .{
            .gpa = gpa,
            .io = io,
            .version = version,
            .options = .{ .data_path = try gpa.dupe(u8, "./data") },
            .store = .init(gpa, source),
            .out = out,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.stopSearch();
        if (self.cross_cache) |*c| c.deinit(self.gpa);
        if (self.zobrist) |*z| z.deinit(self.gpa);
        self.tt.deinit(self.gpa);
        self.store.deinit();
        self.gpa.free(self.options.data_path);
        self.* = undefined;
    }

    /// Writes a line to the engine's output (the adapter's own errors),
    /// between whatever a background search writes.
    pub fn writeLine(self: *Engine, line: []const u8) Writer.Error!void {
        self.out_mutex.lockUncancelable(self.io);
        defer self.out_mutex.unlock(self.io);
        try self.out.writeAll(line);
        try self.out.writeAll("\n");
        try self.out.flush();
    }

    /// Waits for the search running in the background, if any, to finish
    /// and write its bestmove.
    pub fn waitSearch(self: *Engine) void {
        const s = self.search orelse return;
        if (s.future) |*f| f.await(self.io);
        s.destroy();
        self.search = null;
    }

    /// Stops the search running in the background, if any, and waits for
    /// it to write its bestmove.
    pub fn stopSearch(self: *Engine) void {
        const s = self.search orelse return;
        s.stop.store(true, .monotonic);
        self.waitSearch();
    }

    /// Handles one input line (without its trailing newline), writing the
    /// response lines to the engine's output. While a search runs in the
    /// background only stop, isready and quit are taken.
    pub fn handle(self: *Engine, line: []const u8) (Writer.Error || Allocator.Error)!Status {
        var tokens = std.mem.tokenizeAny(u8, line, " \t\r");
        const cmd = tokens.next() orelse return .@"continue";
        if (self.search) |s| {
            if (s.done.load(.acquire)) self.waitSearch();
        }
        if (std.mem.eql(u8, cmd, "quit")) {
            self.stopSearch();
            return .quit;
        }
        if (std.mem.eql(u8, cmd, "stop")) {
            if (self.search) |s| s.stop.store(true, .monotonic);
            return .@"continue";
        }
        self.out_mutex.lockUncancelable(self.io);
        defer self.out_mutex.unlock(self.io);
        defer self.out.flush() catch {};
        const out = self.out;
        if (self.search != null and !std.mem.eql(u8, cmd, "isready")) {
            try out.writeAll("error busy\n");
            return .@"continue";
        }

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
            if (std.mem.eql(u8, tokens.rest(), "raw")) try self.printRawMoves(out) else try self.printMoves(out);
        } else if (std.mem.eql(u8, cmd, "crosssets")) {
            try self.printCrossSets(out);
        } else if (std.mem.eql(u8, cmd, "leave")) {
            try self.printLeave(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "minisim")) {
            try self.miniSim(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "play")) {
            try self.playMove(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "takeback")) {
            try self.takeBack(tokens.rest(), out);
        } else if (std.mem.eql(u8, cmd, "cgp")) {
            try self.printCgp(out);
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
        try out.writeAll("option name rack-tables type combo default auto var auto var true var false\n");
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
        } else if (std.mem.eql(u8, name, "rack-tables")) {
            if (std.mem.eql(u8, value, "auto")) {
                self.options.rack_tables = .auto;
            } else if (std.mem.eql(u8, value, "true")) {
                self.options.rack_tables = .always;
            } else if (std.mem.eql(u8, value, "false")) {
                self.options.rack_tables = .never;
            } else return out.print("error bad-value {s}\n", .{name});
        } else if (std.mem.eql(u8, name, "zobrist-file")) {
            // Debug: the endgame's position keys from a file (the Go
            // engine's Zobrist.Dump), for comparing searches node for node.
            const bytes = self.store.source.load(self.gpa, "", value) catch return out.print("error data-not-found {s}\n", .{value});
            defer self.gpa.free(bytes);
            const z = zobrist_mod.Zobrist.fromJson(self.gpa, bytes) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.BadZobrist => return out.print("error data-invalid {s}\n", .{value}),
            };
            if (self.zobrist) |*old| old.deinit(self.gpa);
            self.zobrist = z;
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
        p.fromCgpWith(text, &self.store, self.options.data_path, .{ .word_maps = self.options.word_maps, .rack_tables = self.options.rack_tables }) catch |err| switch (err) {
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
        gen.rit = p.rit;
        var cross_cache: @import("cross_set.zig").Cache = try .init(self.gpa, @import("cross_set.zig").Cache.default_index_bits);
        defer cross_cache.deinit(self.gpa);
        for (0..games) |gi| {
            const seed = game_mod.deriveSeed(master, if (pairs) gi / 2 else gi);
            var g: game_mod.Game = .start(p, seed, pairs);
            g.cross_cache = &cross_cache;
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
        const is_infer = std.mem.eql(u8, kind, "infer");
        const is_endgame = std.mem.eql(u8, kind, "endgame");
        if (!is_sim and !is_infer and !is_endgame and !std.mem.eql(u8, kind, "static")) return out.print("error bad-go {s}\n", .{kind});
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        if (is_endgame) return self.goEndgame(p, tokens.rest(), out);
        if (is_sim) return self.goSim(p, tokens.rest(), out);
        if (is_infer) return self.goInfer(p, tokens.rest(), out);
        var rack = p.racks[0];
        if (rack.num_tiles > movegen.rack_tile_limit) return out.writeAll("error rack-too-big\n");

        var gen = movegen.Generator.init(self.gpa, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        gen.word_map = p.word_map;
        gen.rit = p.rit;
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
    /// Debug: `gen raw` lists every tile play for the player on turn in
    /// the order the move generator makes them (rows, then columns; anchor
    /// by anchor), with their scores, then `genok count <n>`. Searches that
    /// keep the first of equal moves depend on this order.
    fn printRawMoves(self: *Engine, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        var gen = movegen.Generator.init(self.gpa, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        var rack = p.racks[0];
        try gen.genAll(&rack, false);
        var n: usize = 0;
        for (gen.plays.items) |*m| {
            if (m.kind != .play) continue;
            try out.writeAll("move ");
            try m.format(p.dist, out);
            try out.print(" score {d}\n", .{m.score});
            n += 1;
        }
        try out.print("genok count {d}\n", .{n});
    }

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

    /// `go sim`: a Monte Carlo sim of the best plays by static equity, on
    /// the `threads` option's workers, in the background. Plain, it rolls
    /// every play out until stopped (or for `iterations`); with `stop` it
    /// is a best-arm run. `time` stops it after that many milliseconds.
    /// Progress lines as it runs, then one `info sim` line per play, best
    /// first, and `bestmove`.
    fn goSim(self: *Engine, p: *const Position, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var plies: u32 = 2;
        var iterations: u64 = 0;
        var num_plays: usize = 15;
        var seed: u64 = 1;
        var stop: ?u32 = null;
        var cap: u64 = default_sim_cap;
        var best_cap: u64 = default_sim_best_cap;
        var time_ms: u64 = 0;
        var cycle = false;
        var infer_budget: u32 = 0;
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            const v = tokens.next() orelse return out.writeAll("error bad-go\n");
            const n = std.fmt.parseInt(u64, v, 10) catch return out.writeAll("error bad-go\n");
            if (std.mem.eql(u8, key, "plies") and n >= 1 and n <= sim_mod.max_plies) {
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
            } else if (std.mem.eql(u8, key, "time")) {
                time_ms = n;
            } else if (std.mem.eql(u8, key, "cycle") and n <= 1) {
                cycle = n == 1;
            } else if (std.mem.eql(u8, key, "infer") and n <= 1_000_000) {
                infer_budget = @intCast(n);
            } else return out.print("error bad-go {s}\n", .{key});
        }
        if (p.racks[0].num_tiles > movegen.rack_tile_limit) return out.writeAll("error rack-too-big\n");
        const win_table = self.store.winTable(self.options.data_path) catch |err| return self.dataError(err, out);

        const ranked = try rankedMoves(self.gpa, p);
        defer self.gpa.free(ranked);
        const moves = try self.gpa.alloc(Move, @min(num_plays, ranked.len));
        errdefer self.gpa.free(moves);
        for (moves, ranked[0..moves.len]) |*m, r| m.* = r.m;

        const game_seed = game_mod.deriveSeed(seed, 0);
        const g = try self.gpa.create(game_mod.Game);
        errdefer self.gpa.destroy(g);
        g.* = game_mod.Game.fromPosition(p, game_seed) catch return out.writeAll("error bad-position\n");
        const turn = positionTurn(p);
        const sim = try sim_mod.Sim.create(self.gpa, g, game_seed, turn, moves, plies, win_table, self.options.threads);
        errdefer sim.destroy();
        var reading: ?Search.Reading = null;
        if (infer_budget > 0) {
            const copy = try self.gpa.create(Position);
            copy.* = p.*;
            reading = .{ .pos = copy, .seed = game_seed, .turn = turn, .opts = .{ .budget = infer_budget, .threads = self.options.threads }, .win_table = win_table };
        }
        errdefer if (reading) |r| self.gpa.destroy(r.pos);
        const search = try self.gpa.create(Search);
        search.* = .{
            .engine = self,
            .dist = p.dist,
            .game = g,
            .moves = moves,
            .sim = sim,
            .bai = if (stop) |c| .{ .delta = sim_mod.stopDelta(c).?, .cap_per_play = cap, .best_play_cap = best_cap, .bag_cycle = cycle } else null,
            .iterations = iterations,
            .time_ms = time_ms,
            .started = std.Io.Clock.Timestamp.now(self.io, .awake),
            .playing = g.playing,
            .reading = reading,
        };
        self.search = search;
        // Writes happen once this command's lock is released.
        search.future = self.io.concurrent(Search.run, .{search}) catch null;
        if (search.future == null) {
            // No concurrency: the search runs here, and the command waits.
            self.out_mutex.unlock(self.io);
            defer self.out_mutex.lockUncancelable(self.io);
            search.run();
            self.waitSearch();
        }
    }

    /// `go endgame plies <n> [firstwin] [noid] [noheur] [nolists]
    /// [nooutreply] [nokeepone] [noshortcuts] [checklists]`: solves the
    /// endgame (the bag must be empty) to n plies on one thread, writing
    /// `info endgame depth <d> value <v> nodes <n> pv <moves>` after each
    /// depth (v: the change in the spread of the player on turn), then
    /// `bestmove`.
    fn goEndgame(self: *Engine, p: *const Position, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var plies: u32 = 0;
        var opts: endgame.Options = .{ .tt_bytes = @as(u64, self.options.hash_mb) << 20 };
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            if (std.mem.eql(u8, key, "plies")) {
                const v = tokens.next() orelse return out.writeAll("error bad-endgame\n");
                plies = std.fmt.parseInt(u32, v, 10) catch return out.writeAll("error bad-endgame\n");
            } else if (std.mem.eql(u8, key, "firstwin")) {
                opts.first_win = true;
            } else if (std.mem.eql(u8, key, "noid")) {
                opts.iterative_deepening = false;
            } else if (std.mem.eql(u8, key, "noheur")) {
                opts.heuristics = false;
            } else if (std.mem.eql(u8, key, "nolists")) {
                opts.lists = false;
            } else if (std.mem.eql(u8, key, "nooutreply")) {
                opts.out_reply = false;
            } else if (std.mem.eql(u8, key, "nokeepone")) {
                opts.keep_one = false;
            } else if (std.mem.eql(u8, key, "noshortcuts")) {
                opts.child_shortcuts = false;
            } else if (std.mem.eql(u8, key, "checklists")) {
                opts.check_lists = true;
            } else return out.print("error bad-endgame {s}\n", .{key});
        }
        if (plies == 0 or plies > endgame.max_variant) return out.writeAll("error bad-endgame\n");
        if (p.unseenBy(0) != p.racks[1].num_tiles) return out.writeAll("error bag-not-empty\n");
        if (self.zobrist == null or self.zobrist.?.dim != p.board.dim) {
            if (self.zobrist) |*z| z.deinit(self.gpa);
            self.zobrist = null;
            self.zobrist = try .init(self.gpa, p.board.dim, @splat(0x5a));
        }
        const solver = endgame.Solver.init(self.gpa, p, &self.zobrist.?, &self.tt, opts) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => |e| return out.print("error endgame {s}\n", .{@errorName(e)}),
        };
        defer solver.deinit();
        const Progress = struct {
            out: *Writer,
            solver: *endgame.Solver,
            dist: *const tiles.LetterDistribution,
            failed: ?anyerror = null,

            fn report(ctx: ?*anyopaque, r: *const endgame.Result) void {
                const self_: *@This() = @ptrCast(@alignCast(ctx.?));
                self_.write(r) catch |err| {
                    self_.failed = err;
                };
            }
            fn write(self_: *@This(), r: *const endgame.Result) !void {
                try self_.out.print("info endgame depth {d} value {d} nodes {d} pv", .{ r.depth, r.value, r.nodes });
                var moves: [endgame.max_variant]Move = undefined;
                const n = try self_.solver.lineMoves(&r.line, &moves);
                for (moves[0..n], 0..) |*m, i| {
                    try self_.out.writeAll(if (i == r.line.searched) " | " else " ");
                    try m.format(self_.dist, self_.out);
                }
                try self_.out.writeAll("\n");
                try self_.out.flush();
            }
        };
        var progress: Progress = .{ .out = out, .solver = solver, .dist = p.dist };
        const result = solver.solve(plies, Progress.report, &progress) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => |e| return out.print("error endgame {s}\n", .{@errorName(e)}),
        };
        if (progress.failed) |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            return error.WriteFailed;
        }
        try out.writeAll("bestmove ");
        if (result.line.len == 0) return out.writeAll("pass\n");
        var moves: [endgame.max_variant]Move = undefined;
        const n = try solver.lineMoves(&result.line, &moves);
        if (n == 0) try out.writeAll("pass") else try moves[0].format(p.dist, out);
        try out.writeAll("\n");
    }

    /// `go infer`: what the opponent kept after the position's last move
    /// (CGP lm), as a posterior over every leave they could have kept.
    fn goInfer(self: *Engine, p: *const Position, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        var opts: infer_mod.Options = .{};
        var seed: u64 = 1;
        var trace = false;
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            const v = tokens.next() orelse return out.writeAll("error bad-go\n");
            if (std.mem.eql(u8, key, "tau") or std.mem.eql(u8, key, "taucap")) {
                const x = std.fmt.parseFloat(f64, v) catch return out.writeAll("error bad-go\n");
                if (!(x >= 0)) return out.writeAll("error bad-go\n");
                if (key.len == 3) opts.tau = x else opts.tau_cap = x;
                continue;
            }
            const n = std.fmt.parseInt(u64, v, 10) catch return out.writeAll("error bad-go\n");
            if (std.mem.eql(u8, key, "budget") and n >= 1 and n <= 1_000_000) {
                opts.budget = @intCast(n);
            } else if (std.mem.eql(u8, key, "seed")) {
                seed = n;
            } else if (std.mem.eql(u8, key, "iterations") and n >= 1) {
                opts.sim_iters = n;
            } else if (std.mem.eql(u8, key, "plays") and n >= 1 and n < classic.max_plays) {
                opts.mini_plays = @intCast(n);
            } else if (std.mem.eql(u8, key, "plies") and n >= 1 and n <= sim_mod.max_plies) {
                opts.mini_plies = @intCast(n);
            } else if (std.mem.eql(u8, key, "rounds") and n <= 100) {
                opts.max_rounds = @intCast(n);
            } else if (std.mem.eql(u8, key, "trace") and n <= 1) {
                trace = n == 1;
            } else return out.print("error bad-go {s}\n", .{key});
        }
        const win_table = self.store.winTable(self.options.data_path) catch |err| return self.dataError(err, out);
        opts.threads = self.options.threads;
        var res = infer_mod.infer(self.gpa, self.io, p, game_mod.deriveSeed(seed, 0), positionTurn(p), win_table, opts) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return out.print("error infer {t}\n", .{err}),
        };
        defer res.deinit(self.gpa);
        if (trace) {
            for (res.draws) |*d| {
                try out.print("info infer draw round {d} leave ", .{d.round});
                for (d.leave.slice()) |ml| try out.writeAll(p.dist.letter(ml, false));
                try out.print(" likelihood {d}\n", .{d.likelihood});
            }
        }
        try out.print("info infer leaves {d} kept {d} tau {d} sims {d} measured {d} enumerated {d}\n", .{ res.leaves.len, res.rack_length, res.tau, res.sims, res.measured, @intFromBool(res.enumerated) });
        for (res.leaves) |*l| {
            try out.writeAll("info infer leave ");
            for (l.slice()) |ml| try out.writeAll(p.dist.letter(ml, false));
            try out.print(" weight {d}\n", .{l.weight});
        }
        try out.writeAll("inferok\n");
    }

    /// Debug: the inference's mini-sim on the position: the top plays by
    /// static equity, simmed on one thread under the classic Stop99 rule
    /// (checked every 64 iterations), reported as `go sim` reports.
    fn miniSim(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        var plies: u32 = 2;
        var iterations: u64 = 200;
        var num_plays: usize = 10;
        var seed: u64 = 1;
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        while (tokens.next()) |key| {
            const v = tokens.next() orelse return out.writeAll("error bad-minisim\n");
            const n = std.fmt.parseInt(u64, v, 10) catch return out.writeAll("error bad-minisim\n");
            if (std.mem.eql(u8, key, "plies") and n >= 1 and n <= sim_mod.max_plies) {
                plies = @intCast(n);
            } else if (std.mem.eql(u8, key, "iterations")) {
                iterations = n;
            } else if (std.mem.eql(u8, key, "plays") and n >= 1 and n <= classic.max_plays) {
                num_plays = @intCast(n);
            } else if (std.mem.eql(u8, key, "seed")) {
                seed = n;
            } else return out.print("error bad-minisim {s}\n", .{key});
        }
        if (p.racks[0].num_tiles > movegen.rack_tile_limit) return out.writeAll("error rack-too-big\n");
        const win_table = self.store.winTable(self.options.data_path) catch |err| return self.dataError(err, out);
        const ranked = try rankedMoves(self.gpa, p);
        defer self.gpa.free(ranked);
        const moves = try self.gpa.alloc(Move, @min(num_plays, ranked.len));
        defer self.gpa.free(moves);
        for (moves, ranked[0..moves.len]) |*m, r| m.* = r.m;
        const game_seed = game_mod.deriveSeed(seed, 0);
        var g = game_mod.Game.fromPosition(p, game_seed) catch return out.writeAll("error bad-position\n");
        const sim = try sim_mod.Sim.create(self.gpa, &g, game_seed, 0, moves, plies, win_table, 1);
        defer sim.destroy();
        if (g.playing) try classic.run(sim, iterations, .{});
        const order = try self.gpa.alloc(*const sim_mod.Play, sim.plays.len);
        defer self.gpa.free(order);
        sim.ranked(order);
        for (order) |play| {
            try out.print("info sim iterations {d} play ", .{play.stats.win.n});
            try play.move.format(p.dist, out);
            try out.print(" win {d} equity {d} stderr {d}\n", .{ play.stats.win.mean(), play.stats.equity.mean(), play.stats.equity.stdErr() });
        }
        try out.writeAll("bestmove ");
        if (order.len > 0) try order[0].move.format(p.dist, out) else try out.writeAll("pass");
        try out.writeAll("\n");
    }

    /// Debug: `play <move> [rack <tiles>]` plays a tile play for the player
    /// on turn the way games and sims do, the cross-sets brought up to
    /// date incrementally rather than worked out again; with `rack`, the
    /// mover's rack afterwards (what they drew). Writes `playok score <s>`.
    fn playMove(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        const text = tokens.next() orelse return out.writeAll("error bad-play\n");
        const parsed = move_mod.parse(text, p.dist, p.board.dim) catch return out.writeAll("error bad-play\n");
        var rack: ?tiles.Rack = null;
        if (tokens.next()) |key| {
            if (!std.mem.eql(u8, key, "rack")) return out.writeAll("error bad-play\n");
            rack = parseRack(p.dist, tokens.next() orelse "") orelse return out.writeAll("error bad-play\n");
        }
        if (tokens.next() != null) return out.writeAll("error bad-play\n");
        if (self.cross_cache == null) self.cross_cache = try .init(self.gpa, cross_set.Cache.default_index_bits);
        var next = p.*;
        const score = next.playMove(&parsed, &self.cross_cache.?) orelse return out.writeAll("error bad-play\n");
        if (rack) |r| next.racks[1] = r;
        p.* = next;
        try out.print("playok score {d}\n", .{score});
    }

    /// Debug: `takeback [rack <tiles>]` takes the last move back (from the
    /// CGP's lm or the last `play`), as inference does; with `rack`, the
    /// rack of the player then on turn, which is otherwise unknown. Writes
    /// `takebackok`.
    fn takeBack(self: *Engine, args: []const u8, out: *Writer) Writer.Error!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        var tokens = std.mem.tokenizeAny(u8, args, " \t\r");
        var rack: ?tiles.Rack = null;
        if (tokens.next()) |key| {
            if (!std.mem.eql(u8, key, "rack")) return out.writeAll("error bad-takeback\n");
            rack = parseRack(p.dist, tokens.next() orelse "") orelse return out.writeAll("error bad-takeback\n");
        }
        if (tokens.next() != null) return out.writeAll("error bad-takeback\n");
        var before = p.beforeLastMove() orelse return out.writeAll("error no-last-move\n");
        if (rack) |r| before.racks[0] = r;
        p.* = before;
        try out.writeAll("takebackok\n");
    }

    /// Debug: `cgp` writes the position's first four CGP fields (board,
    /// racks, scores, turns without a score) as `cgp <fields>`.
    fn printCgp(self: *Engine, out: *Writer) Writer.Error!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        try out.writeAll("cgp ");
        try @import("cgp.zig").writeFields(&p.board, p.dist, p.racks, p.scores, p.scoreless_turns, out);
        try out.writeAll("\n");
    }

    fn parseRack(ld: *const tiles.LetterDistribution, text: []const u8) ?tiles.Rack {
        var mls: [movegen.rack_tile_limit]tiles.MachineLetter = undefined;
        const n = ld.toMachineLetters(text, &mls) catch return null;
        return .fromLetters(mls[0..n]);
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
/// A sim running in the background: it owns its game, moves and sim.
const Search = struct {
    engine: *Engine,
    dist: *const tiles.LetterDistribution,
    game: *game_mod.Game,
    moves: []Move,
    sim: *sim_mod.Sim,
    bai: ?sim_mod.BaiOptions,
    iterations: u64,
    time_ms: u64,
    started: std.Io.Clock.Timestamp,
    /// False if the game is already over: nothing to sim.
    playing: bool,
    stop: std.atomic.Value(bool) = .init(false),
    /// The sim is done (computed); the search is done (bestmove written).
    computed: std.Io.Event = .unset,
    done: std.atomic.Value(bool) = .init(false),
    future: ?std.Io.Future(void) = null,
    bai_result: ?bai_mod.Result = null,
    err: ?anyerror = null,
    /// Inference first, the sim dealing the opponent's racks from what it
    /// reads.
    reading: ?Reading = null,

    const Reading = struct {
        pos: *Position,
        seed: [32]u8,
        turn: u64,
        opts: infer_mod.Options,
        win_table: *const sim_mod.WinTable,
        result: ?infer_mod.Result = null,
    };

    /// How often progress lines are written.
    const progress_ms = 1000;

    fn destroy(s: *Search) void {
        const gpa = s.engine.gpa;
        s.sim.destroy();
        if (s.reading) |*r| {
            if (r.result) |*res| res.deinit(gpa);
            gpa.destroy(r.pos);
        }
        gpa.destroy(s.game);
        gpa.free(s.moves);
        gpa.destroy(s);
    }

    fn compute(s: *Search) void {
        const io = s.engine.io;
        defer s.computed.set(io);
        if (!s.playing) return;
        if (s.reading) |*r| s.infer(r);
        if (s.bai) |opts_in| {
            var opts = opts_in;
            opts.stop = &s.stop;
            s.bai_result = s.sim.runBai(io, opts) catch |e| {
                s.err = e;
                return;
            };
        } else {
            s.sim.run(io, .{ .iterations = s.iterations, .stop = &s.stop }) catch |e| {
                s.err = e;
            };
        }
    }

    /// Reads the opponent's leave and has the sim deal from it; a position
    /// with nothing to read (no last move, a pass, a bingo, an empty bag)
    /// is simmed with random racks, as the elite bot does.
    fn infer(s: *Search, r: *Reading) void {
        const e = s.engine;
        var line: [160]u8 = undefined;
        const res = infer_mod.infer(e.gpa, e.io, r.pos, r.seed, r.turn, r.win_table, r.opts) catch |err| {
            if (err == error.OutOfMemory) {
                s.err = err;
                return;
            }
            const text = std.fmt.bufPrint(&line, "info infer skipped {t}", .{err}) catch return;
            e.writeLine(text) catch {};
            return;
        };
        r.result = res;
        s.sim.setPosterior(res.leaves) catch |err| {
            s.err = err;
            return;
        };
        const text = std.fmt.bufPrint(&line, "info infer leaves {d} kept {d} tau {d} sims {d} measured {d} enumerated {d}", .{ res.leaves.len, res.rack_length, res.tau, res.sims, res.measured, @intFromBool(res.enumerated) }) catch return;
        e.writeLine(text) catch {};
    }

    fn run(s: *Search) void {
        const io = s.engine.io;
        var computing = io.concurrent(compute, .{s}) catch null;
        if (computing == null) compute(s);
        const order = s.engine.gpa.alloc(*const sim_mod.Play, s.moves.len) catch null;
        defer if (order) |o| s.engine.gpa.free(o);
        while (!s.computed.isSet()) {
            var wait_ms: i64 = progress_ms;
            if (s.time_ms > 0) {
                const left = @as(i64, @intCast(s.time_ms)) - s.elapsedMs();
                if (left <= 0) {
                    s.stop.store(true, .monotonic);
                    wait_ms = progress_ms;
                } else wait_ms = @min(wait_ms, left);
            }
            s.computed.waitTimeout(io, .{ .duration = .{ .raw = .fromMilliseconds(wait_ms), .clock = .awake } }) catch {};
            if (s.computed.isSet()) break;
            if (order) |o| s.writeProgress(o);
        }
        if (computing) |*f| f.await(io);
        s.writeResult(order);
        s.done.store(true, .release);
    }

    fn elapsedMs(s: *const Search) i64 {
        return s.started.durationTo(std.Io.Clock.Timestamp.now(s.engine.io, .awake)).raw.toMilliseconds();
    }

    /// `info nodes <n> nps <n> time <ms> rollouts <n> best <move> win <w>
    /// equity <e>`.
    fn writeProgress(s: *Search, order: []*const sim_mod.Play) void {
        const e = s.engine;
        const pr = s.sim.progress(e.io, order);
        const ms = @max(s.elapsedMs(), 1);
        const nodes = s.sim.nodes();
        e.out_mutex.lockUncancelable(e.io);
        defer e.out_mutex.unlock(e.io);
        const out = e.out;
        out.print("info nodes {d} nps {d} time {d} rollouts {d}", .{ nodes, @divTrunc(@as(i64, @intCast(nodes)) * 1000, ms), ms, pr.rollouts }) catch return;
        if (pr.best) |m| {
            out.writeAll(" best ") catch return;
            m.format(s.dist, out) catch return;
            out.print(" win {d} equity {d}", .{ pr.win, pr.equity }) catch return;
        }
        out.writeAll("\n") catch return;
        out.flush() catch {};
    }

    /// The per-play lines, best first, and bestmove.
    fn writeResult(s: *Search, order_buf: ?[]*const sim_mod.Play) void {
        const e = s.engine;
        e.out_mutex.lockUncancelable(e.io);
        defer e.out_mutex.unlock(e.io);
        const out = e.out;
        defer out.flush() catch {};
        if (s.err) |err| out.print("error sim-failed {t}\n", .{err}) catch {};
        if (s.bai_result) |res| {
            out.print("info sim status {s} rollouts {d}\n", .{ res.status.name(), res.samples }) catch return;
        }
        const order = order_buf orelse {
            out.writeAll("bestmove pass\n") catch {};
            return;
        };
        s.sim.ranked(order);
        for (order) |play| {
            out.print("info sim iterations {d} play ", .{play.stats.win.n}) catch return;
            play.move.format(s.dist, out) catch return;
            out.print(" win {d} equity {d} stderr {d}\n", .{ play.stats.win.mean(), play.stats.equity.mean(), play.stats.equity.stdErr() }) catch return;
        }
        out.writeAll("bestmove ") catch return;
        if (order.len > 0) order[0].move.format(s.dist, out) catch return else out.writeAll("pass") catch return;
        out.writeAll("\n") catch {};
    }
};

/// A best-arm sim's budget in rollouts per play, and the rollouts after
/// which it settles for the leading play: the elite bot's.
const default_sim_cap = 1000;
const default_sim_best_cap = 10000;

/// The turn number a position's seeds use: 1 for one reached by a last
/// move the CGP gives (lm), as a game of one move from the position before
/// it counts its turns, otherwise 0.
fn positionTurn(p: *const Position) u64 {
    return if (p.last_move != null) 1 else 0;
}

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
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var engine = try Engine.init(testing.allocator, testing.io, "0.0.0", .empty, &out.writer);
    defer engine.deinit();
    for (lines) |line| {
        if (try engine.handle(line) == .quit) break;
    }
    engine.waitSearch();
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
        \\option name rack-tables type combo default auto var auto var true var false
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
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var engine = try Engine.init(testing.allocator, testing.io, "0.0.0", .empty, &out.writer);
    defer engine.deinit();

    _ = try engine.handle("setoption name threads value 8");
    _ = try engine.handle("setoption name hash-mb value 1024");
    _ = try engine.handle("setoption name data-path value /opt/my data\r");
    try testing.expectEqualStrings("", out.written());
    try testing.expectEqual(8, engine.options.threads);
    try testing.expectEqual(1024, engine.options.hash_mb);
    try testing.expectEqualStrings("/opt/my data", engine.options.data_path);

    _ = try engine.handle("setoption name threads value 0");
    _ = try engine.handle("setoption name threads value 513");
    _ = try engine.handle("setoption name hash-mb value lots");
    _ = try engine.handle("setoption name colour value blue");
    _ = try engine.handle("setoption threads 8");
    _ = try engine.handle("setoption name threads value");
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
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var engine = try Engine.init(testing.allocator, testing.io, "0.0.0", testdata.file_source, &out.writer);
    defer engine.deinit();
    var buf: [256]u8 = undefined;
    _ = try engine.handle(try std.fmt.bufPrint(&buf, "setoption name data-path value {s}", .{build_options.data_path}));

    _ = try engine.handle("go static");
    _ = try engine.handle("go mystery");
    // The opening: FERAL and FARLE score 24, but FERAL puts a vowel next
    // to a double-letter square.
    _ = try engine.handle("position cgp 15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AEFLR/ 0/0 0 lex NWL23;");
    if (std.mem.startsWith(u8, out.written(), "error no-position\nerror bad-go mystery\nerror data-not-found")) return error.SkipZigTest;
    _ = try engine.handle("go static");
    _ = try engine.handle("gen");
    var lines = std.mem.splitScalar(u8, out.written(), '\n');
    try testing.expectEqualStrings("error no-position", lines.next().?);
    try testing.expectEqualStrings("error bad-go mystery", lines.next().?);
    try testing.expectEqualStrings("info move 8D.FARLE score 24 equity 24", lines.next().?);
    try testing.expectEqualStrings("bestmove 8D.FARLE", lines.next().?);
    // gen: best first, FARLE before FLARE on the tiebreak (same score,
    // same square, F-A before F-L).
    try testing.expectEqualStrings("move 8D.FARLE score 24 equity 24", lines.next().?);
    try testing.expectEqualStrings("move 8D.FLARE score 24 equity 24", lines.next().?);
    try testing.expectEqualStrings("move 8D.FERAL score 24 equity 23.3", lines.next().?);
}

test "go sim runs in the background" {
    const testdata = @import("testdata.zig");
    const build_options = @import("build_options");
    var out: Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var engine = try Engine.init(testing.allocator, testing.io, "0.0.0", testdata.file_source, &out.writer);
    defer engine.deinit();
    var buf: [256]u8 = undefined;
    _ = try engine.handle(try std.fmt.bufPrint(&buf, "setoption name data-path value {s}", .{build_options.data_path}));
    _ = try engine.handle("setoption name threads value 2");
    _ = try engine.handle("position cgp 15/15/15/15/15/15/15/5TURnERY3/15/15/15/15/15/15/15 BEIMOQS/ 0/76 0 lex NWL20;");
    if (out.written().len > 0) return error.SkipZigTest;

    // No limit: runs until stopped. Commands other than stop, isready and
    // quit are refused meanwhile.
    _ = try engine.handle("go sim plies 2 plays 5");
    try testing.expect(engine.search != null);
    _ = try engine.handle("isready");
    _ = try engine.handle("gen");
    _ = try engine.handle("stop");
    engine.waitSearch();
    var lines = std.mem.splitScalar(u8, out.written(), '\n');
    var saw_ready = false;
    var saw_busy = false;
    var plays: usize = 0;
    var last: []const u8 = "";
    while (lines.next()) |l| {
        if (l.len == 0) continue;
        if (std.mem.eql(u8, l, "readyok")) saw_ready = true;
        if (std.mem.eql(u8, l, "error busy")) saw_busy = true;
        if (std.mem.startsWith(u8, l, "info sim iterations")) plays += 1;
        last = l;
    }
    try testing.expect(saw_ready and saw_busy);
    try testing.expectEqual(5, plays);
    try testing.expect(std.mem.startsWith(u8, last, "bestmove "));

    // An iteration cap, then a time limit, end it by themselves.
    out.clearRetainingCapacity();
    _ = try engine.handle("go sim plies 2 plays 4 iterations 30");
    engine.waitSearch();
    try testing.expect(std.mem.indexOf(u8, out.written(), "info sim iterations 30 play ") != null);
    out.clearRetainingCapacity();
    _ = try engine.handle("go sim plies 2 plays 4 time 300");
    engine.waitSearch();
    try testing.expect(std.mem.indexOf(u8, out.written(), "bestmove ") != null);
    // After a search the engine takes commands again.
    out.clearRetainingCapacity();
    _ = try engine.handle("isready");
    try testing.expectEqualStrings("readyok\n", out.written());
}
