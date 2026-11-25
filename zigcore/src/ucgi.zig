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

pub const engine_name = "MacondoZig";
pub const engine_author = "the Macondo authors";

const threads_max = 512;
const hash_mb_max = 65536;

pub const Options = struct {
    threads: u32 = 1,
    hash_mb: u32 = 256,
    /// Owned by the engine's allocator.
    data_path: []u8,
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
        p.fromCgp(text, &self.store, self.options.data_path) catch |err| switch (err) {
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

    /// Debug: every move for the player on turn, highest score first, then
    /// `genok count <n>`.
    fn printMoves(self: *Engine, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        var gen = movegen.Generator.init(self.gpa, &p.board, p.lexicon.kwg, p.dist);
        defer gen.deinit();
        const ex = movegen.exchangeRules(p.unseenBy(0), p.lexiconName());
        gen.max_can_exchange = ex.max;
        var rack = p.racks[0];
        try gen.genAll(&rack, ex.allowed);
        const plays = gen.plays.items;
        std.mem.sort(Move, plays, {}, struct {
            fn higher(_: void, a: Move, b: Move) bool {
                return a.score > b.score;
            }
        }.higher);
        for (plays) |*m| {
            try out.writeAll("move ");
            try m.format(p.dist, out);
            try out.print(" score {d}\n", .{m.score});
        }
        try out.print("genok count {d}\n", .{plays.len});
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

    /// Debug: `leave <tiles>` prints the leave value of the tiles with the
    /// position's lexicon's leave file.
    fn printLeave(self: *Engine, args: []const u8, out: *Writer) (Writer.Error || Allocator.Error)!void {
        const p = &(self.position orelse return out.writeAll("error no-position\n"));
        const text = std.mem.trim(u8, args, " \t\r");
        var mls: [32]tiles.MachineLetter = undefined;
        const n = p.dist.toMachineLetters(text, &mls) catch return out.writeAll("error bad-tiles\n");
        for (mls[0..n]) |ml| if (tiles.isBlanked(ml)) return out.writeAll("error bad-tiles\n");
        std.mem.sort(tiles.MachineLetter, mls[0..n], {}, std.sort.asc(tiles.MachineLetter));
        const lv = self.store.leaveValues(self.options.data_path, p.lexiconName()) catch |err| return self.dataError(err, out);
        // The leave file holds f32s; print the f32 exactly.
        const v: f32 = @floatCast(lv.klv.leaveValue(mls[0..n]));
        try out.print("leave {d}\n", .{v});
    }
};

/// Writes "TooManyTiles" as "too-many-tiles", for error tokens.
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
