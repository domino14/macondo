//! UCGI command handling (see zigcore/UCGI.md). The engine takes one input
//! line at a time and writes its response lines to the writer it is given;
//! it never reads or writes anything else, so every adapter (stdio, WASM,
//! sockets) drives it the same way.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

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

    pub fn init(gpa: Allocator, version: []const u8) Allocator.Error!Engine {
        return .{
            .gpa = gpa,
            .version = version,
            .options = .{ .data_path = try gpa.dupe(u8, "./data") },
        };
    }

    pub fn deinit(self: *Engine) void {
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
};

fn parseSpin(value: []const u8, min: u32, max: u32) ?u32 {
    const v = std.fmt.parseInt(u32, value, 10) catch return null;
    if (v < min or v > max) return null;
    return v;
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

/// Runs `lines` through a fresh engine and returns everything it wrote.
fn transcript(lines: []const []const u8) ![]u8 {
    var engine = try Engine.init(testing.allocator, "0.0.0");
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
    var engine = try Engine.init(testing.allocator, "0.0.0");
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
