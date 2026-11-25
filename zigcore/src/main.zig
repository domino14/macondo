//! Native stdio adapter: reads UCGI commands from stdin, one per line, and
//! writes responses to stdout. Logs go to stderr. Ends on `quit` or EOF.

const std = @import("std");
const Io = std.Io;
const macondo = @import("macondo");

/// Longest accepted input line. CGP strings are well under this.
const max_line = 64 * 1024;

/// Reads data files from disk for the engine.
const Files = struct {
    io: Io,

    fn source(self: *Files) macondo.data.Source {
        return .{ .ctx = self, .loadFn = load };
    }

    fn load(ctx: *anyopaque, gpa: std.mem.Allocator, data_path: []const u8, rel_path: []const u8) macondo.data.LoadError![]align(4) u8 {
        const self: *Files = @ptrCast(@alignCast(ctx));
        const path = try std.fs.path.join(gpa, &.{ data_path, rel_path });
        defer gpa.free(path);
        return Io.Dir.cwd().readFileAllocOptions(self.io, path, gpa, .unlimited, .of(u32), null) catch |err| switch (err) {
            error.FileNotFound => error.FileNotFound,
            error.OutOfMemory => error.OutOfMemory,
            else => {
                std.log.err("reading {s}: {t}", .{ path, err });
                return error.ReadFailed;
            },
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var files: Files = .{ .io = io };

    var engine = try macondo.ucgi.Engine.init(init.gpa, macondo.version, files.source());
    defer engine.deinit();

    var in_buf: [max_line]u8 = undefined;
    var stdin_reader: Io.File.Reader = .init(.stdin(), io, &in_buf);
    const in = &stdin_reader.interface;

    var out_buf: [4096]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &out_buf);
    const out = &stdout_writer.interface;

    while (true) {
        const line = in.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                // Drop the rest of the oversized line and carry on.
                _ = in.discardDelimiterInclusive('\n') catch {};
                try out.writeAll("error line-too-long\n");
                try out.flush();
                continue;
            },
            else => |e| return e,
        } orelse break;
        const status = try engine.handle(line, out);
        try out.flush();
        if (status == .quit) break;
    }
}
