//! Native stdio adapter: reads UCGI commands from stdin, one per line, and
//! writes responses to stdout. Logs go to stderr. Ends on `quit` or EOF.

const std = @import("std");
const Io = std.Io;
const macondo = @import("macondo");
const Files = @import("files.zig").Files;

/// Longest accepted input line. CGP strings are well under this.
const max_line = 64 * 1024;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var files: Files = .{ .io = io };

    var in_buf: [max_line]u8 = undefined;
    var stdin_reader: Io.File.Reader = .init(.stdin(), io, &in_buf);
    const in = &stdin_reader.interface;

    var out_buf: [4096]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &out_buf);

    // The engine writes (and flushes) responses itself: a search running
    // in the background writes while commands are read.
    var engine = try macondo.ucgi.Engine.init(init.gpa, io, macondo.version, files.source(), &stdout_writer.interface);
    defer engine.deinit();

    while (true) {
        const line = in.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                // Drop the rest of the oversized line and carry on.
                _ = in.discardDelimiterInclusive('\n') catch {};
                try engine.writeLine("error line-too-long");
                continue;
            },
            else => |e| return e,
        } orelse break;
        if (try engine.handle(line) == .quit) break;
    }
}
