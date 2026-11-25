//! Reads the engine's data files from disk: the data source the native
//! adapters (the stdio engine, the bench tool) give the engine.

const std = @import("std");
const Io = std.Io;
const macondo = @import("macondo");

/// Reads data files from disk for the engine.
pub const Files = struct {
    io: Io,

    pub fn source(self: *Files) macondo.data.Source {
        return .{ .ctx = self, .loadFn = load };
    }

    fn load(ctx: *anyopaque, gpa: std.mem.Allocator, data_path: []const u8, rel_path: []const u8) macondo.data.LoadError![]align(4) u8 {
        const self: *Files = @ptrCast(@alignCast(ctx));
        const path = try std.fs.path.join(gpa, &.{ data_path, rel_path });
        defer gpa.free(path);
        return readAll(self.io, gpa, path) catch |err| switch (err) {
            error.FileNotFound => error.FileNotFound,
            error.OutOfMemory => error.OutOfMemory,
            else => {
                std.log.err("reading {s}: {t}", .{ path, err });
                return error.ReadFailed;
            },
        };
    }

    /// The whole file, read into one allocation of its size (word maps run
    /// to hundreds of MB: no growing buffer).
    fn readAll(io: Io, gpa: std.mem.Allocator, path: []const u8) ![]align(4) u8 {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const size = try file.length(io);
        const buf = try gpa.alignedAlloc(u8, .of(u32), @intCast(size));
        errdefer gpa.free(buf);
        if (try file.readPositionalAll(io, buf, 0) != size) return error.ReadFailed;
        return buf;
    }
};
