//! Reads the engine's data files from disk: the data source the native
//! adapters (the stdio engine, the bench tool) give the engine.

const std = @import("std");
const Io = std.Io;
const macondo = @import("macondo");

/// Reads data files from disk for the engine.
pub const Files = struct {
    io: Io,

    pub fn source(self: *Files) macondo.data.Source {
        return .{ .ctx = self, .loadFn = load, .mapFn = map, .unmapFn = unmap, .saveFn = save };
    }

    /// Maps a file read-only (rack info tables: gigabytes, used in place).
    fn map(ctx: *anyopaque, data_path: []const u8, rel_path: []const u8) macondo.data.LoadError![]align(16) const u8 {
        const self: *Files = @ptrCast(@alignCast(ctx));
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ data_path, rel_path }) catch return error.FileNotFound;
        const file = Io.Dir.cwd().openFile(self.io, path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return error.ReadFailed,
        };
        defer file.close(self.io);
        const size = file.length(self.io) catch return error.ReadFailed;
        if (size == 0) return error.ReadFailed;
        const bytes = std.posix.mmap(null, @intCast(size), .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.ReadFailed,
        };
        return bytes;
    }

    fn unmap(_: *anyopaque, bytes: []align(16) const u8) void {
        std.posix.munmap(@alignCast(bytes));
    }

    /// Writes a file the engine made: to a temporary name first, then
    /// renamed, so that a file is never there half written.
    fn save(ctx: *anyopaque, data_path: []const u8, rel_path: []const u8, bytes: []const u8) macondo.data.SaveError!void {
        const self: *Files = @ptrCast(@alignCast(ctx));
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ data_path, rel_path }) catch return error.WriteFailed;
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}.tmp", .{path}) catch return error.WriteFailed;
        const dir = Io.Dir.cwd();
        {
            const file = dir.createFile(self.io, tmp, .{}) catch return error.WriteFailed;
            defer file.close(self.io);
            file.writePositionalAll(self.io, bytes, 0) catch return error.WriteFailed;
        }
        dir.rename(tmp, dir, path, self.io) catch return error.WriteFailed;
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
