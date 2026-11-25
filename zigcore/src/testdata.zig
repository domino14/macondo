//! Test-only access to Macondo's data directory (lexica, letter
//! distributions). The path comes from the `data-path` build option, relative
//! to zigcore/ by default. Tests whose files are missing are skipped, so the
//! suite runs on a machine without the lexica.

const std = @import("std");
const build_options = @import("build_options");
const testing = std.testing;

/// Reads `<data path>/<rel>` with testing.allocator, 4-byte aligned so word
/// graphs can be used in place. Skips the test if the file doesn't exist.
pub fn load(rel: []const u8) ![]align(4) u8 {
    const path = try std.fs.path.join(testing.allocator, &.{ build_options.data_path, rel });
    defer testing.allocator.free(path);
    return std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .unlimited, .of(u32), null) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("skipping: {s} not found\n", .{path});
            return error.SkipZigTest;
        },
        else => return err,
    };
}
