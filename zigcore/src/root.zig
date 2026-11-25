//! Macondo's engine core. Everything under this module is free of I/O:
//! data comes in as byte slices, allocators are passed in, and output goes
//! to a caller-supplied writer. The adapters (src/main.zig for stdio, later
//! WASM exports) are the only code that touches files or the process.

pub const version = @import("build_options").version;
pub const ucgi = @import("ucgi.zig");
pub const tiles = @import("tiles.zig");
pub const kwg = @import("kwg.zig");
pub const board = @import("board.zig");
pub const cross_set = @import("cross_set.zig");
pub const cgp = @import("cgp.zig");
pub const data = @import("data.zig");
pub const position = @import("position.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
