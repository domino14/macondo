//! Macondo's engine core. Everything under this module is free of I/O:
//! data comes in as byte slices, allocators are passed in, and output goes
//! to a caller-supplied writer. The adapters (src/main.zig for stdio, later
//! WASM exports) are the only code that touches files or the process.

pub const version = @import("build_options").version;
pub const ucgi = @import("ucgi.zig");
pub const tiles = @import("data/tiles.zig");
pub const kwg = @import("data/kwg.zig");
pub const board = @import("board/board.zig");
pub const cross_set = @import("board/cross_set.zig");
pub const cgp = @import("board/cgp.zig");
pub const data = @import("data/data.zig");
pub const position = @import("board/position.zig");
pub const move = @import("board/move.zig");
pub const movegen = @import("movegen/movegen.zig");
pub const equity = @import("equity/equity.zig");
pub const leavemap = @import("movegen/leavemap.zig");
pub const shadow = @import("movegen/shadow.zig");
pub const wmp = @import("data/wmp.zig");
pub const chacha8 = @import("gocompat/chacha8.zig");
pub const game = @import("game/game.zig");
pub const wmp_maker = @import("data/wmp_maker.zig");
pub const wmp_movegen = @import("movegen/wmp_movegen.zig");
pub const wordmap_gen = @import("movegen/wordmap_gen.zig");
pub const sim = @import("montecarlo/sim.zig");
pub const carry = @import("movegen/carry.zig");
pub const gomath = @import("gocompat/gomath.zig");
pub const bai = @import("montecarlo/bai.zig");
pub const rit = @import("data/rit.zig");
pub const classic = @import("montecarlo/classic.zig");
pub const gosort = @import("gocompat/gosort.zig");
pub const gorand = @import("gocompat/gorand.zig");
pub const infer = @import("rangefinder/infer.zig");
pub const zobrist = @import("endgame/zobrist.zig");
pub const ttable = @import("endgame/ttable.zig");
pub const abdada = @import("endgame/abdada.zig");
pub const endgame = @import("endgame/endgame.zig");
pub const wordprune = @import("endgame/wordprune.zig");
pub const pathlists = @import("endgame/pathlists.zig");
pub const bestsmall = @import("movegen/bestsmall.zig");
pub const preendgame = @import("preendgame/peg.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
