//! Moves: tile plays, exchanges and passes. Port of Macondo's `move`
//! package, minus the leave (it is the rack minus the tiles, worked out
//! when needed).

const std = @import("std");
const tiles = @import("tiles.zig");
const board = @import("board.zig");
const MachineLetter = tiles.MachineLetter;

pub const Kind = enum(u8) { play, exchange, pass };

pub const Move = struct {
    kind: Kind,
    vertical: bool = false,
    row: u8 = 0,
    col: u8 = 0,
    /// Tiles along the play, 0 for a square already holding a tile; for an
    /// exchange, the tiles thrown in.
    len: u8 = 0,
    tiles_played: u8 = 0,
    score: i32 = 0,
    tiles: [board.max_dim]MachineLetter = undefined,

    pub fn pass() Move {
        return .{ .kind = .pass };
    }

    pub fn tileSlice(self: *const Move) []const MachineLetter {
        return self.tiles[0..self.len];
    }

    /// UCGI notation: `8D.QUIZ` (row first = horizontal, column first =
    /// vertical; `.` for a tile already on the board, lower case for a
    /// designated blank), `exch.ABC`, `pass`.
    pub fn format(self: *const Move, ld: *const tiles.LetterDistribution, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self.kind) {
            .pass => return w.writeAll("pass"),
            .exchange => {
                try w.writeAll("exch.");
                for (self.tileSlice()) |ml| try w.writeAll(ld.letter(ml, false));
            },
            .play => {
                const col_letter: u8 = 'A' + self.col;
                if (self.vertical) {
                    try w.print("{c}{d}.", .{ col_letter, @as(u32, self.row) + 1 });
                } else {
                    try w.print("{d}{c}.", .{ @as(u32, self.row) + 1, col_letter });
                }
                for (self.tileSlice()) |ml| try w.writeAll(ld.letter(ml, true));
            },
        }
    }
};
