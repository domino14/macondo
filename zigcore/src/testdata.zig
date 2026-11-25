//! Test-only access to Macondo's data directory (lexica, letter
//! distributions). The path comes from the `data-path` build option, relative
//! to zigcore/ by default. Tests whose files are missing are skipped, so the
//! suite runs on a machine without the lexica.

const std = @import("std");
const build_options = @import("build_options");
const testing = std.testing;
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const cgp = @import("cgp.zig");
const Kwg = @import("kwg.zig").Kwg;
const LetterDistribution = tiles.LetterDistribution;
const MachineLetter = tiles.MachineLetter;
const Board = board_mod.Board;
const CrossSet = board_mod.CrossSet;

/// Reads `<data path>/<rel>` with testing.allocator, 4-byte aligned so word
/// graphs can be used in place. Skips the test if the file doesn't exist.
pub fn loadFile(rel: []const u8) ![]align(4) u8 {
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

/// An English letter distribution and a lexicon, for tests.
pub const Fixture = struct {
    ld: LetterDistribution,
    kwg_bytes: []align(4) u8,
    kwg: Kwg,

    pub fn load(lexicon: []const u8) !Fixture {
        const dist_bytes = try loadFile("letterdistributions/english");
        defer testing.allocator.free(dist_bytes);
        var buf: [64]u8 = undefined;
        const kwg_bytes = try loadFile(try std.fmt.bufPrint(&buf, "lexica/gaddag/{s}.kwg", .{lexicon}));
        errdefer testing.allocator.free(kwg_bytes);
        return .{
            .ld = try .parse(testing.allocator, "english", dist_bytes),
            .kwg_bytes = kwg_bytes,
            .kwg = try .init(kwg_bytes),
        };
    }

    pub fn deinit(self: *Fixture) void {
        self.ld.deinit(testing.allocator);
        testing.allocator.free(self.kwg_bytes);
    }

    pub fn cs(self: *const Fixture, letters: []const u8) CrossSet {
        var mls: [32]MachineLetter = undefined;
        const n = self.ld.toMachineLetters(letters, &mls) catch unreachable;
        var set: CrossSet = 0;
        for (mls[0..n]) |ml| set |= @as(CrossSet, 1) << @intCast(ml);
        return set;
    }

    pub fn boardFromFen(self: *const Fixture, fen: []const u8) !Board {
        var b = Board.init(&board_mod.crossword_game);
        var rows = std.mem.splitScalar(u8, fen, '/');
        var r: usize = 0;
        while (rows.next()) |row_text| : (r += 1) {
            var row: [board_mod.max_dim]MachineLetter = undefined;
            const n = try cgp.decodeRow(row_text, &self.ld, &row);
            _ = b.setRow(r, row[0..n]);
        }
        b.countTiles();
        return b;
    }

    /// Go's SetRow: clears the row, then sets the letters of `text` from
    /// the left.
    pub fn setRowText(self: *const Fixture, b: *Board, row: usize, text: []const u8) !void {
        var mls: [board_mod.max_dim]MachineLetter = @splat(0);
        _ = try self.ld.toMachineLetters(text, &mls);
        _ = b.setRow(row, mls[0..b.dim]);
        b.countTiles();
    }
};

// Boards from board/sample_testing_boards.go, as CGP rows.
pub const vs_ed = "14E/14N/14d/14U/4GLOWS5R/8PET3E/7FAXING1R/6JAY1TEEMS/2B2BOY4N2/2L1DOE5U2/2ANEW5PI2/2MO1LEU3ON2/2EH7HE2/15/15";
pub const vs_matt = "7ZEP1F3/1FLUKY3R1R3/5EX2A1U3/2SCARIEST1I3/9TOT3/6GO1LO4/6OR1ETA3/6JABS1b3/5QI4A3/5I1N3N3/3ReSPOND1D3/1HOE3V3O3/1ENCOMIA3N3/7T7/3VENGED6";
pub const vs_jeremy = "7N6M/5ZOON4AA/7B5UN/2S4L3LADY/2T4E2QI1I1/2A2PORN3NOR/2BICE2AA1DA1E/6GUVS1OP1F/8ET1LA1U/5J3R1E1UT/4VOTE1I1R1NE/5G1MICKIES1/6FE1T1THEW/6OR3E1XI/6OY6G";
pub const vs_oxy = "1PACIFYING5/1IS12/YE13/1REQUALIFIED3/H1L12/EDS12/NO3T9/1RAINWASHING3/UM3O9/T2E1O9/1WAKEnERS6/1OnETIME7/OOT2E1B7/N6U7/1JACULATING4";
pub const maven_vs_macondo = "D1V1BEZIQuE3W/JUICER4LOUIE/I1V1G3FA1HM1A/N1AH4EL3UN/S2U5OP2RE/3ED4FA2ID/4R1MILTY2N1/3BIGOT2OCTAD/2T1NA4R2R1/2O1KI7Y1/2WOST9/2N2SPAEING3/2L4AX6/2E12/2T12";

const data = @import("data.zig");
const position_mod = @import("position.zig");

/// Reads data files under the data-path build option, for a data.Store.
fn loadForStore(_: *anyopaque, gpa: std.mem.Allocator, data_path: []const u8, rel: []const u8) data.LoadError![]align(4) u8 {
    const path = std.fs.path.join(gpa, &.{ data_path, rel }) catch return error.OutOfMemory;
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, gpa, .unlimited, .of(u32), null) catch |err| switch (err) {
        error.FileNotFound => error.FileNotFound,
        error.OutOfMemory => error.OutOfMemory,
        else => error.ReadFailed,
    };
}

pub const file_source: data.Source = .{ .ctx = undefined, .loadFn = loadForStore };

/// A data store reading the real data files, for tests.
pub fn store() data.Store {
    return .init(testing.allocator, file_source);
}

/// A position from CGP. Skips the test if its data files are missing.
pub fn position(s: *data.Store, cgp_text: []const u8) !position_mod.Position {
    var p: position_mod.Position = undefined;
    p.fromCgp(cgp_text, s, build_options.data_path) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("skipping: {s} not found\n", .{s.lastPath()});
            return error.SkipZigTest;
        },
        else => return err,
    };
    return p;
}
