//! Loaded game data (letter distributions, word graphs, leave values), and
//! the interface through which the engine asks its adapter for file bytes.
//! The engine never opens files itself.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const kwg_mod = @import("kwg.zig");

pub const LoadError = error{ FileNotFound, ReadFailed, OutOfMemory };

/// Supplied by the adapter. `load` returns the bytes of `rel_path` under
/// `data_path`, allocated with `gpa` and 4-byte aligned; the caller owns
/// them.
pub const Source = struct {
    ctx: *anyopaque,
    loadFn: *const fn (ctx: *anyopaque, gpa: Allocator, data_path: []const u8, rel_path: []const u8) LoadError![]align(4) u8,

    pub fn load(self: Source, gpa: Allocator, data_path: []const u8, rel_path: []const u8) LoadError![]align(4) u8 {
        return self.loadFn(self.ctx, gpa, data_path, rel_path);
    }

    /// A source with no files, for tests and builds without data.
    pub const empty: Source = .{ .ctx = undefined, .loadFn = loadNothing };

    fn loadNothing(_: *anyopaque, _: Allocator, _: []const u8, _: []const u8) LoadError![]align(4) u8 {
        return error.FileNotFound;
    }
};

pub const Error = LoadError || error{ InvalidData, InvalidName };

pub const Lexicon = struct {
    bytes: []align(4) u8,
    kwg: kwg_mod.Kwg,
};

pub const Leaves = struct {
    bytes: []align(4) u8,
    klv: kwg_mod.Klv,
};

/// Everything loaded so far, by name. Entries live until deinit, so
/// pointers to them stay valid.
pub const Store = struct {
    gpa: Allocator,
    source: Source,
    /// The path of the file that failed to load last, for error messages.
    last_path: [256]u8 = undefined,
    last_path_len: usize = 0,
    dists: std.StringHashMapUnmanaged(*tiles.LetterDistribution) = .empty,
    lexica: std.StringHashMapUnmanaged(*Lexicon) = .empty,
    leaves: std.StringHashMapUnmanaged(*Leaves) = .empty,

    pub fn init(gpa: Allocator, source: Source) Store {
        return .{ .gpa = gpa, .source = source };
    }

    pub fn deinit(self: *Store) void {
        var d = self.dists.iterator();
        while (d.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            e.value_ptr.*.deinit(self.gpa);
            self.gpa.destroy(e.value_ptr.*);
        }
        self.dists.deinit(self.gpa);
        var l = self.lexica.iterator();
        while (l.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            self.gpa.free(e.value_ptr.*.bytes);
            self.gpa.destroy(e.value_ptr.*);
        }
        self.lexica.deinit(self.gpa);
        var v = self.leaves.iterator();
        while (v.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            e.value_ptr.*.klv.deinit(self.gpa);
            self.gpa.free(e.value_ptr.*.bytes);
            self.gpa.destroy(e.value_ptr.*);
        }
        self.leaves.deinit(self.gpa);
        self.* = undefined;
    }

    pub fn lastPath(self: *const Store) []const u8 {
        return self.last_path[0..self.last_path_len];
    }

    fn loadFile(self: *Store, data_path: []const u8, comptime fmt: []const u8, args: anytype) Error![]align(4) u8 {
        const rel = std.fmt.bufPrint(&self.last_path, fmt, args) catch return error.InvalidName;
        self.last_path_len = rel.len;
        return self.source.load(self.gpa, data_path, rel);
    }

    /// A letter distribution by (case-insensitive) name.
    pub fn distribution(self: *Store, data_path: []const u8, name: []const u8) Error!*tiles.LetterDistribution {
        var lower_buf: [64]u8 = undefined;
        if (name.len > lower_buf.len or !validName(name)) return error.InvalidName;
        const lower = std.ascii.lowerString(&lower_buf, name);
        if (self.dists.get(lower)) |d| return d;

        const bytes = try self.loadFile(data_path, "letterdistributions/{s}", .{lower});
        defer self.gpa.free(bytes);
        const ld = try self.gpa.create(tiles.LetterDistribution);
        errdefer self.gpa.destroy(ld);
        ld.* = tiles.LetterDistribution.parse(self.gpa, lower, bytes) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidData,
        };
        errdefer ld.deinit(self.gpa);
        const key = try self.gpa.dupe(u8, lower);
        errdefer self.gpa.free(key);
        try self.dists.put(self.gpa, key, ld);
        return ld;
    }

    /// A lexicon's word graph, lexica/gaddag/<name>.kwg.
    pub fn lexicon(self: *Store, data_path: []const u8, name: []const u8) Error!*Lexicon {
        if (!validName(name)) return error.InvalidName;
        if (self.lexica.get(name)) |l| return l;

        const bytes = try self.loadFile(data_path, "lexica/gaddag/{s}.kwg", .{name});
        errdefer self.gpa.free(bytes);
        const lex = try self.gpa.create(Lexicon);
        errdefer self.gpa.destroy(lex);
        lex.* = .{ .bytes = bytes, .kwg = kwg_mod.Kwg.init(bytes) catch return error.InvalidData };
        const key = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(key);
        try self.lexica.put(self.gpa, key, lex);
        return lex;
    }

    /// Leave values, lexica/gaddag/<name>.klv2.
    pub fn leaveValues(self: *Store, data_path: []const u8, name: []const u8) Error!*Leaves {
        if (!validName(name)) return error.InvalidName;
        if (self.leaves.get(name)) |l| return l;

        const bytes = try self.loadFile(data_path, "lexica/gaddag/{s}.klv2", .{name});
        errdefer self.gpa.free(bytes);
        const lv = try self.gpa.create(Leaves);
        errdefer self.gpa.destroy(lv);
        lv.* = .{
            .bytes = bytes,
            .klv = kwg_mod.Klv.init(self.gpa, bytes) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.InvalidData,
            },
        };
        errdefer lv.klv.deinit(self.gpa);
        const key = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(key);
        try self.leaves.put(self.gpa, key, lv);
        return lv;
    }
};

/// Names become file names: no path separators or dots allowed.
fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-')) return false;
    }
    return true;
}
