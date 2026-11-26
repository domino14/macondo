//! Loaded game data (letter distributions, word graphs, leave values), and
//! the interface through which the engine asks its adapter for file bytes.
//! The engine never opens files itself.

const std = @import("std");
const WinTable = @import("sim.zig").WinTable;
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const kwg_mod = @import("kwg.zig");
const wmp_mod = @import("wmp.zig");
const wmp_maker = @import("wmp_maker.zig");
const rit_mod = @import("rit.zig");

pub const LoadError = error{ FileNotFound, ReadFailed, OutOfMemory };
pub const SaveError = error{WriteFailed};

/// Whether positions use a rack info table (see rit.zig): `auto` when
/// there is one for the lexicon and leaves, `always` making (and keeping)
/// one if there isn't, `never`.
pub const RackTables = enum { auto, always, never };

/// Supplied by the adapter. `load` returns the bytes of `rel_path` under
/// `data_path`, allocated with `gpa` and 4-byte aligned; the caller owns
/// them.
pub const Source = struct {
    ctx: *anyopaque,
    loadFn: *const fn (ctx: *anyopaque, gpa: Allocator, data_path: []const u8, rel_path: []const u8) LoadError![]align(4) u8,
    /// Optional: a big file mapped read-only (page-aligned), given back
    /// with unmapFn. Without it big files are read like the others.
    mapFn: ?*const fn (ctx: *anyopaque, data_path: []const u8, rel_path: []const u8) LoadError![]align(16) const u8 = null,
    unmapFn: ?*const fn (ctx: *anyopaque, bytes: []align(16) const u8) void = null,
    /// Optional: writes a file the engine made, so that it is there next
    /// time (rack info tables). Without it they are not kept.
    saveFn: ?*const fn (ctx: *anyopaque, data_path: []const u8, rel_path: []const u8, bytes: []const u8) SaveError!void = null,

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
    sibling_tiles: []u64 = &.{},
};

pub const WordMap = struct {
    bytes: []align(4) u8,
    wmp: wmp_mod.Wmp,
};

pub const Leaves = struct {
    bytes: []align(4) u8,
    klv: kwg_mod.Klv,
    /// The file's path under the data path (owned by the store).
    path: []const u8 = "",
};

pub const RackTable = struct {
    bytes: []align(16) const u8,
    mapped: bool,
    rit: rit_mod.Rit,
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
    peg: std.StringHashMapUnmanaged([]f64) = .empty,
    /// Word maps by lexicon; null for a lexicon that can't have one.
    word_maps: std.StringHashMapUnmanaged(?*WordMap) = .empty,
    win_table: ?*WinTable = null,
    rack_tables: std.StringHashMapUnmanaged(*RackTable) = .empty,

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
            self.gpa.free(e.value_ptr.*.sibling_tiles);
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
        var p = self.peg.iterator();
        while (p.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            self.gpa.free(e.value_ptr.*);
        }
        self.peg.deinit(self.gpa);
        var m = self.word_maps.iterator();
        while (m.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            if (e.value_ptr.*) |wm| {
                wm.wmp.deinit(self.gpa);
                self.gpa.free(wm.bytes);
                self.gpa.destroy(wm);
            }
        }
        self.word_maps.deinit(self.gpa);
        if (self.win_table) |t| self.gpa.destroy(t);
        var rt = self.rack_tables.iterator();
        while (rt.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            self.freeTableBytes(e.value_ptr.*.bytes, e.value_ptr.*.mapped);
            self.gpa.destroy(e.value_ptr.*);
        }
        self.rack_tables.deinit(self.gpa);
        self.* = undefined;
    }

    /// The win probabilities sims score rollouts with. Go's engine reads
    /// them for NWL20 whatever the lexicon (strategy/NWL20/winpct.csv,
    /// then the NWL23 and default folders).
    pub fn winTable(self: *Store, data_path: []const u8) Error!*const WinTable {
        if (self.win_table) |t| return t;
        var buf: [128]u8 = undefined;
        for (strategyDirs("NWL20")) |dir| {
            const bytes = self.loadFile(data_path, "{s}", .{try strategyPath(&buf, dir, "winpct.csv")}) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            defer self.gpa.free(bytes);
            const t = try self.gpa.create(WinTable);
            errdefer self.gpa.destroy(t);
            t.parse(bytes) catch return error.InvalidData;
            self.win_table = t;
            return t;
        }
        return error.FileNotFound;
    }

    /// The rack info table of lexicon `name` with `leaves`, per `mode`:
    /// lexica/<lexicon>.<leaves>.zrit, used only if it was made from these
    /// word graph and leave files; with `always`, made on every core if
    /// there is none (about 20 s and 2.3 GB for English) and saved if the
    /// source can. Null if there is none to use.
    pub fn rackInfoTable(self: *Store, data_path: []const u8, name: []const u8, lex: *const Lexicon, leaves: *const Leaves, word_map: *const wmp_mod.Wmp, ld: *const tiles.LetterDistribution, mode: RackTables) Error!?*const rit_mod.Rit {
        if (mode == .never) return null;
        if (!validName(name)) return error.InvalidName;
        var buf: [128]u8 = undefined;
        const rel = try fmtName(&buf, "lexica/{s}.{s}.zrit", .{ name, leavesStem(leaves.path) });
        if (self.rack_tables.get(rel)) |t| return &t.rit;

        var bytes: []align(16) const u8 = &.{};
        var mapped = false;
        if (self.loadTableBytes(data_path, rel)) |loaded| {
            bytes = loaded.bytes;
            mapped = loaded.mapped;
            const ok = if (rit_mod.Rit.init(bytes)) |r| r.madeFrom(lex.bytes, leaves.bytes) else |_| false;
            if (!ok) {
                self.freeTableBytes(bytes, mapped);
                bytes = &.{};
            }
        } else |err| if (err != error.FileNotFound) return err;
        if (bytes.len == 0) {
            if (mode != .always) return null;
            const threads = std.Thread.getCpuCount() catch 1;
            const made = rit_mod.make(self.gpa, .{
                .ld = ld,
                .kwg_bytes = lex.bytes,
                .klv = &leaves.klv,
                .klv_bytes = leaves.bytes,
                .word_map = word_map,
            }, threads) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return null,
            };
            if (self.source.saveFn) |save| save(self.source.ctx, data_path, rel, made) catch {};
            bytes = made;
            mapped = false;
        }
        errdefer self.freeTableBytes(bytes, mapped);
        const t = try self.gpa.create(RackTable);
        errdefer self.gpa.destroy(t);
        t.* = .{ .bytes = bytes, .mapped = mapped, .rit = rit_mod.Rit.init(bytes) catch unreachable };
        const key = try self.gpa.dupe(u8, rel);
        errdefer self.gpa.free(key);
        try self.rack_tables.put(self.gpa, key, t);
        return &t.rit;
    }

    fn loadTableBytes(self: *Store, data_path: []const u8, rel: []const u8) LoadError!struct { bytes: []align(16) const u8, mapped: bool } {
        self.setLastPath(rel);
        if (self.source.mapFn) |map| return .{ .bytes = try map(self.source.ctx, data_path, rel), .mapped = true };
        const raw = try self.source.load(self.gpa, data_path, rel);
        defer self.gpa.free(raw);
        const copy = try self.gpa.alignedAlloc(u8, .@"16", raw.len);
        @memcpy(copy, raw);
        return .{ .bytes = copy, .mapped = false };
    }

    fn freeTableBytes(self: *Store, bytes: []align(16) const u8, mapped: bool) void {
        if (bytes.len == 0) return;
        if (mapped) {
            if (self.source.unmapFn) |unmap| unmap(self.source.ctx, bytes);
        } else {
            self.gpa.free(bytes);
        }
    }

    fn setLastPath(self: *Store, rel: []const u8) void {
        const n = @min(rel.len, self.last_path.len);
        @memcpy(self.last_path[0..n], rel[0..n]);
        self.last_path_len = n;
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
        lex.sibling_tiles = try lex.kwg.buildSiblingTiles(self.gpa);
        errdefer self.gpa.free(lex.sibling_tiles);
        lex.kwg.sibling_tiles = lex.sibling_tiles;
        const key = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(key);
        try self.lexica.put(self.gpa, key, lex);
        return lex;
    }

    /// The word map of lexicon `name` (lexica/<name>.wmp), made from its
    /// word graph in memory if there is no file; null if `ld` can't have
    /// one. Made for a 15x15 board, as Go and MAGPIE make them.
    pub fn wordMap(self: *Store, data_path: []const u8, name: []const u8, lex: *const Lexicon, ld: *const tiles.LetterDistribution) Error!?*const wmp_mod.Wmp {
        if (!validName(name)) return error.InvalidName;
        if (self.word_maps.get(name)) |wm| return if (wm) |w| &w.wmp else null;

        const key = try self.gpa.dupe(u8, name);
        errdefer self.gpa.free(key);
        const bytes: []align(4) u8 = self.loadFile(data_path, "lexica/{s}.wmp", .{name}) catch |err| switch (err) {
            error.FileNotFound => wmp_maker.makeFromKwg(self.gpa, lex.kwg, ld, 15) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                // This distribution can't have a word map.
                else => {
                    try self.word_maps.put(self.gpa, key, null);
                    return null;
                },
            },
            else => |e| return e,
        };
        errdefer self.gpa.free(bytes);
        const wm = try self.gpa.create(WordMap);
        errdefer self.gpa.destroy(wm);
        wm.* = .{ .bytes = bytes, .wmp = wmp_mod.Wmp.init(self.gpa, bytes) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidData,
        } };
        errdefer wm.wmp.deinit(self.gpa);
        try self.word_maps.put(self.gpa, key, wm);
        return &wm.wmp;
    }

    /// Leave values from the file at `rel` under `data_path`, cached by path.
    fn leaveFile(self: *Store, data_path: []const u8, rel: []const u8) Error!*Leaves {
        if (self.leaves.get(rel)) |l| return l;

        const bytes = try self.loadFile(data_path, "{s}", .{rel});
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
        const key = try self.gpa.dupe(u8, rel);
        errdefer self.gpa.free(key);
        try self.leaves.put(self.gpa, key, lv);
        lv.path = key;
        return lv;
    }

    /// The leave values of lexicon `name` itself, lexica/gaddag/<name>.klv2.
    pub fn leaveValues(self: *Store, data_path: []const u8, name: []const u8) Error!*Leaves {
        if (!validName(name)) return error.InvalidName;
        var buf: [128]u8 = undefined;
        return self.leaveFile(data_path, std.fmt.bufPrint(&buf, "lexica/gaddag/{s}.klv2", .{name}) catch return error.InvalidName);
    }

    /// The leave values a game in lexicon `name` is played with, or null if
    /// there are none (plays are then valued by score alone). Searched as
    /// Macondo's dataloaders.LeavesFileForLexicon does: the lexicon's own
    /// lexica/gaddag/<name>.klv2, then that of the lexicon it borrows from
    /// (CSW19 uses CSW24's), then the legacy strategy folders. `super`
    /// asks for the SuperCrosswordGame leaves (super-<name>.klv2).
    pub fn leavesFor(self: *Store, data_path: []const u8, name: []const u8, super: bool) Error!?*Leaves {
        if (!validName(name)) return error.InvalidName;
        const prefix = if (super) "super-" else "";
        const def = defaultForLexicon(name);
        var buf: [128]u8 = undefined;
        if (self.leaveFile(data_path, try fmtName(&buf, "lexica/gaddag/{s}{s}.klv2", .{ prefix, name }))) |l| {
            return l;
        } else |err| if (err != error.FileNotFound) return err;
        if (def.len > 0 and !std.mem.eql(u8, def, name)) {
            if (self.leaveFile(data_path, try fmtName(&buf, "lexica/gaddag/{s}{s}.klv2", .{ prefix, def }))) |l| {
                return l;
            } else |err| if (err != error.FileNotFound) return err;
        }
        const legacy = if (super) "super-leaves.klv2" else "leaves.klv2";
        for (strategyDirs(name)) |dir| {
            if (self.leaveFile(data_path, try strategyPath(&buf, dir, legacy))) |l| {
                return l;
            } else |err| if (err != error.FileNotFound) return err;
        }
        return null;
    }

    /// Pre-endgame equity adjustments for lexicon `name`, indexed by tiles
    /// in the bag after the play plus 7 (preendgame.json in the strategy
    /// folders). Empty if there are none.
    pub fn preEndgameValues(self: *Store, data_path: []const u8, name: []const u8) Error![]const f64 {
        if (!validName(name)) return error.InvalidName;
        var buf: [128]u8 = undefined;
        for (strategyDirs(name)) |dir| {
            if (self.pegFile(data_path, try strategyPath(&buf, dir, "preendgame.json"))) |v| {
                return v;
            } else |err| if (err != error.FileNotFound) return err;
        }
        return &.{};
    }

    fn pegFile(self: *Store, data_path: []const u8, rel: []const u8) Error![]const f64 {
        if (self.peg.get(rel)) |v| return v;
        const bytes = try self.loadFile(data_path, "{s}", .{rel});
        defer self.gpa.free(bytes);
        const parsed = std.json.parseFromSlice([]f64, self.gpa, bytes, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidData,
        };
        defer parsed.deinit();
        const values = try self.gpa.dupe(f64, parsed.value);
        errdefer self.gpa.free(values);
        const key = try self.gpa.dupe(u8, rel);
        errdefer self.gpa.free(key);
        try self.peg.put(self.gpa, key, values);
        return values;
    }
};

/// The name a rack info table gives its leaves: the leave file's name
/// without extension, or for the legacy strategy files (leaves.klv2) their
/// folder's. Two leave files can share a name; a table is used only for
/// the files it was made from in any case.
fn leavesStem(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    const stem = base[0 .. std.mem.lastIndexOfScalar(u8, base, '.') orelse base.len];
    if (!std.mem.endsWith(u8, stem, "leaves")) return stem;
    const dir = std.fs.path.dirname(path) orelse return stem;
    return std.fs.path.basename(dir);
}

/// The strategy folders searched for a lexicon's files, in order: its own,
/// the one of the lexicon it borrows from, and "default". Port of
/// dataloaders.StratFileForLexicon, which for a lexicon that borrows from
/// none looks in strategy/ itself second (the empty name here).
fn strategyDirs(name: []const u8) [3][]const u8 {
    return .{ name, defaultForLexicon(name), "default" };
}

fn strategyPath(buf: *[128]u8, dir: []const u8, file: []const u8) error{InvalidName}![]const u8 {
    if (dir.len == 0) return fmtName(buf, "strategy/{s}", .{file});
    return fmtName(buf, "strategy/{s}/{s}", .{ dir, file });
}

fn fmtName(buf: *[128]u8, comptime fmt: []const u8, args: anytype) error{InvalidName}![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch error.InvalidName;
}

/// The lexicon whose strategy files and leaves a lexicon without its own
/// uses. Port of dataloaders.defaultForLexicon.
pub fn defaultForLexicon(name: []const u8) []const u8 {
    const table = [_]struct { []const u8, []const u8 }{
        .{ "CSW", "CSW24" },  .{ "TWL", "NWL23" },   .{ "NWL", "NWL23" },
        .{ "NSWL", "NWL23" }, .{ "ECWL", "ECWL" },   .{ "CEL", "ECWL" },
        .{ "RD", "RD29" },    .{ "NSF", "NSF23" },   .{ "FRA", "FRA24" },
        .{ "DISC", "DISC2" }, .{ "OSPS", "OSPS50" }, .{ "FILE", "FILE2017" },
    };
    for (table) |e| {
        if (std.mem.startsWith(u8, name, e[0])) return e[1];
    }
    return "";
}

/// Names become file names: no path separators or dots allowed.
fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-')) return false;
    }
    return true;
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

/// A data source serving a fixed set of files from memory.
const MemSource = struct {
    files: []const struct { []const u8, []const u8 },
    /// Every path asked for, in order, joined by spaces.
    asked: std.ArrayList(u8) = .empty,

    fn source(self: *MemSource) Source {
        return .{ .ctx = self, .loadFn = load };
    }

    fn load(ctx: *anyopaque, gpa: Allocator, _: []const u8, rel: []const u8) LoadError![]align(4) u8 {
        const self: *MemSource = @ptrCast(@alignCast(ctx));
        self.asked.appendSlice(testing.allocator, rel) catch return error.OutOfMemory;
        self.asked.append(testing.allocator, ' ') catch return error.OutOfMemory;
        for (self.files) |f| {
            if (std.mem.eql(u8, f[0], rel)) {
                const out = try gpa.alignedAlloc(u8, .of(u32), f[1].len);
                @memcpy(out, f[1]);
                return out;
            }
        }
        return error.FileNotFound;
    }
};

/// The smallest leave file: a two-node word graph and no leaves.
const empty_klv = std.mem.toBytes([_]u32{ 2, is_end, is_end, 0 });
const is_end: u32 = kwg_mod.is_end_bit;

test "lexicons borrow from" {
    try testing.expectEqualStrings("CSW24", defaultForLexicon("CSW19"));
    try testing.expectEqualStrings("NWL23", defaultForLexicon("TWL06"));
    try testing.expectEqualStrings("NWL23", defaultForLexicon("NSWL20"));
    try testing.expectEqualStrings("ECWL", defaultForLexicon("CEL"));
    try testing.expectEqualStrings("FILE2017", defaultForLexicon("FILE2017"));
    try testing.expectEqualStrings("", defaultForLexicon("America"));
}

test "leave files are searched in Macondo's order" {
    var mem: MemSource = .{ .files = &.{
        .{ "lexica/gaddag/CSW24.klv2", &empty_klv },
        .{ "lexica/gaddag/super-NWL23.klv2", &empty_klv },
        .{ "strategy/default/leaves.klv2", &empty_klv },
    } };
    defer mem.asked.deinit(testing.allocator);
    var store: Store = .init(testing.allocator, mem.source());
    defer store.deinit();

    // Its own.
    const csw = (try store.leavesFor("d", "CSW24", false)).?;
    // Borrowed: CSW19 has none of its own.
    try testing.expectEqual(csw, (try store.leavesFor("d", "CSW19", false)).?);
    // SuperCrosswordGame leaves, borrowed.
    _ = (try store.leavesFor("d", "NSWL20", true)).?;
    // None anywhere but the legacy default folder.
    _ = (try store.leavesFor("d", "America", false)).?;
    // None at all.
    try testing.expectEqual(null, try store.leavesFor("d", "America", true));
    try testing.expectEqualStrings("lexica/gaddag/CSW24.klv2 " ++
        "lexica/gaddag/CSW19.klv2 " ++
        "lexica/gaddag/super-NSWL20.klv2 lexica/gaddag/super-NWL23.klv2 " ++
        "lexica/gaddag/America.klv2 strategy/America/leaves.klv2 strategy/leaves.klv2 strategy/default/leaves.klv2 " ++
        "lexica/gaddag/super-America.klv2 strategy/America/super-leaves.klv2 strategy/super-leaves.klv2 strategy/default/super-leaves.klv2 ", mem.asked.items);
}

test "pre-endgame values" {
    var mem: MemSource = .{ .files = &.{
        .{ "strategy/NWL23/preendgame.json", "[0, -8, 0.5]" },
        .{ "strategy/default/preendgame.json", "[]" },
        .{ "strategy/CSW24/preendgame.json", "[1, oops]" },
    } };
    defer mem.asked.deinit(testing.allocator);
    var store: Store = .init(testing.allocator, mem.source());
    defer store.deinit();

    try testing.expectEqualSlices(f64, &.{ 0, -8, 0.5 }, try store.preEndgameValues("d", "NWL23"));
    // Borrowed from NWL23.
    try testing.expectEqualSlices(f64, &.{ 0, -8, 0.5 }, try store.preEndgameValues("d", "TWL06"));
    // The default folder's.
    try testing.expectEqual(0, (try store.preEndgameValues("d", "FRA20")).len);
    try testing.expectError(error.InvalidData, store.preEndgameValues("d", "CSW21"));
}
