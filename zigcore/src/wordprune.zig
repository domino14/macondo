//! A word graph pruned to an endgame's words: those that can be played on
//! the board with the tiles of both racks (port of movegen/wordprune, after
//! MAGPIE's word_prune). Move generation and cross-sets over it find the
//! same plays as over the whole lexicon, faster: a letter only someone
//! holds can make a word, and every such word is in it.
//!
//! The words are the ones the two racks together could make standing alone
//! in an empty stretch of the board, and the ones through or beside the
//! tiles of some row or column. The graph is a GADDAG only (no DAWG), built
//! as a trie, breadth first, in the KWG format.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const kwg_mod = @import("kwg.zig");
const MachineLetter = tiles.MachineLetter;
const Kwg = kwg_mod.Kwg;
const Board = board_mod.Board;

const separator: MachineLetter = 0;

/// A pruned graph; its nodes and sibling tiles are owned.
pub const Pruned = struct {
    nodes: []u32,
    sibling_tiles: []u64,

    pub fn kwg(self: *const Pruned) Kwg {
        return .{ .nodes = self.nodes, .sibling_tiles = self.sibling_tiles };
    }

    pub fn deinit(self: *Pruned, gpa: Allocator) void {
        gpa.free(self.nodes);
        gpa.free(self.sibling_tiles);
        self.* = undefined;
    }
};

/// Words as one buffer of letters and their ends.
const Words = struct {
    letters: std.ArrayList(MachineLetter) = .empty,
    ends: std.ArrayList(u32) = .empty,

    fn add(w: *Words, gpa: Allocator, word: []const MachineLetter) Allocator.Error!void {
        try w.letters.appendSlice(gpa, word);
        try w.ends.append(gpa, @intCast(w.letters.items.len));
    }

    fn get(w: *const Words, i: usize) []const MachineLetter {
        const start = if (i == 0) 0 else w.ends.items[i - 1];
        return w.letters.items[start..w.ends.items[i]];
    }

    fn deinit(w: *Words, gpa: Allocator) void {
        w.letters.deinit(gpa);
        w.ends.deinit(gpa);
    }
};

/// The graph of the words playable on `b` with the tiles of racks `r0` and
/// `r1` (`num_letters` letter types), from the full graph `full`. Null when
/// there are none.
pub fn generate(gpa: Allocator, b: *const Board, r0: *const tiles.Rack, r1: *const tiles.Rack, num_letters: usize, full: Kwg) Allocator.Error!?Pruned {
    var words: Words = .{};
    defer words.deinit(gpa);
    var rack: [tiles.max_alphabet_size]i32 = @splat(0);
    for (0..num_letters) |i| rack[i] = @as(i32, r0.counts[i]) + r1.counts[i];

    const dim = b.dim;
    // Every row and column, unblanked, once each.
    var rows: std.ArrayList([board_mod.max_dim]MachineLetter) = .empty;
    defer rows.deinit(gpa);
    for (0..2 * dim) |l| {
        var row: [board_mod.max_dim]MachineLetter = @splat(0);
        for (0..dim) |i| {
            const ml = if (l < dim) b.getLetter(l, i) else b.getLetter(i, l - dim);
            row[i] = tiles.unblank(ml);
        }
        try rows.append(gpa, row);
    }
    std.mem.sort([board_mod.max_dim]MachineLetter, rows.items, dim, struct {
        fn less(d: usize, a: [board_mod.max_dim]MachineLetter, c: [board_mod.max_dim]MachineLetter) bool {
            return std.mem.order(MachineLetter, a[0..d], c[0..d]) == .lt;
        }
    }.less);
    var unique: usize = 0;
    for (rows.items) |row| {
        if (unique > 0 and std.mem.eql(MachineLetter, rows.items[unique - 1][0..dim], row[0..dim])) continue;
        rows.items[unique] = row;
        unique += 1;
    }
    rows.shrinkRetainingCapacity(unique);

    // The longest stretch of empty squares not next to a tile bounds the
    // words played alone.
    var max_alone: usize = 0;
    for (rows.items) |*row| {
        var empty: isize = 0;
        for (row[0..dim]) |ml| {
            if (ml == 0) {
                empty += 1;
            } else {
                empty -= 1;
                if (empty > 0) max_alone = @max(max_alone, @as(usize, @intCast(empty)));
                empty = -1;
            }
        }
        if (empty > 0) max_alone = @max(max_alone, @as(usize, @intCast(empty)));
    }

    var word: [board_mod.max_dim]MachineLetter = undefined;
    try alone(gpa, full, full.dawgRoot(), &rack, max_alone, &word, 0, false, &words);
    var strip: [board_mod.max_dim]MachineLetter = undefined;
    for (rows.items) |*row| try throughRow(gpa, row[0..dim], full, &rack, &strip, &words);
    if (words.ends.items.len == 0) return null;

    // Sorted and without repeats.
    const order = try gpa.alloc(u32, words.ends.items.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, &words, struct {
        fn less(w: *const Words, a: u32, c: u32) bool {
            return std.mem.order(MachineLetter, w.get(a), w.get(c)) == .lt;
        }
    }.less);
    return try build(gpa, &words, order);
}

/// The words from the DAWG node `node` that the rack spells, up to `max_len`
/// letters (a blank standing for a letter the rack lacks).
fn alone(gpa: Allocator, k: Kwg, node: u32, rack: *[tiles.max_alphabet_size]i32, max_len: usize, word: *[board_mod.max_dim]MachineLetter, n: usize, accepts: bool, out: *Words) Allocator.Error!void {
    if (accepts) try out.add(gpa, word[0..n]);
    if (n == max_len or node == 0) return;
    var i = node;
    while (true) : (i += 1) {
        const ml: MachineLetter = k.tile(i);
        if (ml != separator) {
            const next = k.arcIndex(i);
            const acc = k.accepts(i);
            if (rack[ml] > 0) {
                rack[ml] -= 1;
                word[n] = ml;
                try alone(gpa, k, next, rack, max_len, word, n + 1, acc, out);
                rack[ml] += 1;
            } else if (rack[0] > 0) {
                rack[0] -= 1;
                word[n] = ml;
                try alone(gpa, k, next, rack, max_len, word, n + 1, acc, out);
                rack[0] += 1;
            }
        }
        if (k.isEnd(i)) break;
    }
}

const Row = struct {
    row: []const MachineLetter,
    k: Kwg,
    rack: *[tiles.max_alphabet_size]i32,
    strip: *[board_mod.max_dim]MachineLetter,
    out: *Words,
    gpa: Allocator,
};

/// The words through or beside the tiles of one row.
fn throughRow(gpa: Allocator, row: []const MachineLetter, k: Kwg, rack: *[tiles.max_alphabet_size]i32, strip: *[board_mod.max_dim]MachineLetter, out: *Words) Allocator.Error!void {
    const r: Row = .{ .row = row, .k = k, .rack = rack, .strip = strip, .out = out, .gpa = gpa };
    const dim = row.len;
    const root = k.gaddagRoot();
    var leftmost: usize = 0;
    var col: usize = 0;
    while (col < dim) : (col += 1) {
        if (row[col] == 0) continue;
        while (col + 1 < dim and row[col + 1] != 0) col += 1;
        const ml = row[col];
        var next: u32 = 0;
        var i = root;
        while (true) : (i += 1) {
            if (k.tile(i) == ml) {
                next = k.arcIndex(i);
                break;
            }
            if (k.isEnd(i)) break;
        }
        try goOn(&r, col, col, ml, next, false, col, col, leftmost, 0);
        leftmost = col + 2;
    }
}

fn goOn(r: *const Row, col: usize, anchor: usize, letter: MachineLetter, node: u32, accepts: bool, left_in: usize, right_in: usize, leftmost: usize, played: usize) Allocator.Error!void {
    const dim = r.row.len;
    var left = left_in;
    var right = right_in;
    r.strip[col] = if (r.row[col] != 0) r.row[col] else letter;
    if (col <= anchor) {
        left = col;
        if (accepts and played > 0) try r.out.add(r.gpa, r.strip[left .. right + 1]);
        if (node == 0) return;
        if (col > leftmost) try recursive(r, col - 1, anchor, node, left, right, leftmost, played);
        const no_left = col == 0 or r.row[col - 1] == 0;
        const sep = r.k.nextNodeIdx(node, separator);
        if (sep != 0 and no_left and anchor < dim - 1) try recursive(r, anchor + 1, anchor, sep, left, right, leftmost, played);
    } else {
        right = col;
        const no_right = col == dim - 1 or r.row[col + 1] == 0;
        if (accepts and no_right and played > 0) try r.out.add(r.gpa, r.strip[left .. right + 1]);
        if (node != 0 and col < dim - 1) try recursive(r, col + 1, anchor, node, left, right, leftmost, played);
    }
}

fn recursive(r: *const Row, col: usize, anchor: usize, node: u32, left: usize, right: usize, leftmost: usize, played: usize) Allocator.Error!void {
    const k = r.k;
    const on = r.row[col];
    if (on != 0) {
        var next: u32 = 0;
        var acc = false;
        var i = node;
        while (true) : (i += 1) {
            if (k.tile(i) == on) {
                next = k.arcIndex(i);
                acc = k.accepts(i);
                break;
            }
            if (k.isEnd(i)) break;
        }
        return goOn(r, col, anchor, on, next, acc, left, right, leftmost, played);
    }
    var i = node;
    while (true) : (i += 1) {
        const ml: MachineLetter = k.tile(i);
        if (ml != separator) {
            const next = k.arcIndex(i);
            const acc = k.accepts(i);
            if (r.rack[ml] > 0) {
                r.rack[ml] -= 1;
                try goOn(r, col, anchor, ml, next, acc, left, right, leftmost, played + 1);
                r.rack[ml] += 1;
            } else if (r.rack[0] > 0) {
                r.rack[0] -= 1;
                try goOn(r, col, anchor, ml, next, acc, left, right, leftmost, played + 1);
                r.rack[0] += 1;
            }
        }
        if (k.isEnd(i)) break;
    }
}

const TrieNode = struct {
    first_child: u32 = 0,
    next_sibling: u32 = 0,
    children: u32 = 0,
    tile: MachineLetter = 0,
    accepts: bool = false,
};

const Trie = struct {
    nodes: std.ArrayList(TrieNode) = .empty,

    fn insert(t: *Trie, gpa: Allocator, word: []const MachineLetter) Allocator.Error!void {
        var node: u32 = 0;
        for (word) |ml| node = try t.child(gpa, node, ml);
        t.nodes.items[node].accepts = true;
    }

    fn child(t: *Trie, gpa: Allocator, parent: u32, ml: MachineLetter) Allocator.Error!u32 {
        var prev: u32 = 0;
        var cur = t.nodes.items[parent].first_child;
        while (cur != 0 and t.nodes.items[cur].tile < ml) {
            prev = cur;
            cur = t.nodes.items[cur].next_sibling;
        }
        if (cur != 0 and t.nodes.items[cur].tile == ml) return cur;
        const idx: u32 = @intCast(t.nodes.items.len);
        try t.nodes.append(gpa, .{ .tile = ml, .next_sibling = cur });
        if (prev == 0) t.nodes.items[parent].first_child = idx else t.nodes.items[prev].next_sibling = idx;
        t.nodes.items[parent].children += 1;
        return idx;
    }
};

/// The GADDAG of the words (in `order`, sorted; repeats are skipped).
fn build(gpa: Allocator, words: *const Words, order: []const u32) Allocator.Error!Pruned {
    var t: Trie = .{};
    defer t.nodes.deinit(gpa);
    try t.nodes.ensureTotalCapacity(gpa, 1 + 2 * words.letters.items.len);
    t.nodes.appendAssumeCapacity(.{});
    var buf: [board_mod.max_dim + 1]MachineLetter = undefined;
    var prev: ?[]const MachineLetter = null;
    for (order) |wi| {
        const word = words.get(wi);
        if (prev) |p| if (std.mem.eql(MachineLetter, p, word)) continue;
        prev = word;
        const n = word.len;
        if (n == 0) continue;
        // Reversed, for plays that start at the last letter and go left.
        for (word, 0..) |ml, i| buf[n - 1 - i] = ml;
        try t.insert(gpa, buf[0..n]);
        // word[k-1] .. word[0], separator, word[k] .. word[n-1], for each
        // pivot k.
        var sep: usize = n - 1;
        while (sep >= 1) : (sep -= 1) {
            for (0..sep) |i| buf[i] = word[sep - 1 - i];
            buf[sep] = separator;
            for (sep..n) |i| buf[sep + 1 + (i - sep)] = word[i];
            try t.insert(gpa, buf[0 .. n + 1]);
        }
    }

    // Breadth first; nodes 0 and 1 are the (empty) DAWG and GADDAG roots.
    var nodes: std.ArrayList(u32) = .empty;
    errdefer nodes.deinit(gpa);
    try nodes.ensureTotalCapacity(gpa, t.nodes.items.len + 1);
    nodes.appendSliceAssumeCapacity(&.{ kwg_mod.is_end_bit, kwg_mod.is_end_bit });
    const root = t.nodes.items[0];
    if (root.children > 0) {
        const QueueEntry = struct { node: u32, base: u32 };
        var queue: std.ArrayList(QueueEntry) = .empty;
        defer queue.deinit(gpa);
        const root_base: u32 = @intCast(nodes.items.len);
        nodes.items[1] = root_base | kwg_mod.is_end_bit;
        try nodes.appendNTimes(gpa, 0, root.children);
        try queue.append(gpa, .{ .node = 0, .base = root_base });
        var head: usize = 0;
        while (head < queue.items.len) : (head += 1) {
            const entry = queue.items[head];
            var out = entry.base;
            var ci = t.nodes.items[entry.node].first_child;
            while (ci != 0) : (ci = t.nodes.items[ci].next_sibling) {
                const c = t.nodes.items[ci];
                var node: u32 = @as(u32, c.tile) << kwg_mod.tile_shift;
                if (c.accepts) node |= kwg_mod.accepts_bit;
                if (c.next_sibling == 0) node |= kwg_mod.is_end_bit;
                if (c.children > 0) {
                    const base: u32 = @intCast(nodes.items.len);
                    node |= base;
                    try nodes.appendNTimes(gpa, 0, c.children);
                    try queue.append(gpa, .{ .node = ci, .base = base });
                }
                nodes.items[out] = node;
                out += 1;
            }
        }
    }
    const owned = try nodes.toOwnedSlice(gpa);
    errdefer gpa.free(owned);
    const k: Kwg = .{ .nodes = owned };
    return .{ .nodes = owned, .sibling_tiles = try k.buildSiblingTiles(gpa) };
}

const testing = std.testing;

test "a pruned graph finds the same moves" {
    const testdata = @import("testdata.zig");
    const movegen = @import("movegen.zig");
    const move_mod = @import("move.zig");
    var store = testdata.store();
    defer store.deinit();
    var p = try testdata.position(&store, "9A1PIXY/9S1L3/2ToWNLETS1O3/9U1DA1R/3GERANIAL1U1I/9g2T1C/8WE2OBI/6EMU4ON/6AID3GO1/5HUN4ET1/4ZA1T4ME1/1Q1FAKEY3JOES/FIVE1E5IT1C/5SPORRAN2A/6ORE2N2D BGIV/DEHILOR 384/389 0 lex NWL20;");
    var pruned = (try generate(testing.allocator, &p.board, &p.racks[0], &p.racks[1], p.dist.size, p.lexicon.kwg)).?;
    defer pruned.deinit(testing.allocator);
    // (12,777 nodes of 1,205,122.)
    try testing.expect(pruned.nodes.len < p.lexicon.kwg.nodes.len / 50);
    for (0..2) |player| {
        var lists: [2]std.ArrayList(move_mod.Move) = .{ .empty, .empty };
        defer for (&lists) |*l| l.deinit(testing.allocator);
        for ([_]Kwg{ p.lexicon.kwg, pruned.kwg() }, 0..) |k, i| {
            var gen = movegen.Generator.init(testing.allocator, &p.board, k, p.dist);
            defer gen.deinit();
            var rack = p.racks[player];
            try gen.genAll(&rack, false);
            try lists[i].appendSlice(testing.allocator, gen.plays.items);
        }
        try testing.expectEqual(lists[0].items.len, lists[1].items.len);
        for (lists[0].items, lists[1].items) |a, c| {
            try testing.expect(a.row == c.row and a.col == c.col and a.vertical == c.vertical and a.score == c.score and
                std.mem.eql(MachineLetter, a.tileSlice(), c.tileSlice()));
        }
    }
}
