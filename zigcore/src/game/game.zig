//! A game in progress: board, bag, racks, scores and turns, played the way
//! Macondo's game package plays them, so that the same seed deals the same
//! tiles and the same bots play the same game in both engines. Ports of
//! word-golib's Bag and the turn logic of game.go and pairedbag.go (only
//! what autoplay and the sims need: no history or challenges yet).

const std = @import("std");
const tiles = @import("../data/tiles.zig");
const board_mod = @import("../board/board.zig");
const cross_set = @import("../board/cross_set.zig");
const chacha8 = @import("../gocompat/chacha8.zig");
const equity = @import("../equity/equity.zig");
const movegen = @import("../movegen/movegen.zig");
const position = @import("../board/position.zig");
const move_mod = @import("../board/move.zig");
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;
const Move = move_mod.Move;

pub const rack_tile_limit = movegen.rack_tile_limit;
pub const default_max_scoreless_turns = 6;

/// The most tiles any distribution has.
const max_bag = 256;

/// The tiles not on a rack or the board. Port of tilemapping.Bag: the
/// order of `tiles` is what makes draws reproducible.
pub const Bag = struct {
    tiles: [max_bag]MachineLetter = undefined,
    len: usize = 0,
    /// Paired mode: draws take the last tiles as they are (see Game).
    fixed_order: bool = false,

    /// The full bag, in machine-letter order (NewBag), unshuffled.
    pub fn init(ld: *const tiles.LetterDistribution) Bag {
        var b: Bag = .{};
        for (ld.counts[0..ld.size], 0..) |n, ml| {
            for (0..n) |_| {
                b.tiles[b.len] = @intCast(ml);
                b.len += 1;
            }
        }
        return b;
    }

    pub fn shuffle(self: *Bag, rng: *chacha8.Rng) void {
        rng.shuffle(MachineLetter, self.tiles[0..self.len]);
    }

    /// Draws `out.len` tiles: each chosen at random from those left and
    /// moved to the end, then the end taken off. Port of Bag.Draw.
    pub fn draw(self: *Bag, rng: *chacha8.Rng, out: []MachineLetter) void {
        const n = out.len;
        std.debug.assert(n <= self.len);
        const k = self.len - n;
        if (!self.fixed_order) {
            var i = self.len;
            while (i > k) : (i -= 1) {
                const xi = rng.intn(i);
                std.mem.swap(MachineLetter, &self.tiles[i - 1], &self.tiles[xi]);
            }
        }
        @memcpy(out, self.tiles[k..self.len]);
        self.len = k;
    }

    /// Puts tiles back at the end (Bag.PutBack without its reshuffle, which
    /// the caller does in fixed order; paired games reinsert instead, see
    /// Game).
    pub fn putBack(self: *Bag, letters: []const MachineLetter) void {
        @memcpy(self.tiles[self.len..][0..letters.len], letters);
        self.len += letters.len;
    }

    /// The tiles of `ld` that are not on `b` or in `racks`, in machine
    /// letter order. Designated blanks on the board count as blanks.
    pub fn unseen(ld: *const tiles.LetterDistribution, b: *const board_mod.Board, racks: []const Rack) error{TooManyTiles}!Bag {
        var counts: [tiles.max_alphabet_size]i32 = @splat(0);
        for (ld.counts[0..ld.size], 0..) |n, ml| counts[ml] = n;
        for (0..b.dim) |r| {
            for (0..b.dim) |c| {
                const ml = b.getLetter(r, c);
                if (ml != 0) counts[if (tiles.isBlanked(ml)) 0 else ml] -= 1;
            }
        }
        for (racks) |rack| {
            for (rack.counts, 0..) |n, ml| counts[ml] -= n;
        }
        var bag: Bag = .{};
        for (counts[0..ld.size], 0..) |n, ml| {
            if (n < 0) return error.TooManyTiles;
            for (0..@intCast(n)) |_| {
                bag.tiles[bag.len] = @intCast(ml);
                bag.len += 1;
            }
        }
        return bag;
    }
};

pub const Game = struct {
    pos: position.Position,
    bag: Bag,
    rng: chacha8.Rng,
    on_turn: u1 = 0,
    turn: u32 = 0,
    scoreless_turns: u32 = 0,
    max_scoreless_turns: u32 = default_max_scoreless_turns,
    playing: bool = true,
    /// Paired-bag mode (MAGPIE's game pairs): the bag is shuffled once and
    /// each seat draws from its own end, so that a game replayed with the
    /// seats swapped deals each seat the same tiles. Port of pairedbag.go.
    paired: bool = false,
    exchange_limit: i32,
    /// Remembers cross-sets by the words beside a square, for the
    /// incremental updates after each play (a sim's or autoplay's own;
    /// copies of a game share it).
    cross_cache: ?*cross_set.Cache = null,

    /// A game on an empty board of `pos`'s layout and lexicon, seeded with
    /// `seed`, with the racks dealt (StartGame after SeedBag). `pos`
    /// supplies the board layout and the data; its tiles and racks are
    /// replaced.
    pub fn start(pos: *const position.Position, seed: [32]u8, paired: bool) Game {
        var g: Game = .{
            .pos = pos.*,
            .bag = .init(pos.dist),
            .rng = .init(seed),
            .paired = paired,
            .exchange_limit = movegen.exchangeLimit(pos.lexiconName()),
        };
        g.pos.board.clear();
        g.pos.board.countTiles();
        g.pos.scores = .{ 0, 0 };
        g.bag.shuffle(&g.rng);
        g.bag.fixed_order = paired;
        for (0..2) |p| {
            var drawn: [rack_tile_limit]MachineLetter = undefined;
            g.drawFor(@intCast(p), &drawn);
            g.pos.racks[p] = .fromLetters(&drawn);
        }
        g.refreshBoard();
        return g;
    }

    /// A game going on from `pos`, player 0 to move, with the unseen tiles
    /// in the bag in machine-letter order (whatever order a Go game built
    /// from the same CGP has them in is incidental: a test sorts it). The
    /// bag draws with an RNG seeded with `seed` (SeedBag).
    pub fn fromPosition(pos: *const position.Position, seed: [32]u8) error{TooManyTiles}!Game {
        var g: Game = .{
            .pos = pos.*,
            .bag = try .unseen(pos.dist, &pos.board, &pos.racks),
            .rng = .init(seed),
            .scoreless_turns = pos.scoreless_turns,
            .max_scoreless_turns = pos.max_scoreless_turns,
            .exchange_limit = movegen.exchangeLimit(pos.lexiconName()),
        };
        g.playing = !(g.bag.len == 0 and (pos.racks[0].num_tiles == 0 or pos.racks[1].num_tiles == 0));
        return g;
    }

    /// The end of the bag seat `p` draws from in paired mode: seat 0 moves
    /// first and takes from the back, seat 1 from the front.
    fn drawsFromFront(p: u1) bool {
        return p == 1;
    }

    /// Draws `out.len` tiles for seat `p`.
    fn drawFor(self: *Game, p: u1, out: []MachineLetter) void {
        if (self.paired and out.len > 0 and drawsFromFront(p)) {
            // Rotate this seat's tiles round to the back, where draw takes
            // them; the rest keep their order.
            rotateFrontToBack(self.bag.tiles[0..self.bag.len], out.len);
        }
        self.bag.draw(&self.rng, out);
    }

    /// Puts exchanged tiles back. Paired mode reinserts each at a random
    /// spot, moving the tile there to the seat's own end; otherwise they go
    /// on the end (no shuffle: the bag isn't in fixed order).
    fn putBackFor(self: *Game, p: u1, letters: []const MachineLetter) void {
        if (self.paired) {
            for (letters) |ml| self.reinsert(p, ml);
            return;
        }
        self.putBack(letters);
    }

    /// Bag.PutBack: the tiles go on the end, and a bag in fixed order (a
    /// sim's) is reshuffled.
    pub fn putBack(self: *Game, letters: []const MachineLetter) void {
        if (letters.len == 0) return;
        self.bag.putBack(letters);
        if (self.bag.fixed_order) self.bag.shuffle(&self.rng);
    }

    /// Takes `letters` out of the bag (designated blanks as blanks), which
    /// is then rebuilt in machine-letter order and shuffled: Bag.RemoveTiles.
    pub fn removeTiles(self: *Game, letters: []const MachineLetter) error{NotInBag}!void {
        var counts: [tiles.max_alphabet_size]u32 = @splat(0);
        for (self.bag.tiles[0..self.bag.len]) |ml| counts[ml] += 1;
        for (letters) |ml| {
            const t = if (tiles.isBlanked(ml)) 0 else ml;
            if (counts[t] == 0) return error.NotInBag;
            counts[t] -= 1;
        }
        self.bag.len = 0;
        for (counts, 0..) |n, ml| {
            for (0..n) |_| {
                self.bag.tiles[self.bag.len] = @intCast(ml);
                self.bag.len += 1;
            }
        }
        self.bag.shuffle(&self.rng);
    }

    /// Gives seat `p` a random rack holding `known` (Game.SetRandomRack):
    /// its rack goes back in the bag, then without `known` it redraws
    /// seven, and with `known` those come out of the bag (which is rebuilt
    /// and shuffled) and the rest of the rack is drawn. Returns how many
    /// were drawn besides `known`, which are written to `extra`.
    pub fn setRandomRack(self: *Game, p: u1, known: []const MachineLetter, extra: *[rack_tile_limit]MachineLetter) error{NotInBag}!usize {
        var old: [rack_tile_limit * 4]MachineLetter = undefined;
        const n = self.pos.racks[p].tiles(&old);
        if (known.len == 0) {
            self.putBack(old[0..n]);
            var drawn: [rack_tile_limit]MachineLetter = undefined;
            const k = self.drawAtMost(&drawn);
            self.pos.racks[p] = .fromLetters(drawn[0..k]);
            return 0;
        }
        self.putBack(old[0..n]);
        self.removeTiles(known) catch |err| {
            self.removeTiles(old[0..n]) catch unreachable;
            return err;
        };
        const want = @max(n, rack_tile_limit) - known.len;
        var rack: [rack_tile_limit * 4]MachineLetter = undefined;
        @memcpy(rack[0..known.len], known);
        const k = self.drawAtMost(rack[known.len..][0..@min(want, rack_tile_limit)]);
        @memcpy(extra[0..k], rack[known.len..][0..k]);
        self.pos.racks[p] = .fromLetters(rack[0 .. known.len + k]);
        return k;
    }

    /// Draws at most `out.len` tiles for the player on turn's seat (not
    /// paired), returning how many.
    pub fn drawAtMost(self: *Game, out: []MachineLetter) usize {
        const n = @min(out.len, self.bag.len);
        self.bag.draw(&self.rng, out[0..n]);
        return n;
    }

    /// Port of reinsert: the tile lands in one of the len+1 slots.
    fn reinsert(self: *Game, p: u1, ml: MachineLetter) void {
        const t = &self.bag.tiles;
        const n = self.bag.len;
        const slot = self.rng.intn(n + 1);
        if (!drawsFromFront(p)) {
            if (slot == n) {
                t[n] = ml;
            } else {
                t[n] = t[slot];
                t[slot] = ml;
            }
        } else if (slot == 0) {
            std.mem.copyBackwards(MachineLetter, t[1 .. n + 1], t[0..n]);
            t[0] = ml;
        } else {
            const displaced = t[slot - 1];
            std.mem.copyBackwards(MachineLetter, t[1 .. n + 1], t[0..n]);
            t[0] = displaced;
            t[slot] = ml;
        }
        self.bag.len = n + 1;
    }

    pub fn tilesInBag(self: *const Game) i32 {
        return @intCast(self.bag.len);
    }

    /// The equity context for the player on turn.
    pub fn equityContext(self: *const Game) equity.Context {
        const p = &self.pos;
        return .{
            .ld = p.dist,
            .klv = p.leaves,
            .peg = p.peg,
            .board_dim = p.board.dim,
            .board_empty = p.board.tiles_played == 0,
            .tiles_in_bag = self.tilesInBag(),
            .opp_rack_score = p.racks[1 - self.on_turn].scoreOn(p.dist),
        };
    }

    /// Whether the player on turn may exchange, and how many tiles. Port of
    /// GenBestStaticTurn's rule (the opponent's rack is known).
    pub fn exchangeRules(self: *const Game) struct { allowed: bool, max: u8 } {
        const unseen = @as(i32, self.pos.racks[1 - self.on_turn].num_tiles) + self.tilesInBag();
        return .{
            .allowed = unseen - rack_tile_limit >= self.exchange_limit,
            .max = movegen.maxCanExchange(unseen - rack_tile_limit, self.exchange_limit),
        };
    }

    /// Plays `m` for the player on turn: places it and scores it, draws,
    /// and ends the game on an emptied rack or too many scoreless turns.
    /// Port of playMove without history (challenge rule VOID).
    pub fn play(self: *Game, m: *const Move) void {
        self.playMove(m, true);
    }

    /// play without bringing cross-sets and anchors up to date, for a
    /// position nothing will be generated from (the last ply of a sim's
    /// rollout: PlayMoveNoCrossSet).
    pub fn playLast(self: *Game, m: *const Move) void {
        self.playMove(m, false);
    }

    fn playMove(self: *Game, m: *const Move, comptime update_board: bool) void {
        const p = self.on_turn;
        const rack = &self.pos.racks[p];
        switch (m.kind) {
            .play => {
                self.placeTiles(m, update_board);
                self.scoreless_turns = 0;
                self.pos.scores[p] += m.score;
                for (m.tileSlice()) |ml| {
                    if (ml != 0) rack.take(if (tiles.isBlanked(ml)) 0 else ml);
                }
                const n = @min(m.tiles_played, self.bag.len);
                var drawn: [rack_tile_limit]MachineLetter = undefined;
                self.drawFor(p, drawn[0..n]);
                for (drawn[0..n]) |ml| rack.add(ml);
                if (rack.num_tiles == 0) {
                    self.playing = false;
                    self.pos.scores[p] += 2 * self.pos.racks[1 - p].scoreOn(self.pos.dist);
                }
            },
            .pass => {
                self.scoreless_turns += 1;
            },
            .exchange => {
                var drawn: [rack_tile_limit]MachineLetter = undefined;
                const n = m.len;
                self.drawFor(p, drawn[0..n]);
                self.putBackFor(p, m.tileSlice());
                for (m.tileSlice()) |ml| rack.take(ml);
                for (drawn[0..n]) |ml| rack.add(ml);
                self.scoreless_turns += 1;
            },
        }
        if (self.playing and self.scoreless_turns == self.max_scoreless_turns) {
            // Each player loses what is left on their rack.
            self.playing = false;
            for (0..2) |q| self.pos.scores[q] -= self.pos.racks[q].scoreOn(self.pos.dist);
        } else if (self.playing) {
            self.on_turn = 1 - p;
        }
        self.turn += 1;
    }

    /// Puts a play's tiles on the board and brings cross-sets and anchors
    /// up to date.
    fn placeTiles(self: *Game, m: *const Move, comptime update_board: bool) void {
        const b = &self.pos.board;
        const was_empty = b.tiles_played == 0;
        for (m.tileSlice(), 0..) |ml, i| {
            if (ml == 0) continue;
            const r = m.row + if (m.vertical) i else 0;
            const c = m.col + if (m.vertical) 0 else i;
            b.setLetter(r, c, ml);
        }
        b.tiles_played += m.tiles_played;
        if (update_board) cross_set.updateForPlay(b, m, self.pos.lexicon.kwg, self.pos.dist, was_empty, self.cross_cache);
    }

    fn refreshBoard(self: *Game) void {
        cross_set.generateAll(&self.pos.board, self.pos.lexicon.kwg, self.pos.dist);
        self.pos.board.updateAllAnchors();
    }
};

/// The seed of game (or game pair) `index` of a run with master seed
/// `master`: SHA-256 of both, big-endian. Port of automatic.DeriveSeed.
pub fn deriveSeed(master: u64, index: u64) [32]u8 {
    var buf: [16]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], master, .big);
    std.mem.writeInt(u64, buf[8..16], index, .big);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&buf, &out, .{});
    return out;
}

/// Plays the best static move for the player on turn (HastyBot) and
/// returns it. `gen` must be set up for this game's board and lexicon.
pub fn playStaticTurn(g: *Game, gen: *movegen.Generator) std.mem.Allocator.Error!movegen.Generator.Best {
    const ex = g.exchangeRules();
    gen.max_can_exchange = ex.max;
    const ctx = g.equityContext();
    var rack = g.pos.racks[g.on_turn];
    const best = try gen.genBest(&rack, ex.allowed, &ctx);
    g.play(&best.move);
    return best;
}

/// Moves the first n tiles to the back, keeping the rest in order.
fn rotateFrontToBack(t: []MachineLetter, n: usize) void {
    if (n == 0 or n >= t.len) return;
    std.mem.rotate(MachineLetter, t, n);
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

test "bag draws" {
    var ld: tiles.LetterDistribution = undefined;
    ld.size = 3;
    ld.counts = @splat(0);
    ld.counts[0] = 2;
    ld.counts[1] = 3;
    ld.counts[2] = 1;
    var b: Bag = .init(&ld);
    try testing.expectEqualSlices(MachineLetter, &.{ 0, 0, 1, 1, 1, 2 }, b.tiles[0..b.len]);
    var rng: chacha8.Rng = .init(@splat(7));
    b.shuffle(&rng);
    var out: [4]MachineLetter = undefined;
    b.draw(&rng, &out);
    try testing.expectEqual(2, b.len);
    // Every tile is drawn or left, once.
    var counts: [3]u8 = @splat(0);
    for (out) |ml| counts[ml] += 1;
    for (b.tiles[0..b.len]) |ml| counts[ml] += 1;
    try testing.expectEqual([3]u8{ 2, 3, 1 }, counts);
}

test "derived seeds are Go's" {
    // automatic.DeriveSeed: SHA-256 of the two big-endian u64s.
    try testing.expectEqualSlices(u8, &try hex("783825822a6f9e62da2190e828e4c9d2576e5977e3a0b3620b092dfb9e9996fa"), &deriveSeed(1, 0));
    try testing.expectEqualSlices(u8, &try hex("7212ebfeb3caa3cf6351e0a9870a00994ad72272ac0e8e22fa1056050c9e40a3"), &deriveSeed(42, 7));
}

fn hex(comptime s: []const u8) ![32]u8 {
    var out: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, s);
    return out;
}

test "rotate front to back" {
    var t = [_]MachineLetter{ 1, 2, 3, 4, 5 };
    rotateFrontToBack(&t, 2);
    try testing.expectEqualSlices(MachineLetter, &.{ 3, 4, 5, 1, 2 }, &t);
}

fn testGame(store: *@import("../data/data.zig").Store, seed: u8) !Game {
    const testdata = @import("../testing/testdata.zig");
    const pos = try testdata.position(store, "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;");
    return Game.start(&pos, @splat(seed), false);
}

test "six scoreless turns end the game, each losing their rack" {
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    var g = try testGame(&store, 3);
    try testing.expectEqual(86, g.bag.len);
    const racks = [2]i32{ g.pos.racks[0].scoreOn(g.pos.dist), g.pos.racks[1].scoreOn(g.pos.dist) };
    for (0..6) |i| {
        try testing.expect(g.playing);
        try testing.expectEqual(@as(u1, @intCast(i % 2)), g.on_turn);
        g.play(&Move.pass());
    }
    try testing.expect(!g.playing);
    try testing.expectEqual([2]i32{ -racks[0], -racks[1] }, g.pos.scores);
    try testing.expectEqual(6, g.turn);
}

test "an exchange keeps the bag and racks whole" {
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    var g = try testGame(&store, 9);
    var before: [tiles.max_alphabet_size]u32 = @splat(0);
    for (g.bag.tiles[0..g.bag.len]) |ml| before[ml] += 1;
    for (0..g.pos.dist.size) |ml| before[ml] += g.pos.racks[0].counts[ml];
    // Throw in the first three tiles of the rack.
    var buf: [7]MachineLetter = undefined;
    _ = g.pos.racks[0].tiles(&buf);
    var m: Move = .{ .kind = .exchange, .len = 3, .tiles_played = 3 };
    @memcpy(m.tiles[0..3], buf[0..3]);
    g.play(&m);
    try testing.expectEqual(86, g.bag.len);
    try testing.expectEqual(7, g.pos.racks[0].num_tiles);
    var after: [tiles.max_alphabet_size]u32 = @splat(0);
    for (g.bag.tiles[0..g.bag.len]) |ml| after[ml] += 1;
    for (0..g.pos.dist.size) |ml| after[ml] += g.pos.racks[0].counts[ml];
    try testing.expectEqual(before, after);
    try testing.expectEqual(1, g.scoreless_turns);
    try testing.expectEqual(1, g.on_turn);
}

/// Whether two boards' squares, cross-sets, cross-scores, extension sets
/// and anchors are the same.
/// Whether two boards agree on everything move generation reads: the
/// tiles, cross-sets, cross-scores and anchors, and the extension sets
/// where they are read (empty squares, and the last tile of a word along
/// the extension sets' lane). Elsewhere an incremental update can leave
/// the sets a tile had when it ended a word that has since grown.
fn sameBoards(a: *const board_mod.Board, b: *const board_mod.Board) bool {
    const n = a.dim * a.dim;
    if (!std.mem.eql(MachineLetter, a.squares[0..n], b.squares[0..n])) return false;
    for (0..2) |d| {
        if (!std.mem.eql(u64, a.cross_sets[d][0..n], b.cross_sets[d][0..n])) return false;
        if (!std.mem.eql(i32, a.cross_scores[d][0..n], b.cross_scores[d][0..n])) return false;
        if (!std.mem.eql(bool, a.anchors[d][0..n], b.anchors[d][0..n])) return false;
        for (0..n) |i| {
            if (a.squares[i] != 0) {
                // Slot 0's sets serve plays down a column.
                const r = i / a.dim;
                const c = i % a.dim;
                const nr = if (d == 0) r + 1 else r;
                const nc = if (d == 0) c else c + 1;
                if (nr < a.dim and nc < a.dim and a.squares[nr * a.dim + nc] != 0) continue;
            }
            if (a.left_ext[d][i] != b.left_ext[d][i] or a.right_ext[d][i] != b.right_ext[d][i]) return false;
        }
    }
    return true;
}

test "incremental cross-set updates match regenerating everything" {
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const cgps = [_][]const u8{
        "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;",
        "21/21/21/21/21/21/21/21/21/21/21/21/21/21/21/21/21/21/21/21/21 / 0/0 0 lex CSW24; ld english_super; bdn SuperCrosswordGame;",
        "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex FRA20; ld french;",
    };
    var moves: usize = 0;
    for (cgps) |text| {
        var pos = testdata.position(&store, text) catch |err| switch (err) {
            error.SkipZigTest => continue,
            else => return err,
        };
        const gen = try testing.allocator.create(movegen.Generator);
        defer testing.allocator.destroy(gen);
        gen.* = .init(testing.allocator, &pos.board, pos.lexicon.kwg, pos.dist);
        defer gen.deinit();
        var cache: cross_set.Cache = try .init(testing.allocator, 6);
        defer cache.deinit(testing.allocator);
        for (0..20) |seed| {
            var g: Game = .start(&pos, @splat(@intCast(seed)), seed % 2 == 1);
            // Half the games with a (small, so often overwritten) cache.
            if (seed % 4 >= 2) g.cross_cache = &cache;
            gen.board = &g.pos.board;
            while (g.playing) {
                _ = try playStaticTurn(&g, gen);
                var full = g.pos.board;
                cross_set.generateAll(&full, pos.lexicon.kwg, pos.dist);
                full.updateAllAnchors();
                try testing.expect(sameBoards(&g.pos.board, &full));
                moves += 1;
            }
        }
    }
    try testing.expect(moves > 1000);
}

test "the last move in a CGP takes back to the position before it" {
    // Static games: after every move, the position as a CGP with the move
    // in lm, read back and taken back, is the position before the move.
    const cgp = @import("../board/cgp.zig");
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const base = try testdata.position(&store, "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL23;");
    var checked: usize = 0;
    var through: usize = 0;
    for (0..20) |i| {
        var g = Game.start(&base, deriveSeed(7, i), false);
        var gen = movegen.Generator.init(testing.allocator, &g.pos.board, g.pos.lexicon.kwg, g.pos.dist);
        defer gen.deinit();
        while (g.playing) {
            const before = g.pos;
            const mover = g.on_turn;
            const best = try playStaticTurn(&g, &gen);
            if (!g.playing) break;
            const next = g.on_turn;
            var buf: [512]u8 = undefined;
            var w: std.Io.Writer = .fixed(&buf);
            try cgp.writeFields(&g.pos.board, g.pos.dist, .{ g.pos.racks[next], .{} }, .{ g.pos.scores[next], g.pos.scores[mover] }, g.scoreless_turns, &w);
            try w.writeAll(" lex NWL23; lm ");
            try move_mod.formatLastMove(&best.move, &g.pos.board, g.pos.dist, checked % 2 == 0, &w);
            // Every other one without its score, which is worked out.
            if (checked % 3 != 0) try w.print(" {d}", .{best.move.score});
            try w.writeByte(';');
            const p = try testdata.position(&store, w.buffered());
            try testing.expectEqual(best.move.score, p.last_move.?.score);
            const back = p.beforeLastMove().?;
            try testing.expectEqualSlices(MachineLetter, &before.board.squares, &back.board.squares);
            try testing.expectEqual([2]i32{ before.scores[mover], before.scores[next] }, back.scores);
            try testing.expectEqual(before.racks[next], back.racks[1]);
            if (std.mem.indexOfScalar(u8, w.buffered(), '(') != null) through += 1;
            checked += 1;
        }
    }
    try testing.expect(checked > 300 and through > 100);
}

test "taking a move back regenerates the board from scratch" {
    // The board of the position before a CGP's last move must be the board
    // that position's own CGP gives, extension sets and all: a square that
    // held a tile of the move can't keep the sets it had then.
    const cgp = @import("../board/cgp.zig");
    const testdata = @import("../testing/testdata.zig");
    var store = testdata.store();
    defer store.deinit();
    const base = try testdata.position(&store, "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;");
    var checked: usize = 0;
    for (0..10) |i| {
        var g = Game.start(&base, deriveSeed(11, i), false);
        var gen = movegen.Generator.init(testing.allocator, &g.pos.board, g.pos.lexicon.kwg, g.pos.dist);
        defer gen.deinit();
        while (g.playing) {
            const mover = g.on_turn;
            var before_buf: [512]u8 = undefined;
            var bw: std.Io.Writer = .fixed(&before_buf);
            try cgp.writeFields(&g.pos.board, g.pos.dist, .{ .{}, g.pos.racks[1 - mover] }, .{ g.pos.scores[mover], g.pos.scores[1 - mover] }, 0, &bw);
            try bw.writeAll(" lex NWL20;");
            const best = try playStaticTurn(&g, &gen);
            if (!g.playing) break;
            if (best.move.kind != .play) continue;
            var buf: [512]u8 = undefined;
            var w: std.Io.Writer = .fixed(&buf);
            try cgp.writeFields(&g.pos.board, g.pos.dist, .{ g.pos.racks[1 - mover], .{} }, .{ g.pos.scores[1 - mover], g.pos.scores[mover] }, g.scoreless_turns, &w);
            try w.writeAll(" lex NWL20; lm ");
            try move_mod.formatLastMove(&best.move, &g.pos.board, g.pos.dist, false, &w);
            try w.writeByte(';');
            const p = try testdata.position(&store, w.buffered());
            const back = p.beforeLastMove().?;
            const fresh = try testdata.position(&store, bw.buffered());
            const n = fresh.board.dim * fresh.board.dim;
            for (0..2) |d| {
                try testing.expectEqualSlices(u64, fresh.board.left_ext[d][0..n], back.board.left_ext[d][0..n]);
                try testing.expectEqualSlices(u64, fresh.board.right_ext[d][0..n], back.board.right_ext[d][0..n]);
            }
            try testing.expect(sameBoards(&fresh.board, &back.board));
            checked += 1;
        }
    }
    try testing.expect(checked > 100);
}
