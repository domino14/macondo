//! A game in progress: board, bag, racks, scores and turns, played the way
//! Macondo's game package plays them, so that the same seed deals the same
//! tiles and the same bots play the same game in both engines. Ports of
//! word-golib's Bag and the turn logic of game.go and pairedbag.go (only
//! what autoplay needs: no history, challenges or backups yet).

const std = @import("std");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const cross_set = @import("cross_set.zig");
const chacha8 = @import("chacha8.zig");
const equity = @import("equity.zig");
const movegen = @import("movegen.zig");
const position = @import("position.zig");
const move_mod = @import("move.zig");
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

    /// Puts tiles back at the end (Bag.PutBack; in fixed order the bag is
    /// reshuffled, which paired games do their own way, see Game).
    pub fn putBack(self: *Bag, letters: []const MachineLetter) void {
        @memcpy(self.tiles[self.len..][0..letters.len], letters);
        self.len += letters.len;
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
        if (!self.paired) return self.bag.putBack(letters);
        for (letters) |ml| self.reinsert(p, ml);
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
        const p = self.on_turn;
        const rack = &self.pos.racks[p];
        switch (m.kind) {
            .play => {
                self.placeTiles(m);
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
    fn placeTiles(self: *Game, m: *const Move) void {
        const b = &self.pos.board;
        for (m.tileSlice(), 0..) |ml, i| {
            if (ml == 0) continue;
            const r = m.row + if (m.vertical) i else 0;
            const c = m.col + if (m.vertical) 0 else i;
            b.setLetter(r, c, ml);
        }
        b.tiles_played += m.tiles_played;
        self.refreshBoard();
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

fn testGame(store: *@import("data.zig").Store, seed: u8) !Game {
    const testdata = @import("testdata.zig");
    const pos = try testdata.position(store, "15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 / 0/0 0 lex NWL20;");
    return Game.start(&pos, @splat(seed), false);
}

test "six scoreless turns end the game, each losing their rack" {
    const testdata = @import("testdata.zig");
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
    const testdata = @import("testdata.zig");
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
