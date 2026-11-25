//! Word-map generation: the plays of an anchor the shadow pass recorded per
//! (playthrough blocks, tiles to play), from the word map's word lists
//! instead of a walk of the GADDAG. Port of Macondo's movegen/wordmap_gen.go
//! (MAGPIE's wordmap_gen, record_wmp_plays_for_word and record_wmp_play).
//!
//! For each subrack of the anchor's size that may have plays, each of its
//! words is fitted at every start column of the anchor at once with
//! per-letter column masks, the blanks are placed every way they can be,
//! and each play is scored and recorded.

const std = @import("std");
const tiles = @import("tiles.zig");
const board_mod = @import("board.zig");
const movegen = @import("movegen.zig");
const shadow = @import("shadow.zig");
const wmp = @import("wmp.zig");
const wmp_movegen = @import("wmp_movegen.zig");
const Generator = movegen.Generator;
const MachineLetter = tiles.MachineLetter;
const Rack = tiles.Rack;

pub fn gen(g: *Generator, rack: *Rack, anchor: *const shadow.Anchor) void {
    const w = &g.wmp_gen;
    const tiles_to_play: usize = anchor.tiles_to_play;
    const word_length: usize = anchor.word_length;
    const leftmost: usize = anchor.leftmost_start_col;
    const rightmost: usize = anchor.rightmost_start_col;

    setPlaythroughFromRow(g, anchor);
    w.playthroughSubracksInit(tiles_to_play, word_length);
    // Every play of the anchor plays this many tiles.
    g.tiles_played = @intCast(tiles_to_play);

    // A subrack's plays all leave the same tiles, so the anchor's score
    // bound (tightened by its score bases) plus the subrack's leave bounds
    // all of them: a subrack whose bound is below the best play found is
    // passed over before its words are looked up.
    const prune_by_leave = g.leave_map.initialized and !g.ctx.board_empty;
    const score_bound = anchor.highest_score;
    const bases = anchor.score_bases;

    const prune_now = prune_by_leave and g.have_best;
    const cutoff: f64 = if (prune_now) g.best_eq else 0;
    var candidates = w.candidateSubracks(prune_now, score_bound, &bases, cutoff);

    const start_cols: u32 = (@as(u32, 2) << @intCast(rightmost)) - (@as(u32, 1) << @intCast(leftmost));
    var letter_cols_loaded = false;
    var letter_cols: [64]u32 = undefined;

    while (candidates != 0) : (candidates &= candidates - 1) {
        const si = @ctz(candidates);
        if (prune_by_leave and g.have_best) {
            const bound = wmp_movegen.MoveGen.subrackScoreBound(score_bound, &bases, w.tileScoreSum(si));
            if (@as(f64, @floatFromInt(bound)) + w.leaveValue(si) < g.best_eq) continue;
        }
        if (!w.subrackWords(si)) continue;
        if (!letter_cols_loaded) {
            loadLetterCols(g, &letter_cols, leftmost, rightmost + word_length - 1);
            letter_cols_loaded = true;
        }
        const words = w.words;
        var off: usize = 0;
        while (off + word_length <= words.len) : (off += word_length) {
            const word = words[off..][0..word_length];
            // Every start column at once: the word fits at a column when each
            // of its letters can go on its square from there.
            var fit = start_cols;
            for (word, 0..) |letter, i| fit &= letter_cols[letter & 63] >> @intCast(i);
            while (fit != 0) : (fit &= fit - 1) {
                const start: usize = @ctz(fit);
                markWord(g, word, start);
                recordPlaysForWord(g, rack, si, w.nonplaythroughSubrack(si), start, word_length, 0, 0);
            }
        }
    }
}

/// For each letter, the columns from `first` to `last` where a word may
/// have it: a tile that is the letter, or an empty square whose cross-set
/// allows it. Most squares allow every letter, so each letter starts with
/// every such column and loses those that don't allow it; a column that
/// allows fewer than half (a tile's, usually) is instead given to the
/// letters it allows. Port of wordmapLoadLetterCols.
fn loadLetterCols(g: *Generator, letter_cols: *[64]u32, first: usize, last: usize) void {
    const num_letters = g.ld.size;
    const universe: u64 = (@as(u64, 1) << @intCast(num_letters)) - 2;
    var col_letters: [board_mod.max_dim]u64 = undefined;
    var most: u32 = 0;
    var few: u32 = 0;
    for (first..last + 1) |col| {
        const sq = g.row[col];
        var letters: u64 = sq.cross_set;
        if (sq.letter != 0) letters = @as(u64, 1) << @intCast(tiles.unblank(sq.letter) & 63);
        letters &= universe;
        if (2 * @popCount(letters) >= num_letters) {
            most |= @as(u32, 1) << @intCast(col);
            col_letters[col] = universe & ~letters;
        } else {
            few |= @as(u32, 1) << @intCast(col);
            col_letters[col] = letters;
        }
    }
    @memset(letter_cols[0..num_letters], most);
    var cols = most;
    while (cols != 0) : (cols &= cols - 1) {
        const col = @ctz(cols);
        var letters = col_letters[col];
        while (letters != 0) : (letters &= letters - 1) letter_cols[@ctz(letters)] &= ~(@as(u32, 1) << @intCast(col));
    }
    cols = few;
    while (cols != 0) : (cols &= cols - 1) {
        const col = @ctz(cols);
        var letters = col_letters[col];
        while (letters != 0) : (letters &= letters - 1) letter_cols[@ctz(letters)] |= @as(u32, 1) << @intCast(col);
    }
}

/// Writes a word that fits at `start` into the strip: its letter on each
/// empty square (no blank yet) and 0 on each tile it plays through.
fn markWord(g: *Generator, word: []const u8, start: usize) void {
    for (word, 0..) |letter, i| {
        g.strip[start + i] = if (g.row[start + i].letter == 0) letter else 0;
    }
}

/// The playthrough tiles of the anchor's blocks, from its rightmost start
/// column on.
fn setPlaythroughFromRow(g: *Generator, anchor: *const shadow.Anchor) void {
    const w = &g.wmp_gen;
    w.resetPlaythrough();
    const want = anchor.playthrough_blocks;
    if (want == 0) return;
    var in_block = false;
    var found: u32 = 0;
    for (anchor.rightmost_start_col..g.board.dim) |col| {
        const ml = g.row[col].letter;
        if (ml == 0) {
            if (in_block) {
                if (found == want) break;
                in_block = false;
            }
            continue;
        }
        w.addPlaythroughLetter(tiles.unblank(ml));
        if (!in_block) {
            in_block = true;
            found += 1;
        }
    }
}

/// Places the subrack's blanks every way they can go among the word's new
/// tiles, then records each play.
fn recordPlaysForWord(g: *Generator, rack: *Rack, si: usize, subrack: wmp.BitRack, start: usize, word_length: usize, blanks_so_far: u32, pos: usize) void {
    if (wmp.count(subrack, 0) == blanks_so_far) return recordPlay(g, rack, si, start, word_length);
    if (pos >= word_length) return;
    const col = start + pos;
    const ml = g.strip[col];
    if (ml == 0) return recordPlaysForWord(g, rack, si, subrack, start, word_length, blanks_so_far, pos + 1);
    const p = blankPossibilities(g, subrack, start, word_length, pos);
    if (p.unblanked) recordPlaysForWord(g, rack, si, subrack, start, word_length, blanks_so_far, pos + 1);
    if (p.blanked) {
        g.strip[col] = tiles.blank(ml);
        recordPlaysForWord(g, rack, si, subrack, start, word_length, blanks_so_far + 1, pos + 1);
        g.strip[col] = ml;
    }
}

/// Whether the new tile at `pos` can be a real tile (the subrack has a copy
/// left after the earlier positions' real copies) or a blank (the word has
/// more of the letter than the subrack). Port of get_blank_possibilities.
fn blankPossibilities(g: *const Generator, subrack: wmp.BitRack, start: usize, word_length: usize, pos: usize) struct { unblanked: bool, blanked: bool } {
    const target = g.strip[start + pos];
    const in_subrack: u32 = wmp.count(subrack, target);
    var before: u32 = 0;
    var at_or_after: u32 = 0;
    for (0..word_length) |p| {
        const ml = g.strip[start + p];
        if (ml == 0 or tiles.isBlanked(ml) or ml != target) continue;
        if (p < pos) before += 1 else at_or_after += 1;
    }
    return .{ .unblanked = in_subrack > before, .blanked = before + at_or_after > in_subrack };
}

/// Scores the play in the strip and records it. Every play of subrack `si`
/// leaves the same tiles, so with the leave map its equity is the score plus
/// the subrack's leave value (the leave map's value for that leave, the
/// same number); otherwise its tiles are taken from the rack for the
/// recorder to value it, and put back.
fn recordPlay(g: *Generator, rack: *Rack, si: usize, start: usize, word_length: usize) void {
    const score = scoreLanePlay(g, start, word_length);
    if (g.leave_map.initialized and !g.ctx.board_empty) {
        const eq = @as(f64, @floatFromInt(score)) + g.wmp_gen.leaveValue(si);
        if (g.have_best and eq < g.best_eq) return;
        return g.recordPlayWithEquity(rack, start, start + word_length - 1, score, eq);
    }
    for (g.strip[start..][0..word_length]) |ml| {
        if (ml == 0) continue;
        const r: MachineLetter = if (tiles.isBlanked(ml)) 0 else ml;
        rack.take(r);
        if (g.leave_map.initialized) g.leave_map.take(r, rack.counts[r]);
    }
    g.recordPlay(rack, start, start + word_length - 1, score);
    var i = word_length;
    while (i > 0) {
        i -= 1;
        const ml = g.strip[start + i];
        if (ml == 0) continue;
        const r: MachineLetter = if (tiles.isBlanked(ml)) 0 else ml;
        if (g.leave_map.initialized) g.leave_map.add(r, rack.counts[r]);
        rack.add(r);
    }
}

/// The score of the tiles in the strip from `start` (0 for a square
/// played through). Port of scoreLanePlay.
fn scoreLanePlay(g: *const Generator, start: usize, word_length: usize) i32 {
    var played: i32 = 0;
    var playthrough: i32 = 0;
    var hooked_cross: i32 = 0;
    var played_cross: i32 = 0;
    var word_mul: i32 = 1;
    for (g.strip[start..][0..word_length], start..) |ml, col| {
        const sq = g.row[col];
        if (ml == 0) {
            playthrough += g.ld.score(sq.letter);
            continue;
        }
        const tile_score = g.ld.score(ml);
        const wm: i32 = sq.word_mul;
        const lm: i32 = sq.letter_mul;
        hooked_cross += sq.cross_score * wm;
        if (sq.cross_score > 0 or sq.cross_set != board_mod.trivial_cross_set) played_cross += tile_score * wm * lm;
        played += tile_score * lm;
        word_mul *= wm;
    }
    const bingo: i32 = if (g.tiles_played == movegen.rack_tile_limit) movegen.bingo_bonus else 0;
    return (played + playthrough) * word_mul + hooked_cross + played_cross + bingo;
}
