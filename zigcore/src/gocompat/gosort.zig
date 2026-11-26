//! Go's sort.Slice (pattern-defeating quicksort, from sort/zsortfunc.go),
//! step for step, so that an unstable sort leaves elements its comparison
//! can't separate in the order Go leaves them. Where the order of ties
//! decides what a search measures next, a port must sort as Go does.

const std = @import("std");

/// Sorts `items` as Go's sort.Slice(items, less) does. `less(ctx, a, b)`.
pub fn slice(comptime T: type, items: []T, ctx: anytype, comptime less: fn (@TypeOf(ctx), T, T) bool) void {
    const Data = struct {
        items: []T,
        ctx: @TypeOf(ctx),
        fn lt(d: @This(), i: usize, j: usize) bool {
            return less(d.ctx, d.items[i], d.items[j]);
        }
        fn swap(d: @This(), i: usize, j: usize) void {
            std.mem.swap(T, &d.items[i], &d.items[j]);
        }
    };
    const d: Data = .{ .items = items, .ctx = ctx };
    const limit: usize = if (items.len == 0) 0 else std.math.log2_int(usize, items.len) + 1;
    pdqsort(d, 0, items.len, limit);
}

const Hint = enum { unknown, increasing, decreasing };

fn insertionSort(d: anytype, a: usize, b: usize) void {
    var i = a + 1;
    while (i < b) : (i += 1) {
        var j = i;
        while (j > a and d.lt(j, j - 1)) : (j -= 1) d.swap(j, j - 1);
    }
}

fn siftDown(d: anytype, lo: usize, hi: usize, first: usize) void {
    var root = lo;
    while (true) {
        var child = 2 * root + 1;
        if (child >= hi) break;
        if (child + 1 < hi and d.lt(first + child, first + child + 1)) child += 1;
        if (!d.lt(first + root, first + child)) return;
        d.swap(first + root, first + child);
        root = child;
    }
}

fn heapSort(d: anytype, a: usize, b: usize) void {
    const first = a;
    const hi = b - a;
    var i: isize = @as(isize, @intCast(hi)) - 1;
    i = @divTrunc(i, 2);
    while (i >= 0) : (i -= 1) siftDown(d, @intCast(i), hi, first);
    var k: isize = @as(isize, @intCast(hi)) - 1;
    while (k >= 0) : (k -= 1) {
        d.swap(first, first + @as(usize, @intCast(k)));
        siftDown(d, 0, @intCast(k), first);
    }
}

fn pdqsort(d: anytype, a_in: usize, b_in: usize, limit_in: usize) void {
    const max_insertion = 12;
    var a = a_in;
    var b = b_in;
    var limit = limit_in;
    var was_balanced = true;
    var was_partitioned = true;
    while (true) {
        const length = b - a;
        if (length <= max_insertion) {
            insertionSort(d, a, b);
            return;
        }
        if (limit == 0) {
            heapSort(d, a, b);
            return;
        }
        if (!was_balanced) {
            breakPatterns(d, a, b);
            limit -= 1;
        }
        var cp = choosePivot(d, a, b);
        if (cp.hint == .decreasing) {
            reverseRange(d, a, b);
            cp.pivot = (b - 1) - (cp.pivot - a);
            cp.hint = .increasing;
        }
        if (was_balanced and was_partitioned and cp.hint == .increasing) {
            if (partialInsertionSort(d, a, b)) return;
        }
        if (a > 0 and !d.lt(a - 1, cp.pivot)) {
            a = partitionEqual(d, a, b, cp.pivot);
            continue;
        }
        const p = partition(d, a, b, cp.pivot);
        was_partitioned = p.already;
        const mid = p.mid;
        const left_len = mid - a;
        const right_len = b - mid;
        const balance_threshold = length / 8;
        if (left_len < right_len) {
            was_balanced = left_len >= balance_threshold;
            pdqsort(d, a, mid, limit);
            a = mid + 1;
        } else {
            was_balanced = right_len >= balance_threshold;
            pdqsort(d, mid + 1, b, limit);
            b = mid;
        }
    }
}

fn partition(d: anytype, a: usize, b: usize, pivot: usize) struct { mid: usize, already: bool } {
    d.swap(a, pivot);
    // i and j as signed: j can step below a.
    var i: isize = @intCast(a + 1);
    var j: isize = @as(isize, @intCast(b)) - 1;
    const ia: usize = a;
    while (i <= j and d.lt(@intCast(i), ia)) i += 1;
    while (i <= j and !d.lt(@intCast(j), ia)) j -= 1;
    if (i > j) {
        d.swap(@intCast(j), ia);
        return .{ .mid = @intCast(j), .already = true };
    }
    d.swap(@intCast(i), @intCast(j));
    i += 1;
    j -= 1;
    while (true) {
        while (i <= j and d.lt(@intCast(i), ia)) i += 1;
        while (i <= j and !d.lt(@intCast(j), ia)) j -= 1;
        if (i > j) break;
        d.swap(@intCast(i), @intCast(j));
        i += 1;
        j -= 1;
    }
    d.swap(@intCast(j), ia);
    return .{ .mid = @intCast(j), .already = false };
}

fn partitionEqual(d: anytype, a: usize, b: usize, pivot: usize) usize {
    d.swap(a, pivot);
    var i: isize = @intCast(a + 1);
    var j: isize = @as(isize, @intCast(b)) - 1;
    while (true) {
        while (i <= j and !d.lt(a, @intCast(i))) i += 1;
        while (i <= j and d.lt(a, @intCast(j))) j -= 1;
        if (i > j) break;
        d.swap(@intCast(i), @intCast(j));
        i += 1;
        j -= 1;
    }
    return @intCast(i);
}

fn partialInsertionSort(d: anytype, a: usize, b: usize) bool {
    const max_steps = 5;
    const shortest_shifting = 50;
    var i = a + 1;
    for (0..max_steps) |_| {
        while (i < b and !d.lt(i, i - 1)) i += 1;
        if (i == b) return true;
        if (b - a < shortest_shifting) return false;
        d.swap(i, i - 1);
        if (i - a >= 2) {
            var j = i - 1;
            while (j >= 1) : (j -= 1) {
                if (!d.lt(j, j - 1)) break;
                d.swap(j, j - 1);
            }
        }
        if (b - i >= 2) {
            var j = i + 1;
            while (j < b) : (j += 1) {
                if (!d.lt(j, j - 1)) break;
                d.swap(j, j - 1);
            }
        }
    }
    return false;
}

fn breakPatterns(d: anytype, a: usize, b: usize) void {
    const length = b - a;
    if (length >= 8) {
        var random: u64 = length;
        const modulus: u64 = @as(u64, 1) << @intCast(std.math.log2_int(usize, length) + 1);
        var idx = a + (length / 4) * 2 - 1;
        while (idx <= a + (length / 4) * 2 + 1) : (idx += 1) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            var other: usize = @intCast(random & (modulus - 1));
            if (other >= length) other -= length;
            d.swap(idx, a + other);
        }
    }
}

fn choosePivot(d: anytype, a: usize, b: usize) struct { pivot: usize, hint: Hint } {
    const shortest_ninther = 50;
    const max_swaps = 4 * 3;
    const l = b - a;
    var swaps: usize = 0;
    var i = a + l / 4 * 1;
    var j = a + l / 4 * 2;
    var k = a + l / 4 * 3;
    if (l >= 8) {
        if (l >= shortest_ninther) {
            i = median(d, i - 1, i, i + 1, &swaps);
            j = median(d, j - 1, j, j + 1, &swaps);
            k = median(d, k - 1, k, k + 1, &swaps);
        }
        j = median(d, i, j, k, &swaps);
    }
    return switch (swaps) {
        0 => .{ .pivot = j, .hint = .increasing },
        max_swaps => .{ .pivot = j, .hint = .decreasing },
        else => .{ .pivot = j, .hint = .unknown },
    };
}

fn median(d: anytype, a_in: usize, b_in: usize, c_in: usize, swaps: *usize) usize {
    var a = a_in;
    var b = b_in;
    var c = c_in;
    order2(d, &a, &b, swaps);
    order2(d, &b, &c, swaps);
    order2(d, &a, &b, swaps);
    return b;
}

fn order2(d: anytype, a: *usize, b: *usize, swaps: *usize) void {
    if (d.lt(b.*, a.*)) {
        swaps.* += 1;
        std.mem.swap(usize, a, b);
    }
}

fn reverseRange(d: anytype, a: usize, b: usize) void {
    var i = a;
    var j = b - 1;
    while (i < j) {
        d.swap(i, j);
        i += 1;
        j -= 1;
    }
}

// Golden orders from Go: sort.Slice on pairs by key only (ties keep a
// position the test can see), for patterns that reach every branch.
test "sorts as Go's sort.Slice" {
    const Pair = struct { key: u32, id: u32 };
    const golden = @import("gosort_golden.zig");
    for (golden.cases) |c| {
        var items: [golden.max_len]Pair = undefined;
        for (c.keys, 0..) |k, i| items[i] = .{ .key = k, .id = @intCast(i) };
        const xs = items[0..c.keys.len];
        slice(Pair, xs, {}, struct {
            fn lt(_: void, x: Pair, y: Pair) bool {
                return x.key < y.key;
            }
        }.lt);
        for (xs, c.ids) |p, id| try std.testing.expectEqual(id, p.id);
    }
}
