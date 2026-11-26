//! Go's math.Exp and math.Log as they run on amd64 (exp_amd64.s and
//! log_amd64.s), bit for bit. Sims use them where Go does (the best-arm
//! reward, the stopping rules), so that a Zig sim makes exactly the
//! decisions a Go sim makes. Go's exp takes its fused multiply-add path on
//! CPUs with AVX and FMA, and its results then differ in the last bit now
//! and then; this follows the path the target CPU gives Go.

const std = @import("std");
const builtin = @import("builtin");

/// Whether Go's exp takes its FMA path (math.useFMA: AVX and FMA).
const use_fma = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHasAll(builtin.cpu.features, .{ .avx, .fma });

const ln2u = 0.69314718055966295651160180568695068359375;
const ln2l = 0.28235290563031577122588448175013436025525412068e-12;
const log2e = 1.4426950408889634073599246810018920;
const overflow = 7.09782712893384e+02;
const taylor = [_]f64{
    0.5,                      1.0,                      2.0,
    1.6666666666666666667e-1, 4.1666666666666666667e-2, 8.3333333333333333333e-3,
    1.3888888888888888889e-3, 1.9841269841269841270e-4, 2.4801587301587301587e-5,
};

/// Rounds to the nearest integer, ties to even (CVTSD2SL), for |x| < 2^31.
fn rint(x: f64) i32 {
    const big = 0x1p52;
    const r = if (x >= 0) (x + big) - big else (x - big) + big;
    return @intFromFloat(r);
}

fn bits(b: u64) f64 {
    return @bitCast(b);
}

pub fn exp(x: f64) f64 {
    const xb: u64 = @bitCast(x);
    const abs_bits = xb & ~(@as(u64, 1) << 63);
    if (abs_bits >= 0x7FF0000000000000) {
        // -Inf gives 0; NaN and +Inf themselves.
        return if (xb == 0xFFF0000000000000) 0 else x;
    }
    if (x > overflow) return std.math.inf(f64);
    const scaled = log2e * x;
    // CVTSD2SL gives the "integer indefinite" for out-of-range values.
    const e: i32 = if (@abs(scaled) < 0x1p31) rint(scaled) else std.math.minInt(i32);
    const k: f64 = @floatFromInt(e);
    var x0 = x;
    var x1: f64 = undefined;
    if (use_fma) {
        x0 = @mulAdd(f64, -k, ln2u, x0);
        x0 = @mulAdd(f64, -k, ln2l, x0);
        x0 *= 0.0625;
        x1 = taylor[8];
        x1 = @mulAdd(f64, x1, x0, taylor[7]);
        x1 = @mulAdd(f64, x1, x0, taylor[6]);
        x1 = @mulAdd(f64, x1, x0, taylor[5]);
        x1 = @mulAdd(f64, x1, x0, taylor[4]);
        x1 = @mulAdd(f64, x1, x0, taylor[3]);
        x1 = @mulAdd(f64, x1, x0, taylor[0]);
        x1 = @mulAdd(f64, x1, x0, taylor[1]);
        x0 *= x1;
        x1 = x0 + taylor[2];
        x0 *= x1;
        x1 = x0 + taylor[2];
        x0 *= x1;
        x1 = x0 + taylor[2];
        x0 *= x1;
        x1 = x0 + taylor[2];
        x0 = @mulAdd(f64, x0, x1, taylor[1]);
    } else {
        x0 -= ln2u * k;
        x0 -= ln2l * k;
        x0 *= 0.0625;
        x1 = taylor[8] * x0;
        x1 += taylor[7];
        x1 *= x0;
        x1 += taylor[6];
        x1 *= x0;
        x1 += taylor[5];
        x1 *= x0;
        x1 += taylor[4];
        x1 *= x0;
        x1 += taylor[3];
        x1 *= x0;
        x1 += taylor[0];
        x1 *= x0;
        x1 += taylor[1];
        x0 *= x1;
        for (0..4) |_| {
            x1 = taylor[2] + x0;
            x0 *= x1;
        }
        x0 += taylor[1];
    }
    // fr * 2**exponent
    var b: i64 = @as(i64, e) + 0x3FF;
    if (b <= 0) {
        if (b < -52) return 0;
        b += 0x3FE;
        x0 *= bits(@as(u64, @intCast(b)) << 52);
        b = 1;
    } else if (b >= 0x7FF) {
        return std.math.inf(f64);
    }
    return x0 * bits(@as(u64, @intCast(b)) << 52);
}

const hsqrt2 = 7.07106781186547524401e-01;
const ln2hi = 6.93147180369123816490e-01;
const ln2lo = 1.90821492927058770002e-10;
const l = [_]f64{
    6.666666666666735130e-01, 3.999999999940941908e-01, 2.857142874366239149e-01,
    2.222219843214978396e-01, 1.818357216161805012e-01, 1.531383769920937332e-01,
    1.479819860511658591e-01,
};

pub fn log(x: f64) f64 {
    const xb: u64 = @bitCast(x);
    if (xb & ~(@as(u64, 1) << 63) == 0) return -std.math.inf(f64);
    if (@as(i64, @bitCast(xb)) < 0) return bits(0x7FF8000000000001);
    if (xb >= 0x7FF0000000000000) return x;
    // Frexp on the bits (no denormal handling, as the assembly).
    var f1 = bits((xb & 0x000FFFFFFFFFFFFF) | 0x3FE0000000000000);
    var k: f64 = @floatFromInt(@as(i32, @intCast((xb >> 52) & 0x7FF)) - 0x3FE);
    if (!(hsqrt2 < f1)) {
        k -= 1;
        f1 *= 2;
    }
    const f = f1 - 1;
    const s = f / (2 + f);
    const s2 = s * s;
    const s4 = s2 * s2;
    var t = l[6] * s4;
    t += l[4];
    t *= s4;
    t += l[2];
    t *= s4;
    t += l[0];
    const t1 = s2 * t;
    var u = l[5] * s4;
    u += l[3];
    u *= s4;
    u += l[1];
    const t2 = s4 * u;
    const r = t1 + t2;
    const hfsq = 0.5 * f * f;
    return k * ln2hi - ((hfsq - (s * (hfsq + r) + k * ln2lo)) - f);
}

// Tests ----------------------------------------------------------------------

const testing = std.testing;

// (input, Go's result) bit patterns, from Go 1.27 on an x86-64 CPU with
// FMA.
const exp_cases = [_][2]u64{
    .{ 0x0000000000000000, 0x3ff0000000000000 },
    .{ 0x3ff0000000000000, 0x4005bf0a8b145769 },
    .{ 0xbff0000000000000, 0x3fd78b56362cef38 },
    .{ 0x3fe0000000000000, 0x3ffa61298e1e069c },
    .{ 0xbfe0000000000000, 0x3fe368b2fc6f960a },
    .{ 0x01a56e1fc2f8f359, 0x3ff0000000000000 },
    .{ 0x4085e00000000000, 0x7f0d945df4f8ec8e },
    .{ 0x40862d999999999a, 0x7ff0000000000000 },
    .{ 0xc085e00000000000, 0x00d14f2b0fb9307f },
    .{ 0xc087480000000000, 0x0000000000000001 },
    .{ 0xc087500000000000, 0x0000000000000000 },
    .{ 0x4004000000000000, 0x40285d6fd931e0bb },
    .{ 0xc00e000000000000, 0x3f981509354f0d29 },
    .{ 0x3fb0000000000000, 0x3ff1082b577d34ee },
    .{ 0x3ddb7cdfd9d7bdbb, 0x3ff000000006df38 },
    .{ 0xbddb7cdfd9d7bdbb, 0x3feffffffff24190 },
    .{ 0x400c3b9fb2aac950, 0x40410bff018da73f },
    .{ 0xbfe8b67b6a298790, 0x3fdd90c33ed1955b },
    .{ 0x3fc5e19f5ed82a80, 0x3ff2fb9cf9146269 },
    .{ 0xbff67760e1b46ffc, 0x3fcf6f1060c43ec0 },
    .{ 0x4017d2fdfb47af48, 0x4078214b73f92b4d },
    .{ 0xc001dc823d963a54, 0x3fbb7430d9531521 },
    .{ 0x40198ac0a6880e2e, 0x40828973d70c9222 },
    .{ 0xc009e413e873123a, 0x3fa41ffa55e8e128 },
    .{ 0x4022ebfd31c8d8ff, 0x40c917d309a4e420 },
    .{ 0xc0117bec0ce68c48, 0x3f89e225a80c49e1 },
    .{ 0x4000c330eb7120ab, 0x40204182090d4fa9 },
    .{ 0x40211cc820977a5c, 0x40b44ef661371778 },
    .{ 0xc01ae3d4dd352eac, 0x3f53b7fe89ade25b },
    .{ 0x40121c77f95b4c88, 0x405723829f1066de },
    .{ 0x400f994325d456d6, 0x4049f6a180c8e8f5 },
    .{ 0x401520a8c7e135b4, 0x406897bf775b5332 },
    .{ 0x401181666f796642, 0x4053e31b2d454001 },
    .{ 0x3ffb65ebb7a34354, 0x40162b2a14554261 },
    .{ 0x402232c99859f687, 0x40c17a0a47e1b3ea },
    .{ 0x3fd59b6463b20930, 0x3ff66cecb063c6b1 },
    .{ 0x40060716cd0286fd, 0x402f64cc68ad7bd8 },
    .{ 0x40113b98e7d6032a, 0x405293a02f8a78f9 },
    .{ 0x3ff1c59cf8fa0dc2, 0x40084afffb783fd4 },
    .{ 0xc011515605c12051, 0x3f8afb867995f174 },
    .{ 0x4014f95190e071ef, 0x4067aa760f4a876c },
    .{ 0xbffdf37ecd2b77a0, 0x3fc3b0800f7160df },
    .{ 0x401d218b46963590, 0x4096bbfb0cc9eb28 },
    .{ 0x3fc237aadcec0650, 0x3ff2727ba6a0a237 },
    .{ 0xc020e3019caaef8d, 0x3f2c3909da982885 },
    .{ 0x40099d4aaa0edb92, 0x403893a89d673b64 },
    .{ 0xc02251b77d785720, 0x3f1b942a262f6b1f },
    .{ 0x401b62a8f2ab47a3, 0x408d6384a80f2764 },
    .{ 0xc0131d0f36fd0799, 0x3f813910cf516a36 },
    .{ 0xc015b7441fee5c6e, 0x3f71f8bddcfec3fb },
    .{ 0xc02291e11cd82efe, 0x3f1854985ec4fcf2 },
    .{ 0x401e88d17b06a90c, 0x40a024fff825a299 },
    .{ 0x40140a1de1a3efb6, 0x4062bc5fc53e2b8c },
    .{ 0x402318c47102b7a4, 0x40cb62efb68ced2c },
    .{ 0xc02114c35396339a, 0x3f299bf0a52a88be },
    .{ 0xc0173af5a474bf4e, 0x3f689d4fd31f6137 },
    .{ 0x407dd8278d42e78a, 0x6afddfb6f312a12d },
    .{ 0x4049cef8a8f26d46, 0x44961fbb5f776dca },
    .{ 0xc06d68e227487f17, 0x2ab7b075ff4ca71a },
    .{ 0xc06dfd2dfb7da526, 0x2a4d73fabddc356d },
    .{ 0x4080780d1d9ab2ab, 0x6f73d42c7c9830cf },
    .{ 0x404fee888e1a36fa, 0x45b193b55d450a21 },
    .{ 0xc0732d85c0d88222, 0x2443eb10bb9588a0 },
    .{ 0x4063ccb9e04c883e, 0x4e36efb26bd1bc61 },
};
const log_cases = [_][2]u64{
    .{ 0x3ff0000000000000, 0x0000000000000000 },
    .{ 0x4000000000000000, 0x3fe62e42fefa39ef },
    .{ 0x3fe0000000000000, 0xbfe62e42fefa39ef },
    .{ 0x3fe6a09e667f3bcd, 0xbfd62e42fefa39ee },
    .{ 0x3fe6a09e667f3bcc, 0xbfd62e42fefa39f1 },
    .{ 0x01a56e1fc2f8f359, 0xc085963447f87fb5 },
    .{ 0x7e37e43c8800759c, 0x4085963447f87fb5 },
    .{ 0x4008000000000000, 0x3ff193ea7aad030a },
    .{ 0x4024000000000000, 0x40026bb1bbb55516 },
    .{ 0x3f50624dd2f1a9fc, 0xc01ba18a998fffa0 },
    .{ 0x40fe240c9fbe76c9, 0x40277281cad8a844 },
    .{ 0x4041254c618c9d6a, 0x400c4776b787907c },
    .{ 0x404a1979a8fe2203, 0x400fa3f88040da97 },
    .{ 0x40519ed0922a54d4, 0x4011057a77407819 },
    .{ 0x3ff77b22846f93d2, 0x3fd88cf23387e81c },
    .{ 0x40461c6b2535c901, 0x400e505411efe8a1 },
    .{ 0x404253dbd0131243, 0x400ccffb72c38e1a },
    .{ 0x400a5c6b5b126e8d, 0x3ff31440ca445328 },
    .{ 0x405580625b5fa339, 0x4011d152b748bf63 },
    .{ 0x40558d750e7c3dfb, 0x4011d3c0937c4823 },
    .{ 0x4058949b205a4d8b, 0x40125a5d8c55833f },
    .{ 0x4054aab69e4eefb4, 0x4011a8c7f56aced6 },
    .{ 0x40558eca18a3a962, 0x4011d3ffdceaf706 },
    .{ 0x40558976a476f34d, 0x4011d302c1384cdc },
    .{ 0x4050b9ed1a9d27cc, 0x4010d0275c3eb08e },
    .{ 0x4054cc1cda3fca7e, 0x4011af39a0633b0e },
    .{ 0x4051fca5ebd72e11, 0x40111a8f9c0b9046 },
    .{ 0x405492a31b73bdee, 0x4011a41c4f19662d },
    .{ 0x402b259dca16cdbc, 0x4004dd6c81396d8a },
    .{ 0x404dc268ae6ff07a, 0x4010585d13965bbd },
    .{ 0x4053fa28feb1d0ae, 0x40118606d185f53d },
    .{ 0x4058e61efd9b5193, 0x4012678b953523be },
    .{ 0x403f5456e0bde56a, 0x400b8e74b94d3b00 },
    .{ 0x4056a7f15acdb2e8, 0x401206e1ac51c34a },
    .{ 0x404097ec02510048, 0x400c0469cbf2fcf9 },
    .{ 0x404ea2cbd94e6c73, 0x4010761690b77387 },
    .{ 0x4058c9e5fd7fa854, 0x401263004fec9faa },
    .{ 0x405037e18a625965, 0x4010b09274ae51de },
    .{ 0x40586e655f3693e8, 0x40125420e77a0c98 },
    .{ 0x4053d5eb727fe468, 0x40117ebe93b6d70f },
    .{ 0x40542588806382fe, 0x40118eacae52bf7d },
    .{ 0x4044156a4cbcf950, 0x400d8b5f93e89771 },
    .{ 0x4024809c013a0582, 0x40029e80a66c47a9 },
    .{ 0x40567cebb3d27366, 0x4011ff41ed4cf7a4 },
    .{ 0x40565c1be11714dd, 0x4011f9678d5e6e91 },
    .{ 0x4040c2a8c3c0deb1, 0x400c18ea2cf344db },
    .{ 0x40580fbde5e5e689, 0x40124483487399c4 },
    .{ 0x404d90465b32d51a, 0x4010519a4c29687b },
    .{ 0x4054a15f3d4b7a52, 0x4011a6f8b4c47e66 },
    .{ 0x401948f4d8398cf4, 0x3ffd80ae814cc4e2 },
    .{ 0x405461218b13a051, 0x40119a7083cbc197 },
    .{ 0x16c96fb3509e642b, 0xc07c85205e1bcdaf },
    .{ 0x58bc44d99abed0d3, 0x407130e3289abb16 },
    .{ 0x1eddfff5b1198b25, 0xc076ebd4977c4d89 },
    .{ 0x34b06fdd77a63a4d, 0xc05f2f54c1022d40 },
    .{ 0x4db102ee6e091551, 0x406311b7ba443337 },
    .{ 0x1877de966272f9f4, 0xc07b5ab43fba96dc },
    .{ 0x6afd7ea80f8a703e, 0x407dd7f33c7a90a4 },
    .{ 0x197b7cd8911e55f1, 0xc07aa7000ebe04a1 },
    .{ 0x52e0b2bc2ca62a94, 0x406a421f2fbf3f09 },
    .{ 0x64b7c5dda380b4dc, 0x40797f76c8fee7a5 },
};

test "exp and log are Go's" {
    if (!use_fma) return error.SkipZigTest;
    for (exp_cases) |c| {
        testing.expectEqual(c[1], @as(u64, @bitCast(exp(bits(c[0]))))) catch |err| {
            std.debug.print("exp({e})\n", .{bits(c[0])});
            return err;
        };
    }
    for (log_cases) |c| {
        testing.expectEqual(c[1], @as(u64, @bitCast(log(bits(c[0]))))) catch |err| {
            std.debug.print("log({e})\n", .{bits(c[0])});
            return err;
        };
    }
}

/// Go's math.Lgamma(x) for x > 0 (the log of the gamma function), with
/// Go's arithmetic (its Log): the leave prior's binomial coefficients.
pub fn lgamma(x: f64) f64 {
    const ymin = 1.461632144968362245;
    const two58 = 1 << 58;
    const tiny = 1.0 / @as(f64, 1 << 70);
    const tc = 1.46163214496836224576e+00;
    const tf = -1.21486290535849611461e-01;
    const tt = -3.63867699703950536541e-18;
    std.debug.assert(x > 0);
    if (std.math.isNan(x) or std.math.isInf(x)) return x;
    if (x < tiny) return -log(x);
    if (x == 1 or x == 2) return 0;
    if (x < 2) {
        var lg: f64 = 0;
        var y: f64 = undefined;
        var i: u2 = undefined;
        if (x <= 0.9) {
            lg = -log(x);
            if (x >= (ymin - 1 + 0.27)) {
                y = 1 - x;
                i = 0;
            } else if (x >= (ymin - 1 - 0.27)) {
                y = x - (tc - 1);
                i = 1;
            } else {
                y = x;
                i = 2;
            }
        } else {
            if (x >= (ymin + 0.27)) {
                y = 2 - x;
                i = 0;
            } else if (x >= (ymin - 0.27)) {
                y = x - tc;
                i = 1;
            } else {
                y = x - 1;
                i = 2;
            }
        }
        switch (i) {
            0 => {
                const zz = y * y;
                const p1 = lgam_a[0] + zz * (lgam_a[2] + zz * (lgam_a[4] + zz * (lgam_a[6] + zz * (lgam_a[8] + zz * lgam_a[10]))));
                const p2 = zz * (lgam_a[1] + zz * (lgam_a[3] + zz * (lgam_a[5] + zz * (lgam_a[7] + zz * (lgam_a[9] + zz * lgam_a[11])))));
                const p = y * p1 + p2;
                lg += (p - 0.5 * y);
            },
            1 => {
                const zz = y * y;
                const w = zz * y;
                const p1 = lgam_t[0] + w * (lgam_t[3] + w * (lgam_t[6] + w * (lgam_t[9] + w * lgam_t[12])));
                const p2 = lgam_t[1] + w * (lgam_t[4] + w * (lgam_t[7] + w * (lgam_t[10] + w * lgam_t[13])));
                const p3 = lgam_t[2] + w * (lgam_t[5] + w * (lgam_t[8] + w * (lgam_t[11] + w * lgam_t[14])));
                const p = zz * p1 - (tt - w * (p2 + y * p3));
                lg += (tf + p);
            },
            else => {
                const p1 = y * (lgam_u[0] + y * (lgam_u[1] + y * (lgam_u[2] + y * (lgam_u[3] + y * (lgam_u[4] + y * lgam_u[5])))));
                const p2 = 1 + y * (lgam_v[1] + y * (lgam_v[2] + y * (lgam_v[3] + y * (lgam_v[4] + y * lgam_v[5]))));
                lg += (-0.5 * y + p1 / p2);
            },
        }
        return lg;
    }
    if (x < 8) {
        const i: u32 = @intFromFloat(x);
        const y = x - @as(f64, @floatFromInt(i));
        const p = y * (lgam_s[0] + y * (lgam_s[1] + y * (lgam_s[2] + y * (lgam_s[3] + y * (lgam_s[4] + y * (lgam_s[5] + y * lgam_s[6]))))));
        const q = 1 + y * (lgam_r[1] + y * (lgam_r[2] + y * (lgam_r[3] + y * (lgam_r[4] + y * (lgam_r[5] + y * lgam_r[6])))));
        var lg = 0.5 * y + p / q;
        var zz: f64 = 1.0;
        if (i >= 7) zz *= (y + 6);
        if (i >= 6) zz *= (y + 5);
        if (i >= 5) zz *= (y + 4);
        if (i >= 4) zz *= (y + 3);
        if (i >= 3) {
            zz *= (y + 2);
            lg += log(zz);
        }
        return lg;
    }
    if (x < two58) {
        const t = log(x);
        const zz = 1 / x;
        const y = zz * zz;
        const w = lgam_w[0] + zz * (lgam_w[1] + y * (lgam_w[2] + y * (lgam_w[3] + y * (lgam_w[4] + y * (lgam_w[5] + y * lgam_w[6])))));
        return (x - 0.5) * (t - 1) + w;
    }
    return x * (log(x) - 1);
}

const lgam_a = [_]f64{ @bitCast(@as(u64, 0x3fb3c467e37db0c8)), @bitCast(@as(u64, 0x3fd4a34cc4a60fad)), @bitCast(@as(u64, 0x3fb13e001a5562a7)), @bitCast(@as(u64, 0x3f951322ac92547b)), @bitCast(@as(u64, 0x3f7e404fb68fefe8)), @bitCast(@as(u64, 0x3f67add8ccb7926b)), @bitCast(@as(u64, 0x3f538a94116f3f5d)), @bitCast(@as(u64, 0x3f40b6c689b99c00)), @bitCast(@as(u64, 0x3f2cf2eced10e54d)), @bitCast(@as(u64, 0x3f1c5088987dfb07)), @bitCast(@as(u64, 0x3efa7074428cfa52)), @bitCast(@as(u64, 0x3f07858e90a45837)) };
const lgam_r = [_]f64{ 1.0, @bitCast(@as(u64, 0x3ff645a762c4ab74)), @bitCast(@as(u64, 0x3fe71a1893d3dcdc)), @bitCast(@as(u64, 0x3fc601edccfbdf27)), @bitCast(@as(u64, 0x3f9317ea742ed475)), @bitCast(@as(u64, 0x3f497ddaca41a95b)), @bitCast(@as(u64, 0x3edebaf7a5b38140)) };
const lgam_s = [_]f64{ @bitCast(@as(u64, 0xbfb3c467e37db0c8)), @bitCast(@as(u64, 0x3fcb848b36e20878)), @bitCast(@as(u64, 0x3fd4d98f4f139f59)), @bitCast(@as(u64, 0x3fc2bb9cbee5f2f7)), @bitCast(@as(u64, 0x3f9b481c7e939961)), @bitCast(@as(u64, 0x3f5e26b67368f239)), @bitCast(@as(u64, 0x3f00bfecdd17e945)) };
const lgam_t = [_]f64{ @bitCast(@as(u64, 0x3fdef72bc8ee38a2)), @bitCast(@as(u64, 0xbfc2e4278dc6c509)), @bitCast(@as(u64, 0x3fb08b4294d5419b)), @bitCast(@as(u64, 0xbfa0c9a8df35b713)), @bitCast(@as(u64, 0x3f9266e7970af9ec)), @bitCast(@as(u64, 0xbf851f9fba91ec6a)), @bitCast(@as(u64, 0x3f78fce0e370e344)), @bitCast(@as(u64, 0xbf6e2effb3e914d7)), @bitCast(@as(u64, 0x3f6282d32e15c915)), @bitCast(@as(u64, 0xbf56fe8ebf2d1af1)), @bitCast(@as(u64, 0x3f4cdf0cef61a8e9)), @bitCast(@as(u64, 0xbf41a6109c73e0ec)), @bitCast(@as(u64, 0x3f34af6d6c0ebbf7)), @bitCast(@as(u64, 0xbf347f24ecc38c38)), @bitCast(@as(u64, 0x3f35fd3ee8c2d3f4)) };
const lgam_u = [_]f64{ @bitCast(@as(u64, 0xbfb3c467e37db0c8)), @bitCast(@as(u64, 0x3fe4401e8b005dff)), @bitCast(@as(u64, 0x3ff7475cd119bd6f)), @bitCast(@as(u64, 0x3fef497644ea8450)), @bitCast(@as(u64, 0x3fcd4eaef6010924)), @bitCast(@as(u64, 0x3f8b678bbf2bab09)) };
const lgam_v = [_]f64{ 1.0, @bitCast(@as(u64, 0x4003a5d7c2bd619c)), @bitCast(@as(u64, 0x40010725a42b18f5)), @bitCast(@as(u64, 0x3fe89dfbe45050af)), @bitCast(@as(u64, 0x3fbaae55d6537c88)), @bitCast(@as(u64, 0x3f6a5abb57d0cf61)) };
const lgam_w = [_]f64{ @bitCast(@as(u64, 0x3fdacfe390c97d69)), @bitCast(@as(u64, 0x3fb555555555553b)), @bitCast(@as(u64, 0xbf66c16c16b02e5c)), @bitCast(@as(u64, 0x3f4a019f98cf38b6)), @bitCast(@as(u64, 0xbf4380cb8c0fe741)), @bitCast(@as(u64, 0x3f4b67ba4cdad5d1)), @bitCast(@as(u64, 0xbf5ab89d0b9e43e4)) };

test "lgamma matches Go" {
    // math.Lgamma from Go, bit for bit.
    for (@import("golgamma_golden.zig").cases) |c| try std.testing.expectEqual(c[1], lgamma(c[0]));
}
