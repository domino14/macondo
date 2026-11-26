const std = @import("std");
const zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const data_path = b.option([]const u8, "data-path", "Macondo data directory for tests (default ../data)") orelse "../data";
    const tsan = b.option(bool, "tsan", "Build with ThreadSanitizer (for `zig build test`: finds data races)") orelse false;

    const options = b.addOptions();
    options.addOption([]const u8, "version", zon.version);
    options.addOption([]const u8, "data_path", data_path);

    // The engine: everything except the I/O adapters. Tools, tests and the
    // later WASM build import this module.
    // Frame pointers let the bench's profiler (BENCH_PROFILE) record who
    // called each sampled function.
    const frame_pointers = b.option(bool, "frame-pointers", "Keep frame pointers, for the bench's profiler") orelse false;
    const omit_fp: ?bool = if (frame_pointers) false else null;
    const core = b.addModule("macondo", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .sanitize_thread = tsan,
        .omit_frame_pointer = omit_fp,
        .imports = &.{
            .{ .name = "build_options", .module = options.createModule() },
        },
    });

    // The native stdio adapter.
    const exe = b.addExecutable(.{
        .name = "macondo-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/adapter/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "macondo", .module = core },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the UCGI engine on stdio");
    run_step.dependOn(&run_cmd.step);

    // Times the best-move search on a file of positions. Linking libc lets
    // preloading profilers (gprofng) see inside it.
    const bench_libc = b.option(bool, "bench-libc", "Link the bench tool against libc, for profilers") orelse false;
    const bench = b.addExecutable(.{
        .name = "macondo-zig-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/adapter/bench.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = bench_libc,
            .omit_frame_pointer = omit_fp,
            .imports = &.{
                .{ .name = "macondo", .module = core },
            },
        }),
    });
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    run_bench.setCwd(b.path("."));
    run_bench.addPassthruArgs();
    const bench_step = b.step("bench", "Time the best-move search on a file of positions");
    bench_step.dependOn(&run_bench.step);

    const core_tests = b.addTest(.{ .root_module = core });
    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const test_step = b.step("test", "Run unit tests");
    // Tests read data files relative to zigcore/.
    const run_core_tests = b.addRunArtifact(core_tests);
    run_core_tests.setCwd(b.path("."));
    const run_exe_tests = b.addRunArtifact(exe_tests);
    run_exe_tests.setCwd(b.path("."));
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
