//! Recording controls for the output work a hidden-row learner does not consume.
const std = @import("std");
const more = @import("cuda_train_more.zig");
const engine = @import("cuda_engine.zig");
const t = std.testing;
test "dense read cost: keys omit logits but target and loss scoring still need them" {
    const keys = more.keepFor(.keys);
    try t.expect(!std.meta.eql(keys, @as(engine.Keep, .logits)));
    {
        try t.expectEqual(engine.Keep.logits, more.keepFor(.target));
        try t.expectEqual(engine.Keep.logits, more.keepFor(.loss));
    }
}

const Recorder = struct {
    g: struct { hidden: usize = 64 } = .{},
    learning: ?*const @import("cuda_learn_hooks.zig").Keep = null,
    h: u64 = 100,
    x: u64 = 200,
    logits: u64 = 300,
    final_norm: u64 = 400,
    head: u64 = 500,
    stream: @import("cuda").Stream = undefined,
    norms: usize = 0,
    heads: usize = 0,
    src: u64 = 0,
    pub fn norm(r: *@This(), residual: bool, weight: u64, rows: usize) !void {
        try t.expect(residual and weight == r.final_norm and rows == 3);
        r.norms += 1;
    }
    pub fn normTail(r: *@This(), weight: u64, total: usize, heads: usize) !void {
        try t.expect(weight == r.final_norm and total == 3 and heads == 1);
        r.norms += heads;
    }
    pub fn proj(r: *@This(), weight: u64, src: u64, dst: u64, rows: usize, prompt: bool) !void {
        try t.expect(weight == r.head and dst == r.logits and rows == 1 and !prompt);
        r.heads += 1;
        r.src = src;
    }
};
test "dense read cost: hidden taps skip only the output tail and a scoring pass retains it" {
    var r: Recorder = .{};
    try @import("cuda_forward.zig").finish(&r, 3, 1, true);
    try t.expectEqual(@as(usize, 0), r.norms);
    try t.expectEqual(@as(usize, 0), r.heads);
    try @import("cuda_forward.zig").finish(&r, 3, 1, false);
    try t.expectEqual(@as(usize, 1), r.norms);
    try t.expectEqual(@as(usize, 1), r.heads);
    try t.expectEqual(@as(u64, 456), r.src);
}

const cuda = @import("cuda");
const NormRecorder = struct {
    var active: *@This() = undefined;
    affine: struct { pdl: bool = false } = .{},
    h_window: ?cuda.abi.AccessPolicyWindow = null,
    stream: cuda.Stream = undefined,
    g: struct { hidden: usize = 5120, eps: f32 = 1e-6 } = .{},
    int8: struct { on: bool = false } = .{},
    fns: struct { norm: cuda.Function } = undefined,
    tables: struct { ptr: u64 = 0x80000 } = .{},
    h: u64 = 0x10000,
    r: u64 = 0x20000,
    x: u64 = 0x30000,
    xs: u64 = 0x40000,
    rows: usize = 0,
    pointers: [4]u64 = @splat(0),
    fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const r = active;
        if (bx != 256) return 1;
        const a = raw.?;
        r.rows = gx;
        inline for (.{ 0, 1, 3, 7 }, 0..) |at, i| r.pointers[i] = @as(*const u64, @ptrCast(@alignCast(a[at].?))).*;
        return 0;
    }
    pub fn runAfter(r: *@This(), f: cuda.Function, grid: [3]usize, threads: [3]usize, args: anytype) !void {
        try t.expectEqual(r.fns.norm.handle, f.handle);
        try t.expectEqual([3]usize{ 256, 1, 1 }, threads);
        try t.expectEqual(@as(usize, 1), grid[1]);
        try t.expectEqual(@as(usize, 1), grid[2]);
        try t.expectEqual(@as(u64, 0x50000), args[2]);
        try t.expectEqual(r.g.eps, args[4]);
        try t.expectEqual(@as(c_int, 5120), args[5]);
        try t.expectEqual(r.tables.ptr + engine.Engine.params_at, args[6]);
        r.rows = grid[0];
        r.pointers = .{ args[0], args[1], args[3], args[7] };
    }
};

test "dense read cost: prompt sums retain int8 and native boundary consumers" {
    const norm = @import("cuda_norm.zig");
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = NormRecorder.launch;
    var r: NormRecorder = .{};
    r.fns = .{ .norm = .{ .d = &driver, .handle = @ptrFromInt(7) } };
    r.stream = .{ .d = &driver, .handle = null };
    NormRecorder.active = &r;
    for ([_]bool{ false, true }) |int8| for ([_]bool{ false, true }) |prompt| {
        r.int8.on = int8;
        for ([_]usize{ 1, 16, 17, 496 }) |rows| for ([_]bool{ false, true }) |residual| {
            try norm.normInput(&r, residual, 0x50000, rows, prompt);
            try t.expectEqual(rows, r.rows);
            try t.expectEqual([4]u64{ r.h, if (residual) r.r else 0, r.x, if (prompt and rows > 16 and !int8) 0 else r.xs }, r.pointers);
        };
    };
    try norm.normTail(&r, 0x50000, 496, 1);
    const offset = 495 * 5120 * 2;
    try t.expectEqual(@as(usize, 1), r.rows);
    try t.expectEqual([4]u64{ r.h + offset, r.r + offset, r.x + offset, 0 }, r.pointers);
}
