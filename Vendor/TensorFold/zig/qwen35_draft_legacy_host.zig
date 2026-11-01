const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const lanes = @import("lanes");
const P = q.draft.precision;
const T = std.testing;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
fn function(d: *cuda.Driver) cuda.Function { return .{ .d = d, .handle = @ptrFromInt(1) }; }
const AbsorbRecord = struct {
    var launches: usize = 0;
    var sums: usize = 0;
    var outputs: [16]u64 = undefined;
    var count: usize = 0;
    var puts: usize = 0;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        if (@intFromPtr(f) == 3) puts += 1;
        if (@intFromPtr(f) == 4) sums += 1;
        if (@intFromPtr(f) != 2) return 0;
        const a = args.?;
        const x = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
        if (x != 10000) return 0; // the tap projection precedes the shared normalized K/V input
        const xs = @as(*const u64, @ptrCast(@alignCast(a[1].?))).*;
        std.debug.assert(xs == 20000);
        const parts = @as(*const cuda.qlinear.Many, @ptrCast(@alignCast(a[2].?))).*;
        launches += 1;
        for (parts.p[0..@intCast(parts.count)]) |part| {
            outputs[count] = part.out;
            count += 1;
        }
        return 0;
    }
};

test "Draft frames: legacy live absorption groups all K/V outputs with disjoint row strides and no duplicate sums" {
    _ = setenv("TF_DRAFT_GROUP_LIVE", "1", 1);
    defer _ = unsetenv("TF_DRAFT_GROUP_LIVE");
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = AbsorbRecord.launch;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.fns.sums = .{ .d = &driver, .handle = @ptrFromInt(4) };
    e.fns.norm = function(&driver);
    e.affine.gemv_fn = function(&driver);
    e.affine.many_fn = .{ .d = &driver, .handle = @ptrFromInt(2) };
    e.affine.gemv_blocks = 96;
    e.affine.group_fn = .{ .d = &driver, .handle = @ptrFromInt(2) };
    e.affine.split = null;
    e.affine.narrow = null;
    e.affine.cluster_decode = true;
    e.affine.pdl = false;
    e.affine.wide_verify = false;
    var d: q.draft.Draft = undefined;
    d.gpa = T.allocator;
    d.io = T.io;
    d.e = &e;
    d.streams = .empty;
    defer d.streams.deinit(T.allocator);
    d.c = .{ .hidden = 5120, .intermediate = 17408, .layers = 1, .vocab = 256, .heads = 32, .kv_heads = 8, .head_dim = 128, .block = 16, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
    d.fns.gather_taps = function(&driver);
    d.fns.norm_rope = function(&driver);
    d.fns.ring_put = .{ .d = &driver, .handle = @ptrFromInt(3) };
    d.taps_in = 60000;
    d.th = 70000;
    d.thn = 10000;
    d.xs = 20000;
    d.kabs = 30000;
    d.vabs = 40000;
    d.kn = 50000;
    d.w.hidden_norm = 1;
    d.w.fc = .{ .w = 1, .s = 2, .b = 3, .n = 256, .k = 256, .npad = 256 };
    var layers: [8]@TypeOf(d.w.layers[0]) = undefined;
    for (&layers) |*layer| {
        layer.k = .{ .w = 1, .s = 2, .b = 3, .n = 1024, .k = 5120, .npad = 1024 };
        layer.v = layer.k;
        layer.k_norm = 1;
    }
    var ctx: q.draft.Ctx = .{ .ring = .{ .d = &driver, .ptr = 1, .len = 1 } };
    var s = try lanes.Stream.init(T.allocator, .{ .id = "absorb", .prompt = &.{1}, .max_new = 32 });
    defer s.deinit(T.allocator);
    try d.streams.put(T.allocator, &s, &ctx);
    const follow: [16]u32 = @splat(1);
    for ([_]usize{ 1, 2, 3, 4, 8 }) |n_layers| for ([_]usize{ 1, 2, 16 }) |rows| {
        d.w.layers = layers[0..n_layers];
        d.projection_role = .proposal;
        AbsorbRecord.launches = 0;
        AbsorbRecord.sums = 0;
        AbsorbRecord.count = 0;
        AbsorbRecord.puts = 0;
        try d.drafter().absorb(&.{.{ .stream = &s, .features = .{ .buffer = 1, .offset = 0, .rows = @intCast(rows), .row_bytes = 10240, .space = .device, .dtype = .bf16 }, .start = 0, .follow = follow[0..rows], .prompt = false }});
        try T.expectEqual((n_layers * 2 + 3) / 4, AbsorbRecord.launches);
        try T.expectEqual(n_layers * 2, AbsorbRecord.count);
        try T.expectEqual(n_layers, AbsorbRecord.puts);
        try T.expectEqual(@as(usize, 1), AbsorbRecord.sums); // only the initial tap projection needs a sums launch
        for (0..n_layers) |i| {
            try T.expectEqual(@as(u64, 30000 + i * rows * 1024 * 2), AbsorbRecord.outputs[i * 2]);
            try T.expectEqual(@as(u64, 40000 + i * rows * 1024 * 2), AbsorbRecord.outputs[i * 2 + 1]);
        }
        try T.expectEqual(P.Role.proposal, d.projection_role);
        try T.expect(e.affine.cluster_decode);
    };
}

const MlpRecord = struct {
    var sums: usize = 0;
    var pairs: usize = 0;
    var activations: usize = 0;
    var downs: usize = 0;
    var down_sums: u64 = 0;
    fn ptr(a: [*]?*anyopaque, i: usize) u64 { return @as(*const u64, @ptrCast(@alignCast(a[i].?))).*; }
    fn record(f: cuda.abi.Function, args: ?[*]?*anyopaque) cuda.abi.Result {
        const id = @intFromPtr(f);
        const a = args.?;
        if (id == 1) sums += 1;
        if (id == 4) activations += 1;
        if (id == 5) {
            pairs += 1;
            std.debug.assert(ptr(a, 0) == 100000 and ptr(a, 1) == 200000);
            std.debug.assert(ptr(a, 3) == 400000 and ptr(a, 4) == 300000);
            const parts = @as(*const cuda.qlinear.Parts, @ptrCast(@alignCast(a[2].?))).*;
            std.debug.assert(parts.count == 2 and parts.p[0].n == 17408 and parts.p[1].n == 17408);
        }
        if (id == 2) { downs += 1; down_sums = ptr(a, 1); }
        return 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result { return record(f, args); }
    fn launchEx(_: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result { return record(f, args); }
    fn reset() void { sums = 0; pairs = 0; activations = 0; downs = 0; down_sums = 0; }
};

test "Draft frames: paired MLP keeps input sums live and passes disjoint output sums to down, with wide fallback" {
    _ = setenv("TF_DRAFT_MLP_PAIR", "1", 1);
    defer _ = unsetenv("TF_DRAFT_MLP_PAIR");
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = MlpRecord.launch;
    driver.api.cuLaunchKernelEx = MlpRecord.launchEx;
    var e: q.engine.Engine = undefined;
    e.kit = null;
    e.d = &driver;
    e.stream = .{ .d = &driver, .handle = null };
    e.fns.sums = .{ .d = &driver, .handle = @ptrFromInt(1) };
    e.fns.mlp_act = .{ .d = &driver, .handle = @ptrFromInt(4) };
    e.affine.gemv_fn = .{ .d = &driver, .handle = @ptrFromInt(2) };
    e.affine.group_fn = .{ .d = &driver, .handle = @ptrFromInt(3) };
    e.affine.gemv_blocks = 96;
    e.affine.split = null;
    e.affine.narrow = null;
    e.affine.cluster_decode = false;
    e.affine.wide_verify = false;
    e.affine.pdl = false;
    e.mlp_pair = .{ .pair_fn = .{ .d = &driver, .handle = @ptrFromInt(5) }, .pdl = false };
    var d: q.draft.Draft = undefined;
    d.e = &e;
    d.c.intermediate = 17408;
    d.projection_role = .proposal;
    d.xin = 100000;
    d.xs = 200000;
    d.gate = 300000;
    d.act = 400000;
    d.up = 500000;
    d.o = 600000;
    var l: @TypeOf(d.w.layers[0]) = undefined;
    l.gate = .{ .w = 1, .s = 2, .b = 3, .n = 17408, .k = 5120, .npad = 17408 };
    l.up = l.gate;
    l.down = .{ .w = 1, .s = 2, .b = 3, .n = 5120, .k = 17408, .npad = 5120 };
    for ([_]usize{ 1, 8, 16 }) |rows| {
        MlpRecord.reset();
        try q.draft.frame.mlp(&d, l, rows, null);
        try T.expectEqual(@as(usize, 1), MlpRecord.sums);
        try T.expectEqual(@as(usize, 1), MlpRecord.pairs);
        try T.expectEqual(@as(usize, 0), MlpRecord.activations);
        try T.expectEqual(@as(usize, 1), MlpRecord.downs);
        try T.expectEqual(d.gate, MlpRecord.down_sums);
    }
    MlpRecord.reset();
    try q.draft.frame.mlp(&d, l, 32, null);
    try T.expectEqual(@as(usize, 0), MlpRecord.pairs);
    try T.expectEqual(@as(usize, 1), MlpRecord.activations);
    try T.expectEqual(@as(usize, 2), MlpRecord.downs);
    try T.expectEqual(d.xs + 16 * (17408 / 64) * 4, MlpRecord.down_sums);
    _ = setenv("TF_DRAFT_MLP_PAIR", "0", 1);
    MlpRecord.reset();
    try q.draft.frame.mlp(&d, l, 16, null);
    try T.expectEqual(@as(usize, 0), MlpRecord.pairs);
    try T.expectEqual(@as(usize, 1), MlpRecord.activations);
    try T.expectEqual(d.xs, MlpRecord.down_sums);
}
