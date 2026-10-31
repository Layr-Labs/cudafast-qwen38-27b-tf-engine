//! The real 4-bit verify projection dispatch through a recording CUDA driver.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const t = std.testing;
pub const Trace = struct {
    const Call = struct { id: usize, x: u64, xs: u64, out: u64, rows: c_int, sk: c_int, row_tiles: c_int, grid: u32, shared: u32, parts: c_int = 1, weight: u64 = 0 };
    pub var calls: [128]Call = undefined;
    pub var count: usize = 0;
    pub var prompts: usize = 0;
    pub var last_parts: cuda.qlinear.Parts = undefined;
    fn record(f: cuda.abi.Function, gx: u32, shared: u32, args: ?[*]?*anyopaque) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id == 4 or id == 15) {
            prompts += 1;
            return 0;
        }
        if (id == 14) {
            const parts = @as(*const cuda.prompt_many.Parts, @ptrCast(@alignCast(args.?[1].?))).*;
            prompts += @intCast(parts.count);
            return 0;
        }
        if (id == 1) return 0; // the input-sum producer
        const a = args.?;
        const x = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
        const xs = @as(*const u64, @ptrCast(@alignCast(a[1].?))).*;
        const rows = @as(*const c_int, @ptrCast(@alignCast(a[3].?))).*;
        var out: u64 = 0;
        var sk: c_int = 0;
        var tiles: c_int = 1;
        var part_count: c_int = 1;
        var weight: u64 = 0;
        if (id == 8) {
            const segs = @as(*const cuda.qlinear.Segs, @ptrCast(@alignCast(a[2].?))).*;
            for (segs.s[0..@intCast(segs.count)]) |seg| {
                calls[count] = .{ .id = id, .x = x, .xs = xs, .out = seg.p.out, .rows = rows, .sk = seg.p.sk, .row_tiles = 1, .grid = gx, .shared = shared };
                count += 1;
            }
            return 0;
        }
        if (id == 2) {
            const p = @as(*const cuda.qlinear.GemvPart, @ptrCast(@alignCast(a[2].?))).*;
            out = p.out;
            sk = p.sk;
            weight = p.w;
        } else if (id == 3 or id == 5 or id == 6 or id == 7 or id == 10 or id == 11 or id == 12) {
            const p = @as(*const cuda.qlinear.Parts, @ptrCast(@alignCast(a[2].?))).*;
            last_parts = p;
            out = p.p[0].out;
            part_count = p.count;
            weight = p.p[0].w;
            sk = p.p[0].sk;
            tiles = @as(*const c_int, @ptrCast(@alignCast(a[6].?))).*;
        } else return 0;
        calls[count] = .{ .id = id, .x = x, .xs = xs, .out = out, .rows = rows, .sk = sk, .row_tiles = tiles, .grid = gx, .shared = shared, .parts = part_count, .weight = weight };
        count += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return record(f, gx, shared, args);
    }
    fn launchEx(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return record(f, c.grid_x, c.shared_bytes, args);
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
pub fn engine(d: *cuda.Driver, gb10: bool) q.engine.Engine {
    return device(d, gb10, false);
}
fn device(d: *cuda.Driver, gb10: bool, wide: bool) q.engine.Engine {
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuLaunchKernelEx = Trace.launchEx;
    var e: q.engine.Engine = undefined;
    e.int8 = .{};
    e.kit = null;
    e.d = d;
    e.stream = .{ .d = d, .handle = null };
    e.fns.sums = function(d, 1);
    e.affine.gemv_fn = function(d, 2);
    e.affine.group_fn = function(d, 3);
    e.affine.stream16_fns = @splat(null);
    e.affine.gdn_stream16_fn = null;
    e.affine.stream16_dict_fn = null;
    e.affine.gdn_stream16_dict_fn = null;
    e.affine.gdn_stream32_fn = null;
    e.affine.gdn_stream32_dict_fn = null;
    e.affine.dict = null;
    e.affine.short_lane = false;
    e.affine.sms = 48;
    e.affine.prefill_fn = function(d, 4);
    e.affine.prefill22_fn = function(d, 4);
    e.prompt_many = .{ .many_fn = function(d, 14), .mlp_fn = function(d, 4), .split_fn = function(d, 15), .slots = 96 };
    e.affine.half_prompt = false;
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify_fn")) e.affine.verify_fn = function(d, 5);
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify64_fn")) e.affine.verify64_fn = function(d, 6);
    if (@hasField(cuda.qlinear.Affine4Kernels, "verify128_fn")) e.affine.verify128_fn = function(d, 7);
    e.affine.gemv_blocks = 1;
    e.affine.split = null;
    e.affine.pdl = gb10;
    e.affine.group_wide = false;
    e.affine.wide_fns = @splat(null);
    e.affine.cluster_decode = false;
    e.affine.narrow = null;
    e.affine.wide_verify = wide;
    return e;
}
fn project(e: *q.engine.Engine, rows: usize, n: usize) !void {
    Trace.count = 0;
    const p: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = n, .k = 8192, .npad = n } };
    try e.projSums(p, 0x10000, 0x20000, 0x30000, rows, false, 0);
}

test "verify matmuls: up to 16 rows already share one weight pass" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for ([_]usize{ 1, 2, 8, 15, 16 }) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(rows, @as(usize, @intCast(Trace.calls[0].rows)));
    }
}

test "verify matmuls: 17 through 32 GB10 rows share one weight pass and keep the original K slices" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for (17..33) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        const c = Trace.calls[0];
        try t.expectEqual(@as(usize, 5), c.id);
        try t.expectEqual(rows, @as(usize, @intCast(c.rows)));
        try t.expectEqual(@as(c_int, @intCast(cuda.qlinear.splitK(4096, 8192))), c.sk);
        try t.expectEqual(@as(c_int, 1), c.row_tiles);
        try t.expectEqual(@as(u32, 32640), c.shared);
        try t.expectEqual(@as(u32, @intCast(64 * c.sk)), c.grid);
    }
}

test "verify matmuls: 33 through 128 GB10 rows share one weight pass in four or eight row tiles" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for (33..129) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        const c = Trace.calls[0];
        try t.expectEqual(@as(usize, if (rows <= 64) 6 else 7), c.id);
        try t.expectEqual(rows, @as(usize, @intCast(c.rows)));
        try t.expectEqual(@as(u64, 0x10000), c.x);
        try t.expectEqual(@as(u64, 0x30000), c.out);
        try t.expectEqual(@as(c_int, @intCast(cuda.qlinear.splitK(4096, 8192))), c.sk);
        try t.expectEqual(@as(u32, if (rows <= 64) 43008 else 57600), c.shared);
        try t.expectEqual(@as(u32, @intCast(64 * c.sk)), c.grid);
    }
}

test "verify matmuls: graph row buckets and segment counts do not multiply projection launches" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    for ([_]usize{ 1, 2, 17 }) |segments| {
        const shape = q.engine.Engine.Shape.init(17, segments, true, false, 1);
        try project(&e, shape.total, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(@as(c_int, 32), Trace.calls[0].rows);
    }
    // Large output matrices keep splitK=1 and still read one panel for all 32 rows.
    try project(&e, 32, 17408);
    try t.expectEqual(@as(usize, 1), Trace.count);
    try t.expectEqual(@as(c_int, 1), Trace.calls[0].sk);
}

test "verify matmuls: sm_120 takes the 32-row tile without programmatic dependent launch" {
    var d: cuda.Driver = undefined;
    var e = device(&d, false, true);
    for (17..33) |rows| {
        try project(&e, rows, 4096);
        try t.expectEqual(@as(usize, 1), Trace.count);
        try t.expectEqual(@as(usize, 5), Trace.calls[0].id);
        try t.expectEqual(rows, @as(usize, @intCast(Trace.calls[0].rows)));
    }
    try project(&e, 16, 4096);
    try t.expectEqual(@as(usize, 2), Trace.calls[0].id);
}

test "verify matmuls: other devices retain the original sixteen-row dispatch" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, false);
    try project(&e, 32, 4096);
    try t.expectEqual(@as(usize, 2), Trace.count);
    try t.expectEqual(@as(usize, 2), Trace.calls[0].id);
    try t.expectEqual(@as(u64, 0x30000 + 16 * 4096 * 2), Trace.calls[1].out);
}

const Resolve = struct {
    fn alloc(out: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        out.* = 0x100000;
        return 0;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn memset(_: cuda.abi.DevicePtr, _: c_uint, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    var requested = false;
    var shared: [3]c_int = @splat(0); // verify32, verify64, verify128
    fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const s = cuda.qlinear.symbols;
        const n = std.mem.span(name);
        const id: usize = if (std.mem.eql(u8, n, s.verify32)) 5 else if (std.mem.eql(u8, n, s.verify64)) 6 else if (std.mem.eql(u8, n, s.verify128)) 7 else 2;
        out.* = @ptrFromInt(id);
        requested = requested or @intFromPtr(out.*) == 5;
        return 0;
    }
    fn attribute(f: cuda.abi.Function, what: cuda.abi.FunctionAttribute, value: c_int) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id >= 5 and id <= 7 and what == .max_dynamic_shared_size_bytes) shared[id - 5] = value;
        return 0;
    }
    fn occupancy(out: *c_int, _: cuda.abi.Function, _: c_int, _: usize) callconv(.c) cuda.abi.Result {
        out.* = 2;
        return 0;
    }
};

test "verify matmuls: real module resolution sets the new function's shared-memory allowance" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Resolve.alloc;
    d.api.cuMemFree_v2 = Resolve.free;
    d.api.cuMemsetD32_v2 = Resolve.memset;
    d.api.cuModuleGetFunction = Resolve.get;
    d.api.cuFuncSetAttribute = Resolve.attribute;
    d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor = Resolve.occupancy;
    const m: cuda.Module = .{ .d = &d, .handle = null };
    Resolve.requested = false;
    Resolve.shared = @splat(0);
    var a = try cuda.qlinear.Affine4Kernels.resolve(&d, m, m, m, m, 48, true);
    defer a.deinit();
    try t.expect(Resolve.requested);
    try t.expectEqual([3]c_int{ 32640, 43008, 57600 }, Resolve.shared);
    try t.expectEqual(@as(usize, 96), a.gemv_blocks);
    try t.expectEqual(@as(usize, 128), a.verifyRows());
}

const WidePack = struct {
    var count: usize = 0;
    var size: usize = 0;
    var synced = false;
    var original: [8]u64 = undefined;
    var dest: [8]u64 = undefined;
    fn alloc(out: *cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
        size = n;
        out.* = 0x100000;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        synced = true;
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const a = args.?;
        original[count] = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
        dest[count] = @as(*const u64, @ptrCast(@alignCast(a[1].?))).*;
        count += 1;
        return 0;
    }
};

test "wide weights: one owned view keeps narrow and prompt consumers intact" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    d.api.cuMemAlloc_v2 = WidePack.alloc;
    d.api.cuMemFree_v2 = Resolve.free;
    d.api.cuModuleGetFunction = Resolve.get;
    d.api.cuFuncSetAttribute = Resolve.attribute;
    d.api.cuStreamSynchronize = WidePack.sync;
    d.api.cuLaunchKernel = WidePack.launch;
    e.mods[0] = .{ .d = &d, .handle = null };
    e.mods[3] = .{ .d = &d, .handle = null };
    e.head = .{ .a4 = .{ .w = 1, .s = 2, .b = 3, .n = 256, .npad = 256, .k = 5120 } };
    const down: q.engine.Proj = .{ .a4 = .{ .w = 2, .s = 3, .b = 4, .n = 5120, .npad = 5120, .k = 17408 } };
    const input: q.engine.Proj = .{ .a4 = .{ .w = 3, .s = 4, .b = 5, .n = 128, .npad = 128, .k = 5120 } };
    var key = input;
    key.a4.w = 4;
    var value = input;
    value.a4.w = 5;
    var layers: [1]q.engine.Layer = undefined;
    layers[0].mlp = .{ .dense = .{ .gate = input, .up = input, .down = down } };
    layers[0].mixer = .{ .attention = .{ .q = input, .k = key, .v = value, .o = input, .q_norm = 0, .k_norm = 0 } };
    e.layers = &layers;
    WidePack.count = 0;
    WidePack.synced = false;
    var buffer = (try q.engine.wide_weight_layout.prepare(&e)).?;
    defer buffer.free();
    try t.expectEqual(@as(usize, 5), WidePack.count);
    try t.expectEqual(@as(usize, (256 * 5120 + 5120 * 17408 + 3 * 128 * 5120) / 2), WidePack.size);
    try t.expectEqualSlices(u64, &.{ 1, 2, 3, 4, 5 }, WidePack.original[0..5]);
    try t.expect(WidePack.synced);
    try t.expectEqual(@as(u64, 1), e.head.a4.w);
    try t.expectEqual(@as(u64, 2), layers[0].mlp.dense.down.a4.w);
    try t.expectEqual(@as(u64, 0), layers[0].mlp.dense.gate.a4.wide_w);
    try t.expectEqual(@as(u64, 0), layers[0].mixer.attention.o.a4.wide_w);
    try t.expectEqual(@as(u8, 3), e.head.a4.wide_kind);
    try t.expectEqual(@as(u8, 2), layers[0].mlp.dense.down.a4.wide_kind);
    try t.expectEqual(@as(u8, 1), layers[0].mixer.attention.q.a4.wide_kind);
}

test "wide dispatch: original grouped projections preserve each split count" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    e.affine.group_wide = true;
    const large: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = 10240, .npad = 10240, .k = 5120 } };
    var mid = large;
    mid.a4.n = 6144;
    mid.a4.npad = 6144;
    var small = large;
    small.a4.n = 48;
    small.a4.npad = 128;
    Trace.count = 0;
    try e.projMany(&.{ large, mid, small, small }, 0x1000, &.{ 0x2000, 0x3000, 0x4000, 0x5000 }, 32, false);
    try t.expectEqual(@as(usize, 2), Trace.count);
    try t.expectEqual(@as(c_int, 2), Trace.calls[0].parts);
    try t.expectEqual(@as(c_int, 2), Trace.calls[0].sk);
    try t.expectEqual(@as(u32, 512), Trace.calls[0].grid);
    try t.expectEqual(@as(c_int, 2), Trace.calls[1].parts);
    try t.expectEqual(@as(c_int, 8), Trace.calls[1].sk);
    try t.expectEqual(@as(u32, 16), Trace.calls[1].grid);
}

test "wide dispatch: alternate words are used only at 17 to 32 rows" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    e.affine.wide_fns = .{ function(&d, 10), function(&d, 11), function(&d, 12) };
    const ns = [_]usize{ 12288, 5120, 248320 };
    const ks = [_]usize{ 5120, 17408, 5120 };
    const grids = [_]u32{ 96, 80, 1940 };
    const shared = [_]u32{ 26496, 32768, 17664 };
    for (ns, ks, grids, shared, 0..) |n, k, grid, smem, i| {
        const p: q.engine.Proj = .{ .a4 = .{ .w = 10, .wide_w = 0x9000, .wide_kind = @intCast(i + 1), .s = 20, .b = 30, .n = n, .npad = n, .k = k } };
        for ([_]usize{ 17, 24, 31, 32 }) |rows| {
            Trace.count = 0;
            try e.projSums(p, 0x1000, 0x2000, 0x3000, rows, false, 0);
            try t.expectEqual(@as(usize, 1), Trace.count);
            try t.expectEqual(@as(usize, 10) + i, Trace.calls[0].id);
            try t.expectEqual(@as(u64, 0x9000), Trace.calls[0].weight);
            try t.expectEqual(grid, Trace.calls[0].grid);
            try t.expectEqual(smem, Trace.calls[0].shared);
        }
        Trace.count = 0;
        try e.projSums(p, 0x1000, 0x2000, 0x3000, 16, false, 0);
        try t.expectEqual(@as(u64, 10), Trace.calls[0].weight);
        try t.expect(Trace.calls[0].id < 10);
    }
}

test "attention mixed launch: Q and K/V keep the original ordered slices and weights" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    e.affine.cluster_decode = true;
    e.xs = 0x8000;
    const query: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = 12288, .npad = 12288, .k = 5120 } };
    var key = query; key.a4.n = 1024; key.a4.npad = 1024; key.a4.w = 11;
    var value = key; value.a4.w = 12;
    Trace.count = 0;
    try e.projMany(&.{query, key, value}, 0x1000, &.{0x2000, 0x3000, 0x4000}, 16, false);
    try t.expectEqual(@as(usize, 1), Trace.count);
    try t.expectEqual(@as(usize, 3), Trace.calls[0].id);
    try t.expectEqual(@as(u32, 448), Trace.calls[0].grid);
    try t.expectEqual(@as(c_int, 3), Trace.last_parts.count);
    for (Trace.last_parts.p[0..3], [_]c_int{1, 8, 8}, [_]u64{10, 11, 12}, [_]c_int{0, 24, 40}) |part, sk, words, first| {
        try t.expectEqual(sk, part.sk);
        try t.expectEqual(words, part.w);
        try t.expectEqual(first, part.first);
    }
}

const PackTrace = struct {
    var count: usize = 0;
    var bytes: usize = 0;
    var synced = false;
    var touched = false;
    var original: [2]u64 = undefined;
    var alternate: [2]u64 = undefined;
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result { return 0; }
    fn alloc(out: *cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
        bytes = n; out.* = 0x100000; return 0;
    }
    fn memset(ptr: cuda.abi.DevicePtr, value: u8, n: usize) callconv(.c) cuda.abi.Result {
        touched = ptr == 0x100000 and value == 0 and n == bytes;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result { synced = true; return 0; }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const a = args.?;
        original[count] = @as(*const u64, @ptrCast(@alignCast(a[0].?))).*;
        alternate[count] = @as(*const u64, @ptrCast(@alignCast(a[1].?))).*;
        count += 1; return 0;
    }
};

const PairTrace = struct {
    var id: usize = 0;
    var grid: u32 = 0;
    var shared: u32 = 0;
    var parts: cuda.qlinear.Parts = undefined;
    var rows: c_int = 0;
    var sums: u64 = 0;
    var cluster: u32 = 1;
    fn launch(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        id = @intFromPtr(f);
        grid = c.grid_x;
        shared = c.shared_bytes;
        cluster = 1;
        for (c.attrs.?[0..c.num_attrs]) |attr| {
            if (attr.id == .cluster_dimension) cluster = attr.value.cluster_dim.x;
        }
        const a = args.?;
        parts = @as(*const cuda.qlinear.Parts, @ptrCast(@alignCast(a[2].?))).*;
        sums = @as(*const u64, @ptrCast(@alignCast(a[4].?))).*;
        rows = @as(*const c_int, @ptrCast(@alignCast(a[5].?))).*;
        return 0;
    }
};

test "fused MLP: alternate words require both views and exactly sixteen rows" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernelEx = PairTrace.launch;
    var pair: cuda.mlp_pair.Kernels = .{ .pair_fn = function(&d, 20), .kmajor_fn = function(&d, 21), .pdl = true };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const gate: cuda.qlinear.Affine4 = .{ .w = 10, .verify_w = 100, .s = 20, .b = 30, .n = 17408, .k = 5120, .npad = 17408 };
    var up = gate;
    up.w = 40; up.verify_w = 400;
    for ([_]usize{ 1, 15, 16, 17, 32 }) |rows| {
        try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, rows);
        const major = rows == 16;
        try t.expectEqual(@as(usize, if (major) 21 else 20), PairTrace.id);
        try t.expectEqual(@as(u32, @intCast(2 * ((rows + 15) / 16) * (17408 / @as(usize, if (major) 128 else 64)))), PairTrace.grid);
        try t.expectEqual(@as(u32, if (major) 17536 else 37376), PairTrace.shared);
        try t.expectEqual(@as(u64, if (major) 100 else 10), PairTrace.parts.p[0].w);
        try t.expectEqual(@as(u64, if (major) 400 else 40), PairTrace.parts.p[1].w);
        try t.expectEqual(@as(u64, 20), PairTrace.parts.p[0].scales);
        try t.expectEqual(@as(u64, 30), PairTrace.parts.p[0].biases);
        try t.expectEqual(@as(u64, 4000), PairTrace.sums);
        try t.expectEqual(@as(c_int, @intCast(rows)), PairTrace.rows);
    }
    up.verify_w = 0;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 20), PairTrace.id);
    try t.expectEqual(@as(u64, 10), PairTrace.parts.p[0].w);
    up.verify_w = 400; pair.kmajor_fn = null;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 20), PairTrace.id);
}

test "compact MLP: sixteen pair-coded rows use both half panels and preserve fallbacks" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernelEx = PairTrace.launch;
    var pair: cuda.mlp_pair.Kernels = .{
        .pair_fn = function(&d, 20), .kmajor_fn = function(&d, 21), .pdl = true,
        .kmajor_single = true, .single_dict_fn = function(&d, 22), .compact_pair_fn = function(&d, 23),
    };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const code = cuda.qlinear.dict_kmajor | (1000 << 5) | 10;
    const gate: cuda.qlinear.Affine4 = .{ .w = 10, .verify_w = 100, .s = 20, .b = 30,
        .sb = 40, .tab = 50, .code = code, .n = 17408, .k = 5120, .npad = 17408 };
    var up = gate; up.verify_w = 400;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 23), PairTrace.id);
    try t.expectEqual(@as(u32, 136), PairTrace.grid);
    try t.expectEqual(@as(u32, 17536), PairTrace.shared);
    try t.expectEqual(@as(u32, 1), PairTrace.cluster);
    try t.expectEqual(@as(c_int, 136), PairTrace.parts.p[0].tiles);
    try t.expectEqual(@as(u64, 100), PairTrace.parts.p[0].w);
    try t.expectEqual(@as(u64, 400), PairTrace.parts.p[1].w);
    try t.expectEqual(@as(u64, 40), PairTrace.parts.p[0].scales);
    try t.expectEqual(@as(u64, 50), PairTrace.parts.p[0].biases);
    try t.expectEqual(@as(u32, code & ~cuda.qlinear.dict_kmajor), @as(u32, @bitCast(PairTrace.parts.p[0].first)));
    up.code = cuda.qlinear.dict_kmajor | (120 << 18) | (100 << 5) | 8;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 22), PairTrace.id);
    try t.expectEqual(@as(u32, 68), PairTrace.grid);
    up = gate; pair.compact_pair_fn = null;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 22), PairTrace.id);
    try t.expectEqual(@as(u32, 68), PairTrace.grid);
    pair.compact_pair_fn = function(&d, 23);
    for ([_]usize{ 1, 15, 17, 32 }) |rows| {
        try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, rows);
        try t.expectEqual(@as(usize, 20), PairTrace.id);
    }
}

test "verify weights: preparation owns one weight-only view and retains original consumers" {
    for ([_]bool{ false, true }) |single| {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    d.api.cuMemAlloc_v2 = PackTrace.alloc;
    d.api.cuMemFree_v2 = PackTrace.free;
    d.api.cuModuleGetFunction = Resolve.get;
    d.api.cuFuncSetAttribute = Resolve.attribute;
    d.api.cuStreamSynchronize = PackTrace.sync;
    d.api.cuLaunchKernel = PackTrace.launch;
    e.mods[0] = .{ .d = &d, .handle = null };
    e.mods[3] = .{ .d = &d, .handle = null };
    const gate: q.engine.Proj = .{ .a4 = .{ .w = 10, .s = 20, .b = 30, .n = 17408, .k = 5120, .npad = 17408 } };
    var up = gate; up.a4.w = 40;
    var layers: [1]q.engine.Layer = undefined;
    layers[0].mlp = .{ .dense = .{ .gate = gate, .up = up, .down = gate } };
    e.layers = &layers;
    PackTrace.count = 0; PackTrace.synced = false;
    var buffer = (try q.engine.verify_weight_layout.prepare(&e, single)).?;
    defer buffer.free();
    try t.expectEqual(@as(usize, 2), PackTrace.count);
    try t.expectEqual(@as(usize, 17408 * 5120), PackTrace.bytes);
    try t.expectEqual([2]u64{10, 40}, PackTrace.original);
    try t.expectEqual([2]u64{buffer.ptr, buffer.ptr + 17408 * 5120 / 2}, PackTrace.alternate);
    try t.expect(PackTrace.synced);
    try t.expect(e.mlp_pair.kmajor_fn != null);
    try t.expectEqual(single, e.mlp_pair.kmajor_single);
    try t.expectEqual(gate.a4.w, layers[0].mlp.dense.gate.a4.w);
    try t.expectEqual(up.a4.w, layers[0].mlp.dense.up.a4.w);
    try t.expectEqual(PackTrace.alternate[1], layers[0].mlp.dense.up.a4.verify_w);
    try t.expectEqual(@as(u64, 0), layers[0].mlp.dense.down.a4.verify_w);
    }
}

test "fused MLP: single-CTA variant has no cluster and preserves narrow fallback" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernelEx = PairTrace.launch;
    var pair: cuda.mlp_pair.Kernels = .{ .pair_fn = function(&d, 20), .kmajor_fn = function(&d, 22), .kmajor_single = true, .pdl = true };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const gate: cuda.qlinear.Affine4 = .{ .w = 10, .verify_w = 100, .s = 20, .b = 30, .n = 17408, .k = 5120, .npad = 17408 };
    var up = gate;
    up.w = 40; up.verify_w = 400;
    for ([_]usize{ 1, 15, 16, 17, 32 }) |rows| {
        try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, rows);
        const single = rows == 16;
        try t.expectEqual(@as(usize, if (single) 22 else 20), PairTrace.id);
        try t.expectEqual(@as(u32, if (single) 68 else @intCast(2 * ((rows + 15) / 16) * 272)), PairTrace.grid);
        try t.expectEqual(@as(u32, if (single) 1 else 2), PairTrace.cluster);
        try t.expectEqual(@as(u32, if (single) 41088 else 37376), PairTrace.shared);
        try t.expectEqual(@as(u64, if (single) 100 else 10), PairTrace.parts.p[0].w);
        try t.expectEqual(@as(u64, if (single) 400 else 40), PairTrace.parts.p[1].w);
    }
    up.verify_w = 0;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 20), PairTrace.id);
    try t.expectEqual(@as(u32, 2), PairTrace.cluster);
    up.verify_w = 400; pair.kmajor_fn = null;
    try pair.pair(stream, 1000, 2000, gate, up, 3000, 4000, 16);
    try t.expectEqual(@as(usize, 20), PairTrace.id);
}

const PromptPersistentTrace = struct {
    var id: usize = 0;
    var grid: u32 = 0;
    var block: u32 = 0;
    var shared: u32 = 0;
    var parts: cuda.prompt_many.Parts = undefined;
    var act: u64 = 0;
    var scalar: [7]c_int = undefined;
    fn value(comptime T: type, args: [*]?*anyopaque, index: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args[index].?))).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, bytes: c_uint, _: cuda.abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        id = @intFromPtr(f);
        grid = gx;
        block = bx;
        shared = bytes;
        if (id >= 40 and id <= 43) {
            const args = raw.?;
            parts = value(cuda.prompt_many.Parts, args, 1);
            act = value(u64, args, 2);
            for (&scalar, 0..) |*v, i| v.* = value(c_int, args, 3 + i);
        }
        return 0;
    }
    fn kernels(d: *cuda.Driver) cuda.prompt_many.Kernels {
        d.api.cuLaunchKernel = launch;
        return .{ .many_fn = function(d, 14), .mlp_fn = function(d, 4), .split_fn = function(d, 15), .slots = 96, .persistent_fns = .{ function(d, 40), function(d, 41), function(d, 42), function(d, 43) } };
    }
    fn matrix(n: usize, k: usize) cuda.qlinear.Affine4 {
        return .{ .w = 10, .s = 20, .b = 30, .n = n, .k = k, .npad = (n + 63) / 64 * 64 };
    }
};

test "prompt persistence: target MLP retains canonical parts and passes total tiles independently of its bounded grid" {
    var d: cuda.Driver = undefined;
    var kernels = PromptPersistentTrace.kernels(&d);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const gate = PromptPersistentTrace.matrix(17408, 5120);
    var up = gate;
    up.w = 40;
    for ([_]usize{ 1, 128, 255, 256, 496, 512, 513 }) |rows| {
        try kernels.mlp(stream, 1000, gate, up, 3000, rows);
        const selected = true;
        try t.expectEqual(@as(usize, if (selected) 40 else 4), PromptPersistentTrace.id);
        const total = (rows + 127) / 128 * 272;
        try t.expectEqual(@as(u32, @intCast(if (selected) @min(total, 96) else total)), PromptPersistentTrace.grid);
        try t.expectEqual(@as(u32, 128), PromptPersistentTrace.block);
        try t.expectEqual(@as(u32, 41984), PromptPersistentTrace.shared);
        if (selected) {
            try t.expectEqualSlices(c_int, &.{ @intCast(rows), 5120, 5120, @intCast(@min((rows + 127) / 128, 25)), -1, @intCast(total), 1 }, &PromptPersistentTrace.scalar);
            try t.expectEqual(@as(u64, 3000), PromptPersistentTrace.act);
            try t.expectEqual(@as(c_int, 2), PromptPersistentTrace.parts.count);
            try t.expectEqual(@as(u64, 10), PromptPersistentTrace.parts.p[0].w);
            try t.expectEqual(@as(u64, 40), PromptPersistentTrace.parts.p[1].w);
            try t.expectEqual(@as(u64, 20), PromptPersistentTrace.parts.p[0].scales);
        }
    }
    kernels.persistent_fns[0] = null;
    try kernels.mlp(stream, 1000, gate, up, 3000, 496);
    try t.expectEqual(@as(usize, 4), PromptPersistentTrace.id);
    kernels.persistent_fns[0] = function(&d, 40);
    kernels.slots = 80;
    try kernels.mlp(stream, 1000, gate, up, 3000, 496);
    try t.expectEqual(@as(usize, 4), PromptPersistentTrace.id);
}

test "prompt persistence: only measured projection roles select row rings and preserve split columns and part boundaries" {
    var d: cuda.Driver = undefined;
    var kernels = PromptPersistentTrace.kernels(&d);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]usize{ 6144, 17408 }) |k| {
        const outputs = [_]cuda.prompt_many.Out{.{ .q = PromptPersistentTrace.matrix(5120, k), .out = 3000 }};
        try kernels.prompt(stream, 1000, &outputs, 496);
        try t.expectEqual(@as(usize, 42), PromptPersistentTrace.id);
        try t.expectEqual(@as(u32, 96), PromptPersistentTrace.grid);
        try t.expectEqual(@as(c_int, 4096), PromptPersistentTrace.scalar[4]);
        try t.expectEqual(@as(c_int, 192), PromptPersistentTrace.scalar[5]);
        try t.expectEqual(@as(c_int, 1), PromptPersistentTrace.scalar[6]);
        try t.expectEqual(@as(u64, 3000), PromptPersistentTrace.parts.p[0].out);
        try t.expectEqual(@as(c_int, 1), PromptPersistentTrace.parts.count);
        try kernels.prompt(stream, 1000, &outputs, 128);
        try t.expect(PromptPersistentTrace.id == 14 or PromptPersistentTrace.id == 15);
    }
    const short = [_]cuda.prompt_many.Out{.{ .q = PromptPersistentTrace.matrix(5120, 6144), .out = 3000 }};
    try kernels.prompt(stream, 1000, &short, 256);
    try t.expectEqual(@as(usize, 42), PromptPersistentTrace.id);
    try t.expectEqual(@as(u32, 96), PromptPersistentTrace.grid);
    try t.expectEqual(@as(c_int, 0), PromptPersistentTrace.scalar[4]);
    try t.expectEqual(@as(c_int, 160), PromptPersistentTrace.scalar[5]);
    var recurrent: [4]cuda.prompt_many.Out = undefined;
    for (&recurrent, [_]usize{ 10240, 6144, 48, 48 }, 0..) |*o, n, i| o.* = .{ .q = PromptPersistentTrace.matrix(n, 5120), .out = 3000 + i * 100 };
    try kernels.prompt(stream, 1000, &recurrent, 496);
    try t.expectEqual(@as(usize, 43), PromptPersistentTrace.id);
    try t.expectEqual(@as(c_int, 520), PromptPersistentTrace.scalar[5]);
    for (PromptPersistentTrace.parts.p, [_]c_int{ 0, 320, 512, 516 }, recurrent) |p, first, o| {
        try t.expectEqual(first, p.first);
        try t.expectEqual(o.out, p.out);
        try t.expectEqual(@as(c_int, @intCast(o.q.n)), p.n);
        try t.expectEqual(@as(c_int, @intCast(o.q.npad)), p.npad);
    }
    recurrent[3].q.k = 6144;
    try t.expectError(error.InputMismatch, kernels.prompt(stream, 1000, &recurrent, 496));
    const attention = [_]cuda.prompt_many.Out{ .{ .q = PromptPersistentTrace.matrix(12288, 5120), .out = 3000 }, .{ .q = PromptPersistentTrace.matrix(1024, 5120), .out = 4000 }, .{ .q = PromptPersistentTrace.matrix(1024, 5120), .out = 5000 } };
    try kernels.prompt(stream, 1000, &attention, 496);
    try t.expectEqual(@as(usize, 43), PromptPersistentTrace.id);
    try t.expectEqual(@as(u32, 96), PromptPersistentTrace.grid);
    try t.expectEqual(@as(c_int, 448), PromptPersistentTrace.scalar[5]);
    const head = [_]cuda.prompt_many.Out{.{ .q = PromptPersistentTrace.matrix(248320, 5120), .out = 3000 }};
    try kernels.prompt(stream, 1000, &head, 496);
    try t.expect(PromptPersistentTrace.id == 14 or PromptPersistentTrace.id == 15);
}


const OutputStreamTrace = struct {
    var id: usize = 0;
    var grid: u32 = 0;
    var threads: u32 = 0;
    var cluster: bool = false;
    var parts: cuda.qlinear.Parts = undefined;
    fn launch(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        id = @intFromPtr(f); grid = c.grid_x; threads = c.block_x; cluster = false;
        if (id == 42) return 0;
        for (c.attrs.?[0..c.num_attrs]) |attr| if (attr.id == .cluster_dimension) { cluster = true; };
        parts = @as(*const cuda.qlinear.Parts, @ptrCast(@alignCast(args.?[2].?))).*;
        return 0;
    }
};
test "output stream: original slices require a prepared view and the native sixteen-row shape" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    d.api.cuLaunchKernelEx = OutputStreamTrace.launch;
    e.affine.stream16_fns = .{ function(&d, 20), function(&d, 21) };
    for ([_]usize{ 6144, 17408 }) |k| {
        var q4: cuda.qlinear.Affine4 = .{ .w = 10, .stream_w = 100, .s = 20, .b = 30, .n = 5120, .npad = 5120, .k = k };
        for ([_]usize{ 1, 15, 16, 32 }) |rows| {
            try e.affine.cluster(e.stream, 1000, 2000, q4, 3000, rows, 4);
            const selected = rows == 16;
            try t.expectEqual(@as(usize, if (selected) (if (k == 6144) 21 else 20) else 3), OutputStreamTrace.id);
            try t.expectEqual(@as(u32, if (selected) 512 else 128), OutputStreamTrace.threads);
            try t.expectEqual(!selected, OutputStreamTrace.cluster);
            try t.expectEqual(@as(u64, if (selected) 100 else 10), OutputStreamTrace.parts.p[0].w);
            try t.expectEqual(@as(c_int, 4), OutputStreamTrace.parts.p[0].sk);
            try t.expectEqual(@as(u64, 20), OutputStreamTrace.parts.p[0].scales);
            if (selected) try t.expectEqual(@as(u32, 80), OutputStreamTrace.grid);
        }
        q4.stream_w = 0;
        try e.affine.cluster(e.stream, 1000, 2000, q4, 3000, 16, 4);
        try t.expectEqual(@as(usize, 3), OutputStreamTrace.id);
    }
}

test "GDN stream: complete native input group retains its split order and requires every view" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    d.api.cuLaunchKernelEx = OutputStreamTrace.launch;
    e.affine.gdn_stream16_fn = function(&d, 26);
    e.affine.many_fn = function(&d, 42);
    var outs: [4]cuda.qlinear.Affine4Kernels.Out = undefined;
    for (&outs, [_]usize{ 10240, 6144, 48, 48 }, 0..) |*o, n, index| {
        o.* = .{ .q = .{ .w = 10 + index, .stream_w = 100 + index, .s = 20 + index, .b = 30 + index, .n = n, .npad = (n + 127) / 128 * 128, .k = 5120 }, .out = 1000 + index };
    }
    try e.affine.decodeMany(e.stream, 2000, 3000, &outs, 16);
    try t.expectEqual(@as(usize, 26), OutputStreamTrace.id);
    try t.expectEqual(@as(u32, 66), OutputStreamTrace.grid);
    try t.expectEqual(@as(u32, 128), OutputStreamTrace.threads);
    try t.expect(!OutputStreamTrace.cluster);
    for (OutputStreamTrace.parts.p[0..4], [_]c_int{ 2, 2, 8, 8 }, [_]c_int{ 0, 40, 64, 65 }, 0..) |part, sk, first, index| {
        try t.expectEqual(sk, part.sk);
        try t.expectEqual(first, part.first);
        try t.expectEqual(@as(u64, 100 + index), part.w);
        try t.expectEqual(@as(u64, 20 + index), part.scales);
        try t.expectEqual(@as(u64, 1000 + index), part.out);
    }
    outs[3].q.stream_w = 0;
    try e.affine.decodeMany(e.stream, 2000, 3000, &outs, 16);
    try t.expectEqual(@as(usize, 42), OutputStreamTrace.id);
    try t.expectError(error.WindowTooWide, e.affine.decodeMany(e.stream, 2000, 3000, &outs, 17));
}

const StreamPackTrace = struct {
    var count: usize = 0;
    var original: [6]u64 = undefined;
    var alternate: [6]u64 = undefined;
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        original[count] = @as(*const u64, @ptrCast(@alignCast(args.?[0].?))).*;
        alternate[count] = @as(*const u64, @ptrCast(@alignCast(args.?[1].?))).*;
        count += 1;
        return 0;
    }
    fn projection(n: usize, k: usize, id: u64) q.engine.Proj {
        return .{ .a4 = .{ .w = id, .s = id + 100, .b = id + 200, .n = n, .npad = (n + 127) / 128 * 128, .k = k } };
    }
};
test "projection streams: weight-only views cover all roles without replacing original packed words" {
    var d: cuda.Driver = undefined;
    var e = engine(&d, true);
    d.api.cuMemAlloc_v2 = PackTrace.alloc;
    d.api.cuMemFree_v2 = PackTrace.free;
    d.api.cuMemsetD8_v2 = PackTrace.memset;
    d.api.cuModuleGetFunction = Resolve.get;
    d.api.cuFuncSetAttribute = Resolve.attribute;
    d.api.cuStreamSynchronize = PackTrace.sync;
    d.api.cuLaunchKernel = StreamPackTrace.launch;
    e.mods[0] = .{ .d = &d, .handle = null }; e.mods[3] = .{ .d = &d, .handle = null };
    const input = StreamPackTrace.projection(10240, 5120, 10);
    var layers: [1]q.engine.Layer = undefined;
    layers[0].mlp = .{ .dense = .{ .gate = input, .up = input, .down = StreamPackTrace.projection(5120, 17408, 50) } };
    layers[0].mixer = .{ .delta = .{ .qkv = input, .z = StreamPackTrace.projection(6144, 5120, 20),
        .a = StreamPackTrace.projection(48, 5120, 30), .b = StreamPackTrace.projection(48, 5120, 40),
        .out = StreamPackTrace.projection(5120, 6144, 60), .conv = 0, .norm = 0, .a_log = 0, .dt_bias = 0 } };
    e.layers = &layers;
    StreamPackTrace.count = 0; PackTrace.touched = false; PackTrace.synced = false;
    var buffer = (try q.engine.projection_stream_layout.prepare(&e, true, true)).?;
    defer buffer.free();
    try t.expectEqual(@as(usize, 6), StreamPackTrace.count);
    try t.expectEqual([6]u64{ 50, 60, 10, 20, 30, 40 }, StreamPackTrace.original);
    try t.expect(PackTrace.touched and PackTrace.synced);
    const m = layers[0].mixer.delta;
    try t.expectEqual(@as(u64, 10), m.qkv.a4.w);
    try t.expectEqual(@as(u64, 60), m.out.a4.w);
    try t.expectEqual(StreamPackTrace.alternate[2], m.qkv.a4.stream_w);
    try t.expectEqual(StreamPackTrace.alternate[5], m.b.a4.stream_w);
    try t.expect(e.affine.stream16_fns[0] != null and e.affine.stream16_fns[1] != null and e.affine.gdn_stream16_fn != null);
}
