//! The real 4-bit verify projection dispatch through a recording CUDA driver.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const t = std.testing;

test "leaf slot stores: retain traversal and every parent's state after dropping unread leaf writes" {
    const R = q.round;
    var original: [R.bytes]u8 align(8) = undefined;
    var reduced: [R.bytes]u8 align(8) = undefined;
    var rng: u64 = 20261031;
    for ([_]usize{ 17, 24, 32 }) |rows| for (0..100) |trial| {
        var parents: [32]i32 = undefined;
        var expected: [32]u64 = undefined;
        for (parents[0..rows], 0..) |*parent, row| {
            rng = rng *% 6364136223846793005 +% 1442695040888963407;
            parent.* = if (row == 0) -1 else if (trial == 0) 0 else @intCast(rng % row);
            const prior = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
            expected[row] = prior *% 65537 +% row +% 1;
        }
        const parts = [_]q.engine.Part{.{ .cache = undefined, .rows = rows, .parents = parents[0..rows] }};
        const old_peak = try R.build(&original, &parts, true);
        const new_peak = try R.buildLeaves(&reduced, &parts, true);
        try t.expect(new_peak <= old_peak);
        if (trial == 0) {
            try t.expectEqual(@as(usize, 2), old_peak);
            try t.expectEqual(@as(usize, 1), new_peak);
        }
        var actual: [32]u64 = undefined;
        var slots: [16]u64 = undefined;
        for (R.steps(&original)[0..rows], R.steps(&reduced)[0..rows]) |old, step| {
            try t.expectEqual(old.row, step.row);
            try t.expectEqual(old.read_slot, step.read_slot);
            const prior = if (step.read_slot < 0) @as(u64, 23) else slots[@intCast(step.read_slot)];
            const value = prior *% 65537 +% step.row +% 1;
            actual[step.row] = value;
            if (step.write_slot >= 0) {
                try t.expectEqual(old.write_slot, step.write_slot);
                try t.expect(@as(usize, @intCast(step.write_slot)) < new_peak);
                slots[@intCast(step.write_slot)] = value;
            } else for (parents[0..rows]) |parent| try t.expect(parent != step.row);
        }
        try t.expectEqualSlices(u64, expected[0..rows], actual[0..rows]);
    };
}
pub const Trace = struct {
    const Call = struct { id: usize, x: u64, xs: u64, out: u64, rows: c_int, sk: c_int, row_tiles: c_int, grid: u32, shared: u32, parts: c_int = 1, weight: u64 = 0 };
    pub var calls: [128]Call = undefined;
    pub var count: usize = 0;
    pub var prompts: usize = 0;
    fn record(f: cuda.abi.Function, gx: u32, shared: u32, args: ?[*]?*anyopaque) cuda.abi.Result {
        const id = @intFromPtr(f);
        if (id == 4) {
            prompts += 1;
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
    e.affine.prefill_fn = function(d, 4);
    e.affine.prefill22_fn = function(d, 4);
    e.affine.half_prompt = false;
    e.prompt_many = .{ .many_fn = function(d, 4), .mlp_fn = function(d, 4), .split_fn = function(d, 4), .slots = 96 };
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
