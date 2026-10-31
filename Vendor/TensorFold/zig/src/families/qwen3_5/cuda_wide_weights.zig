//! Input-independent word-panel views used only by 17 to 32-row verification.
const cuda = @import("cuda");
const layers = @import("cuda_layers.zig");

fn candidates(layer: *layers.Layer) [4]?*layers.Proj {
    var list: [4]?*layers.Proj = @splat(null);
    switch (layer.mlp) {
        .dense => |*m| list[0] = &m.down,
        else => {},
    }
    switch (layer.mixer) {
        .attention => |*m| {
            list[1] = &m.q;
            list[2] = &m.k;
            list[3] = &m.v;
        },
        else => {},
    }
    return list;
}

fn eligible(q: cuda.qlinear.Affine4, kind: u8) bool {
    return if (kind == 2) q.n == 5120 and q.k == 17408 and q.npad % 256 == 0 else q.k == 5120 and q.npad % 128 == 0;
}

fn bytes(p: layers.Proj, kind: u8) usize {
    return switch (p) {
        .a4 => |q| if (eligible(q, kind)) q.npad * q.k / 2 else 0,
        else => 0,
    };
}

fn pack(e: anytype, function: cuda.Function, p: layers.Proj, kind: u8, dest: u64) !usize {
    const size = bytes(p, kind);
    if (size == 0) return 0;
    const q = p.a4;
    const words = size / 4;
    try e.run(function, .{ (words + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ q.w, dest, @as(c_int, @intCast(q.k / 64)), @as(c_int, @intCast(q.npad / 64)) });
    return size;
}

fn publish(p: *layers.Proj, kind: u8, dest: u64) usize {
    const size = bytes(p.*, kind);
    if (size == 0) return 0;
    p.a4.wide_w = dest;
    p.a4.wide_kind = kind;
    return size;
}

/// One owned allocation; original words and metadata remain intact for every other consumer.
pub fn prepare(e: anytype) !?cuda.DeviceBuffer {
    var size = bytes(e.head, 3);
    for (e.layers) |*layer| for (candidates(layer), 0..) |maybe, i| {
        if (maybe) |p| size += bytes(p.*, if (i == 0) 2 else 1);
    };
    if (size == 0) return null;
    const function = try e.mods[3].function("tf_pack_wide_kmajor");
    var functions: [3]?cuda.Function = undefined;
    const names = [_][:0]const u8{ "tf_qmm_wide_attention", "tf_qmm_wide_down", "tf_qmm_wide_head" };
    const shared = [_]u32{ 26496, 32768, 17664 };
    for (names, shared, &functions) |name, smem, *f| {
        f.* = try e.mods[0].function(name);
        try f.*.?.allowDynamicShared(smem);
    }
    var buffer = try cuda.DeviceBuffer.alloc(e.d, size);
    errdefer buffer.free();
    errdefer cuda.completion.drain(e.d, e.stream, "wide weight packing failed before draining");
    var at = try pack(e, function, e.head, 3, buffer.ptr);
    for (e.layers) |*layer| for (candidates(layer), 0..) |maybe, i| {
        if (maybe) |p| at += try pack(e, function, p.*, if (i == 0) 2 else 1, buffer.ptr + at);
    };
    try e.stream.synchronize();
    at = publish(&e.head, 3, buffer.ptr);
    for (e.layers) |*layer| for (candidates(layer), 0..) |maybe, i| {
        if (maybe) |p| at += publish(p, if (i == 0) 2 else 1, buffer.ptr + at);
    };
    e.affine.wide_fns = functions;
    return buffer;
}
