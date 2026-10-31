//! Immutable word-only projection streams, prepared without any request content.
const cuda = @import("cuda");
const layers = @import("cuda_layers.zig");

fn candidates(layer: *layers.Layer, input: bool, output: bool) [6]?*layers.Proj {
    var list: [6]?*layers.Proj = @splat(null);
    if (output) switch (layer.mlp) { .dense => |*m| { list[0] = &m.down; }, else => {} };
    switch (layer.mixer) {
        .delta => |*m| {
            if (output) list[1] = &m.out;
            if (input and m.qkv == .a4 and m.z == .a4 and m.a == .a4 and m.b == .a4 and
                m.qkv.outputs() == 10240 and m.z.outputs() == 6144 and m.a.outputs() == 48 and m.b.outputs() == 48 and
                m.qkv.inputs() == 5120 and m.z.inputs() == 5120 and m.a.inputs() == 5120 and m.b.inputs() == 5120) {
                list[2] = &m.qkv; list[3] = &m.z; list[4] = &m.a; list[5] = &m.b;
            }
        },
        .attention => |*m| {
            if (output) list[1] = &m.o;
            if (input and m.q == .a4 and m.k == .a4 and m.v == .a4 and
                m.q.outputs() == 12288 and m.k.outputs() == 1024 and m.v.outputs() == 1024 and
                m.q.inputs() == 5120 and m.k.inputs() == 5120 and m.v.inputs() == 5120) {
                list[2] = &m.q; list[3] = &m.k; list[4] = &m.v;
            }
        },
    }
    return list;
}

fn size(p: layers.Proj) usize {
    if (p != .a4) return 0;
    const q = p.a4;
    const out = q.n == 5120 and (q.k == 6144 or q.k == 17408) and cuda.qlinear.splitK(q.n, q.k) == 4;
    const in = q.k == 5120 and (q.n == 10240 or q.n == 6144 or q.n == 48 or q.n == 12288 or q.n == 1024);
    if (!out and !in) return 0;
    return q.npad * q.k / 2;
}

/// Existing views remain available to every other projection role and row count.
pub fn prepare(e: anytype, input: bool, output: bool) !?cuda.DeviceBuffer {
    var bytes: usize = 0;
    for (e.layers) |*layer| for (candidates(layer, input, output)) |maybe| { if (maybe) |p| { bytes += size(p.*); } };
    if (bytes == 0) return null;
    const pack = try e.mods[3].function("tf_pack_kmajor");
    const grouped = try e.mods[0].function("tf_qmm_gdn_stream16");
    try grouped.allowDynamicShared(22656);
    // its twin for dictionary-coded parts: a 256-column CTA's decoded scales and biases and its part's table
    const grouped_dict: ?cuda.Function = if (e.affine.dict == null) null else blk: {
        const f = try e.mods[0].function("tf_qmm_gdn_stream16_dict");
        try f.allowDynamicShared(22656 + 4 * 256 + @as(u32, @max(cuda.qlinear.dict_tab_max, cuda.qlinear.gdn_pair_tab_max)));
        break :blk f;
    };
    const down = try e.mods[0].function("tf_qmm_output_stream16");
    const out = try e.mods[0].function("tf_qmm_output_prefetch16");
    try down.allowDynamicShared(35328);
    // its twin for a dictionary-coded down/out/o projection: the table past the stages
    const down_dict: ?cuda.Function = if (e.affine.dict == null) null else blk: {
        const f = try e.mods[0].function("tf_qmm_output_stream16_dict");
        try f.allowDynamicShared(35328 + @as(u32, @max(cuda.qlinear.dict_tab_max, cuda.qlinear.out_pair_tab_max)));
        break :blk f;
    };
    try out.allowDynamicShared(35328);
    var buffer = try cuda.DeviceBuffer.alloc(e.d, bytes);
    errdefer buffer.free();
    try buffer.fill8(0, null);
    errdefer cuda.completion.drain(e.d, e.stream, "projection stream packing failed before draining");
    var at: usize = 0;
    for (e.layers) |*layer| for (candidates(layer, input, output)) |maybe| {
        if (maybe) |p| {
            const nbytes = size(p.*);
            if (nbytes == 0) continue;
            const q = p.a4;
            const words = q.npad * q.k / 8;
            try e.run(pack, .{ (words + 255) / 256, 1, 1 }, .{ 256, 1, 1 },
                .{ q.w, buffer.ptr + at, @as(c_int, @intCast(q.k / 64)), @as(c_int, @intCast(q.npad / 64)) });
            at += nbytes;
        }
    };
    try e.stream.synchronize();
    at = 0;
    for (e.layers) |*layer| for (candidates(layer, input, output)) |maybe| {
        if (maybe) |p| {
            const nbytes = size(p.*);
            if (nbytes == 0) continue;
            p.a4.stream_w = buffer.ptr + at;
            at += nbytes;
        }
    };
    if (output) e.affine.stream16_fns = .{ down, out };
    if (output) e.affine.stream16_dict_fn = down_dict;
    if (input) {
        e.affine.gdn_stream16_fn = grouped;
        e.affine.gdn_stream16_dict_fn = grouped_dict;
        const off32 = if (@import("sys").c.getenv("TF_GDN_STREAM32")) |v| @import("std").mem.eql(u8, @import("std").mem.span(v), "0") else false;
        if (!off32) {
            const g32 = try e.mods[0].function("tf_qmm_gdn_stream32");
            try g32.allowDynamicShared(cuda.qlinear.Affine4Kernels.gdn_stream32_smem);
            e.affine.gdn_stream32_fn = g32;
            if (e.affine.dict != null) {
                const gd32 = try e.mods[0].function("tf_qmm_gdn_stream32_dict");
                try gd32.allowDynamicShared(cuda.qlinear.Affine4Kernels.gdn_stream32_smem + 4 * 128 + @as(u32, @max(cuda.qlinear.dict_tab_max, cuda.qlinear.gdn_pair_tab_max)));
                e.affine.gdn_stream32_dict_fn = gd32;
            }
        }
    }
    if (input and e.affine.dict != null) {
        const off = if (@import("sys").c.getenv("TF_ATTN_STREAM")) |v| @import("std").mem.eql(u8, @import("std").mem.span(v), "0") else false;
        if (!off) {
            const f = try e.mods[0].function("tf_qmm_attention_stream16_dict");
            try f.allowDynamicShared(28032);
            try f.setAttribute(.preferred_shared_memory_carveout, 100);
            try validateAttention(e, f);
            e.affine.attention_stream16_fn = f;
        }
    }
    return buffer;
}

// A deterministic nonzero input validates every new attention projection against its original cluster route.
fn validateAttention(e: anytype, f: cuda.Function) !void {
    const std = @import("std");
    const N = 12288 + 2 * 1024;
    const x_bytes = 16 * 5120 * 2;
    const xs_bytes = 16 * (5120 / 64) * 4;
    const out_bytes = 16 * N * 2;
    const old_at = x_bytes + xs_bytes;
    const new_at = old_at + out_bytes;
    const bad_at = new_at + out_bytes;
    var scratch = try cuda.DeviceBuffer.alloc(e.d, bad_at + 4);
    defer scratch.free();
    errdefer cuda.completion.drain(e.d, e.stream, "attention view validation failed before draining");
    const fill = try e.mods[0].function("tf_attention_validate_fill");
    const check = try e.mods[0].function("tf_attention_validate_equal");
    try scratch.fill8(0, e.stream.handle);
    try e.run(fill, .{ (16 * 5120 + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ scratch.ptr, @as(c_int, 16 * 5120) });
    try e.run(e.fns.sums, .{ 16, 2, 1 }, .{ 64, 1, 1 }, .{ scratch.ptr, scratch.ptr + x_bytes, @as(c_int, 5120), @as(u64, 0) });
    var original = e.affine;
    original.attention_stream16_fn = null;
    var candidate = e.affine;
    candidate.attention_stream16_fn = f;
    var checked: usize = 0;
    for (e.layers) |layer| switch (layer.mixer) {
        .attention => |m| {
            if (m.q != .a4 or m.k != .a4 or m.v != .a4) continue;
            var old: [3]cuda.qlinear.Affine4Kernels.Out = undefined;
            var fresh: [3]cuda.qlinear.Affine4Kernels.Out = undefined;
            var offset: usize = 0;
            for ([_]cuda.qlinear.Affine4{ m.q.a4, m.k.a4, m.v.a4 }, 0..) |q, i| {
                old[i] = .{ .q = q, .out = scratch.ptr + old_at + offset };
                fresh[i] = .{ .q = q, .out = scratch.ptr + new_at + offset };
                offset += 16 * q.n * 2;
            }
            if (!candidate.attentionStreamable(&fresh, 16)) continue;
            try original.decodeAttentionMixed(e.stream, scratch.ptr, scratch.ptr + x_bytes, &old, 16);
            try candidate.decodeAttentionMixed(e.stream, scratch.ptr, scratch.ptr + x_bytes, &fresh, 16);
            try e.run(check, .{ (16 * N + 255) / 256, 1, 1 }, .{ 256, 1, 1 },
                .{ scratch.ptr + old_at, scratch.ptr + new_at, @as(c_int, 16 * N), scratch.ptr + bad_at });
            checked += 1;
        },
        else => {},
    };
    try e.stream.synchronize();
    var mismatches: u32 = 0;
    try scratch.download(bad_at, std.mem.asBytes(&mismatches));
    if (mismatches != 0) return error.InvalidAttentionStream;
    std.debug.print("info: exact attention stream: {d} layers, {d} BF16 outputs verified\n", .{ checked, checked * 16 * N });
}
