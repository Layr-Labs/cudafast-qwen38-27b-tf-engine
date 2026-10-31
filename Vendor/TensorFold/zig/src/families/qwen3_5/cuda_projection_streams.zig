//! Immutable word-only output-projection streams, prepared without any request content.
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
        .attention => |*m| { if (output) list[1] = &m.o; },
    }
    return list;
}

fn size(p: layers.Proj) usize {
    if (p != .a4) return 0;
    const q = p.a4;
    const out = q.n == 5120 and (q.k == 6144 or q.k == 17408) and cuda.qlinear.splitK(q.n, q.k) == 4;
    const in = q.k == 5120 and (q.n == 10240 or q.n == 6144 or q.n == 48);
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
        try f.allowDynamicShared(22656 + 4 * 256 + cuda.qlinear.dict_tab_max);
        break :blk f;
    };
    const down = try e.mods[0].function("tf_qmm_output_stream16");
    const out = try e.mods[0].function("tf_qmm_output_prefetch16");
    try down.allowDynamicShared(35328);
    // its twin for a dictionary-coded down projection: four slices' decoded groups and the table past the stages
    const down_dict: ?cuda.Function = if (e.affine.dict == null) null else blk: {
        const f = try e.mods[0].function("tf_qmm_output_stream16_dict");
        try f.allowDynamicShared(35328 + 4 * cuda.qlinear.dict_dec + cuda.qlinear.dict_tab_max);
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
    if (input) e.affine.gdn_stream16_fn = grouped;
    if (input) e.affine.gdn_stream16_dict_fn = grouped_dict;
    return buffer;
}
