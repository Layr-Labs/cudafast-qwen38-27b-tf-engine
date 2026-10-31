//! Optional input-independent word-panel view for the narrow gate/up projection pair.
const cuda = @import("cuda");

fn eligible(q: cuda.qlinear.Affine4) bool {
    return q.n == 17408 and q.k == 5120;
}

/// Original packed words remain intact for prompts, single rows and every other consumer.
/// The returned allocation belongs to the engine and contains weights only.
pub fn prepare(e: anytype, single: bool) !?cuda.DeviceBuffer {
    var bytes: usize = 0;
    for (e.layers) |layer| switch (layer.mlp) {
        .dense => |mlp| for ([_]@TypeOf(mlp.gate){ mlp.gate, mlp.up }) |p| switch (p) {
            .a4 => |q| { if (eligible(q)) { bytes += q.npad * q.k / 2; } },
            else => {},
        },
        else => {},
    };
    if (bytes == 0) return null;
    const pack = try e.mods[3].function("tf_pack_kmajor");
    const project = try e.mods[0].function(if (single) "tf_qmm_mlp_single16" else "tf_qmm_mlp_pair_kmajor16");
    try project.allowDynamicShared(if (single) cuda.mlp_pair.single_smem else cuda.mlp_pair.kmajor_smem);
    var buffer = try cuda.DeviceBuffer.alloc(e.d, bytes);
    errdefer buffer.free();
    errdefer cuda.completion.drain(e.d, e.stream, "verify weight packing failed before draining");
    var at: usize = 0;
    for (e.layers) |*layer| switch (layer.mlp) {
        .dense => |*mlp| for ([_]@TypeOf(&mlp.gate){ &mlp.gate, &mlp.up }) |p| switch (p.*) {
            .a4 => |*q| {
                if (!eligible(q.*)) continue;
                q.verify_w = buffer.ptr + at;
                const words = q.npad * q.k / 8;
                try e.run(pack, .{ (words + 255) / 256, 1, 1 }, .{ 256, 1, 1 },
                    .{ q.w, q.verify_w, @as(c_int, @intCast(q.k / 64)), @as(c_int, @intCast(q.npad / 64)) });
                at += words * 4;
            },
            else => {},
        },
        else => {},
    };
    try e.stream.synchronize();
    e.mlp_pair.kmajor_fn = project;
    e.mlp_pair.kmajor_single = single;
    // the same panels for 17 to 32-row windows (one weight pass); TF_MLP32=0 keeps them on the 16-row tiles
    const off32 = if (@import("sys").c.getenv("TF_MLP32")) |v| @import("std").mem.eql(u8, @import("std").mem.span(v), "0") else false;
    if (single and !off32) {
        const f32w = try e.mods[0].function("tf_qmm_mlp_single32");
        try f32w.allowDynamicShared(cuda.mlp_pair.single32_smem);
        e.mlp_pair.single32_fn = f32w;
        if (e.dict != null) {
            const f32d = try e.mods[0].function("tf_qmm_mlp_single32_dict");
            try f32d.allowDynamicShared(cuda.mlp_pair.single32_smem + 2 * cuda.qlinear.dict_tab_max);
            e.mlp_pair.single32_dict_fn = f32d;
        }
    }
    return buffer;
}
