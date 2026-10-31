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
    // tf_qmm_mlp_single16's 32-row twins (qmm_stream32.cuh); TF_STREAM32=0 keeps 17 to 32-row windows on tf_qmm_mlp_pair
    const wide = @import("sys").c.getenv("TF_STREAM32");
    if (single and !(wide != null and @import("std").mem.eql(u8, @import("std").mem.span(wide.?), "0"))) {
        const f = try e.mods[0].function("tf_qmm_mlp_single32");
        try f.allowDynamicShared(cuda.mlp_pair.single32_smem);
        e.mlp_pair.single32_fn = f;
        if (e.mlp_pair.single_dict_fn != null) {
            const fd = try e.mods[0].function("tf_qmm_mlp_single32_dict");
            try fd.allowDynamicShared(cuda.mlp_pair.single32_smem + 2 * cuda.qlinear.dict_tab_max);
            e.mlp_pair.single32_dict_fn = fd;
            const fp = try e.mods[0].function("tf_qmm_mlp_single32_pair");
            try fp.allowDynamicShared(cuda.mlp_pair.single32_pair_smem);
            e.mlp_pair.single32_pair_fn = fp;
            const fm = try e.mods[0].function("tf_qmm_mlp_single32_mixed");
            try fm.allowDynamicShared(cuda.mlp_pair.single32_smem + 2 * cuda.qlinear.dict_tab_max);
            e.mlp_pair.single32_mixed_fn = fm;
        }
    }
    return buffer;
}
