//! Optional input-independent word-panel view for the narrow gate/up projection pair.
const cuda = @import("cuda");

fn eligible(q: cuda.qlinear.Affine4) bool {
    return q.n == 17408 and q.k == 5120;
}

/// The panels' sizes in the order `prepare` places them (for the fast pool's screen); returns how many.
pub fn copySizes(e: anytype, out: []usize) usize {
    var n: usize = 0;
    for (e.layers) |layer| switch (layer.mlp) {
        .dense => |mlp| for ([_]@TypeOf(mlp.gate){ mlp.gate, mlp.up }) |p| switch (p) {
            .a4 => |q| if (eligible(q)) {
                out[n] = q.npad * q.k / 2;
                n += 1;
            },
            else => {},
        },
        else => {},
    };
    return n;
}

/// Original packed words remain intact for prompts, single rows and every other consumer.
/// The returned allocation belongs to the engine and contains weights only; with the engine's fast pool the panels
/// are placed in its pieces instead and nothing is returned.
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
    var buffer: ?cuda.DeviceBuffer = null;
    if (e.fast == null) {
        buffer = try cuda.DeviceBuffer.alloc(e.d, bytes);
    }
    errdefer if (buffer) |*b| b.free();
    errdefer cuda.completion.drain(e.d, e.stream, "verify weight packing failed before draining");
    // committed in address order before the packing kernels write every word (as the projection streams' buffer is);
    // the fast pool's pieces were committed when it screened them
    if (buffer) |b| try b.fill8(0, e.stream.handle);
    var at: usize = 0;
    for (e.layers) |*layer| switch (layer.mlp) {
        .dense => |*mlp| for ([_]@TypeOf(&mlp.gate){ &mlp.gate, &mlp.up }) |p| switch (p.*) {
            .a4 => |*q| {
                if (!eligible(q.*)) continue;
                q.verify_w = if (e.fast) |*pool| (pool.alloc(q.npad * q.k / 2) orelse return error.FastPoolShort) else buffer.?.ptr + at;
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
    // tf_qmm_mlp_single16's 32-row twins (qmm_stream32.cuh) on the same panels, taken ahead of the kernels above for 17 to
    // 32-row windows; TF_STREAM32=0 loads none (those windows keep the kernels above)
    const wide = @import("sys").c.getenv("TF_STREAM32");
    if (single and !(wide != null and @import("std").mem.eql(u8, @import("std").mem.span(wide.?), "0"))) {
        const f = try e.mods[0].function("tf_qmm_mlp_twin32");
        try f.allowDynamicShared(cuda.mlp_pair.twin32_smem);
        e.mlp_pair.twin32_fn = f;
        if (e.mlp_pair.single_dict_fn != null) {
            const fd = try e.mods[0].function("tf_qmm_mlp_twin32_dict");
            try fd.allowDynamicShared(cuda.mlp_pair.twin32_smem + 2 * cuda.qlinear.dict_tab_max);
            e.mlp_pair.twin32_dict_fn = fd;
            const fp = try e.mods[0].function("tf_qmm_mlp_twin32_pair");
            try fp.allowDynamicShared(cuda.mlp_pair.twin32_pair_smem);
            e.mlp_pair.twin32_pair_fn = fp;
            // pair-coded gate and up on 136 half CTAs, three an SM; TF_MLP32_HALF=0 keeps tf_qmm_mlp_twin32_pair
            const half = @import("sys").c.getenv("TF_MLP32_HALF");
            if (!(half != null and @import("std").mem.eql(u8, @import("std").mem.span(half.?), "0"))) {
                const fh = try e.mods[0].function("tf_qmm_mlp_half32_pair");
                try fh.allowDynamicShared(cuda.mlp_pair.half32_pair_smem);
                e.mlp_pair.twin32_half_fn = fh;
            }
            const fm = try e.mods[0].function("tf_qmm_mlp_twin32_mixed");
            try fm.allowDynamicShared(cuda.mlp_pair.twin32_smem + 2 * cuda.qlinear.dict_tab_max);
            e.mlp_pair.twin32_mixed_fn = fm;
        }
    }
    return buffer;
}
