//! Optional input-independent word-panel view for the narrow gate/up projection pair.
const cuda = @import("cuda");

fn eligible(q: cuda.qlinear.Affine4) bool {
    return q.n == 17408 and q.k == 5120;
}

/// TF_MLP_COPACK=0: no co-packed records; the compact gate/up kernel reads the panels and the index stream apart.
fn copackOn(e: anytype) bool {
    if (e.mlp_pair.copack_fn == null) return false;
    const v = @import("sys").c.getenv("TF_MLP_COPACK") orelse return true;
    return !@import("std").mem.eql(u8, @import("std").mem.span(v), "0");
}

/// One projection's co-packed records (tf_copack_pair): k / 64 groups of npad / 128 records, each two 2048-byte panels
/// and 16 * bits index bytes.
fn copackBytes(q: cuda.qlinear.Affine4) usize {
    return (q.k / 64) * (q.npad / 128) * (4096 + 16 * @as(usize, q.code & 31));
}

/// The panels' sizes in the order `prepare` places them (for the fast pool's screen); returns how many.
/// Whether a layer's gate and up take co-packed records (copackOn, both affine-4 and copackable).
fn pairCopack(e: anytype, mlp: anytype) bool {
    return copackOn(e) and mlp.gate == .a4 and mlp.up == .a4 and cuda.mlp_pair.Kernels.copackable(mlp.gate.a4, mlp.up.a4);
}

/// The fast pool's copies in the order `prepare` places them (for its screen); returns how many. A layer whose gate
/// and up take co-packed records places those records where its panels would go (the 16-row compact kernel reads
/// them), and its panels (17 to 32-row windows only) in the returned buffer; any other layer places its panels.
pub fn copySizes(e: anytype, out: []usize) usize {
    var n: usize = 0;
    for (e.layers) |layer| switch (layer.mlp) {
        .dense => |mlp| if (pairCopack(e, mlp)) {
            out[n] = copackBytes(mlp.gate.a4);
            out[n + 1] = copackBytes(mlp.up.a4);
            n += 2;
        } else for ([_]@TypeOf(mlp.gate){ mlp.gate, mlp.up }) |p| switch (p) {
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
/// The returned allocation belongs to the engine and contains weights only: every copy without the engine's fast
/// pool; with it, the panels of the layers whose co-packed records took their place in the pool.
pub fn prepare(e: anytype, single: bool) !?cuda.DeviceBuffer {
    var any = false;
    var bytes: usize = 0;
    for (e.layers) |layer| switch (layer.mlp) {
        .dense => |mlp| {
            const cp = pairCopack(e, mlp);
            for ([_]@TypeOf(mlp.gate){ mlp.gate, mlp.up }) |p| switch (p) {
                .a4 => |q| if (eligible(q)) {
                    any = true;
                    if (cp or e.fast == null) bytes += q.npad * q.k / 2;
                    if (cp and e.fast == null) bytes += copackBytes(q);
                },
                else => {},
            };
        },
        else => {},
    };
    if (!any) return null;
    const pack = try e.mods[3].function("tf_pack_kmajor");
    const copack_fn: ?cuda.Function = if (copackOn(e)) try e.mods[3].function("tf_copack_pair") else null;
    const project = try e.mods[0].function(if (single) "tf_qmm_mlp_single16" else "tf_qmm_mlp_pair_kmajor16");
    try project.allowDynamicShared(if (single) cuda.mlp_pair.single_smem else cuda.mlp_pair.kmajor_smem);
    var buffer: ?cuda.DeviceBuffer = null;
    if (bytes > 0) {
        buffer = try cuda.DeviceBuffer.alloc(e.d, bytes);
    }
    errdefer if (buffer) |*b| b.free();
    errdefer cuda.completion.drain(e.d, e.stream, "verify weight packing failed before draining");
    // committed in address order before the packing kernels write every word (as the projection streams' buffer is);
    // the fast pool's pieces were committed when it screened them
    if (buffer) |b| try b.fill8(0, e.stream.handle);
    var at: usize = 0;
    for (e.layers) |*layer| switch (layer.mlp) {
        .dense => |*mlp| {
            const cp = pairCopack(e, mlp.*);
            for ([_]@TypeOf(&mlp.gate){ &mlp.gate, &mlp.up }) |p| switch (p.*) {
                .a4 => |*q| {
                    if (!eligible(q.*)) continue;
                    const panel = q.npad * q.k / 2;
                    if (cp or e.fast == null) {
                        q.verify_w = buffer.?.ptr + at;
                        at += panel;
                    } else q.verify_w = (e.fast.?.alloc(panel) orelse return error.FastPoolShort);
                    const words = q.npad * q.k / 8;
                    try e.run(pack, .{ (words + 255) / 256, 1, 1 }, .{ 256, 1, 1 },
                        .{ q.w, q.verify_w, @as(c_int, @intCast(q.k / 64)), @as(c_int, @intCast(q.npad / 64)) });
                    if (!cp) continue;
                    // the compact kernel's co-packed records: the panels just packed, then the K-major index bytes
                    const nbytes = copackBytes(q.*);
                    if (e.fast) |*pool| {
                        q.copack_w = pool.alloc(nbytes) orelse return error.FastPoolShort;
                    } else {
                        q.copack_w = buffer.?.ptr + at;
                        at += nbytes;
                    }
                    try e.run(copack_fn.?, .{ 4096, 1, 1 }, .{ 256, 1, 1 }, .{ q.verify_w, q.sb, q.copack_w, @as(c_int, @intCast(q.k / 64)),
                        @as(c_int, @intCast(q.npad / 64)), @as(c_int, @intCast(q.code & 31)) });
                },
                else => {},
            };
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
