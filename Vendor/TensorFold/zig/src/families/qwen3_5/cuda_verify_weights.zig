//! Optional input-independent word-panel view for the narrow gate/up projection pair.
const cuda = @import("cuda");

fn eligible(q: cuda.qlinear.Affine4) bool {
    return q.n == 17408 and q.k == 5120;
}

/// Original packed words remain intact for prompts, single rows and every other consumer.
/// The returned allocation belongs to the engine and contains weights only.
pub fn prepare(e: anytype) !?cuda.DeviceBuffer {
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
    const project = try e.mods[0].function("tf_qmm_verify_kmajor16");
    try project.allowDynamicShared(13440);
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
    e.affine.kmajor_fn = project;
    return buffer;
}
