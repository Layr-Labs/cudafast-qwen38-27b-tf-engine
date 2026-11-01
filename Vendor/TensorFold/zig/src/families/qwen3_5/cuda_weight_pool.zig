//! The decode round's packed words on the fastest screened pieces of free memory (cuda_fastpool.zig).
//!
//! A 2-row verify round reads every packed panel of the model once. The narrow gate/up pair reads its own K-major
//! copies (`verify_w`, cuda_verify_weights.zig) and the ordered-slice streams serve 16-row windows; every other
//! panel a round reads - the down projection, the delta rule's qkv/z/out, attention's q/k/v/o and the vocabulary
//! head - is the arena's own buffer, streamed wherever the loader happened to commit it. This copies those buffers
//! word for word onto the pool's pieces, so the same bytes stream at the pieces' speed instead of the arena's.
//!
//! A copy holds every word of the original at the same offsets, and the packed layout is input-independent, so every
//! kernel reads the values it read before: prompts, single rows, the dictionary coding and the drafter's views are
//! untouched. `prepare` repoints the projection's own word buffer, so no kernel, launch or dispatch changes;
//! TF_WEIGHT_POOL=0 (or a pool that could not be screened) leaves every panel in the arena.
const std = @import("std");
const cuda = @import("cuda");
const layers = @import("cuda_layers.zig");

const Affine4 = cuda.qlinear.Affine4;

fn wordsOf(p: *layers.Proj) ?*Affine4 {
    return switch (p.*) {
        .a4 => |*q| q,
        else => null,
    };
}

/// The panels a round reads out of the arena: every packed projection's words but the gate and up, whose K-major
/// copy the pair reads (cuda_verify_weights.zig's eligible(): the intermediate rows by the hidden width).
fn copied(q: Affine4) bool {
    return q.w != 0 and q.k != 0 and q.npad != 0 and !(q.n == 17408 and q.k == 5120);
}

fn bytesOf(q: Affine4) usize {
    return q.npad * q.k / 2;
}

/// One layer's placed panels, in the order `prepare` copies them.
fn layerList(layer: *layers.Layer, list: *[7]?*Affine4) usize {
    var count: usize = 0;
    switch (layer.mlp) {
        .dense => |*m| for ([_]*layers.Proj{ &m.down, &m.gate, &m.up }) |p| if (wordsOf(p)) |q| {
            if (copied(q.*)) {
                list[count] = q;
                count += 1;
            }
        },
        else => {},
    }
    switch (layer.mixer) {
        .delta => |*m| for ([_]*layers.Proj{ &m.qkv, &m.z, &m.out }) |p| if (wordsOf(p)) |q| {
            if (copied(q.*)) {
                list[count] = q;
                count += 1;
            }
        },
        .attention => |*m| for ([_]*layers.Proj{ &m.q, &m.k, &m.v, &m.o }) |p| if (wordsOf(p)) |q| {
            if (copied(q.*)) {
                list[count] = q;
                count += 1;
            }
        },
    }
    return count;
}

/// The panels' sizes in the order `prepare` places them (for the fast pool's screen); returns how many.
pub fn copySizes(e: anytype, out: []usize) usize {
    var n: usize = 0;
    for (e.layers) |*layer| {
        var list: [7]?*Affine4 = @splat(null);
        for (list[0..layerList(layer, &list)]) |maybe| {
            out[n] = bytesOf(maybe.?.*);
            n += 1;
        }
    }
    if (wordsOf(&e.head)) |q| {
        if (copied(q.*)) {
            out[n] = bytesOf(q.*);
            n += 1;
        }
    }
    return n;
}

/// Copies each panel onto the pool's pieces and points the projection's words at the copy; a copy that cannot be
/// placed or launched leaves that panel where it is (the same words, read at the arena's speed).
pub fn prepare(e: anytype) void {
    const pool = if (e.fast) |*p| p else return;
    for (e.layers) |*layer| {
        var list: [7]?*Affine4 = @splat(null);
        for (list[0..layerList(layer, &list)]) |maybe| place(e, pool, maybe.?);
    }
    if (wordsOf(&e.head)) |q| place(e, pool, q);
    e.stream.synchronize() catch {};
}

/// `q`'s words onto the next piece with room: the copy is written on the engine's stream, in order with the boot's
/// packing kernels that read it, and only then does the projection point at it.
fn place(e: anytype, pool: anytype, q: *Affine4) void {
    if (!copied(q.*)) return;
    const dst = pool.alloc(bytesOf(q.*)) orelse return;
    e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, q.w, bytesOf(q.*), e.stream.handle), "cuMemcpyDtoDAsync") catch return;
    q.w = dst;
}

test "weight pool: the gate and up panels are the pair's own copies, every other panel is placed" {
    const t = std.testing;
    const down: Affine4 = .{ .w = 1, .s = 0, .b = 0, .n = 5120, .k = 17408, .npad = 5120 };
    try t.expect(copied(down));
    try t.expectEqual(@as(usize, 5120 * 17408 / 2), bytesOf(down));
    const gate: Affine4 = .{ .w = 1, .s = 0, .b = 0, .n = 17408, .k = 5120, .npad = 17408 };
    try t.expect(!copied(gate));
    try t.expect(!copied(.{ .w = 0, .s = 0, .b = 0, .n = 5120, .k = 5120, .npad = 5120 }));
}
