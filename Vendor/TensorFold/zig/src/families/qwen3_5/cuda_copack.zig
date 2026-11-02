//! Co-packed compact-pair records: the narrow (at most 16-row) gate/up pair's weights and dictionary windows, one
//! contiguous record a (128-column tile, K group).
//!
//! tf_qmm_mlp_single16_compact_pair fetches each stage from four places: gate's and up's two 2 KB K-major tiles (the
//! next group's are a whole panel row, npad / 64 tiles, away) and gate's and up's dictionary index windows in their
//! own streams. Here every stage's bytes are copied once, at load, into a record laid out exactly as the stage's shared
//! bytes, records ordered tile by tile and group by group, so each CTA streams one contiguous run of records.
//! tf_qmm_mlp_single16_compact_copack loads the same shared bytes from it; the arithmetic is the record kernel's.
//!
//! Every layer's records are checked when built: the record kernel and the co-packed one run on the same probe window
//! and their act and sums must be equal byte for byte, else that layer keeps the record path. The records are placed
//! last on the fast pool (after the verify panels and projection streams); no room means no records for that layer.
//! TF_MLP_COPACK=0 turns it off.
const std = @import("std");
const cuda = @import("cuda");
const qlinear = cuda.qlinear;

pub fn enabled() bool {
    const v = @import("sys").c.getenv("TF_MLP_COPACK") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn bits(q: qlinear.Affine4) usize {
    return (q.code & ~qlinear.dict_kmajor) & 31;
}

/// Bytes of one record: the stage's four weight tiles and its two dictionary windows (16-byte chunks).
pub fn recBytes(gate: qlinear.Affine4, up: qlinear.Affine4) usize {
    return 256 * 64 / 2 + 16 * (bits(gate) + bits(up));
}

fn bytesFor(gate: qlinear.Affine4, up: qlinear.Affine4) usize {
    return gate.n / 128 * (gate.k / 64) * recBytes(gate, up);
}

/// The layers whose pair the compact kernel takes (before the panels exist: shape and both K-major pair codes).
fn eligible(gate: qlinear.Affine4, up: qlinear.Affine4) bool {
    return gate.n == 17408 and gate.k == 5120 and up.n == gate.n and up.k == gate.k and up.npad == gate.npad and
        gate.code & qlinear.dict_kmajor != 0 and up.code & qlinear.dict_kmajor != 0 and
        qlinear.pairCode(gate.code & ~qlinear.dict_kmajor) and qlinear.pairCode(up.code & ~qlinear.dict_kmajor);
}

/// The records' sizes in the order `prepare` places them (for the fast pool's screen, after every other copy).
pub fn copySizes(e: anytype, out: []usize) usize {
    if (!enabled()) return 0;
    var n: usize = 0;
    for (e.layers) |layer| switch (layer.mlp) {
        .dense => |mlp| switch (mlp.gate) {
            .a4 => |g| switch (mlp.up) {
                .a4 => |u| if (eligible(g, u) and n < out.len) {
                    out[n] = bytesFor(g, u);
                    n += 1;
                },
                else => {},
            },
            else => {},
        },
        else => {},
    };
    return n;
}

/// Builds and checks every eligible layer's records on the fast pool; a layer that fails either keeps the record path.
pub fn prepare(e: anytype) !void {
    if (!enabled()) return;
    const pool = if (e.fast) |*p| p else return;
    const kk = &e.mlp_pair;
    if (kk.copack_fn == null) return;
    const pack = try e.mods[0].function("tf_qmm_mlp_copack_pack");
    const rows: usize = 16;
    const k: usize = 5120;
    const n: usize = 17408;
    // the probe window: deterministic bf16 inputs of either sign and group sums (any values: both kernels read them)
    var x = try cuda.DeviceBuffer.alloc(e.d, rows * k * 2);
    defer x.free();
    var xs = try cuda.DeviceBuffer.alloc(e.d, rows * (k / 64) * 4);
    defer xs.free();
    var act = try cuda.DeviceBuffer.alloc(e.d, 2 * rows * n * 2);
    defer act.free();
    var sums = try cuda.DeviceBuffer.alloc(e.d, 2 * rows * (n / 64) * 4);
    defer sums.free();
    const host = try e.gpa.alloc(u8, 2 * rows * n * 2 + 2 * rows * (n / 64) * 4 + rows * k * 2);
    defer e.gpa.free(host);
    {
        var seed: u32 = 0x9E3779B9;
        const xb = std.mem.bytesAsSlice(u16, host[0 .. rows * k * 2]);
        for (xb) |*v| {
            seed = seed *% 1664525 +% 1013904223;
            // a bf16 in [-2, 2): sign, exponent 126..127, 7 mantissa bits
            v.* = @intCast((seed >> 31) << 15 | (126 + (seed >> 30 & 1)) << 7 | (seed >> 16 & 0x7F));
        }
        try x.upload(0, host[0 .. rows * k * 2]);
        const fs = std.mem.bytesAsSlice(f32, host[0 .. rows * (k / 64) * 4]);
        for (fs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 97)) - 48)) * 0.25;
        try xs.upload(0, host[0 .. rows * (k / 64) * 4]);
    }
    var built: usize = 0;
    var short: usize = 0;
    var bad: usize = 0;
    for (e.layers) |*layer| switch (layer.mlp) {
        .dense => |*mlp| switch (mlp.gate) {
            .a4 => |*g| switch (mlp.up) {
                .a4 => |u| {
                    if (!eligible(g.*, u) or !kk.compactFor(g.*, u, rows)) continue;
                    const rec = recBytes(g.*, u);
                    const ptr = pool.alloc(bytesFor(g.*, u)) orelse {
                        short += 1;
                        continue;
                    };
                    const groups = g.k / 64;
                    const parts = cuda.mlp_pair.Kernels.compactParts(g.*, u);
                    try e.run(pack, .{ g.n / 128 * groups, 1, 1 }, .{ 128, 1, 1 }, .{ parts, @as(c_int, @intCast(groups)), ptr, @as(c_int, @intCast(rec)) });
                    // the self-check: the record kernel, then the co-packed one, on the same window
                    var plain = g.*;
                    plain.copack = 0;
                    var packed_ = g.*;
                    packed_.copack = ptr;
                    packed_.copack_rec = @intCast(rec);
                    try act.fill8(0, e.stream.handle);
                    try sums.fill8(0, e.stream.handle);
                    try kk.pair(e.stream, x.ptr, xs.ptr, plain, u, act.ptr, sums.ptr, rows);
                    try kk.pair(e.stream, x.ptr, xs.ptr, packed_, u, act.ptr + rows * n * 2, sums.ptr + rows * (n / 64) * 4, rows);
                    try e.stream.synchronize();
                    const a_bytes = 2 * rows * n * 2;
                    const s_bytes = 2 * rows * (n / 64) * 4;
                    try act.download(0, host[0..a_bytes]);
                    try sums.download(0, host[a_bytes..][0..s_bytes]);
                    const half_a = a_bytes / 2;
                    const half_s = s_bytes / 2;
                    const same = std.mem.eql(u8, host[0..half_a], host[half_a..a_bytes]) and
                        std.mem.eql(u8, host[a_bytes..][0..half_s], host[a_bytes + half_s ..][0..half_s]);
                    if (!same) {
                        bad += 1;
                        continue;
                    }
                    g.copack = ptr;
                    g.copack_rec = @intCast(rec);
                    built += 1;
                },
                else => {},
            },
            else => {},
        },
        else => {},
    };
    std.debug.print("info: compact pair co-packed records: {d} layers built and checked, {d} without room, {d} mismatching (kept on the record path)\n", .{ built, short, bad });
}
