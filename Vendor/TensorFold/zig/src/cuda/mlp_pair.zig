//! A dense MLP's decode gate and up projections and its activation in one launch (qmm_group.cu's tf_qmm_mlp_pair).
//!
//! Clusters of two CTAs a 64-column tile of 16 rows: gate's tile and up's, each with group_kernel<64, 16, 64, 1, 4, 8>'s
//! arithmetic (what decode() runs for a projection whose splitK is 1), then act = bf16(silu(gate) * up) and each row's
//! group sums of act as tf_q35_mlp_act writes them. The two projections' rows are never stored.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const qlinear = @import("qlinear.zig");
const Function = module.Function;
const Stream = stream_.Stream;
const Affine4 = qlinear.Affine4;

pub const symbol = "tf_qmm_mlp_pair";
/// The stages' memory and, past it, the (16, 64) bf16 gate tile rank 0 stores into rank 1's.
pub const smem: u32 = qlinear.group_smem + 16 * 64 * 2;
pub const kmajor_smem: u32 = 13440 + 16 * 128 * 2;
pub const single_smem: u32 = 41088;

pub const Kernels = struct {
    pair_fn: Function,
    pdl: bool,
    kmajor_fn: ?Function = null,
    kmajor_single: bool = false,

    /// From the loaded qmm_group module; `pdl` as the decode tiles' launches (GB10).
    pub fn resolve(group_mod: module.Module, pdl: bool) !Kernels {
        const f = try group_mod.function(symbol);
        try f.allowDynamicShared(smem);
        return .{ .pair_fn = f, .pdl = pdl };
    }

    /// Whether `pair` takes this gate and up: one shape whose decode() is group_kernel's one-slice tile.
    pub fn fits(gate: Affine4, up: Affine4) bool {
        return gate.n == up.n and gate.k == up.k and gate.npad == up.npad and gate.n % 64 == 0 and gate.k % 64 == 0 and
            qlinear.splitK(gate.n, gate.k) == 1;
    }

    /// act (rows, n) = bf16(silu(x through gate) * (x through up)), x's group sums in xs; act's group sums into as.
    pub fn pair(kk: *const Kernels, s: Stream, x: u64, xs: u64, gate: Affine4, up: Affine4, act: u64, as: u64, rows: usize) !void {
        if (!fits(gate, up) or rows == 0) return error.InputMismatch;
        const rows_t = (rows + 15) / 16;
        // the word-panel view for a full 16-row window, and the single-CTA one for 17 to 32 rows too (two row tiles
        // reading each panel, every row's MMA chain as at 16)
        const wide = kk.kmajor_single and rows > 16 and rows <= 32;
        const major = (rows == 16 or wide) and gate.n == 17408 and gate.k == 5120 and
            gate.verify_w != 0 and up.verify_w != 0 and kk.kmajor_fn != null;
        const single = major and kk.kmajor_single;
        const tiles = gate.n / @as(usize, if (single) 256 else if (major) 128 else 64);
        var parts: qlinear.Parts = std.mem.zeroes(qlinear.Parts);
        parts.p[0] = .{ .w = gate.w, .scales = gate.s, .biases = gate.b, .out = 0, .n = @intCast(gate.n), .npad = @intCast(gate.npad), .sk = 1, .tiles = @intCast(tiles), .first = 0 };
        parts.p[1] = .{ .w = up.w, .scales = up.s, .biases = up.b, .out = 0, .n = @intCast(up.n), .npad = @intCast(up.npad), .sk = 1, .tiles = @intCast(tiles), .first = 0 };
        parts.count = 2;
        if (major) {
            parts.p[0].w = gate.verify_w;
            parts.p[1].w = up.verify_w;
        }
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        g.add(act);
        g.add(as);
        for ([_]usize{ rows, gate.k, gate.k, rows_t }) |v| g.add(@as(c_int, @intCast(v)));
        try launch_.launch(if (major) kk.kmajor_fn.? else kk.pair_fn, .{ .grid = .{ .x = @intCast(@as(usize, if (single) 1 else 2) * rows_t * tiles) }, .block = .{ .x = 128 }, .shared = if (single) single_smem else if (major) kmajor_smem else smem, .cluster = if (single) null else .{ .x = 2 }, .pdl = kk.pdl }, s, &g);
    }
};
