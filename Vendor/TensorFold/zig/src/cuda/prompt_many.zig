//! Prompt GEMMs of up to four affine-4 projections of one input in one launch (qmm_prompt_many.cuh).
//!
//! Each output keeps qmm_prefill's prefill_kernel bits: the same bf16 weights, the same m16n8k16 MMA chain over
//! K into one fp32 accumulator, the same bf16 store. The tile is prefill_kernel's 128 x 128 on 2 x 2 warps with
//! two stages (its RTX 5090 instantiation), so two CTAs share a GB10 SM, and the projections' tiles share one
//! launch: a small projection (GDN's a and b, attention's k and v) runs beside a large one instead of alone.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const qlinear = @import("qlinear.zig");
const Function = module.Function;
const Stream = stream_.Stream;
const Affine4 = qlinear.Affine4;

pub const symbol = "tf_qmm_prompt_many";
pub const parts_max = 4;
pub const tile_m = 128;
pub const tile_n = 128;
pub const threads = 128;
/// Tile<64, 128, 128, 2, 2, 2>::SMEM: two stages of 128 input rows, 128 columns' words, scales and biases.
pub const smem: u32 = 2 * (128 * 128 + 128 * 32 + 2 * 128 * 2);

/// qmm_prompt_many.cuh's ManyPart and ManyParts, passed by value.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, first: c_int };
pub const Parts = extern struct { p: [parts_max]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 48 and @sizeOf(Parts) == 200);
    std.debug.assert(smem == 41984);
}

/// One projection of the shared input and its output rows.
pub const Out = struct { q: Affine4, out: u64 };

pub const Kernels = struct {
    many_fn: Function,

    /// From a loaded qmm_prefill module.
    pub fn resolve(prefill_mod: module.Module) !Kernels {
        const f = try prefill_mod.function(symbol);
        try f.allowDynamicShared(smem);
        return .{ .many_fn = f };
    }

    /// out_i (rows, n_i) = x (rows, k) through each of `outs`, every one with prefill_kernel's bits.
    pub fn prompt(kk: *const Kernels, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > parts_max or rows == 0) return error.TooManyParts;
        const k = outs[0].q.k;
        const rows_t = (rows + tile_m - 1) / tile_m;
        // prefill_kernel's row band: the row tiles of a column panel that share it from L2
        const band = (12 << 20) / (tile_m * k * 2);
        const group = @max(1, @min(rows_t, band));
        var parts: Parts = std.mem.zeroes(Parts);
        var blocks: usize = 0;
        for (outs, 0..) |o, i| {
            if (o.q.k != k) return error.InputMismatch;
            parts.p[i] = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = @intCast(o.q.n), .npad = @intCast(o.q.npad), .first = @intCast(blocks) };
            blocks += rows_t * ((o.q.n + tile_n - 1) / tile_n);
        }
        parts.count = @intCast(outs.len);
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        for ([_]usize{ rows, k, k, group }) |v| g.add(@as(c_int, @intCast(v)));
        try launch_.launch(kk.many_fn, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = smem }, s, &g);
    }
};
