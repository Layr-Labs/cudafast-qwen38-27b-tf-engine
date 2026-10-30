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
/// The same body on 64-row tiles (Tile<64, 64, 128, 2, 2, 2>): every output's MMA chain and store are unchanged.
pub const symbol64 = "tf_qmm_prompt_many64";
pub const parts_max = 4;
pub const tile_m = 128;
pub const tile_n = 128;
pub const threads = 128;
/// Tile<64, 128, 128, 2, 2, 2>::SMEM: two stages of 128 input rows, 128 columns' words, scales and biases.
pub const smem: u32 = 2 * (128 * 128 + 128 * 32 + 2 * 128 * 2);
/// Tile<64, 64, 128, 2, 2, 2>::SMEM: two stages of 64 input rows and the same 128 columns.
pub const smem64: u32 = 2 * (64 * 128 + 128 * 32 + 2 * 128 * 2);

/// qmm_prompt_many.cuh's ManyPart and ManyParts, passed by value.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, first: c_int };
pub const Parts = extern struct { p: [parts_max]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 48 and @sizeOf(Parts) == 200);
    std.debug.assert(smem == 41984);
    std.debug.assert(smem64 == 25600);
}

/// One projection of the shared input and its output rows.
pub const Out = struct { q: Affine4, out: u64 };

pub const Kernels = struct {
    many_fn: Function,
    many64_fn: Function,
    /// The GPU's SMs (0: unknown, always the 128-row tile); the engine sets it after resolve.
    sms: usize = 0,

    /// From a loaded qmm_prefill module.
    pub fn resolve(prefill_mod: module.Module) !Kernels {
        const f = try prefill_mod.function(symbol);
        try f.allowDynamicShared(smem);
        const half_fn = try prefill_mod.function(symbol64);
        try half_fn.allowDynamicShared(smem64);
        return .{ .many_fn = f, .many64_fn = half_fn };
    }

    /// 64-row tiles when the 128-row grid's last wave leaves most SMs idle (a 160-CTA down or out projection on
    /// GB10's 48 SMs runs 4 tiles an SM for 3.33 tiles of work). Waves in 64-row tile units, the half-height tile
    /// charged 8% for its doubled dequant and weight reads a multiply.
    pub fn short(kk: *const Kernels, rows: usize, cols: usize) bool {
        if (kk.sms == 0 or rows <= 64) return false;
        const big = (rows + tile_m - 1) / tile_m * cols;
        const small = (rows + 63) / 64 * cols;
        const cost_big = (big + kk.sms - 1) / kk.sms * 2 * 100;
        const cost_small = (small + kk.sms - 1) / kk.sms * 108;
        return cost_small < cost_big;
    }

    /// out_i (rows, n_i) = x (rows, k) through each of `outs`, every one with prefill_kernel's bits.
    pub fn prompt(kk: *const Kernels, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > parts_max or rows == 0) return error.TooManyParts;
        const k = outs[0].q.k;
        var cols: usize = 0;
        for (outs) |o| cols += (o.q.n + tile_n - 1) / tile_n;
        const half = kk.short(rows, cols);
        const tm: usize = if (half) 64 else tile_m;
        const rows_t = (rows + tm - 1) / tm;
        // prefill_kernel's row band: the row tiles of a column panel that share it from L2
        const band = (12 << 20) / (tm * k * 2);
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
        const f = if (half) kk.many64_fn else kk.many_fn;
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = if (half) smem64 else smem }, s, &g);
    }
};
