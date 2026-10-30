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
pub const mlp_symbol = "tf_qmm_prompt_mlp";
pub const split_symbol = "tf_qmm_prompt_split";
pub const sk_symbol = "tf_qmm_prompt_sk";
/// A stream-K CTA's parked accumulators: 128 threads' 128 fp32 each (both warp layouts).
pub const sk_slot_bytes = threads * 128 * 4;
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
    mlp_fn: Function,
    split_fn: Function,
    sk_fn: Function,
    /// CTAs the device runs at once: two of these tiles an SM.
    slots: usize,
    /// Stream-K for grouped prompt GEMMs: CTAs launched (0: off), parked accumulators and flags for that many.
    sk_slots: usize = 0,
    ws: u64 = 0,
    flags: u64 = 0,
    epoch: u32 = 0,

    /// From a loaded qmm_prefill module, on a device of `sms` SMs.
    pub fn resolve(prefill_mod: module.Module, sms: usize) !Kernels {
        const f = try prefill_mod.function(symbol);
        try f.allowDynamicShared(smem);
        const m = try prefill_mod.function(mlp_symbol);
        try m.allowDynamicShared(smem);
        const sp = try prefill_mod.function(split_symbol);
        try sp.allowDynamicShared(smem);
        const k = try prefill_mod.function(sk_symbol);
        try k.allowDynamicShared(smem);
        return .{ .many_fn = f, .mlp_fn = m, .split_fn = sp, .sk_fn = k, .slots = 2 * sms };
    }

    /// The columns one projection's whole 128-wide tiles cover when its last panels run 64 wide (null: none do).
    /// Whole tiles past one wave leave a last wave part full (a 496-row prompt's 5120 columns: 160 tiles on a GB10's
    /// 96 slots); a fifth of the panels at half width evens the SMs' shares.
    pub fn splitCols(kk: *const Kernels, n: usize, rows: usize) ?usize {
        const cols = n / tile_n;
        const tiles = (rows + tile_m - 1) / tile_m * cols;
        if (n % tile_n != 0 or cols < 10 or tiles <= kk.slots or tiles % kk.slots == 0) return null;
        return (cols - cols / 5) * tile_n;
    }

    /// Whether `mlp` takes this gate and up: one shape, whole 64-column tiles.
    pub fn mlpFits(gate: Affine4, up: Affine4) bool {
        return gate.n == up.n and gate.k == up.k and gate.npad == up.npad and gate.n % 64 == 0 and gate.k % 64 == 0;
    }

    /// act (rows, n) = bf16(silu(gate) * up) of x (rows, k) through gate and up: the bits of `prompt` over both,
    /// each stored as bf16, then qwen35.cu's tf_q35_mlp_act over them; neither projection's rows are written.
    pub fn mlp(kk: *const Kernels, s: Stream, x: u64, gate: Affine4, up: Affine4, act: u64, rows: usize) !void {
        if (!mlpFits(gate, up) or rows == 0) return error.InputMismatch;
        const k = gate.k;
        const rows_t = (rows + tile_m - 1) / tile_m;
        const band = (12 << 20) / (tile_m * k * 2);
        const group = @max(1, @min(rows_t, band));
        var parts: Parts = std.mem.zeroes(Parts);
        parts.p[0] = .{ .w = gate.w, .scales = gate.s, .biases = gate.b, .out = 0, .n = @intCast(gate.n), .npad = @intCast(gate.npad), .first = 0 };
        parts.p[1] = .{ .w = up.w, .scales = up.s, .biases = up.b, .out = 0, .n = @intCast(up.n), .npad = @intCast(up.npad), .first = 0 };
        parts.count = 2;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        g.add(act);
        for ([_]usize{ rows, k, k, group }) |v| g.add(@as(c_int, @intCast(v)));
        const blocks = rows_t * (gate.n / 64);
        try launch_.launch(kk.mlp_fn, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = smem }, s, &g);
    }

    /// out_i (rows, n_i) = x (rows, k) through each of `outs`, every one with prefill_kernel's bits.
    pub fn prompt(kk: *Kernels, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > parts_max or rows == 0) return error.TooManyParts;
        const k = outs[0].q.k;
        const rows_t = (rows + tile_m - 1) / tile_m;
        // prefill_kernel's row band: the row tiles of a column panel that share it from L2
        const band = (12 << 20) / (tile_m * k * 2);
        const group = @max(1, @min(rows_t, band));
        if (outs.len == 1) if (kk.splitCols(outs[0].q.n, rows)) |full_cols| {
            const o = outs[0];
            const part: Part = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = @intCast(o.q.n), .npad = @intCast(o.q.npad), .first = 0 };
            var g: launch_.Args = .{};
            g.add(x);
            g.add(part);
            for ([_]usize{ rows, k, k, group, full_cols }) |v| g.add(@as(c_int, @intCast(v)));
            const blocks = rows_t * (full_cols / tile_n) + rows_t * ((o.q.n - full_cols) / (tile_n / 2));
            return launch_.launch(kk.split_fn, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = smem }, s, &g);
        };
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
        // more tiles than resident CTAs and a part-full last wave: the same chains spread evenly over them (stream-K)
        if (kk.sk_slots > 0 and blocks > kk.sk_slots and blocks % kk.sk_slots != 0) {
            kk.epoch +%= 1;
            if (kk.epoch == 0) kk.epoch = 1;
            g.add(@as(c_int, @intCast(blocks)));
            g.add(kk.ws);
            g.add(kk.flags);
            g.add(kk.epoch);
            return launch_.launch(kk.sk_fn, .{ .grid = .{ .x = @intCast(kk.sk_slots) }, .block = .{ .x = threads }, .shared = smem, .cooperative = true }, s, &g);
        }
        try launch_.launch(kk.many_fn, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = smem }, s, &g);
    }
};
