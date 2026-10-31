//! Prompt GEMMs of up to four affine-4 projections of one input in one launch (qmm_prompt_many.cuh).
//!
//! Each output keeps qmm_prefill's prefill_kernel bits: the same bf16 weights, the same m16n8k16 MMA chain over
//! K into one fp32 accumulator, the same bf16 store. The tile is prefill_kernel's 128 x 128 on 1 x 4 warps with
//! two stages, so two CTAs share a GB10 SM, and the projections' tiles share one
//! launch: a small projection (GDN's a and b, attention's k and v) runs beside a large one instead of alone.
//!
//! `variants` are the same chains on other warp layouts, stage counts and tile placements (plain, stream-K, a split
//! last panel); the engine times them at open on the device it runs on (cuda_prompt_tune.zig) and keeps the
//! quickest per projection class and row-tile count in `tuned`. Without a tuned choice a call takes the default.
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
/// A stream-K CTA's parked accumulators: its 128 x 128 tile's fp32 (every warp layout).
pub const sk_slot_bytes = tile_m * tile_n * 4;
pub const parts_max = 4;
pub const tile_m = 128;
pub const tile_n = 128;
pub const threads = 128;
/// Tile<64, 128, 128, 2, 2, 2>::SMEM: two stages of 128 input rows, 128 columns' words, scales and biases.
pub const smem: u32 = 2 * stage_bytes;
const stage_bytes: u32 = 128 * 128 + 128 * 32 + 2 * 128 * 2;

/// qmm_prompt_many.cuh's ManyPart and ManyParts, passed by value.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, first: c_int };
pub const Parts = extern struct { p: [parts_max]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 48 and @sizeOf(Parts) == 200);
    std.debug.assert(smem == 41984);
}

/// One projection of the shared input and its output rows.
pub const Out = struct { q: Affine4, out: u64 };

/// How a variant lays a launch out: one tile a CTA, stream-K over the resident CTAs, a single projection's last
/// panels at half width, or the fused gate, up and activation.
pub const Kind = enum { many, sk, split, mlp };
pub const Variant = struct { name: [:0]const u8, kind: Kind, wm: u32, wn: u32, stages: u32 };
pub const variants = [_]Variant{
    .{ .name = symbol, .kind = .many, .wm = 1, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_many_222", .kind = .many, .wm = 2, .wn = 2, .stages = 2 },
    .{ .name = "tf_pmv_many_242", .kind = .many, .wm = 2, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_many_223", .kind = .many, .wm = 2, .wn = 2, .stages = 3 },
    .{ .name = "tf_pmv_many_143", .kind = .many, .wm = 1, .wn = 4, .stages = 3 },
    .{ .name = sk_symbol, .kind = .sk, .wm = 1, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_sk_222", .kind = .sk, .wm = 2, .wn = 2, .stages = 2 },
    .{ .name = "tf_pmv_sk_242", .kind = .sk, .wm = 2, .wn = 4, .stages = 2 },
    .{ .name = split_symbol, .kind = .split, .wm = 1, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_split_242", .kind = .split, .wm = 2, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_split_143", .kind = .split, .wm = 1, .wn = 4, .stages = 3 },
    .{ .name = mlp_symbol, .kind = .mlp, .wm = 1, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_mlp_222", .kind = .mlp, .wm = 2, .wn = 2, .stages = 2 },
    .{ .name = "tf_pmv_mlp_242", .kind = .mlp, .wm = 2, .wn = 4, .stages = 2 },
    .{ .name = "tf_pmv_mlp_143", .kind = .mlp, .wm = 1, .wn = 4, .stages = 3 },
};
const v_many = 0;
const v_sk = 5;
const v_split = 8;
const v_mlp = 11;

fn variantSmem(v: Variant) u32 {
    return v.stages * stage_bytes;
}

fn variantThreads(v: Variant) u32 {
    return v.wm * v.wn * 32;
}

/// A tuned choice: the variant a projection class at a row-tile bucket takes.
pub const Choice = struct { key: u64, v: u8 };

pub const Kernels = struct {
    fns: [variants.len]Function,
    /// Each variant's CTAs resident on the device at once.
    resident: [variants.len]usize,
    /// CTAs the device runs at once: two of the default tiles an SM.
    slots: usize,
    /// Stream-K: whether the default grouped path takes it, and the parked accumulators and flags for `sk_cap` CTAs.
    sk_on: bool = false,
    sk_cap: usize = 0,
    ws: u64 = 0,
    flags: u64 = 0,
    epoch: u32 = 0,
    tuned: [48]Choice = undefined,
    n_tuned: usize = 0,

    /// From a loaded qmm_prefill module, on a device of `sms` SMs.
    pub fn resolve(prefill_mod: module.Module, sms: usize) !Kernels {
        var kk: Kernels = .{ .fns = undefined, .resident = undefined, .slots = 2 * sms };
        for (variants, &kk.fns, &kk.resident) |v, *f, *r| {
            f.* = try prefill_mod.function(v.name);
            try f.allowDynamicShared(variantSmem(v));
            r.* = @as(usize, try f.occupancy(variantThreads(v), variantSmem(v))) * sms;
            if (v.kind == .sk) kk.sk_cap = @max(kk.sk_cap, r.*);
        }
        return kk;
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

    /// The row-tile counts a choice covers: 1, 2, 3-4, 5-8 and more.
    pub fn bucket(rows: usize) usize {
        const rows_t = (rows + tile_m - 1) / tile_m;
        return if (rows_t <= 2) rows_t else if (rows_t <= 4) 4 else if (rows_t <= 8) 8 else 16;
    }

    /// A projection class: the kind of call, its input width and its parts' widths, at a row-tile bucket.
    pub fn keyOf(fused: bool, k: usize, ns: []const usize, rows: usize) u64 {
        var h = std.hash.Wyhash.init(0x9e3779b97f4a7c15);
        h.update(std.mem.asBytes(&@as(u8, @intFromBool(fused))));
        h.update(std.mem.asBytes(&@as(u64, k)));
        for (ns) |n| h.update(std.mem.asBytes(&@as(u64, n)));
        h.update(std.mem.asBytes(&@as(u64, bucket(rows))));
        return h.final();
    }

    fn outsKey(outs: []const Out, rows: usize) u64 {
        var ns: [parts_max]usize = undefined;
        for (outs, ns[0..outs.len]) |o, *n| n.* = o.q.n;
        return keyOf(false, outs[0].q.k, ns[0..outs.len], rows);
    }

    fn mlpKey(gate: Affine4, rows: usize) u64 {
        return keyOf(true, gate.k, &.{gate.n}, rows);
    }

    pub fn choose(kk: *Kernels, key: u64, v: u8) void {
        for (kk.tuned[0..kk.n_tuned]) |*c| if (c.key == key) {
            c.v = v;
            return;
        };
        if (kk.n_tuned == kk.tuned.len) return;
        kk.tuned[kk.n_tuned] = .{ .key = key, .v = v };
        kk.n_tuned += 1;
    }

    fn chosen(kk: *const Kernels, key: u64) ?u8 {
        for (kk.tuned[0..kk.n_tuned]) |c| if (c.key == key) return c.v;
        return null;
    }

    /// Whether variant `v` can run `outs` (null outs: the fused MLP) at all.
    pub fn takes(kk: *const Kernels, v: usize, outs: ?[]const Out, rows: usize) bool {
        const vt = variants[v];
        if (outs) |os| {
            return switch (vt.kind) {
                .many => true,
                .sk => kk.ws != 0 and kk.resident[v] <= kk.sk_cap and kk.resident[v] > 0,
                .split => os.len == 1 and os[0].q.n % tile_n == 0 and os[0].q.n / tile_n >= 10 and rows > 0,
                .mlp => false,
            };
        }
        return vt.kind == .mlp;
    }

    /// act (rows, n) = bf16(silu(gate) * up) of x (rows, k) through gate and up: the bits of `prompt` over both,
    /// each stored as bf16, then qwen35.cu's tf_q35_mlp_act over them; neither projection's rows are written.
    pub fn mlp(kk: *Kernels, s: Stream, x: u64, gate: Affine4, up: Affine4, act: u64, rows: usize) !void {
        return kk.mlpVariant(kk.chosen(mlpKey(gate, rows)) orelse v_mlp, s, x, gate, up, act, rows);
    }

    pub fn mlpVariant(kk: *Kernels, v: usize, s: Stream, x: u64, gate: Affine4, up: Affine4, act: u64, rows: usize) !void {
        if (!mlpFits(gate, up) or rows == 0) return error.InputMismatch;
        if (variants[v].kind != .mlp) return error.InputMismatch;
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
        for ([_]usize{ rows, k, k, group }) |n| g.add(@as(c_int, @intCast(n)));
        const blocks = rows_t * (gate.n / 64);
        const vt = variants[v];
        try launch_.launch(kk.fns[v], .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = variantThreads(vt) }, .shared = variantSmem(vt) }, s, &g);
    }

    /// out_i (rows, n_i) = x (rows, k) through each of `outs`, every one with prefill_kernel's bits.
    pub fn prompt(kk: *Kernels, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > parts_max or rows == 0) return error.TooManyParts;
        if (kk.chosen(outsKey(outs, rows))) |v| if (kk.takes(v, outs, rows)) return kk.promptVariant(v, s, x, outs, rows);
        return kk.promptVariant(kk.defaultFor(outs, rows), s, x, outs, rows);
    }

    /// The untuned choice: a single wide projection's split last panels when its last wave is part full, else the
    /// grouped tile, stream-K when there are more tiles than resident CTAs and a part-full last wave.
    pub fn defaultFor(kk: *const Kernels, outs: []const Out, rows: usize) usize {
        if (outs.len == 1) if (kk.splitCols(outs[0].q.n, rows) != null) return v_split;
        const rows_t = (rows + tile_m - 1) / tile_m;
        var blocks: usize = 0;
        for (outs) |o| blocks += rows_t * ((o.q.n + tile_n - 1) / tile_n);
        const sk = kk.resident[v_sk];
        if (kk.sk_on and kk.ws != 0 and blocks > sk and blocks % sk != 0) return v_sk;
        return v_many;
    }

    pub fn promptVariant(kk: *Kernels, v: usize, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > parts_max or rows == 0) return error.TooManyParts;
        const vt = variants[v];
        const k = outs[0].q.k;
        const rows_t = (rows + tile_m - 1) / tile_m;
        // prefill_kernel's row band: the row tiles of a column panel that share it from L2
        const band = (12 << 20) / (tile_m * k * 2);
        const group = @max(1, @min(rows_t, band));
        const cfg_threads = variantThreads(vt);
        const cfg_smem = variantSmem(vt);
        if (vt.kind == .split) {
            const o = outs[0];
            const cols = o.q.n / tile_n;
            if (outs.len != 1 or o.q.n % tile_n != 0 or cols < 10) return error.InputMismatch;
            const full_cols = (cols - cols / 5) * tile_n;
            const part: Part = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = @intCast(o.q.n), .npad = @intCast(o.q.npad), .first = 0 };
            var g: launch_.Args = .{};
            g.add(x);
            g.add(part);
            for ([_]usize{ rows, k, k, group, full_cols }) |n| g.add(@as(c_int, @intCast(n)));
            const blocks = rows_t * (full_cols / tile_n) + rows_t * ((o.q.n - full_cols) / (tile_n / 2));
            return launch_.launch(kk.fns[v], .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = cfg_threads }, .shared = cfg_smem }, s, &g);
        }
        if (vt.kind == .mlp) return error.InputMismatch;
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
        for ([_]usize{ rows, k, k, group }) |n| g.add(@as(c_int, @intCast(n)));
        if (vt.kind == .sk) {
            // the same chains spread evenly over the resident CTAs (never more CTAs than tiles: a tile spans at most two)
            const grid = @min(kk.resident[v], blocks);
            if (kk.ws == 0 or grid > kk.sk_cap) return error.InputMismatch;
            kk.epoch +%= 1;
            if (kk.epoch == 0) kk.epoch = 1;
            g.add(@as(c_int, @intCast(blocks)));
            g.add(kk.ws);
            g.add(kk.flags);
            g.add(kk.epoch);
            return launch_.launch(kk.fns[v], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = cfg_threads }, .shared = cfg_smem, .cooperative = true }, s, &g);
        }
        try launch_.launch(kk.fns[v], .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = cfg_threads }, .shared = cfg_smem }, s, &g);
    }
};
