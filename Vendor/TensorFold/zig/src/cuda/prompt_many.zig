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
pub const parts_max = 4;
pub const tile_m = 128;
pub const tile_n = 128;
pub const threads = 128;
/// Tile<64, 128, 128, 2, 2, 2>::SMEM: two stages of 128 input rows, 128 columns' words, scales and biases.
pub const smem: u32 = 2 * (128 * 128 + 128 * 32 + 2 * 128 * 2);
/// Three stages of the same stage bytes (qmm_prompt_pipe3.cuh): the PTILE deeper cp.async pipeline.
pub const smem3: u32 = 3 * (128 * 128 + 128 * 32 + 2 * 128 * 2);
/// Tile<64, 64, 128, 1, 4, 2>: two stages of 64 input rows (PFL3 m64 bodies, TF_PFL3_M64=1).
pub const smem64: u32 = 2 * (64 * 128 + 128 * 32 + 2 * 128 * 2);

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
    /// CTAs the device runs at once: two of these tiles an SM.
    slots: usize,
    /// G6B: slot cap override for the rows2/rows3 persistent prompt GEMMs only (0: crown, `slots`).
    /// Scheduling only — the tile set and each tile's arithmetic are unchanged, so the outputs keep
    /// the persistent kernels' bits at any cap. mlp (persistent_fns[0]) stays on `slots`.
    g6: usize = 0,
    /// Native MLP, two-row normal/split projection, and three-row grouped projection.
    persistent_fns: [4]?Function = @splat(null),
    /// PTILE 4x2-warp non-persistent many body (qmm_prefill.cu tf_qmm_prompt_manyw42), TF_PTILE_W42=1.
    manyw42_fn: ?Function = null,
    /// PFL3 BM=64 non-persistent bodies (tf_qmm_prompt_{many,mlp,split}64), TF_PFL3_M64=1, rows<=64 only.
    many64_fn: ?Function = null,
    mlp64_fn: ?Function = null,
    split64_fn: ?Function = null,
    /// PTILE 3-stage bodies (qmm_prompt_pipe3.cuh), same four roles, selected by TF_PTILE_STAGES=3.
    many3_fn: ?Function = null,
    mlp3_fn: ?Function = null,
    split3_fn: ?Function = null,
    persistent3_fns: [4]?Function = @splat(null),
    st3: bool = false,

    /// Dynamic shared bytes of the active pipeline depth.
    pub fn smem_now(kk: *const Kernels) u32 {
        return if (kk.st3) smem3 else smem;
    }

    /// From a loaded qmm_prefill module, on a device of `sms` SMs.
    pub fn resolve(prefill_mod: module.Module, sms: usize) !Kernels {
        const f = try prefill_mod.function(symbol);
        try f.allowDynamicShared(smem);
        const m = try prefill_mod.function(mlp_symbol);
        try m.allowDynamicShared(smem);
        const sp = try prefill_mod.function(split_symbol);
        try sp.allowDynamicShared(smem);
        var result: Kernels = .{ .many_fn = f, .mlp_fn = m, .split_fn = sp, .slots = 2 * sms };
        if (std.c.getenv("TF_G6GRID")) |v| {
            const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
            if (n >= 96 and n <= 8 * sms and n % 4 == 0) result.g6 = n; // 4: the crown M-tile cohort
        }
        const disabled = if (std.c.getenv("TF_PROMPT_PERSISTENT")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        const st3 = if (std.c.getenv("TF_PTILE_STAGES")) |v| std.mem.eql(u8, std.mem.span(v), "3") else false;
        const w42 = if (std.c.getenv("TF_PTILE_W42")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (w42) {
            result.manyw42_fn = try prefill_mod.function("tf_qmm_prompt_manyw42");
            try result.manyw42_fn.?.allowDynamicShared(smem);
        }
        // PFL3 m64 default ON in this candidate (owner directive V10H60+PFL3): TF_PFL3_M64=0 restores the crown persistent-MLP path.
        const m64 = if (std.c.getenv("TF_PFL3_M64")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        if (m64) {
            result.many64_fn = try prefill_mod.function("tf_qmm_prompt_many64");
            try result.many64_fn.?.allowDynamicShared(smem64);
            result.mlp64_fn = try prefill_mod.function("tf_qmm_prompt_mlp64");
            try result.mlp64_fn.?.allowDynamicShared(smem64);
            result.split64_fn = try prefill_mod.function("tf_qmm_prompt_split64");
            try result.split64_fn.?.allowDynamicShared(smem64);
        }
        if (st3) {
            result.many3_fn = try prefill_mod.function("tf_qmm_prompt_many3");
            try result.many3_fn.?.allowDynamicShared(smem3);
            result.mlp3_fn = try prefill_mod.function("tf_qmm_prompt_mlp3");
            try result.mlp3_fn.?.allowDynamicShared(smem3);
            result.split3_fn = try prefill_mod.function("tf_qmm_prompt_split3");
            try result.split3_fn.?.allowDynamicShared(smem3);
            const names3 = [_][:0]const u8{ "tf_qmm_persist3_mlp", "tf_qmm_persist3_rows2", "tf_qmm_persist3_rows2s", "tf_qmm_persist3_rows3" };
            for (names3, &result.persistent3_fns) |name, *dest| {
                const kernel = try prefill_mod.function(name);
                try kernel.allowDynamicShared(smem3);
                dest.* = kernel;
            }
            result.st3 = true;
        }
        if (sms == 48 and !disabled) {
            const names = [_][:0]const u8{
                "_ZN17tf_qmm_persistent6kernelILi128ELi1ELb1ELb0EEEvPK13__nv_bfloat16N14tf_qmm_prefill9ManyPartsEPS1_iiiiiii",
                "_ZN23tf_qmm_persistent_rows26kernelILi128ELi1ELb0ELb0EEEvPK13__nv_bfloat16N14tf_qmm_prefill9ManyPartsEPS1_iiiiiii",
                "_ZN23tf_qmm_persistent_rows26kernelILi128ELi1ELb0ELb1EEEvPK13__nv_bfloat16N14tf_qmm_prefill9ManyPartsEPS1_iiiiiii",
                "_ZN23tf_qmm_persistent_rows36kernelILi128ELi1ELb0ELb0EEEvPK13__nv_bfloat16N14tf_qmm_prefill9ManyPartsEPS1_iiiiiii",
            };
            for (names, &result.persistent_fns) |name, *dest| {
                const kernel = try prefill_mod.function(name);
                try kernel.allowDynamicShared(smem);
                dest.* = kernel;
            }
        }
        return result;
    }

    fn persistent(kk: *const Kernels, f: Function, s: Stream, x: u64, parts: Parts, act: u64, rows: usize, k: usize, group: usize, full_cols: c_int, blocks: usize) !void {
        return persistentCap(kk, kk.slots, f, s, x, parts, act, rows, k, group, full_cols, blocks);
    }

    /// G6B: the same launch at an explicit slot cap (rows2/rows3 only; `persistent` keeps crown).
    fn persistentCap(kk: *const Kernels, cap: usize, f: Function, s: Stream, x: u64, parts: Parts, act: u64, rows: usize, k: usize, group: usize, full_cols: c_int, blocks: usize) !void {
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        g.add(act);
        for ([_]usize{ rows, k, k, group }) |v| g.add(@as(c_int, @intCast(v)));
        g.add(full_cols);
        g.add(@as(c_int, @intCast(blocks)));
        g.add(@as(c_int, 1));
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(@min(blocks, cap)) }, .block = .{ .x = threads }, .shared = kk.smem_now() }, s, &g);
    }

    fn persistentRows(kk: *const Kernels, rows: usize) bool {
        return (kk.slots == 96 or kk.g6 != 0) and rows >= 256;
    }

    /// G6B: the effective slot cap of the rows2/rows3 persistent launches (crown: `slots`).
    fn g6cap(kk: *const Kernels) usize {
        return if (kk.g6 != 0) kk.g6 else kk.slots;
    }

    fn outputRole(outs: []const Out) bool {
        return outs.len == 1 and outs[0].q.n == 5120 and (outs[0].q.k == 4096 or outs[0].q.k == 6144 or outs[0].q.k == 17408 or outs[0].q.k == 25600);
    }

    fn recurrentRole(outs: []const Out) bool {
        return outs.len >= 2 and outs[0].q.k == 5120;
    }

    /// The columns one projection's whole 128-wide tiles cover when its last panels run 64 wide (null: none do).
    /// Whole tiles past one wave leave a last wave part full (a 496-row prompt's 5120 columns: 160 tiles on a GB10's
    /// 96 slots); a fifth of the panels at half width evens the SMs' shares.
    pub fn splitCols(kk: *const Kernels, n: usize, rows: usize) ?usize {
        const cols = n / tile_n;
        const tiles = (rows + tile_m - 1) / tile_m * cols;
        if (n % tile_n != 0 or cols < 10 or tiles % kk.slots == 0) return null;
        if (tiles < kk.slots) return 0;
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
        const band = (32 << 20) / (tile_m * k * 2);
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
        if (kk.mlp64_fn != null and rows <= 64 and gate.n == 17408 and gate.k == 5120) {
            // rows<=64: one 64-row band, so parts/group/full grid keep the crown values and the BM=64 body
            // skips the padding rows the BM=128 tile issues MMAs and ldmatrix loads for (mb4: -26% at T=32).
            return launch_.launch(kk.mlp64_fn.?, .{ .grid = .{ .x = @intCast(((rows + 63) / 64) * (gate.n / 64)) }, .block = .{ .x = threads }, .shared = smem64 }, s, &g);
        }
        const mlpk = if (kk.st3 and kk.mlp3_fn != null) kk.mlp3_fn.? else kk.mlp_fn;
        if (kk.slots == 96 and blocks > kk.slots and gate.n == 17408 and gate.k == 5120) {
            if (kk.st3 and kk.persistent3_fns[0] != null) if (kk.persistent3_fns[0]) |f|
                return kk.persistent(f, s, x, parts, act, rows, k, group, -1, blocks);
            if (kk.persistent_fns[0]) |f|
                return kk.persistent(f, s, x, parts, act, rows, k, group, -1, blocks);
        }
        try launch_.launch(mlpk, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = kk.smem_now() }, s, &g);
    }

    /// out_i (rows, n_i) = x (rows, k) through each of `outs`, every one with prefill_kernel's bits.
    pub fn prompt(kk: *const Kernels, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > parts_max or rows == 0) return error.TooManyParts;
        const k = outs[0].q.k;
        const rows_t = (rows + tile_m - 1) / tile_m;
        // prefill_kernel's row band: the row tiles of a column panel that share it from L2
        const band = (32 << 20) / (tile_m * k * 2);
        const group = @max(1, @min(rows_t, band));
        if (outs.len == 1) if (kk.splitCols(outs[0].q.n, rows)) |full_cols| {
            const o = outs[0];
            const part: Part = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = @intCast(o.q.n), .npad = @intCast(o.q.npad), .first = 0 };
            var g: launch_.Args = .{};
            g.add(x);
            g.add(part);
            for ([_]usize{ rows, k, k, group, full_cols }) |v| g.add(@as(c_int, @intCast(v)));
            const blocks = rows_t * (full_cols / tile_n) + rows_t * ((o.q.n - full_cols) / (tile_n / 2));
            if (kk.persistentRows(rows) and outputRole(outs)) {
                if (kk.st3 and kk.persistent3_fns[2] != null) if (kk.persistent3_fns[2]) |f| {
                    var parts: Parts = std.mem.zeroes(Parts);
                    parts.count = 1;
                    parts.p[0] = part;
                    return kk.persistentCap(kk.g6cap(), f, s, x, parts, 0, rows, k, group, @intCast(full_cols), blocks);
                };
                if (kk.persistent_fns[2]) |f| {
                    var parts: Parts = std.mem.zeroes(Parts);
                    parts.count = 1;
                    parts.p[0] = part;
                    return kk.persistentCap(kk.g6cap(), f, s, x, parts, 0, rows, k, group, @intCast(full_cols), blocks);
                }
            }
            if (kk.split64_fn != null and rows <= 64) {
                // rows<=64: rows_t=1 so blocks/group keep the crown values (full_cols from splitCols holds).
                return launch_.launch(kk.split64_fn.?, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = smem64 }, s, &g);
            }
            const splitk = if (kk.st3 and kk.split3_fn != null) kk.split3_fn.? else kk.split_fn;
            return launch_.launch(splitk, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = kk.smem_now() }, s, &g);
        };
        var parts: Parts = std.mem.zeroes(Parts);
        var blocks: usize = 0;
        for (outs, 0..) |o, i| {
            if (o.q.k != k) return error.InputMismatch;
            parts.p[i] = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = @intCast(o.q.n), .npad = @intCast(o.q.npad), .first = @intCast(blocks) };
            blocks += rows_t * ((o.q.n + tile_n - 1) / tile_n);
        }
        parts.count = @intCast(outs.len);
        const selected: ?usize = if (outputRole(outs)) 1 else if (recurrentRole(outs)) 3 else null;
        if (kk.persistentRows(rows) and blocks > kk.slots) if (selected) |index| {
            if (kk.st3 and kk.persistent3_fns[index] != null) if (kk.persistent3_fns[index]) |f|
                return kk.persistentCap(kk.g6cap(), f, s, x, parts, 0, rows, k, group, -1, blocks);
            if (kk.persistent_fns[index]) |f|
                return kk.persistentCap(kk.g6cap(), f, s, x, parts, 0, rows, k, group, -1, blocks);
        };
        if (kk.many64_fn != null and rows <= 64) {
            // rows<=64: rows_t=1, so parts' first offsets and group hold; the BM=64 body needs no rebuild.
            var g64: launch_.Args = .{};
            g64.add(x);
            g64.add(parts);
            for ([_]usize{ rows, k, k, group }) |v| g64.add(@as(c_int, @intCast(v)));
            return launch_.launch(kk.many64_fn.?, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = smem64 }, s, &g64);
        }
        const manyk = if (kk.st3 and kk.many3_fn != null) kk.many3_fn.? else kk.many_fn;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        for ([_]usize{ rows, k, k, group }) |v| g.add(@as(c_int, @intCast(v)));
        try launch_.launch(manyk, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = threads }, .shared = kk.smem_now() }, s, &g);
    }
};
