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
    /// Native MLP, two-row normal/split projection, and three-row grouped projection.
    persistent_fns: [4]?Function = @splat(null),
    /// PEER-11 (terrapinelf/00bbc635): a 496-row prompt's planned tiles (qmm_prompt_plan.cuh); null off GB10's 48 SMs.
    plan_fn: ?Function = null,
    plan_mlp_fn: ?Function = null,
    /// TF_PROMPT_PLAN's bits (default all): 1 the 5120-column projections, 2 attention's q, k and v, 4 the MLP, 8 the
    /// recurrent layers' qkv, z, a and b.
    plan_mask: u32 = 1,
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
        const disabled = if (std.c.getenv("TF_PROMPT_PERSISTENT")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        const st3 = if (std.c.getenv("TF_PTILE_STAGES")) |v| std.mem.eql(u8, std.mem.span(v), "3") else false;
        const w42 = if (std.c.getenv("TF_PTILE_W42")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (w42) {
            result.manyw42_fn = try prefill_mod.function("tf_qmm_prompt_manyw42");
            try result.manyw42_fn.?.allowDynamicShared(smem);
        }
        const m64 = if (std.c.getenv("TF_PFL3_M64")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
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
            // CAP3-03: TF_ROWS2BAL=1 swaps the persistent rows2 split schedule for the wave-balanced
            // map (qmm_prompt_persist_rows2v3.cuh): same tf_qmm_pipe tile bodies and bits, only the
            // tile->CTA order changes (the short row band carries a plain full; pair makespan 1.875 vs
            // the crown's 2.0 full-tile equivalents at the scored M=496/N=5120 shape). Default off.
            if (std.c.getenv("TF_ROWS2BAL")) |v| {
                if (std.mem.eql(u8, std.mem.span(v), "1")) {
                    const bal = try prefill_mod.function("_ZN22tf_qmm_persist_rows2v36kernelILi128ELi1ELb0ELb1EEEvPK13__nv_bfloat16N14tf_qmm_prefill9ManyPartsEPS1_iiiiiii");
                    try bal.allowDynamicShared(smem);
                    result.persistent_fns[2] = bal;
                }
            }
            // PEER-11 (terrapinelf/00bbc635): planned-tile kernels for the 496-row prompt, TF_PROMPT_PLAN-masked.
            const pf = try prefill_mod.function("tf_qmm_prompt_plan");
            try pf.allowDynamicShared(smem);
            const pm = try prefill_mod.function("tf_qmm_prompt_plan_mlp");
            try pm.allowDynamicShared(smem);
            result.plan_fn = pf;
            result.plan_mlp_fn = pm;
            if (std.c.getenv("TF_PROMPT_PLAN")) |v| result.plan_mask = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1;
        }
        return result;
    }

    /// PEER-11 (terrapinelf/00bbc635): qmm_prompt_plan_tables.cuh's plans: (first item, items), each for exactly a
    /// 496-row prompt on 96 slots.
    const plan_rows = 496;
    const plans = [_][2]c_int{ .{ 0, 192 }, .{ 192, 576 }, .{ 768, 1248 }, .{ 2016, 576 } };

    /// The plan for these projections of a `rows`-row prompt (null: the persistent schedules): plan 0 one 5120-column
    /// output (out, o, down), plan 1 attention's q (with its gate), k and v of a 5120-wide input, plan 3 a recurrent
    /// layer's qkv, z, a and b of a 5120-wide input.
    fn planFor(kk: *const Kernels, outs: []const Out, rows: usize) ?usize {
        if (kk.plan_fn == null or kk.slots != 96 or rows != plan_rows) return null;
        if (outputRole(outs) and kk.plan_mask & 1 != 0) return 0;
        if (outs.len == 3 and outs[0].q.k == 5120 and outs[0].q.n == 12288 and outs[1].q.n == 1024 and outs[2].q.n == 1024 and
            outs[1].q.k == 5120 and outs[2].q.k == 5120 and kk.plan_mask & 2 != 0) return 1;
        if (outs.len == 4 and kk.plan_mask & 8 != 0 and outs[0].q.n == 10240 and outs[1].q.n == 6144 and outs[2].q.n == 48 and
            outs[3].q.n == 48 and for (outs) |o| {
            if (o.q.k != 5120) break false;
        } else true) return 3;
        return null;
    }

    /// PEER-11 (terrapinelf/00bbc635): launch plan `plan` on the 96 persistent CTAs.
    fn planned(kk: *const Kernels, f: Function, s: Stream, x: u64, parts: Parts, act: ?u64, rows: usize, k: usize, plan: usize) !void {
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        if (act) |a| g.add(a);
        for ([_]usize{ rows, k, k }) |v| g.add(@as(c_int, @intCast(v)));
        g.add(plans[plan][0]);
        g.add(plans[plan][1]);
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(kk.slots) }, .block = .{ .x = threads }, .shared = smem }, s, &g);
    }

    fn persistent(kk: *const Kernels, f: Function, s: Stream, x: u64, parts: Parts, act: u64, rows: usize, k: usize, group: usize, full_cols: c_int, blocks: usize) !void {
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        g.add(act);
        for ([_]usize{ rows, k, k, group }) |v| g.add(@as(c_int, @intCast(v)));
        g.add(full_cols);
        g.add(@as(c_int, @intCast(blocks)));
        g.add(@as(c_int, 1));
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(@min(blocks, kk.slots)) }, .block = .{ .x = threads }, .shared = kk.smem_now() }, s, &g);
    }

    fn persistentRows(kk: *const Kernels, rows: usize) bool {
        return kk.slots == 96 and rows >= 256;
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
        // PEER-11 (terrapinelf/00bbc635): the MLP's planned tiles (plan 2) at the 496-row prompt.
        if (kk.slots == 96 and rows == plan_rows and gate.n == 17408 and gate.k == 5120 and kk.plan_mask & 4 != 0) if (kk.plan_mlp_fn) |f|
            return kk.planned(f, s, x, parts, act, rows, k, 2);
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
        // PEER-11 (terrapinelf/00bbc635): planned tiles for a 496-row prompt's projections (plan 0/1/3).
        if (kk.planFor(outs, rows)) |plan| {
            var plan_parts: Parts = std.mem.zeroes(Parts);
            for (outs, 0..) |o, i| {
                if (o.q.k != k) return error.InputMismatch;
                plan_parts.p[i] = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = @intCast(o.q.n), .npad = @intCast(o.q.npad), .first = 0 };
            }
            plan_parts.count = @intCast(outs.len);
            return kk.planned(kk.plan_fn.?, s, x, plan_parts, null, rows, k, plan);
        }
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
                    return kk.persistent(f, s, x, parts, 0, rows, k, group, @intCast(full_cols), blocks);
                };
                if (kk.persistent_fns[2]) |f| {
                    var parts: Parts = std.mem.zeroes(Parts);
                    parts.count = 1;
                    parts.p[0] = part;
                    return kk.persistent(f, s, x, parts, 0, rows, k, group, @intCast(full_cols), blocks);
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
                return kk.persistent(f, s, x, parts, 0, rows, k, group, -1, blocks);
            if (kk.persistent_fns[index]) |f|
                return kk.persistent(f, s, x, parts, 0, rows, k, group, -1, blocks);
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
