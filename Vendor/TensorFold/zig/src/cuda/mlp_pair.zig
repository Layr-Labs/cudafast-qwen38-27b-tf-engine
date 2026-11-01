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
/// The dictionary twins (qmm_dict.cuh): gate's and up's scales and biases from their dictionary indices, the same bits.
pub const symbol_dict = "tf_qmm_mlp_pair_dict";
pub const symbol_single_dict = "tf_qmm_mlp_single16_dict";
/// The stages' memory and, past it, the (16, 64) bf16 gate tile rank 0 stores into rank 1's.
pub const smem: u32 = qlinear.group_smem + 16 * 64 * 2;
pub const kmajor_smem: u32 = 13440 + 16 * 128 * 2;
pub const single_smem: u32 = 41088;
/// Pair-only half panel: two weight/index/sum stages, direct input fragments.
pub const compact_pair_smem: u32 = 17536;
/// tf_qmm_mlp_single32: two stages of 32 input rows, four 2 KB weight tiles, 256 scales and biases, 32 sums.
pub const single32_smem: u32 = 2 * (32 * 128 + 256 * 32 + 2 * 256 * 2 + 32 * 4);
/// tf_qmm_mlp_single16_dict past single_smem: four warps' decoded values, then gate's and up's tables.
pub const single_dict_dec: u32 = 512;
/// Gate's and up's tables at most for the twin to keep tf_qmm_mlp_single16's two CTAs an SM (GB10: 100 KiB an SM,
/// 1 KiB of it reserved a CTA), so its 68 CTAs stay one wave; a layer past it keeps the original kernel.
pub const single_tab_max: u32 = 102400 / 2 - 1024 - single_smem - single_dict_dec;
/// tf_qmm_mlp_twin32 (qmm_stream32.cuh): tf_qmm_mlp_single16 on 17 to 32 rows, eight warps a CTA; its stages, the
/// inputs of both m16 row tiles and eight warps' pass values, then (the dictionary twin) gate's and up's tables.
pub const twin32_smem: u32 = 41216 + 1024;
pub const twin32_tab_max: u32 = 102400 / 2 - 1024 - twin32_smem;
/// tf_qmm_mlp_twin32_pair: its stages and inputs, then eight warps' four passes' values (no table in shared memory)
pub const twin32_pair_smem: u32 = 45312 + 4096; // each stage holds its group's inputs (qmm_stream32.cuh MBytes2)

pub const Kernels = struct {
    pair_fn: Function,
    pdl: bool,
    kmajor_fn: ?Function = null,
    kmajor_single: bool = false,
    single32_fn: ?Function = null, // tf_qmm_mlp_single32: 17 to 32-row windows on the K-major panels, one weight pass
    single32_dict_fn: ?Function = null, // tf_qmm_mlp_single32_dict: dictionary-coded twin on the K-major panels
    dict_fn: ?Function = null, // GB10: tf_qmm_mlp_pair's twin, taken when gate and up are both coded (tile-major)
    compact_pair_fn: ?Function = null,
    single_dict_fn: ?Function = null, // tf_qmm_mlp_single16's, when both are coded on the K-major panel's layout
    // tf_qmm_mlp_single16's 32-row twins (qmm_stream32.cuh), on the same K-major panels; TF_STREAM32=0: none loaded
    twin32_fn: ?Function = null, // tf_qmm_mlp_twin32: verify windows of 17 to 32 rows on tf_qmm_mlp_single16's panel
    twin32_dict_fn: ?Function = null, // its dictionary twin
    twin32_pair_fn: ?Function = null, // ... on pair-coded gate and up (both)
    twin32_mixed_fn: ?Function = null, // ... one pair-coded, one value-coded
    twin32_half_fn: ?Function = null, // tf_qmm_mlp_half32_pair: both pair-coded, on 136 half CTAs (TF_MLP32_HALF=0: off)

    /// From the loaded qmm_group module; `pdl` as the decode tiles' launches (GB10).
    pub fn resolve(group_mod: module.Module, pdl: bool) !Kernels {
        const f = try group_mod.function(symbol);
        try f.allowDynamicShared(smem);
        var k: Kernels = .{ .pair_fn = f, .pdl = pdl };
        if (pdl) {
            const fd = try group_mod.function(symbol_dict);
            try fd.allowDynamicShared(smem + qlinear.dict_dec + qlinear.dict_tab_max);
            k.dict_fn = fd;
            const fs = try group_mod.function(symbol_single_dict);
            try fs.allowDynamicShared(single_smem + single_dict_dec + 2 * qlinear.dict_tab_max);
            k.single_dict_fn = fs;
            const fc = try group_mod.function("tf_qmm_mlp_single16_compact_pair");
            try fc.allowDynamicShared(compact_pair_smem);
            // 50% rounds upward to the 64 KiB carveout on GB10. This is a hint,
            // allowing the driver to avoid disruptive per-node reconfiguration.
            try fc.setAttribute(.preferred_shared_memory_carveout, 50);
            k.compact_pair_fn = fc;
        }
        return k;
    }

    /// Both projections' twin of `layout`: gate and up coded, in it (qlinear.dict_kmajor: the K-major panel's).
    fn coded(gate: Affine4, up: Affine4, kmajor: bool) bool {
        const want: u32 = if (kmajor) qlinear.dict_kmajor else 0;
        return gate.code != 0 and up.code != 0 and gate.code & qlinear.dict_kmajor == want and up.code & qlinear.dict_kmajor == want;
    }

    /// tf_qmm_mlp_single16's shared table bytes for q: a pair code's table is read through L2 there (none).
    fn singleTab(q: Affine4) u32 {
        return if (qlinear.pairCode(q.code)) 0 else q.tabBytes();
    }

    /// Whether `pair` takes this gate and up: one shape whose decode() is group_kernel's one-slice tile.
    pub fn fits(gate: Affine4, up: Affine4) bool {
        return gate.n == up.n and gate.k == up.k and gate.npad == up.npad and gate.n % 64 == 0 and gate.k % 64 == 0 and
            qlinear.splitK(gate.n, gate.k) == 1;
    }

    /// act (rows, n) = bf16(silu(x through gate) * (x through up)), x's group sums in xs; act's group sums into as.
    pub fn pair(kk: *const Kernels, s: Stream, x: u64, xs: u64, gate: Affine4, up: Affine4, act: u64, as: u64, rows: usize) !void {
        if (!fits(gate, up) or rows == 0) return error.InputMismatch;
        if (rows > 16 and rows <= 32 and gate.n == 17408 and gate.k == 5120 and gate.verify_w != 0 and up.verify_w != 0 and
            kk.kmajor_single and kk.twin32_fn != null) return kk.pair32(s, x, xs, gate, up, act, as, rows);
        if (rows > 16 and rows <= 32 and gate.n == 17408 and gate.k == 5120 and gate.verify_w != 0 and up.verify_w != 0)
            if (kk.single32_fn) |fw| {
                const sdict32 = kk.single32_dict_fn != null and coded(gate, up, true) and singleTab(gate) + singleTab(up) <= single_tab_max;
                var wide: qlinear.Parts = std.mem.zeroes(qlinear.Parts);
                for ([2]Affine4{ gate, up }, wide.p[0..2]) |q, *p| p.* = .{
                    .w = q.verify_w,
                    .scales = if (sdict32) q.sb else q.s,
                    .biases = if (sdict32) q.tab else q.b,
                    .out = 0,
                    .n = @intCast(q.n),
                    .npad = @intCast(q.npad),
                    .sk = 1,
                    .tiles = @intCast(q.n / 128),
                    .first = if (sdict32) @bitCast(q.code & ~qlinear.dict_kmajor) else 0,
                };
                wide.count = 2;
                var a: launch_.Args = .{};
                a.add(x);
                a.add(xs);
                a.add(wide);
                a.add(act);
                a.add(as);
                for ([_]usize{ rows, gate.k, gate.k, 1 }) |v| a.add(@as(c_int, @intCast(v)));
                const f32_use = if (sdict32) kk.single32_dict_fn.? else fw;
                const sh32: u32 = if (sdict32) single32_smem + singleTab(gate) + singleTab(up) else single32_smem;
                return launch_.launch(f32_use, .{ .grid = .{ .x = @intCast(gate.n / 128) }, .block = .{ .x = 128 }, .shared = sh32, .pdl = kk.pdl }, s, &a);
            };
        const rows_t = (rows + 15) / 16;
        // a window of at most 16 rows: the single16 tile bounds its own rows (`min(BM, M - m0)`), so it reads the
        // K-major panels and their dictionary index stream (a byte a 64 weights) instead of the tile-major panels'
        // separate scale and bias arrays (four bytes a 64 weights) on every window narrower than 16 rows
        const major = rows <= 16 and gate.n == 17408 and gate.k == 5120 and
            gate.verify_w != 0 and up.verify_w != 0 and kk.kmajor_fn != null;
        const single = major and kk.kmajor_single;
        const compact = single and kk.single_dict_fn != null and kk.compact_pair_fn != null and coded(gate, up, true) and
            qlinear.pairCode(gate.code) and qlinear.pairCode(up.code);
        const tiles = gate.n / @as(usize, if (compact) 128 else if (single) 256 else if (major) 128 else 64);
        // the twins: tf_qmm_mlp_pair's (npad carries the code) and tf_qmm_mlp_single16's (first carries it; npad
        // stays, the panel's group pitch); tf_qmm_mlp_pair_kmajor16 has none
        const dict = !major and kk.dict_fn != null and coded(gate, up, false);
        const sdict = single and kk.single_dict_fn != null and coded(gate, up, true) and singleTab(gate) + singleTab(up) <= single_tab_max;
        var parts: qlinear.Parts = std.mem.zeroes(qlinear.Parts);
        for ([2]Affine4{ gate, up }, parts.p[0..2]) |q, *p| p.* = .{
            .w = q.w,
            .scales = if (dict or sdict) q.sb else q.s,
            .biases = if (dict or sdict) q.tab else q.b,
            .out = 0,
            .n = @intCast(q.n),
            .npad = if (dict) @bitCast(q.code) else @intCast(q.npad),
            .sk = 1,
            .tiles = @intCast(tiles),
            .first = if (sdict) @bitCast(q.code & ~qlinear.dict_kmajor) else 0,
        };
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
        const f = if (compact) kk.compact_pair_fn.? else if (sdict) kk.single_dict_fn.? else if (major) kk.kmajor_fn.? else if (dict) kk.dict_fn.? else kk.pair_fn;
        const shared: u32 = if (compact) compact_pair_smem else if (sdict) single_smem + single_dict_dec + singleTab(gate) + singleTab(up) else if (single) single_smem else if (major) kmajor_smem else if (dict) smem + qlinear.dict_dec + @max(gate.tabBytes(), up.tabBytes()) else smem;
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(@as(usize, if (single) 1 else 2) * rows_t * tiles) }, .block = .{ .x = 128 }, .shared = shared, .cluster = if (single) null else .{ .x = 2 }, .pdl = kk.pdl }, s, &g);
    }

    /// pair()'s tf_qmm_mlp_single16 launch on 17 to 32 rows: tf_qmm_mlp_twin32, both m16 row tiles off one panel stream.
    fn pair32(kk: *const Kernels, s: Stream, x: u64, xs: u64, gate: Affine4, up: Affine4, act: u64, as: u64, rows: usize) !void {
        const sdict = kk.twin32_dict_fn != null and coded(gate, up, true) and singleTab(gate) + singleTab(up) <= twin32_tab_max;
        const tiles = gate.n / 256;
        var parts: qlinear.Parts = std.mem.zeroes(qlinear.Parts);
        for ([2]Affine4{ gate, up }, parts.p[0..2]) |q, *p| p.* = .{
            .w = q.verify_w,
            .scales = if (sdict) q.sb else q.s,
            .biases = if (sdict) q.tab else q.b,
            .out = 0,
            .n = @intCast(q.n),
            .npad = @intCast(q.npad),
            .sk = 1,
            .tiles = @intCast(tiles),
            .first = if (sdict) @bitCast(q.code & ~qlinear.dict_kmajor) else 0,
        };
        parts.count = 2;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        g.add(act);
        g.add(as);
        for ([_]usize{ rows, gate.k, gate.k, 1 }) |v| g.add(@as(c_int, @intCast(v)));
        const shared: u32 = if (sdict) twin32_smem + singleTab(gate) + singleTab(up) else twin32_smem;
        const g_pair = gate.code >> 18 & 0x1FFF == 0;
        const u_pair = up.code >> 18 & 0x1FFF == 0;
        const twin = if (!sdict) kk.twin32_fn.? else if (g_pair and u_pair) kk.twin32_pair_fn.? else if (g_pair or u_pair) kk.twin32_mixed_fn.? else kk.twin32_dict_fn.?;
        const shared32: u32 = if (sdict and g_pair and u_pair) twin32_pair_smem else shared;
        if (sdict and g_pair and u_pair and kk.twin32_half_fn != null) {
            try launch_.launch(kk.twin32_half_fn.?, .{ .grid = .{ .x = @intCast(2 * tiles) }, .block = .{ .x = 128 }, .shared = half32_pair_smem, .pdl = kk.pdl }, s, &g);
            return;
        }
        try launch_.launch(twin, .{ .grid = .{ .x = @intCast(tiles) }, .block = .{ .x = 256 }, .shared = shared32, .pdl = kk.pdl }, s, &g);
    }
};

/// tf_qmm_mlp_half32_pair (qmm_stream32.cuh): two stages (a group's four 64-column tiles of words, both pair index runs,
/// the 32 row sums and both m16 row tiles' inputs), then four warps' two passes' values; three CTAs an SM
pub const half32_pair_smem: u32 = 25856 + 1024;
