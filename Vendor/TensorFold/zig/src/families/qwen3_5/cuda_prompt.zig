//! A one-part prompt chunk's attention tiles and delta-rule chain (uncaptured: the host passes the stream's pointers).
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const Seg = @import("cuda_cache.zig").Seg;
const head_dim = geometry.head_dim;

/// One bf16 key slot beside four warps of queries; fp8 retains its two-slot, eight-warp kernel.
pub const smem: u32 = 49152;
const fp8_smem: u32 = 98304;
const block_rows = 64;

pub const Fns = struct { bf16: cuda.Function, fp8: cuda.Function, chain: cuda.Function };

pub const symbols = struct {
    pub const bf16 = "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi4ELi1ELi1EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi";
    pub const fp8 = "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
    pub const chain = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li64ELi16EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii";
};
/// gdn_prefill.cu's value rows a block (two threads a row).
const chain_rows = 64;

pub fn resolve(attention: cuda.Module, gdn: cuda.Module) !Fns {
    const f: Fns = .{ .bf16 = try attention.function(symbols.bf16), .fp8 = try attention.function(symbols.fp8), .chain = try gdn.function(symbols.chain) };
    try f.bf16.allowDynamicShared(smem);
    try f.fp8.allowDynamicShared(fp8_smem);
    return f;
}

/// Attention layer `layer` into e.y: a chunk's queries (e.qn) over cache positions 0 .. len + row (rope wrote its own).
pub fn run(e: *E.Engine, layer: usize, rows: usize) !void {
    return runFrom(e, layer, 0, rows);
}

/// `run` for the chunk's rows from `first` on: each row's queries over its own keys, the same arithmetic a row
/// (a tile past a row's position is wholly masked and leaves its maximum, sum and output as they were).
pub fn runFrom(e: *E.Engine, layer: usize, first: usize, end: usize) !void {
    const g = e.g;
    const rows = end - first;
    const at: u64 = first * g.q_heads * head_dim * 2;
    var sg = std.mem.bytesToValue(Seg, e.staged.bytes[E.Engine.segs_at..][0..@sizeOf(Seg)]);
    sg.len += @intCast(first);
    const kvh: u64 = g.kv_heads;
    const cap: u64 = @intCast(sg.cap);
    const heads = kvh * cap; // rows of one layer's keys (or values)
    const group = g.q_heads / g.kv_heads;
    var a: cuda.Args = .{};
    if (e.kv8) {
        const base = sg.kv + layer * 2 * heads * (head_dim + 4);
        const keys = base;
        const values = keys + heads * head_dim;
        const kscale = values + heads * head_dim;
        for ([_]u64{ e.qn + at, keys, values, kscale, kscale + heads * 4, e.y + at }) |v| a.add(v);
    } else {
        const base = sg.kv + layer * 2 * heads * head_dim * 2;
        for ([_]u64{ e.qn + at, base, base + heads * head_dim * 2, e.y + at }) |v| a.add(v);
    }
    for ([_]usize{ @intCast(sg.len), rows, g.q_heads, g.kv_heads, group }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(@as(f32, 1.0 / 16.0));
    a.add(@as(c_int, @intCast(cap)));
    const hpc: usize = if (e.kv8) 2 else 1;
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast((rows + block_rows - 1) / block_rows), .y = @intCast(g.kv_heads * (group / hpc)) }, .block = .{ .x = if (e.kv8) 256 else 128 }, .shared = if (e.kv8) fp8_smem else smem };
    try cuda.launch.launch(if (e.kv8) e.prompt_attn.fp8 else e.prompt_attn.bf16, cfg, e.stream, &a);
}

/// Linear layer `layer`'s delta rule: e.q, e.k, e.v, e.g_, e.beta into e.y, the stream's recurrence updated in place.
pub fn delta(e: *E.Engine, layer: usize, rows: usize) !void {
    const g = e.g;
    const sg = std.mem.bytesToValue(Seg, e.staged.bytes[E.Engine.segs_at..][0..@sizeOf(Seg)]);
    const state = sg.rec + layer * g.v_heads * geometry.dk * geometry.dk * 4;
    var a: cuda.Args = .{};
    for ([_]u64{ e.q, e.k, e.v, e.g_, e.beta, state, state, e.y }) |v| a.add(v);
    for ([_]usize{ rows, g.k_heads, g.v_heads }) |v| a.add(@as(c_int, @intCast(v)));
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(g.v_heads), .y = @intCast(geometry.dk / chain_rows) }, .block = .{ .x = 2 * chain_rows } };
    try cuda.launch.launch(e.prompt_attn.chain, cfg, e.stream, &a);
}
