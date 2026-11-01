//! The landed CUDA forward with optional learner taps and pocket contributions.
const E = @import("cuda_engine.zig");
const cuda = @import("cuda");
const sig = @import("cuda_signatures.zig");
const hooks = @import("cuda_learn_hooks.zig");
const geometry = @import("cuda_geometry.zig");
const head_dim = geometry.head_dim;
const max_chunks = 64;
/// Prompt experts keep the shipped arithmetic boundary, independently of verification capacity.
pub const prompt_moe_cutoff: usize = 16;
/// qwen35_attn_tc.cuh's bf16 tile buffers: two of a 32-key tile's keys and values, 512 bytes a row each.
pub const attn_tiles = 4 * 32 * 512;
/// tf_q35_attn_partial's one key and value tile (Bf16Tiles1), and the chunk blocks it launches a (row, key head):
/// each takes every partial_chunks-th of the MAXCH chunks, so a short stream's rows dispatch no idle blocks.
const attn_tile = attn_tiles / 2;
const partial_chunks = 12;
/// qwen35_attn_tc.cuh's GROUP_LEN: a stream at least this long has its rows on tf_q35_attn_rows.
pub const group_len = 8192;
const ids_at = E.Engine.ids_at;
const segs_at = E.Engine.segs_at;
const rowseg_at = E.Engine.rowseg_at;
const rowpar_at = E.Engine.rowpar_at;
const rowanc_at = E.Engine.rowanc_at;
const params_at = E.Engine.params_at;
const mpos_at = E.Engine.mpos_at;
const src_at = E.Engine.src_at;

pub fn promptMoe(prompt: bool, rows: usize) bool {
    return prompt and rows > prompt_moe_cutoff;
}

pub fn body(e: *E.Engine, shape: anytype) !void {
    const g = e.g;
    const total: usize = shape.total;
    const nseg: usize = shape.nseg;
    const record = shape.record;
    const prompt = shape.prompt;
    const heads: usize = if (record) total else 1;
    // Independent prompt rows share a block; decode keeps one warp per row. A 16..64-row window's gdn_pre rows share
    // blocks too (four row warps a block: its warps are independent, the same bits); TF_GDN_ROWS4=0 keeps one warp a
    // block there. gdn_post keeps a warp a block in windows (four a block measured slower).
    const pre_rows: usize = if ((prompt and total > 16) or (total >= 8 and rows4On())) 4 else if (total >= 2 and rows4On()) 2 else 1;
    const pre_blocks = (total + pre_rows - 1) / pre_rows;
    // gdn_post's decode windows: two independent row warps a block from 2 rows, as gdn_pre's (TF_POST_ROWS2=0: one)
    const post_rows: usize = if (prompt and total > 16) 4 else if (total >= 2 and postRows2On()) 2 else 1;
    const post_blocks = (total + post_rows - 1) / post_rows;
    const t_ids = e.tables.ptr + ids_at;
    const t_segs = e.tables.ptr + segs_at;
    const t_rows = e.tables.ptr + rowseg_at;
    const t_par = e.tables.ptr + rowpar_at;
    const t_anc = e.tables.ptr + rowanc_at;
    const t_params = e.tables.ptr + params_at;
    const geo = e.geo;
    const group = g.q_heads / g.kv_heads;

    try e.embed(t_ids, e.h, total, t_params);
    // the producers of the out, o and down inputs write their group sums; a prompt GEMM past 16 rows needs none
    const sums: u64 = if (prompt and total > 16) 0 else e.xs;
    // an image chunk's rows take the tower's features, and rotate at their staged t, h, w positions
    const mpos: u64 = if (e.features != 0) e.tables.ptr + mpos_at else 0;
    if (e.features != 0) try e.run(e.fns.splice, .{ total, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, e.features, e.tables.ptr + src_at, @as(c_int, @intCast(g.hidden)), t_params });
    // a prompt chunk keeps its last row: past its keys and values, the last layer runs that row alone
    const tail = prompt and @import("cuda_prompt_tail.zig").fits(e, total, nseg, shape.hidden, shape.paged);
    var di: usize = 0;
    var ai: c_int = 0;
    for (e.layers, 0..) |l, i| {
        try e.normInput(i != 0, l.input_norm, total, prompt);
        if (e.learning) |k| try k.input(i, e.h, total);
        switch (l.mixer) {
            .delta => |m| {
                const layer: c_int = @intCast(di);
                try e.projMany(&.{ m.qkv, m.z, m.a, m.b }, e.x, &.{ e.qkv, e.z, e.a, e.b }, total, prompt);
                if (e.learning) |k| try k.delta(i, e.qkv, e.z, e.b, e.a, total);
                // a prompt chunk: the chain convolves the value heads and takes their decay and strength in its stages,
                // so gdn_pre runs the q and k heads and conv_commit follows the chain (TF_CHAIN_CONV=0: as before)
                const conv_in_chain = prompt and nseg == 1 and !record and chainConvOn();
                const pre_heads = if (conv_in_chain) 2 * g.k_heads else 2 * g.k_heads + g.v_heads;
                try e.runAfter(e.fns.gdn_pre, .{ 1, pre_heads, pre_blocks }, .{ 32, pre_rows, 1 }, .{ geo, e.qkv, t_segs, t_rows, t_par, layer, m.conv, e.a, e.b, m.a_log, m.dt_bias, e.q, e.k, e.v, e.g_, e.beta, t_params });
                // one recorded window of 2+ rows: gdn_pre logged its tails, conv_commit would only return
                const commit = !(record and nseg == 1 and total > 1);
                if (commit and !conv_in_chain) try e.runAfter(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                if (conv_in_chain) {
                    try @import("cuda_prompt.zig").deltaConv(e, di, total, m);
                    if (commit) try e.runAfter(e.fns.conv_commit, .{ (g.convWidth() + 255) / 256, nseg, 1 }, .{ 256, 1, 1 }, .{ geo, e.qkv, t_segs, t_par, layer, t_params });
                } else if (prompt and nseg == 1) try @import("cuda_prompt.zig").delta(e, di, total) else try e.deltaScan(record, shape.slots, nseg, layer);
                // a prompt chunk: eight rows a warp, their loads in flight together (TF_PROMPT_ILP=0: a row a warp)
                if (prompt and total > 16 and promptIlpOn())
                    try e.runAfter(e.fns.gdn_post_ilp, .{ 1, g.v_heads, (total + 31) / 32 }, .{ 32, 4, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params, sums })
                else
                    try e.runAfter(e.fns.gdn_post, .{ 1, g.v_heads, post_blocks }, .{ 32, post_rows, 1 }, .{ geo, e.y, e.z, m.norm, g.eps, e.mix, t_params, sums });
                try e.projAgain(m.out, e.mix, e.r, total, prompt);
                di += 1;
            },
            .attention => |m| {
                const aw = g.attnWidth();
                if (tail and i == e.layers.len - 1)
                    try @import("cuda_prompt_tail.zig").query(e, m, total)
                else
                    try e.projMany(&.{ m.q, m.k, m.v }, e.x, &.{ e.qraw, e.kraw, e.vraw }, total, prompt);
                if (e.learning) |k| try k.attention(i, e.qraw, e.kraw, e.vraw, total);
                if (prompt and total > 16 and !e.kv8) {
                    // Preserve the original two head sums and bf16 boundary before rotary/cache stores.
                    try e.runAfter(e.fns.head_rope_warp, .{ (total * (g.q_heads + g.kv_heads) + 3) / 4, 1, 1 }, .{ 32, 4, 1 }, .{ geo, e.qraw, e.kraw, e.vraw, m.q_norm, m.k_norm, e.qn, t_segs, t_rows, t_anc, @as(c_int, ai), g.eps, t_params, mpos });
                } else if (e.kv8) {
                    try e.runAfter(e.fns.head_norm, .{ total * g.q_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.qraw, m.q_norm, e.qn, @as(c_int, @intCast(g.q_heads)), @as(c_int, @intCast(2 * aw)), @as(c_int, 2 * head_dim), g.eps, t_params });
                    try e.runAfter(e.fns.head_norm, .{ total * g.kv_heads, 1, 1 }, .{ 64, 1, 1 }, .{ e.kraw, m.k_norm, e.kn, @as(c_int, @intCast(g.kv_heads)), @as(c_int, @intCast(g.kv_heads * head_dim)), @as(c_int, head_dim), g.eps, t_params });
                    try launch(e, e.fp8.rope, sig.rope, .{ total, g.q_heads + g.kv_heads, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qn, e.kn, e.vraw, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, mpos });
                } else {
                    // both head norms and the rope in one launch: a block a (row, head), each normed head rotated in place
                    try launchAfter(e, e.fns.norm_rope, sig.norm_rope, .{ total, g.q_heads + g.kv_heads, 1 }, .{ head_dim, 1, 1 }, .{ geo, e.qraw, e.kraw, m.q_norm, m.k_norm, e.qn, e.vraw, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, mpos, @as(f32, g.eps) });
                }
                if (tail and i == e.layers.len - 1) {
                    try @import("cuda_prompt_tail.zig").run(e, @intCast(ai), total);
                    return;
                }
                if (shape.paged) {
                    try @import("cuda_attention.zig").partial(e, shape, ai);
                    try launchAfter(e, e.fns.attn_merge_gate, sig.attn_merge_gate, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.qraw, e.mix, sums });
                } else if (prompt and nseg == 1 and (g.q_heads / g.kv_heads) % 2 == 0) {
                    // a prompt chunk's rows over its own cache: tensor-core tiles in place of the decode partials
                    try @import("cuda_prompt.zig").run(e, @intCast(ai), total);
                } else {
                    const args = .{ geo, e.qn, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po };
                    const groups = (total + 64 / group - 1) / (64 / group);
                    const rows_args = .{ geo, e.qn, t_segs, t_rows, t_anc, @as(c_int, ai), t_params, @as(f32, 1.0 / 16.0), @as(c_int, @intCast(total)), e.pm, e.pl, e.po };
                    if (e.kv8) {
                        try launch(e, e.fp8.attn_partial, sig.attn_partial8, .{ total, max_chunks, g.kv_heads }, .{ 128, 1, 1 }, args);
                        try launch(e, e.fp8.attn_rows, sig.attn_rows8, .{ groups, max_chunks, g.kv_heads }, .{ 256, 1, 1 }, rows_args);
                    } else {
                        // a 2..64-row window: row pairs share each chunk's stream-key tiles (attn_pair, the same bits);
                        // TF_ATTN_PAIR=0 keeps attn_partial's block a row
                        if (record and total >= 2 and 2 * group <= 16 and pairOn())
                            try sharedSized(e, e.fns.attn_pair, sig.attn_pair, .{ (total + 1) / 2, partial_chunks, g.kv_heads }, 128, attn_tile, args)
                        else
                            try sharedSized(e, e.fns.attn_partial, sig.attn_partial, .{ total, partial_chunks, g.kv_heads }, 128, attn_tile, args);
                        // streams of caches no longer than GROUP_LEN have no row for attn_rows (its blocks would only return)
                        if (!shape.short) try shared(e, e.fns.attn_rows, sig.attn_rows, .{ groups, max_chunks, g.kv_heads }, 256, rows_args);
                    }
                    // the merge with the gate and the gated rows' group sums, attn_gate's bits
                    try launchAfter(e, e.fns.attn_merge_gate, sig.attn_merge_gate, .{ g.q_heads, total, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t_segs, t_rows, t_anc, t_params, e.qraw, e.mix, sums });
                }
                // a prompt chunk without group sums: eight entries a thread (TF_PROMPT_ILP=0: an entry a thread)
                if (prompt and nseg == 1 and (g.q_heads / g.kv_heads) % 2 == 0) {
                    if (sums == 0 and aw % 8 == 0 and promptIlpOn())
                        try e.runAfter(e.fns.attn_gate8, .{ (total * aw / 8 + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params })
                    else
                        try launchAfter(e, e.fns.attn_gate, sig.attn_gate, .{ (total * aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ geo, e.y, e.qraw, e.mix, @as(c_int, @intCast(total)), t_params, sums });
                }
                if (e.learning) |k| try k.gated(i, e.mix, total);
                try e.projAgain(m.o, e.mix, e.r, total, prompt);
                ai += 1;
            },
        }
        try e.normInput(true, l.post_norm, total, prompt);
        if (e.learning) |k| try k.post(i, e.h, e.x, total);
        switch (l.mlp) {
            .dense => |m| {
                // a prompt chunk: gate, up and the activation in one launch, with the three kernels' bits
                const fused = prompt and total > 16 and e.learning == null and !e.int8.on and m.gate == .a4 and m.up == .a4 and
                    cuda.prompt_many.Kernels.mlpFits(m.gate.a4, m.up.a4) and m.gate.a4.n == g.intermediate;
                if (fused) {
                    try e.prompt_many.mlp(e.stream, e.x, m.gate.a4, m.up.a4, e.act, total);
                    try e.projAgain(m.down, e.act, e.r, total, prompt);
                } else if (!prompt and total <= E.most_verify and e.learning == null and e.mlp == null and e.affine.split == null and
                    m.gate == .a4 and m.up == .a4 and cuda.mlp_pair.Kernels.fits(m.gate.a4, m.up.a4) and m.gate.a4.n == g.intermediate)
                {
                    // a window: gate, up and the activation in one launch, act's group sums past the norm's (which gate
                    // and up still read; most_verify rows of each fit xs), then down over them; every bit as the two
                    // projections, mlp_act and down give
                    const act_sums = e.xs + E.most_verify * (g.hidden / 64) * 4;
                    try e.mlp_pair.pair(e.stream, e.x, e.xs, m.gate.a4, m.up.a4, e.act, act_sums, total);
                    try e.project(m.down, e.act, act_sums, e.r, total, prompt, t_params, false);
                } else {
                    try e.projMany(&.{ m.gate, m.up }, e.x, &.{ e.gate, e.up }, total, prompt);
                    if (e.learning) |k| try k.mlp(i, e.gate, e.up, total);
                    const n: c_longlong = @intCast(total * g.intermediate);
                    try e.runAfter(e.fns.mlp_act, .{ (total * g.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, n, t_params, @as(c_int, @intCast(g.intermediate)), sums });
                    try e.projAgain(m.down, e.act, e.r, total, prompt);
                }
                if (e.mlp) |k| try k.f(k.ctx, i, e.x, e.r, total, e.taught) else try e.pockets.add(e.stream, i, e.x, e.r, total, e.taught);
            },
            .moe => |m| try e.routedRows(m.router, m.experts, e.x, e.r, total, promptMoe(prompt, total), t_params),
        }
        if (shape.taps) for (e.taps[0..e.n_taps], 0..) |t, ti| if (t == i) {
            try e.runAfter(e.fns.tap, .{ total, (g.hidden + 2047) / 2048, 1 }, .{ 256, 1, 1 }, .{ e.h, e.r, t_segs, t_rows, @as(c_int, @intCast(ti)), @as(c_int, @intCast(g.hidden)), t_params });
        };
        // a captured forward's first layer is a graph of its own (cuda_engine.zig's cutCapture)
        if (i == 0) try e.cutCapture();
    }
    try finish(e, total, heads, shape.hidden, if (@hasField(@TypeOf(shape), "amax")) shape.amax else false);
}

/// TF_CHAIN_CONV=0: a prompt chunk's gdn_pre keeps the value heads and the chain reads its v, g and beta.
fn chainConvOn() bool {
    const v = @import("sys").c.getenv("TF_CHAIN_CONV") orelse return true;
    return !@import("std").mem.eql(u8, @import("std").mem.span(v), "0");
}

/// TF_GDN_ROWS4=0: 16..64-row windows keep one row warp a block in tf_q35_gdn_pre.
/// TF_PROMPT_ILP=0: a prompt chunk keeps gdn_post's and attn_gate's one-row (one-entry) launches.
/// Draw 6 of this tree.
fn promptIlpOn() bool {
    const v = @import("sys").c.getenv("TF_PROMPT_ILP") orelse return true;
    return !@import("std").mem.eql(u8, @import("std").mem.span(v), "0");
}

fn postRows2On() bool {
    const v = @import("sys").c.getenv("TF_POST_ROWS2") orelse return true;
    return !@import("std").mem.eql(u8, @import("std").mem.span(v), "0");
}

fn rows4On() bool {
    const v = @import("sys").c.getenv("TF_GDN_ROWS4") orelse return true;
    return !@import("std").mem.eql(u8, @import("std").mem.span(v), "0");
}

/// TF_ATTN_PAIR=0: 2..64-row windows keep tf_q35_attn_partial (a block a row).
fn pairOn() bool {
    const v = @import("sys").c.getenv("TF_ATTN_PAIR") orelse return true;
    return !@import("std").mem.eql(u8, @import("std").mem.span(v), "0");
}

/// A launch whose arguments are checked against the kernel's .cu signature: a wrong count or kind does not build.
fn launch(e: *E.Engine, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, block: [3]usize, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    return e.run(f, grid, block, args);
}

/// `launch` with programmatic dependent launch on GB10, for kernels that open with griddepcontrol.wait (qwen35.cu).
fn launchAfter(e: *E.Engine, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, block: [3]usize, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    return e.runAfter(f, grid, block, args);
}

/// A bf16 decode attention launch with its two tile buffers of dynamic shared memory.
fn shared(e: *E.Engine, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, threads: u32, args: anytype) !void {
    return sharedSized(e, f, s, grid, threads, attn_tiles, args);
}

fn sharedSized(e: *E.Engine, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, threads: u32, bytes: u32, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    // the bf16 decode attention kernels open with griddepcontrol.wait (qwen35.cu): PDL on GB10
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = bytes, .pdl = E.glue_pdl and e.affine.pdl };
    try cuda.launch.launch(f, cfg, e.stream, &a);
}

/// Layer inputs are already kept; hidden reads need neither the final norm nor the vocabulary head.
pub fn finish(e: anytype, total: usize, heads: usize, hidden: bool, amax: bool) !void {
    if (hidden) return;
    // an armed prompt cut (cuda_engine.zig Overlap): the drafter's prompt absorb may start once the last layer is done
    if (@hasField(@TypeOf(e.*), "overlap")) if (e.overlap.armed) try e.overlapTail();
    const g = e.g;
    if (e.learning == null and heads < total) {
        try e.normTail(e.final_norm, total, heads);
        if (e.learning) |k| try k.final(e.h, total);
        try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
    } else {
        try e.norm(true, e.final_norm, total);
        if (e.learning) |k| try k.final(e.h, total);
        if (@hasField(@TypeOf(e.*), "head_amax")) if (amax) {
            // a greedy verify: the norm wrote the rows' group sums (projAgain's input); the head's tile writes each
            // row's best a tile and the merge each row's token (cuda_engine.zig forward then skips its argmax)
            const x = e.x + (total - heads) * g.hidden * 2;
            const toks = e.tables.ptr + @import("cuda_round.zig").tokens_at;
            // the 2-bit screen and exact rescore (cuda_head3.zig); a failed certificate reruns this head after the readback
            if (@hasField(@TypeOf(e.*), "head3")) if (e.head3) |*h3| return h3.run(e.stream, x, e.xs, e.head.a4, heads, toks);
            return e.affine.headArgmax(e.stream, x, e.xs, e.head.a4, e.head_amax, heads, toks);
        };
        if (@hasDecl(@TypeOf(e.*), "projAgain"))
            try e.projAgain(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false)
        else
            try e.proj(e.head, e.x + (total - heads) * g.hidden * 2, e.logits, heads, false);
    }
    if (e.learning) |k| try hooks.logits(k, e.logits, e.stream);
}

test "prompt MoE precision: target piece arithmetic keeps the shipped 16-row boundary" {
    const std = @import("std");
    try std.testing.expectEqual(@as(usize, 64), E.most_window);
    var mismatches: usize = 0;
    for ([_]usize{ 16, 17, 32, 64, 65 }) |rows| {
        const expected = rows > 16;
        if (promptMoe(true, rows) != expected) {
            std.debug.print("prompt piece {d}: expected prompt experts {any}\n", .{ rows, expected });
            mismatches += 1;
        }
        if (promptMoe(false, rows)) {
            std.debug.print("verify window {d}: expected decode experts\n", .{rows});
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "prompt MoE precision: every prompt piece through 128 and admitted verify width keeps its recipe" {
    const std = @import("std");
    for (1..129) |rows| try std.testing.expectEqual(rows > 16, promptMoe(true, rows));
    for (1..E.most_window + 1) |rows| try std.testing.expect(!promptMoe(false, rows));
}
