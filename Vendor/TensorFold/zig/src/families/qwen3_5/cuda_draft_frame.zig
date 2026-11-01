//! One DFlash frame for all selected streams, with separate rings, positions, convolution boundaries and trees.
const std = @import("std");
const cuda = @import("cuda");
const dr = @import("lanes").drafter;
const draft = @import("cuda_draft.zig");
const tree = @import("cuda_draft_tree.zig");
const layers = @import("cuda_layers.zig");
const sig = @import("cuda_signatures.zig");
pub const max_streams = @import("cuda_engine.zig").max_segs;
pub const max_rows = max_streams * 16;
pub const chunks = 9;
const lp_words = 43;
/// DFlash2's draft vocabulary as the Metal drafter and qwen27 engine read it: the head's first 98,304 rows, last 288.
const spans = [2][2]usize{ .{ 0, 98304 }, .{ 248032, 248320 } };
const span_width = spans[0][1] + spans[1][1] - spans[1][0];
/// The tail span's 64-row tile: projected from row 248,000, its first 32 columns dropped.
const tail_tile = spans[1][0] / 64 * 64;
pub const Segment = extern struct { ring: u64, begin: i64, position: i64 };
pub const segments_at = max_rows * 4;
pub const words_at = segments_at + max_streams * @sizeOf(Segment);
pub const hp_at = words_at + max_rows * lp_words * 4;
pub fn stagedBytes(rank: usize) !usize {
    return std.math.add(usize, hp_at, try std.math.mul(usize, try std.math.mul(usize, max_rows, rank), 2));
}
pub fn stagedLen(rank: usize) usize {
    return stagedBytes(rank) catch unreachable;
}
pub fn segments(st: []u8) []Segment {
    return @alignCast(std.mem.bytesAsSlice(Segment, st[segments_at..words_at]));
}
pub fn stage(c: draft.Config, items: []const dr.Hold, contexts: []const *draft.Ctx, st: []u8) !void {
    if (items.len == 0 or items.len > max_streams or contexts.len != items.len or (c.block != 8 and c.block != 16) or st.len < try stagedBytes(c.rank) or @intFromPtr(st.ptr) % 8 != 0) return error.BadDraftFrame;
    for (items, contexts, 0..) |it, ctx, index| {
        if (it.position == 0 or it.position > std.math.maxInt(i64) - c.block or ctx.end != it.position - 1 or ctx.begin > ctx.end) return error.DraftContextBehind;
        // a chain drafts a row a depth; a grown tree's depth is its node budget (tree.nodes(), up to the window)
        const most = if (c.block == 16 and it.lanes == null) @max(c.block - 1, tree.nodes()) else c.block - 1;
        if (it.pending >= c.vocab or it.depth > most) return error.BadDraftFrame;
        for (contexts[0..index]) |prior| if (ctx == prior or ctx.ring.ptr == prior.ring.ptr) return error.OverlappingDraftContexts;
        if (it.lanes) |shape| {
            if (shape.parents.len > ctx.held.len or shape.parents.len != shape.depths.len or shape.parents.len != shape.ranks.len) return error.BadLanes;
            for (shape.parents, shape.depths, shape.ranks, 0..) |parent, depth, rank, row| {
                if (parent < -1 or parent >= @as(i32, @intCast(row)) or depth >= c.block - 1 or rank >= c.topk) return error.BadLanes;
            }
        }
    }
    const ids = std.mem.bytesAsSlice(u32, st[0 .. c.block * items.len * 4]);
    const sg = segments(st);
    for (items, contexts, 0..) |it, ctx, i| {
        ids[i * c.block] = it.pending;
        @memset(ids[i * c.block + 1 ..][0 .. c.block - 1], c.mask);
        sg[i] = .{ .ring = ctx.ring.ptr, .begin = @intCast(ctx.begin), .position = @intCast(ctx.end) };
    }
}
/// A launch whose arguments are checked against the kernel's .cu signature: a wrong count or kind does not build.
fn launch(d: anytype, f: cuda.Function, comptime s: sig.Sig, grid: [3]usize, block: [3]usize, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    return d.e.runAfter(f, grid, block, args);
}
fn conv(d: anytype, x: u64, base: u64, res: u64, out: u64, rows: usize, branch: c_int, sums: u64) !void {
    try d.e.runAfter(d.fns.dyn_conv_batch, .{ rows, (d.c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ x, d.dyn, base, res, out, @as(c_int, @intCast(d.c.hidden)), @as(c_int, @intCast(d.c.group)), branch, @as(c_int, @intCast(d.c.block)), sums });
}
fn ropeQk(d: anytype, xq: u64, wq: u64, outq: u64, xk: u64, wk: u64, outk: u64, rows: usize) !void {
    const c = d.c;
    const limit: usize = std.math.maxInt(c_int);
    if (rows == 0 or c.heads == 0 or c.kv_heads == 0 or c.block == 0 or rows > limit or c.heads > limit or c.kv_heads > limit or c.block > limit) return error.BadDraftFrame;
    const heads = std.math.add(usize, c.heads, c.kv_heads) catch return error.BadDraftFrame;
    const blocks = std.math.mul(usize, rows, heads) catch return error.BadDraftFrame;
    if (blocks > limit) return error.BadDraftFrame;
    try d.e.runAfter(d.fns.norm_rope_qk_batch, .{ blocks, 1, 1 }, .{ 32, 1, 1 }, .{ xq, wq, outq, @as(c_int, @intCast(c.heads)), xk, wk, outk, @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(rows)), d.segments, @as(c_int, @intCast(c.block)), c.eps, c.theta });
}
pub fn attention(d: anytype, rows: usize, layer: usize) !void {
    const c = d.c;
    const partial = .{ d.q, d.kb, d.vb, d.segments, @as(c_int, @intCast(c.block)), @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.ringRows())), @as(c_int, @intCast(c.window)), @as(c_int, @intCast(layer)), d.pm, d.pl, d.po };
    // four query heads a key head (128 wide): a block takes four rows of a stream and a key head's heads, so each
    // staged key and value serves 16 (row, head) partials; the same bits as tf_q35d_attn_partial's
    if (c.heads == 4 * c.kv_heads and c.head_dim == 128 and c.block % 4 == 0)
        try launch(d, d.fns.attn_partial_rows4, sig.draft_partial_rows, .{ rows / 4, c.kv_heads, chunks }, .{ 128, 1, 1 }, partial)
    else
        try launch(d, d.fns.attn_partial, sig.draft_partial, .{ rows, c.heads, chunks }, .{ 128, 1, 1 }, partial);
    const merge = .{ d.pm, d.pl, d.po, @as(c_int, @intCast(c.heads)), d.att };
    try launch(d, d.fns.attn_merge, sig.draft_merge, .{ rows, c.heads, 1 }, .{ 32, 1, 1 }, merge);
}
pub fn publish(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    defer d.selection_base = 0;
    for (items, contexts, 0..) |it, ctx, i| {
        d.selection_base = i * (d.c.block - 1);
        try d.selectHold(ctx, it);
    }
}
fn paired(l: anytype, B: usize) bool {
    return draft.pairedMlp() and B > 0 and B <= 16 and cuda.mlp_pair.Kernels.fits(l.gate, l.up);
}

/// Whether the MLP's input conv writes the pair's input sums itself (one 256-channel pass, whole groups a block).
fn convSums(d: anytype, l: anytype, B: usize) bool {
    return paired(l, B) and d.c.hidden % 256 == 0 and l.gate.k == d.c.hidden and !std.mem.eql(u8, std.mem.span(@import("sys").c.getenv("TF_DRAFT_CONV_SUMS") orelse ""), "0");
}

/// Reuse the native fused pair while keeping its input sums separate from its output sums; `summed`: the conv that
/// wrote xin already wrote its sums into xs.
pub fn mlp(d: anytype, l: anytype, B: usize, summed: bool) !void {
    const e = d.e;
    const c = d.c;
    if (paired(l, B)) {
        if (!summed) try e.runAfter(e.fns.sums, .{ B, (l.gate.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ d.xin, d.xs, @as(c_int, @intCast(l.gate.k)), @as(u64, 0) });
        // The unused gate buffer has room for the FP32 group sums. Never overwrite xs while pair reads it.
        const act_sums = d.gate;
        try e.mlp_pair.pair(e.stream, d.xin, d.xs, l.gate, l.up, d.act, act_sums, B);
        try draft.precision.affine(e, l.down, d.act, act_sums, d.o, B, false, d.projection_role);
    } else {
        try d.proj(l.gate, d.xin, d.gate, B);
        try d.projAgain(l.up, d.xin, d.up, B);
        try e.runAfter(e.fns.mlp_act, .{ (B * c.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), @as(c_int, @intCast(c.intermediate)), d.xs });
        try d.projAgain(l.down, d.act, d.o, B);
    }
}

/// The current frame's original device operations, captured without prompt-dependent metadata.
fn body(d: anytype, n: usize, narrow_: bool) !?u64 {
    const c = d.c;
    const e = d.e;
    const B = n * c.block;
    try e.embed(d.ids, d.h, B, 0);
    for (d.w.layers, 0..) |l, i| {
        try d.norm(d.h, l.input_norm, d.x, B);
        try d.projAgain(l.attn_conv, d.x, d.dyn, B);
        try conv(d, d.x, l.attn_base, 0, d.xin, B, 0, 0);
        try d.projMany(&.{ l.q, l.k, l.v }, d.xin, &.{ d.qraw, d.kraw, d.vb }, B);
        try ropeQk(d, d.qraw, l.q_norm, d.q, d.kraw, l.k_norm, d.kb, B);
        try attention(d, B, i);
        try d.proj(l.o, d.att, d.o, B);
        try conv(d, d.o, l.attn_base, d.h, d.h2, B, 1, 0);
        try d.norm(d.h2, l.post_norm, d.x, B);
        try d.projAgain(l.mlp_conv, d.x, d.dyn, B);
        const summed = convSums(d, l, B);
        try conv(d, d.x, l.mlp_base, 0, d.xin, B, 0, if (summed) d.xs else 0);
        try mlp(d, l, B, summed);
        try conv(d, d.o, l.mlp_base, d.h2, d.h, B, 1, 0);
    }
    const rows = n * (c.block - 1);
    var masked = d.h + c.hidden * 2;
    if (n > 1) {
        try e.run(d.fns.mask_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.h, d.h2, @as(c_int, @intCast(c.hidden)), @as(c_int, @intCast(c.block)) });
        masked = d.h2;
    }
    try d.norm(masked, d.w.final_norm, d.x, rows);
    const spanned = if (narrow_) try head(d, rows) else try full(d, rows);
    if (spanned != null and d.head2 != null)
        try topkScreened(d, spanned.?, rows)
    else
        try e.draws.topkRows(e.stream, spanned orelse d.logits, if (spanned != null) span_width else c.vocab, @intCast(c.topk), d.parts, d.words, rows);
    try d.projAgain(d.w.selector, d.x, d.hp, rows);
    return spanned;
}

fn run(d: anytype, n: usize, narrow_: bool) !?u64 {
    const e = d.e;
    if (draft.graphsOff()) return body(d, n, narrow_);
    var key: u64 = n | (@as(u64, @intFromBool(narrow_)) << 8) | (@as(u64, @intFromBool(draft.pairedMlp())) << 9);
    key = std.hash.Wyhash.hash(key, std.mem.asBytes(&[_]u64{ d.h, d.x, d.logits, d.hp, d.words, d.ids, d.segments, d.scratch.ptr }));
    if (d.graphs.get(key)) |g| {
        if (g.exec) |x| {
            try x.launchOn(e.stream);
            return g.spanned;
        }
        return body(d, n, narrow_);
    }
    cuda.graph.beginCapture(e.stream, .thread_local) catch return body(d, n, narrow_);
    const spanned = body(d, n, narrow_) catch |err| {
        var broken = cuda.graph.endCapture(e.stream) catch return err;
        broken.deinit();
        return err;
    };
    var exec: ?cuda.graph.Exec = null;
    if (cuda.graph.endCapture(e.stream)) |captured_| {
        var captured = captured_;
        defer captured.deinit();
        exec = captured.instantiate() catch null;
    } else |_| {}
    try d.graphs.put(d.gpa, key, .{ .exec = exec, .spanned = spanned });
    if (exec) |x| try x.launchOn(e.stream) else _ = try body(d, n, narrow_);
    return spanned;
}


pub fn forward(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    const c = d.c;
    const e = d.e;
    const B = items.len * c.block;
    const st = d.staged.bytes;
    try stage(c, items, contexts, st);
    for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    }
    errdefer cuda.completion.drain(e.d, e.stream, "a failed DFlash frame did not drain before its pinned staging was reused");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, st.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.segments, st.ptr + segments_at, items.len * @sizeOf(Segment), e.stream.handle), "cuMemcpyHtoDAsync");
    const rows = items.len * (c.block - 1);
    const spanned = try run(d, items.len, narrow(contexts));
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + words_at, d.words, rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + hp_at, d.hp, rows * c.rank * 2, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    if (spanned != null) for (0..rows) |r| {
        const w = std.mem.bytesAsSlice(u32, st[words_at + r * lp_words * 4 ..][0 .. lp_words * 4]);
        for (w[2..][0..c.topk]) |*id| if (id.* >= spans[0][1]) {
            id.* += @intCast(spans[1][0] - spans[0][1]);
        };
    };
    try publish(d, items, contexts);
}

/// A stream's tokens outside the spans may outnumber this share of its tokens before the frame reads the whole head.
const wide_share = 32;

/// A lane stream's prompt and reply tokens since the last scan: the stream is wide (the frame reads the whole head)
/// once tokens outside the spans are more than 1 in `wide_share` of its tokens. A few rare tokens in a prompt keep the
/// span head: the whole head costs a 248,320-row projection every frame, against 98,592 rows for the spans.
pub fn scan(ctx: *draft.Ctx, toks: []const u32) void {
    for (toks[@min(ctx.scanned, toks.len)..]) |t| if (t >= spans[0][1] and t < spans[1][0]) {
        ctx.outside += 1;
    };
    ctx.scanned = toks.len;
    ctx.wide = ctx.outside * wide_share > ctx.scanned;
}

/// Whether every stream's tokens so far keep to the spans (a warmup's or standalone hold's context reads none).
fn narrow(contexts: []const *draft.Ctx) bool {
    for (contexts) |ctx| if (ctx.wide) return false;
    return true;
}

/// The rows' draft logits over the spans' columns, packed in d.logits after the two projections (null: not this head).
fn head(d: anytype, rows: usize) !?u64 {
    const e = d.e;
    const q = switch (e.head) {
        .a4 => |q| q,
        .an, .ql => return full(d, rows),
        .x3, .bf => return full(d, rows),
    };
    if (q.n != spans[1][1] or q.k % 64 != 0) return full(d, rows);
    const first: cuda.qlinear.Affine4 = .{ .w = q.w, .s = q.s, .b = q.b, .sb = q.sb, .tab = q.tab, .code = q.code, .n = spans[0][1], .k = q.k, .npad = q.npad };
    const tail: cuda.qlinear.Affine4 = .{ .w = q.w + tail_tile * q.k / 2, .s = q.s + tail_tile * 2, .b = q.b + tail_tile * 2, .n = q.n - tail_tile, .k = q.k, .npad = q.npad };
    const a = d.logits; // rows x 98,304, then rows x 320, then the packed rows x 98,592
    const b = a + rows * first.n * 2;
    const out = b + rows * tail.n * 2;
    const skip = spans[1][0] - tail_tile;
    if (d.head2) |h2| {
        // the first span's columns screened straight into the packed rows; the tail exact, as below
        try screen(d, h2, rows, out);
        try e.project(layers.Proj{ .a4 = tail }, d.x, d.xs, b, rows, false, 0, false);
        const m: cuda.abi.Memcpy2D = .{ .src = b + skip * 2, .src_pitch = tail.n * 2, .dst = out + first.n * 2, .dst_pitch = span_width * 2, .width = (tail.n - skip) * 2, .height = rows };
        try e.d.check(e.d.api.cuMemcpy2DAsync_v2(&m, e.stream.handle), "cuMemcpy2DAsync");
        return out;
    }
    // d.x and its group sums d.xs are the final norm's (body): both projections read them, no sums launch of their own
    try e.project(layers.Proj{ .a4 = first }, d.x, d.xs, a, rows, false, 0, false);
    try e.project(layers.Proj{ .a4 = tail }, d.x, d.xs, b, rows, false, 0, false);
    const copies = [2]cuda.abi.Memcpy2D{
        .{ .src = a, .src_pitch = first.n * 2, .dst = out, .dst_pitch = span_width * 2, .width = first.n * 2, .height = rows },
        .{ .src = b + skip * 2, .src_pitch = tail.n * 2, .dst = out + first.n * 2, .dst_pitch = span_width * 2, .width = (tail.n - skip) * 2, .height = rows },
    };
    for (&copies) |*m| try e.d.check(e.d.api.cuMemcpy2DAsync_v2(m, e.stream.handle), "cuMemcpy2DAsync");
    return out;
}

/// The rows' draft logits over the whole vocabulary, in d.logits (over the final norm's d.x and d.xs, as head's).
fn full(d: anytype, rows: usize) !?u64 {
    try d.e.project(d.e.head, d.x, d.xs, d.logits, rows, false, 0, false);
    return null;
}

/// The span head's 2-bit screen (qwen35_head2.cuh): the first span's 98,304 head rows as each 4-bit code's two high bits
/// with (4 s, b + 1.5 s) a group, screened for every column; each row's best screened columns then take their exact
/// logits from the 4-bit head (group_kernel_dict's arithmetic) before the merge, so the rows' drafts are the full head's
/// whenever their best columns are among the screened candidates. TF_HEAD2=0: the span head as before.
const head2_first = spans[0][1];
const head2_tiles = head2_first / 64;
const head2_stages = 8;
const head2_stage_bytes = 16 * 64 * 2 + 256 * 4 + 64 * 4 + 16 * 4;

/// The candidates a row the merge reads (the parts buffer holds the whole vocabulary's slices at k, 976 a row).
const head2_cap = 640;

/// TF_HEAD2_K: the select's target count a row (default 512; 64 .. head2_cap).
fn head2K() usize {
    const v = @import("sys").c.getenv("TF_HEAD2_K") orelse return 512;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return 512;
    return std.math.clamp(n, 64, head2_cap);
}

fn head2On() bool {
    const v = @import("sys").c.getenv("TF_HEAD2") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The screen's buffer (codes, screen scales, exact column scales) packed from the target head, or null.
pub fn head2Init(d: anytype) !?cuda.DeviceBuffer {
    const e = d.e;
    if (!head2On()) return null;
    const q = switch (e.head) {
        .a4 => |q| q,
        else => return null,
    };
    if (q.n != spans[1][1] or q.k != 5120 or q.npad < q.n or q.code == 0) return null;
    const kg = q.k / 64;
    const codes_bytes = head2_tiles * kg * 256 * 4;
    const scales_bytes = head2_tiles * kg * 64 * 4;
    const sb_bytes = head2_first * kg * 4;
    var buf = try cuda.DeviceBuffer.alloc(e.d, codes_bytes + scales_bytes + sb_bytes);
    errdefer buf.free();
    var a: cuda.Args = .{};
    for ([_]u64{ q.w, q.s, q.b }) |v| a.add(v);
    a.add(@as(c_int, @intCast(q.npad)));
    a.add(@as(c_int, @intCast(kg)));
    for ([_]u64{ buf.ptr, buf.ptr + codes_bytes, buf.ptr + codes_bytes + scales_bytes }) |v| a.add(v);
    try cuda.launch.launch(d.fns.head2_pack, .{ .grid = .{ .x = @intCast(head2_tiles), .y = @intCast(kg) }, .block = .{ .x = 256 } }, e.stream, &a);
    try e.stream.synchronize();
    return buf;
}

/// The screened values of the first span's columns into the packed rows `out` (pitch span_width).
fn screen(d: anytype, h2: cuda.DeviceBuffer, rows: usize, out: u64) !void {
    const e = d.e;
    const kg: usize = 80;
    const codes_bytes = head2_tiles * kg * 256 * 4;
    var a: cuda.Args = .{};
    for ([_]u64{ d.x, d.xs, h2.ptr, h2.ptr + codes_bytes }) |v| a.add(v);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(kg)));
    a.add(out);
    a.add(@as(c_int, @intCast(span_width)));
    try cuda.launch.launch(d.fns.head2_screen, .{ .grid = .{ .x = @intCast(head2_tiles) }, .block = .{ .x = 128 }, .shared = head2_stages * head2_stage_bytes }, e.stream, &a);
}

/// topkRows' words over the packed span rows: each row's best `head2K()` columns by their screened values (ties at the
/// cut while room lasts, at most head2_cap), the first span's among them rescored exactly, then the merge's k best.
fn topkScreened(d: anytype, logits: u64, rows: usize) !void {
    const e = d.e;
    const h2 = d.head2.?;
    const q = e.head.a4;
    const kg: usize = q.k / 64;
    const k: usize = d.c.topk;
    const n: usize = head2_cap;
    const pv = d.parts;
    const pi = d.parts + rows * n * 4;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(c_int, @intCast(span_width)));
    a.add(@as(c_int, @intCast(span_width)));
    a.add(@as(c_int, @intCast(head2K())));
    a.add(@as(c_int, @intCast(n)));
    a.add(pv);
    a.add(pi);
    try cuda.launch.launch(d.fns.head2_select, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.stream, &a);
    const sb = h2.ptr + head2_tiles * kg * 256 * 4 + head2_tiles * kg * 64 * 4;
    var r: cuda.Args = .{};
    for ([_]u64{ d.x, d.xs, q.w, sb }) |v| r.add(v);
    r.add(@as(c_int, @intCast(kg)));
    r.add(@as(c_int, @intCast(rows)));
    r.add(pv);
    r.add(pi);
    r.add(@as(c_int, @intCast(n)));
    r.add(@as(c_int, @intCast(head2_first)));
    try cuda.launch.launch(d.fns.head2_rescore, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((n + 31) / 32) }, .block = .{ .x = 128 } }, e.stream, &r);
    var m: cuda.Args = .{};
    m.add(pv);
    m.add(pi);
    m.add(@as(u32, @intCast(n)));
    m.add(@as(u32, @intCast(k)));
    m.add(d.words);
    try cuda.launch.launch(e.draws.topk_merge_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, e.stream, &m);
}
