//! A prompt chunk's last layer past its keys and values: only the kept last row goes on to the head.
//!
//! A prompt forward keeps one row's logits. Every layer's keys, values and recurrences are written for all rows,
//! and a drafter's taps read layers before the last; what the last layer computes after its keys and values
//! (its queries' attention, gate, o projection and MLP) reaches nothing but the last row's logits. So that row
//! alone runs on: the same kernels at the same arithmetic a row (the prompt GEMM at one row: each element's MMA
//! chain does not depend on the rows beside it; the attention's row over its own keys), from the row's pointers.
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const prompt = @import("cuda_prompt.zig");

/// Whether a forward of `total` prompt rows can end on its last row at the last layer.
pub fn fits(e: *const E.Engine, total: usize, nseg: usize, hidden: bool, paged: bool) bool {
    const g = e.g;
    if (total <= 16 or nseg != 1 or hidden or paged) return false;
    if (e.learning != null or e.mlp != null or e.features != 0 or e.kv8 or e.int8.on) return false;
    if ((g.q_heads / g.kv_heads) % 2 != 0) return false;
    const last = e.layers.len - 1;
    if (e.pockets.of(last) != null) return false;
    for (e.taps[0..e.n_taps]) |t| if (t == last) return false;
    const l = e.layers[last];
    const m = switch (l.mixer) {
        .attention => |a| a,
        else => return false,
    };
    if (m.o != .a4 or m.q != .a4) return false;
    const d = switch (l.mlp) {
        .dense => |x| x,
        else => return false,
    };
    if (d.gate != .a4 or d.up != .a4 or d.down != .a4) return false;
    return cuda.prompt_many.Kernels.mlpFits(d.gate.a4, d.up.a4) and d.gate.a4.n == g.intermediate;
}

/// The last layer's query projection (queries and gate) for row total - 1 only: the prompt GEMM at one row, each
/// element's MMA chain as in the every-row launch. The other rows' query rows are not written: the layer's
/// head_rope_warp still normalizes and rotates them (its key heads go to the cache for every row), but nothing past
/// it reads them.
pub fn query(e: *E.Engine, q: cuda.qlinear.Affine4, total: usize) !void {
    const row = total - 1;
    try e.prompt_many.prompt(e.stream, e.x + row * e.g.hidden * 2, &.{.{ .q = q, .out = e.qraw + row * 2 * e.g.attnWidth() * 2 }}, 1);
}

/// The last layer from its attention on (its keys and values already in the cache), then the final norm and
/// head, for row total - 1 only. `layer` is the layer's attention index.
pub fn run(e: *E.Engine, layer: usize, total: usize) !void {
    const g = e.g;
    const l = e.layers[e.layers.len - 1];
    const m = l.mixer.attention;
    const d = l.mlp.dense;
    const row = total - 1;
    const aw = g.attnWidth();
    const hid = g.hidden;
    const params = e.tables.ptr + E.Engine.params_at;
    const y = e.y + row * aw * 2;
    const mix = e.mix + row * aw * 2;
    const h = e.h + row * hid * 2;
    const r = e.r + row * hid * 2;
    const x = e.x + row * hid * 2;
    const act = e.act + row * g.intermediate * 2;
    try prompt.runFrom(e, layer, row, total);
    try e.runAfter(e.fns.attn_gate, .{ (aw + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ e.geo, y, e.qraw + row * 2 * aw * 2, mix, @as(c_int, 1), @as(u64, 0), @as(u64, 0) });
    try e.prompt_many.prompt(e.stream, mix, &.{.{ .q = m.o.a4, .out = r }}, 1);
    try e.runAfter(e.fns.norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, .{ h, r, l.post_norm, x, g.eps, @as(c_int, @intCast(hid)), params, @as(u64, 0) });
    try e.prompt_many.mlp(e.stream, x, d.gate.a4, d.up.a4, act, 1);
    try e.prompt_many.prompt(e.stream, act, &.{.{ .q = d.down.a4, .out = r }}, 1);
    try e.runAfter(e.fns.norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, .{ h, r, e.final_norm, x, g.eps, @as(c_int, @intCast(hid)), params, @as(u64, 0) });
    try e.proj(e.head, x, e.logits, 1, false);
}
