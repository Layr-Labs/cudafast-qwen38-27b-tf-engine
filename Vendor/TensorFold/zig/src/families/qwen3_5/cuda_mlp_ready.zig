//! A verify window's dense MLP with its down projection taking each gate/up tile as it lands, instead of after the
//! whole gate/up grid (qmm_ready.cuh). The two launches are the record's compact pair-only gate/up kernel and its
//! output-stream down projection; their twins add only the readiness words, so act, its group sums and down's
//! output are the same bits: down's four K slices still take their groups in order and add the slices in order.
//!
//! Progress contract: gate/up CTA t publishes tile t (act columns 128t..128t+127 of every row, then their two group
//! sums) with a gpu-scope release after a CTA barrier that follows all of its stores. A down K slice acquires the tile
//! of its next group (gpu scope, one thread) before the CTA barrier that precedes that group's x loads. The value is
//! the forward's generation plus one; the last down CTA of a launch moves the generation after every down CTA has read
//! it, so a later launch pair of the forward never matches an earlier pair's flags, and a forward starts from zeros.
//!
//! No deadlock: the gate/up kernel triggers its dependents only after its own griddepcontrol.wait (at its start),
//! so the down grid cannot start before every gate/up CTA has started, and a gate/up CTA never waits on down. The
//! prior kernels' completion that down's griddepcontrol.wait used to give follows from the same trigger, which every
//! gate/up CTA reaches after its own wait. Without PDL the down grid starts after gate/up finished (flags all set).
//! Both twins or neither: `run` takes the pair only when both selections hold, else the caller's two launches run.
const std = @import("std");
const platform = @import("sys");
const engine = @import("cuda_engine.zig");
const cuda = @import("cuda");

const Engine = engine.Engine;
const Proj = engine.Proj;

/// The gate/up tiles a 16-row window has (17408 intermediate columns, 128 a CTA), one flag each.
pub const tiles = 17408 / 128;
/// Generation, finished down CTAs, then the tile flags (u32 each).
pub const bytes = (2 + tiles) * 4;

/// TF_MLP_READY=0: the record's two launches, each down waiting on the whole gate/up grid.
fn on() bool {
    const S = struct {
        var v: ?bool = null;
    };
    if (S.v == null) S.v = if (platform.c.getenv("TF_MLP_READY")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    return S.v.?;
}

/// A forward of at most 16 rows starts from zeros (on the stream, before its graph or launches).
pub fn clear(e: *Engine, total: usize) !void {
    if (!on() or e.mlp_ready == 0 or total == 0 or total > 16 or !e.affine.pdl) return;
    try e.d.check(e.d.api.cuMemsetD8Async(e.mlp_ready, 0, bytes, e.stream.handle), "cuMemsetD8Async");
}

/// The window's gate, up and activation into e.act and act_sums, then down into e.r, as cuda_forward.zig's
/// mlp_pair.pair then project(down) would launch them; false (nothing launched) when either twin does not apply.
pub fn run(e: *Engine, gate: cuda.qlinear.Affine4, up: cuda.qlinear.Affine4, down: Proj, act_sums: u64, total: usize) !bool {
    if (!on() or e.mlp_ready == 0 or total == 0 or total > 16 or !e.affine.pdl or !e.mlp_pair.pdl) return false;
    if (gate.n != tiles * 128) return false;
    const q = switch (down) {
        .a4 => |a| a,
        else => return false,
    };
    if (e.mlp_pair.compact_ready_fn == null or !e.mlp_pair.compactFits(gate, up, total)) return false;
    if (total > e.affine.verifyRows() or !e.affine.downReadyFits(q, total)) return false;
    try e.mlp_pair.pairWith(e.stream, e.x, e.xs, gate, up, e.act, act_sums, total, e.mlp_ready);
    try e.affine.clusterWith(e.stream, e.act, act_sums, q, e.r, total, cuda.qlinear.splitK(q.n, q.k), e.mlp_ready);
    return true;
}

test "readiness words: one flag a gate/up tile past the generation and count" {
    try std.testing.expectEqual(@as(usize, 136), tiles);
    try std.testing.expectEqual(@as(usize, 552), bytes);
}
