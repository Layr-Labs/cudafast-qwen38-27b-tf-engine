//! Qwen row normalization: omit unused prompt sums and final rows.
const Engine = @import("cuda_engine.zig").Engine;

pub fn norm(e: *Engine, residual: bool, w: u64, rows: usize) !void {
    return e.normInput(residual, w, rows, false);
}

/// Prompt GEMMs consume the normalized rows directly; native and int8 projections still need group sums.
pub fn normInput(e: anytype, residual: bool, w: u64, rows: usize, prompt: bool) !void {
    const sums: u64 = if (prompt and rows > 16 and !e.int8.on) 0 else e.xs;
    try e.runAfter(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ e.h, if (residual) e.r else @as(u64, 0), w, e.x, e.g.eps, @as(c_int, @intCast(e.g.hidden)), e.tables.ptr + Engine.params_at, sums });
}

/// The vocabulary head consumes only these final rows; earlier prompt rows have already supplied their taps.
pub fn normTail(e: anytype, w: u64, total: usize, heads: usize) !void {
    const offset = (total - heads) * e.g.hidden * 2;
    try e.runAfter(e.fns.norm, .{ heads, 1, 1 }, .{ 256, 1, 1 }, .{ e.h + offset, e.r + offset, w, e.x + offset, e.g.eps, @as(c_int, @intCast(e.g.hidden)), e.tables.ptr + Engine.params_at, @as(u64, 0) });
}
