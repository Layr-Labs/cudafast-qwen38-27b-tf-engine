//! Grouped projections share input sums and preserve each caller's arithmetic role.
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Proj = engine.Proj;

pub fn projectMany(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
    if (ls.len == 0 or ls.len != outs.len) return error.UnexpectedTensor;
    const A = cuda.qlinear.Affine4Kernels;
    // a prompt chunk past 16 rows: its affine-4 projections of one input share one prompt-GEMM launch, each block
    // the tile its own launch's same-index block computes (no group sums: the prompt GEMM reads none)
    if (prompt and rows > 16 and !e.int8.on and ls.len <= A.segs_max) {
        var many: [A.segs_max]A.Out = undefined;
        var n: usize = 0;
        for (ls, outs) |l, out| {
            if (l == .a4 and l.a4.k == ls[0].inputs()) {
                many[n] = .{ .q = l.a4, .out = out };
                n += 1;
            } else try e.project(l, x, xs, out, rows, prompt, params, false);
        }
        if (n > 0) try e.affine.promptMany(e.stream, x, many[0..n], rows);
        return;
    }
    // GB10 has no split form: a decode or verify window of one row tile groups its projections by kernel instead
    const grouped = e.affine.split != null or (!prompt and rows <= A.max_rows);
    const segmented = grouped and !(prompt and rows > 16) and ls.len <= A.segs_max and for (ls) |l| {
        if (l != .a4 or l.inputs() != ls[0].inputs()) break false;
    } else true;
    if (!segmented) {
        for (ls, outs, 0..) |l, out, i| try e.project(l, x, xs, out, rows, prompt, params, sums and i == 0);
        return;
    }
    const k = ls[0].inputs();
    if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    var many: [A.segs_max]A.Out = undefined;
    var at: usize = 0;
    while (at < rows) : (at += A.max_rows) {
        for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a4, .out = out + at * l.outputs() * 2 };
        try e.affine.decodeMany(e.stream, x + at * k * 2, xs + at * (k / 64) * 4, many[0..ls.len], @min(A.max_rows, rows - at));
    }
}
