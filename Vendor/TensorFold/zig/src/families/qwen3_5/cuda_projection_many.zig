//! Grouped projections share input sums and preserve each caller's arithmetic role.
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Proj = engine.Proj;

pub fn projectMany(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
    if (ls.len == 0 or ls.len != outs.len) return error.UnexpectedTensor;
    // a prompt chunk's projections of one input in one launch, each with the single prompt GEMM's bits
    const P = cuda.prompt_many;
    const grouped = prompt and rows > 16 and !e.int8.on and ls.len <= P.parts_max and for (ls) |l| {
        if (l != .a4 or l.inputs() != ls[0].inputs()) break false;
    } else true;
    if (grouped) {
        var parts: [P.parts_max]P.Out = undefined;
        for (ls, outs, parts[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a4, .out = out };
        return e.prompt_many.prompt(e.stream, x, parts[0..ls.len], rows);
    }
    const A = cuda.qlinear.Affine4Kernels;
    // GB10 has no split form: a decode or verify window of one row tile groups its projections by kernel instead
    const by_kernel = e.affine.split != null or (!prompt and rows <= A.max_rows);
    const segmented = by_kernel and !(prompt and rows > 16) and ls.len <= A.segs_max and for (ls) |l| {
        if (l != .a4 or l.inputs() != ls[0].inputs()) break false;
    } else true;
    if (!segmented) {
        for (ls, outs, 0..) |l, out, i| try e.project(l, x, xs, out, rows, prompt, params, sums and i == 0);
        return;
    }
    const k = ls[0].inputs();
    if (sums) try e.runAfter(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    var many: [A.segs_max]A.Out = undefined;
    var at: usize = 0;
    while (at < rows) : (at += A.max_rows) {
        for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a4, .out = out + at * l.outputs() * 2 };
        try e.affine.decodeMany(e.stream, x + at * k * 2, xs + at * (k / 64) * 4, many[0..ls.len], @min(A.max_rows, rows - at));
    }
}
