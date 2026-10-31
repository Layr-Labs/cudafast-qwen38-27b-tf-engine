//! At engine open: each prompt GEMM class of the model (GDN's qkv/z/a/b group and its out projection, attention's
//! q/k/v group and o, the dense MLP's fused gate/up and its down) timed in every prompt_many variant at row-tile
//! buckets 1, 2, 4, 8 and 16, on this device, and the quickest kept (prompt_many.zig `tuned`).
//!
//! Content-free: the inputs are the scratch rows as they are (zeros at open), the weights are the model's; nothing
//! is built from or keyed by a prompt. Every variant keeps each element's MMA chain, so the choice changes speed only.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const PM = cuda.prompt_many;

const rows_list = [_]usize{ 128, 256, 512, 1024, 2048 };
const reps = 3;

const Class = struct { name: []const u8, x: u64, outs: [PM.parts_max]PM.Out = undefined, n: usize = 0, mlp: ?struct { gate: cuda.qlinear.Affine4, up: cuda.qlinear.Affine4, act: u64 } = null };

fn a4(p: E.Proj) ?cuda.qlinear.Affine4 {
    return switch (p) {
        .a4 => |q| q,
        else => null,
    };
}

fn group(name: []const u8, x: u64, ps: []const E.Proj, outs: []const u64) ?Class {
    var c: Class = .{ .name = name, .x = x };
    for (ps, outs) |p, o| {
        c.outs[c.n] = .{ .q = a4(p) orelse return null, .out = o };
        c.n += 1;
    }
    return c;
}

fn classes(e: *E.Engine, buf: *[6]Class) []Class {
    var n: usize = 0;
    var delta = false;
    var attn = false;
    var dense = false;
    for (e.layers) |l| {
        switch (l.mixer) {
            .delta => |m| if (!delta) {
                delta = true;
                if (group("gdn qkv/z/a/b", e.x, &.{ m.qkv, m.z, m.a, m.b }, &.{ e.qkv, e.z, e.a, e.b })) |c| {
                    buf[n] = c;
                    n += 1;
                }
                if (group("gdn out", e.mix, &.{m.out}, &.{e.r})) |c| {
                    buf[n] = c;
                    n += 1;
                }
            },
            .attention => |m| if (!attn) {
                attn = true;
                if (group("attn q/k/v", e.x, &.{ m.q, m.k, m.v }, &.{ e.qraw, e.kraw, e.vraw })) |c| {
                    buf[n] = c;
                    n += 1;
                }
            },
        }
        switch (l.mlp) {
            .dense => |m| if (!dense) {
                dense = true;
                const gate = a4(m.gate) orelse continue;
                const up = a4(m.up) orelse continue;
                if (PM.Kernels.mlpFits(gate, up) and gate.n == e.g.intermediate) {
                    buf[n] = .{ .name = "mlp gate/up", .x = e.x, .mlp = .{ .gate = gate, .up = up, .act = e.act } };
                    n += 1;
                }
                if (group("mlp down", e.act, &.{m.down}, &.{e.r})) |c| {
                    buf[n] = c;
                    n += 1;
                }
            },
            else => {},
        }
    }
    return buf[0..n];
}

fn runOnce(e: *E.Engine, c: Class, v: usize, rows: usize) !void {
    if (c.mlp) |m| return e.prompt_many.mlpVariant(v, e.stream, c.x, m.gate, m.up, m.act, rows);
    return e.prompt_many.promptVariant(v, e.stream, c.x, c.outs[0..c.n], rows);
}

fn timeOf(e: *E.Engine, c: Class, v: usize, rows: usize, t0: cuda.Event, t1: cuda.Event) !f32 {
    try runOnce(e, c, v, rows); // loads, first-launch costs
    try t0.record(e.stream);
    for (0..reps) |_| try runOnce(e, c, v, rows);
    try t1.record(e.stream);
    try t1.synchronize();
    return (try cuda.Event.elapsedMs(t0, t1)) / reps;
}

pub fn run(e: *E.Engine) !void {
    var buf: [6]Class = undefined;
    const cs = classes(e, &buf);
    var t0 = try cuda.Event.init(e.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(e.d, true);
    defer t1.deinit();
    for (cs) |c| {
        var ns: [PM.parts_max]usize = undefined;
        for (c.outs[0..c.n], ns[0..c.n]) |o, *n| n.* = o.q.n;
        for (rows_list) |rows| {
            const key = if (c.mlp) |m| PM.Kernels.keyOf(true, m.gate.k, &.{m.gate.n}, rows) else PM.Kernels.keyOf(false, c.outs[0].q.k, ns[0..c.n], rows);
            const incumbent = if (c.mlp != null) 11 else e.prompt_many.defaultFor(c.outs[0..c.n], rows);
            const base = try timeOf(e, c, incumbent, rows, t0, t1);
            var best = incumbent;
            var best_ms = base;
            for (0..PM.variants.len) |v| {
                if (v == incumbent or !e.prompt_many.takes(v, if (c.mlp != null) null else c.outs[0..c.n], rows)) continue;
                const ms = try timeOf(e, c, v, rows, t0, t1);
                if (ms < best_ms) {
                    best = v;
                    best_ms = ms;
                }
            }
            // a variant replaces the default only when clearly quicker: timing noise must not flip choices
            if (best != incumbent and best_ms < base * 0.985) e.prompt_many.choose(key, @intCast(best)) else best = incumbent;
            if (rows == 512) std.debug.print("info: prompt GEMM {s} at {d} rows: {s} {d:.3} ms (default {s} {d:.3} ms)\n", .{ c.name, rows, PM.variants[best].name, if (best == incumbent) base else best_ms, PM.variants[incumbent].name, base });
        }
    }
    try e.stream.synchronize();
}
