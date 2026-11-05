//! TF_DTPROF=1: GPU-clock marks around a decode round's drafter frame and verify forward on the engine stream.
//! Per round it prints the frame span, the frame-end to verify-entry gap, the verify staging span and the verify
//! graph span (ms), measured from events recorded in stream order. Off (no events) unless the env is set.
const std = @import("std");
const cuda = @import("cuda");

pub const Mark = enum(u8) { frame_begin, frame_end, verify_entry, verify_launch, verify_end, post_sync, absorb_entry, absorb_exit };
const n_marks = 8;

var on: ?bool = null;
var made = false;
var ev: [n_marks]cuda.abi.Event = @splat(null);
var hit: [n_marks]bool = @splat(false);
var api: ?*const cuda.abi.Api = null;
var rounds: usize = 0;
var sums: [4]f64 = @splat(0);
var prev_end = false;
var tail: [n_marks]bool = @splat(false);
var turn_n: usize = 0;
var turn_sum: f64 = 0;

pub fn enabled() bool {
    if (on == null) on = @import("sys").c.getenv("TF_DTPROF") != null;
    return on.?;
}

fn elapsed(a: Mark, b: Mark) ?f32 {
    if (!hit[@intFromEnum(a)] or !hit[@intFromEnum(b)]) return null;
    var ms: f32 = 0;
    if (api.?.cuEventElapsedTime(&ms, ev[@intFromEnum(a)], ev[@intFromEnum(b)]) != cuda.abi.success) return null;
    return ms;
}

/// Records `m` on `stream`. At frame_begin the previous round (complete: the host synchronized since) is reported.
pub fn mark(d: anytype, stream: anytype, m: Mark) void {
    if (!enabled()) return;
    if (!made) {
        made = true;
        api = &d.api;
        for (&ev) |*x| if (d.api.cuEventCreate(x, 0) != cuda.abi.success) {
            on = false;
            return;
        };
    }
    if (m == .frame_begin) {
        report();
        hit = @splat(false);
        // the previous verify's end stays recorded: the host's turn from it to this frame is read at verify entry
        hit[@intFromEnum(Mark.verify_end)] = prev_end;
        for ([_]Mark{ .post_sync, .absorb_entry, .absorb_exit }) |t| hit[@intFromEnum(t)] = tail[@intFromEnum(t)];
    }
    if (m == .verify_entry) {
        if (elapsed(.verify_end, .frame_begin)) |t| {
            turn_n += 1;
            turn_sum += t;
            const a = elapsed(.verify_end, .post_sync) orelse -1;
            const b = elapsed(.post_sync, .absorb_entry) orelse -1;
            const c = elapsed(.absorb_entry, .absorb_exit) orelse -1;
            const dd = elapsed(.absorb_exit, .frame_begin) orelse -1;
            std.debug.print("[dtprof] turn {d:.3} ms (verify end to frame begin) | mean over {d}: {d:.3} | sync {d:.3} conclude {d:.3} absorb {d:.3} stage {d:.3}\n", .{ t, turn_n, turn_sum / @as(f64, @floatFromInt(turn_n)), a, b, c, dd });
        }
    }
    if (m == .verify_end) {
        prev_end = true;
        tail = @splat(false);
    }
    if (m == .post_sync or m == .absorb_entry or m == .absorb_exit) tail[@intFromEnum(m)] = true;
    if (d.api.cuEventRecord(ev[@intFromEnum(m)], stream.handle) == cuda.abi.success) hit[@intFromEnum(m)] = true;
}

fn report() void {
    const f = elapsed(.frame_begin, .frame_end) orelse return;
    const g = elapsed(.frame_end, .verify_entry) orelse return;
    const s = elapsed(.verify_entry, .verify_launch) orelse return;
    const v = elapsed(.verify_launch, .verify_end) orelse return;
    rounds += 1;
    for (&sums, [_]f32{ f, g, s, v }) |*x, y| x.* += y;
    const n: f64 = @floatFromInt(rounds);
    std.debug.print("[dtprof] frame {d:.3} gap {d:.3} stage {d:.3} verify {d:.3} ms | mean over {d}: {d:.3} {d:.3} {d:.3} {d:.3}\n", .{ f, g, s, v, rounds, sums[0] / n, sums[1] / n, sums[2] / n, sums[3] / n });
}

/// Sub-marks inside one absorb (TF_DTPROF): 0 at its start, then each step's end; the previous absorb's steps are
/// printed when the next one starts (the host synchronized in between).
const n_sub = 8;
var sub_ev: [n_sub]cuda.abi.Event = @splat(null);
var sub_hit: [n_sub]bool = @splat(false);
var sub_made = false;
const sub_names = [n_sub][]const u8{ "start", "gather", "fc", "norm", "kv", "layers", "-", "-" };

pub fn sub(d: anytype, stream: anytype, i: usize) void {
    if (!enabled()) return;
    if (!sub_made) {
        sub_made = true;
        if (api == null) api = &d.api;
        for (&sub_ev) |*x| if (d.api.cuEventCreate(x, 0) != cuda.abi.success) return;
    }
    if (i == 0) {
        var buf: [256]u8 = undefined;
        var w: usize = 0;
        var prev: usize = 0;
        var any = false;
        for (1..n_sub) |k| if (sub_hit[0] and sub_hit[k]) {
            var ms: f32 = 0;
            if (api.?.cuEventElapsedTime(&ms, sub_ev[prev], sub_ev[k]) == cuda.abi.success) {
                const t = std.fmt.bufPrint(buf[w..], " {s} {d:.3}", .{ sub_names[k], ms }) catch break;
                w += t.len;
                any = true;
            }
            prev = k;
        };
        if (any) std.debug.print("[dtprof] absorb steps:{s}\n", .{buf[0..w]});
        sub_hit = @splat(false);
    }
    if (d.api.cuEventRecord(sub_ev[i], stream.handle) == cuda.abi.success) sub_hit[i] = true;
}

/// TF_DTPROF: the device tree kernel's own span (tree_begin/tree_end), printed at the next begin.
var tree_ev: [2]cuda.abi.Event = .{ null, null };
var tree_hit = false;
var tree_n: usize = 0;
var tree_sum: f64 = 0;
pub fn tree(d: anytype, stream: anytype, end: bool) void {
    if (!enabled()) return;
    if (tree_ev[0] == null) {
        if (api == null) api = &d.api;
        for (&tree_ev) |*x| if (d.api.cuEventCreate(x, 0) != cuda.abi.success) return;
    }
    if (!end and tree_hit) {
        var ms: f32 = 0;
        if (api.?.cuEventElapsedTime(&ms, tree_ev[0], tree_ev[1]) == cuda.abi.success) {
            tree_n += 1;
            tree_sum += ms;
            std.debug.print("[dtprof] tree kernel {d:.3} ms | mean over {d}: {d:.3}\n", .{ ms, tree_n, tree_sum / @as(f64, @floatFromInt(tree_n)) });
        }
        tree_hit = false;
    }
    _ = d.api.cuEventRecord(tree_ev[@intFromBool(end)], stream.handle);
    if (end) tree_hit = true;
}
