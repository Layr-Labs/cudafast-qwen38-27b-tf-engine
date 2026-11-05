//! TF_DROUND: the step from a verify to the next drafter frame on the device. Right after the verify's graph (before
//! the host synchronizes on its picks) the engine enqueues: the accepted path and commits (tf_q35d_dround_accept), the
//! drafter's absorb of the path's rows (Draft.absorbDev), the frame on the device-written ids and segment, and the
//! device tree (cuda_dtree.zig, TF_DTREE). The host's later absorb and hold take that work when they ask for exactly
//! it, and do their own otherwise (the frame and tree are pure; a ring row the host absorbs again gets its bits).
//!   TF_DROUND=1  use: the host's absorb and hold take the device round's (the tree too, with TF_DTREE=1).
//!   TF_DROUND=2  check: the host absorbs and holds as before; the path, the ring rows, the frame's words and
//!                selector rows and the tree are compared with the device round's, every round.
//! Unset or 0: nothing runs.
const std = @import("std");
const cuda = @import("cuda");
const draft = @import("cuda_draft.zig");
const frame = @import("cuda_draft_frame.zig");
const dtree = @import("cuda_dtree.zig");
const dr = @import("lanes").drafter;

pub const Mode = enum { off, use, check };

/// qwen35_dround.cuh's DroundState.
const State = extern struct { n: c_int, rows: [64]c_int, pending: u32, hist_len: c_int, begin: i64, end: i64 };
const state_at = 0;
const wtok_at = 512;
const wpar_at = 768;
const block_len = 1024;

var mode_: ?Mode = null;
pub var drafter: ?*draft.Draft = null;
var ready = false;
var broken = false;
var block: ?cuda.HostBuffer = null;
var block_dev: u64 = 0;

const Did = struct {
    armed: bool = false,
    absorbed: bool = false,
    ctx: ?*draft.Ctx = null,
    tap: u64 = 0,
    start: u64 = 0,
    rows: usize = 0,
    narrow: bool = true,
    spanned: bool = false,
    tree: bool = false,
};
var did: Did = .{};
var rounds: usize = 0;
var taken_absorb: usize = 0;
var taken_frame: usize = 0;
var bad: usize = 0;
// check mode: the device round's ring rows, words and selector rows, and tree, before the host's own
var snap_ring: [5 * 2 * 32 * 1024 * 2]u8 = undefined;
var snap_ring_len: usize = 0;
var snap_frame: []u8 = &.{};
var snap_tree: struct { n: usize = 0, held: [64]u32 = undefined, parents: [64]i32 = undefined, own: [64]f64 = undefined } = .{};
var check_frame = false;
var snap_pending = false;

pub fn mode() Mode {
    if (mode_ == null) {
        const v = @import("sys").c.getenv("TF_DROUND");
        mode_ = if (v) |s| (if (std.mem.eql(u8, std.mem.span(s), "1")) Mode.use else if (std.mem.eql(u8, std.mem.span(s), "2")) Mode.check else Mode.off) else .use;
    }
    return mode_.?;
}

fn setup(d: *draft.Draft) bool {
    if (ready) return !broken;
    ready = true;
    broken = true;
    var b = cuda.HostBuffer.allocMapped(d.e.d, block_len) catch return false;
    block_dev = b.device() catch {
        b.free();
        return false;
    };
    block = b;
    snap_frame = d.gpa.alloc(u8, frame.stagedLen(d.c.rank)) catch return false;
    broken = false;
    std.debug.print("info: dround ({s}): the verify-to-frame step on the device\n", .{@tagName(mode())});
    return true;
}

fn state() *State {
    return @ptrCast(@alignCast(block.?.bytes[state_at..].ptr));
}

/// The engine, after a verify's graph and its picks' kernels (toks: the rows' picks on the device), before it reads them.
pub fn afterVerify(e: anytype, part_cache: anytype, start: u64, rows: usize, parents: ?[]const i32, ids: []const u32, toks: u64) void {
    did = .{};
    if (mode() == .off) return;
    const d = drafter orelse return;
    if (d.e != e or d.streams.count() != 1 or d.c.block != 16 or rows < 1 or rows > 32) return;
    if (dtree.mode() == .off or dtree.hist_len_last == 0 or dtree.hist_len_last + rows > dtree.hist_capacity) return;
    if (!setup(d)) return;
    var vi = d.streams.valueIterator();
    const ctx = vi.next().?.*;
    const tap = part_cache.tap orelse return;
    const b = block.?.bytes;
    const wt = std.mem.bytesAsSlice(u32, b[wtok_at..][0 .. 64 * 4]);
    const wp = std.mem.bytesAsSlice(i32, b[wpar_at..][0 .. 64 * 4]);
    for (0..rows) |r| {
        wt[r] = ids[r];
        wp[r] = if (parents) |ps| ps[r] else @as(i32, @intCast(r)) - 1;
    }
    const st_dev = block_dev + state_at;
    did.ctx = ctx;
    did.tap = tap.ptr;
    did.start = start;
    did.rows = rows;
    did.narrow = frame.narrowOf(ctx);
    run(d, ctx, tap.ptr, start, rows, toks, st_dev) catch |err| {
        std.debug.print("[dround] enqueue failed: {s}\n", .{@errorName(err)});
        did = .{};
        return;
    };
    did.armed = true;
    rounds += 1;
}

fn run(d: *draft.Draft, ctx: *draft.Ctx, tap: u64, start: u64, rows: usize, toks: u64, st_dev: u64) !void {
    const e = d.e;
    try frame.stageDev(d, ctx);
    try e.run(d.fns.dround_accept, .{ 1, 1, 1 }, .{ 32, 1, 1 }, .{ toks, block_dev + wtok_at, block_dev + wpar_at, @as(c_int, @intCast(rows)), @as(c_longlong, @intCast(start)), @as(c_longlong, @intCast(ctx.begin)), @as(c_longlong, @intCast(ctx.end)), @as(c_longlong, @intCast(d.c.window)), dtree.histDev(), @as(c_int, @intCast(dtree.hist_len_last)), st_dev, d.ids, d.segments });
    try d.absorbDev(ctx, tap, start, rows, st_dev);
    const spanned = try frame.runDev(d, ctx);
    did.spanned = spanned != null;
    did.tree = try dtree.launchDev(d, did.spanned, st_dev);
}

fn pathMatches(it: dr.Absorb) bool {
    const s = state();
    const n: usize = @intCast(s.n);
    if (it.rows) |rs| {
        if (rs.len != n) return false;
        for (rs, 0..) |r, i| if (r != @as(u32, @intCast(s.rows[i]))) return false;
    } else {
        if (it.features.rows != n) return false;
        for (0..n) |i| if (s.rows[i] != @as(c_int, @intCast(i))) return false;
    }
    return true;
}

fn ringRows(d: *draft.Draft, ctx: *draft.Ctx, n: usize, out: []u8) !usize {
    const e = d.e;
    const w = d.c.kvWidth() * 2;
    var at: usize = 0;
    try e.stream.synchronize();
    for (0..d.c.layers) |l| for ([2]u64{ d.ringKPub(ctx, l), d.ringVPub(ctx, l) }) |base| for (0..n) |i| {
        const slot = (did.start + i) % d.c.ringRows();
        if (at + w > out.len) return error.SnapTooSmall;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(out[at..].ptr, base + slot * w, w), "cuMemcpyDtoH");
        at += w;
    };
    return at;
}

/// The drafter's absorb of one item: true when the device round already absorbed exactly it (the context's bounds set).
pub fn takeAbsorb(d: *draft.Draft, ctx: *draft.Ctx, it: dr.Absorb) bool {
    if (!did.armed or did.absorbed or ctx != did.ctx or it.prompt) return false;
    const s = state();
    const ok = it.features.buffer == did.tap and it.features.offset == did.start and pathMatches(it);
    if (mode() == .check) {
        if (!ok) {
            bad += 1;
            std.debug.print("[dround] CHECK absorb differs: host start {d} device {d}, host rows {d} device n {d}\n", .{ it.features.offset, did.start, if (it.rows) |r| r.len else it.features.rows, s.n });
            did.armed = false;
            return false;
        }
        snap_ring_len = ringRows(d, ctx, @intCast(s.n), &snap_ring) catch 0;
        did.absorbed = true;
        snap_pending = true;
        return false;
    }
    if (!ok) {
        did.armed = false;
        return false;
    }
    ctx.begin = @intCast(s.begin);
    ctx.end = @intCast(s.end);
    did.absorbed = true;
    taken_absorb += 1;
    return true;
}

/// check mode, after the host's absorb of the same rows: its ring rows against the device round's.
pub fn afterAbsorb(d: *draft.Draft, ctx: *draft.Ctx) void {
    if (mode() != .check or !snap_pending or snap_ring_len == 0) return;
    snap_pending = false;
    var mine: [snap_ring.len]u8 = undefined;
    const n = ringRows(d, ctx, @intCast(state().n), &mine) catch return;
    const same = n == snap_ring_len and std.mem.eql(u8, mine[0..n], snap_ring[0..n]);
    const s = state();
    if (!same) bad += 1;
    if (!same or (ctx.begin != s.begin or ctx.end != s.end)) std.debug.print("[dround] CHECK ring rows {s}, bounds host {d}..{d} device {d}..{d}\n", .{ if (same) "same" else "DIFFER", ctx.begin, ctx.end, s.begin, s.end });
}

/// The drafter's hold: true when the device round's frame (and tree) is this hold's, published from it.
pub fn takeFrame(d: *draft.Draft, items: []const dr.Hold, contexts: []const *draft.Ctx) !bool {
    if (!did.armed or !did.absorbed) return false;
    defer did.armed = false;
    if (items.len != 1 or contexts[0] != did.ctx or items[0].lanes != null) return false;
    const it = items[0];
    const s = state();
    var ok = it.pending == s.pending and it.position == @as(u64, @intCast(s.end)) + 1 and frame.narrowOf(contexts[0]) == did.narrow and it.sampling == null;
    // the device's history after its commits is the stream's tokens
    const hl: usize = @intCast(s.hist_len);
    if (ok and hl == it.tokens.len) {
        const hist = dtree.histHost(hl);
        if (!std.mem.eql(u32, hist, it.tokens)) ok = false;
    } else ok = false;
    if (mode() == .check) {
        if (!ok) {
            bad += 1;
            std.debug.print("[dround] CHECK hold differs: pending host {d} device {d}, position host {d} device end {d}, tokens host {d} device {d}\n", .{ it.pending, s.pending, it.position, s.end, it.tokens.len, s.hist_len });
            return false;
        }
        @memcpy(snap_frame, d.staged.bytes[0..snap_frame.len]);
        snap_tree.n = 0;
        if (did.tree) {
            const r = dtree.resultPub();
            snap_tree.n = r.n;
            for (0..r.n) |i| {
                snap_tree.held[i] = r.held[i];
                snap_tree.parents[i] = r.parents[i];
                snap_tree.own[i] = r.own[i];
            }
        }
        check_frame = true;
        return false;
    }
    if (!ok) return false;
    dtree.hist_len_last = hl;
    taken_frame += 1;
    try frame.finishDev(d, items, contexts, did.spanned);
    if (taken_frame == 1 or taken_frame % 64 == 0) std.debug.print("[dround] rounds {d} absorbs taken {d} frames taken {d}\n", .{ rounds, taken_absorb, taken_frame });
    return true;
}

/// check mode, after the host's own frame and tree: the device round's against them.
pub fn afterFrame(d: *draft.Draft, ctx: *draft.Ctx) void {
    if (mode() != .check or !check_frame) return;
    check_frame = false;
    const rows = d.c.block - 1;
    const wb = rows * 43 * 4;
    const hb = rows * d.c.rank * 2;
    const st = d.staged.bytes;
    // the host's words are remapped after the readback: compare the selector rows bit for bit, the words' ids
    // after the same remap (the device copy was taken before it), logits bit for bit
    const hp_same = std.mem.eql(u8, st[frame.hp_at..][0..hb], snap_frame[frame.hp_at..][0..hb]);
    var words_same = true;
    const hw = std.mem.bytesAsSlice(u32, st[frame.words_at..][0..wb]);
    const dw = std.mem.bytesAsSlice(u32, snap_frame[frame.words_at..][0..wb]);
    for (0..rows) |r| for (0..43) |j| {
        var x = dw[r * 43 + j];
        if (did.spanned and j >= 2 and j < 2 + d.c.topk and x >= 98304) x += 248032 - 98304;
        if (x != hw[r * 43 + j]) words_same = false;
    };
    var tree_same = true;
    if (did.tree) {
        if (snap_tree.n != ctx.grown) {
            tree_same = false;
        } else for (0..ctx.grown) |i| {
            if (snap_tree.held[i] != ctx.held[i] or snap_tree.parents[i] != ctx.parents[i] or @as(u64, @bitCast(snap_tree.own[i])) != @as(u64, @bitCast(ctx.own[i]))) tree_same = false;
        }
    }
    if (!(hp_same and words_same and tree_same)) bad += 1;
    std.debug.print("[dround] check round {d}: path n {d} words {s} selector {s} tree {s} | bad so far {d}\n", .{ rounds, state().n, if (words_same) "same" else "DIFFER", if (hp_same) "same" else "DIFFER", if (!did.tree) "-" else if (tree_same) "same" else "DIFFER", bad });
    dtree.hist_len_last = @intCast(state().hist_len);
}
