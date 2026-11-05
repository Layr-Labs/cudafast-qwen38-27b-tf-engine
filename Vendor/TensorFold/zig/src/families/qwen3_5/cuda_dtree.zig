//! TF_DTREE: the drafter's tree grown on the device (qwen35_dtree.cuh tf_q35d_dtree_grow) right after the frame, from
//! the frame's own words and selector rows, the codebooks and the stream's history in mapped host memory.
//!   TF_DTREE=2  check: the host grows as before and every round's device tree is compared with it, node for node
//!               (tokens, parents, own log-chances' bits, count); mismatches are printed and counted.
//!   TF_DTREE=1  the hold takes the device's tree instead of growing it on the host.
//! Unset or 0: nothing runs. A hold the kernel does not cover (several streams, lanes, a sampled stream, a branch or
//! node budget past its open list) keeps the host's tree in every mode.
const std = @import("std");
const cuda = @import("cuda");
const tree = @import("cuda_draft_tree.zig");
const dr = @import("lanes").drafter;

pub const Mode = enum { off, use, check };

const hist_cap = 1 << 16;
const max_rows = 15;
const max_open = 32 * 12;

/// qwen35_dtree.cuh's Args.
const Args = extern struct {
    words: u64,
    hp: u64,
    pred: u64,
    succ: u64,
    landing: u64,
    hist: u64,
    held: u64,
    parents: u64,
    own: u64,
    grown: u64,
    edge_w: f64,
    scale: f64,
    trust: f64,
    lam: f64,
    rank: c_int,
    topk: c_int,
    rows: c_int,
    spanned: c_int,
    most: c_int,
    branch: c_int,
    hist_len: c_int,
    copy: c_int,
    pending: u32,
    st: u64 = 0, // cuda_dround.zig's device state: pending and the history's length from the device
    prof: u64 = 0, // TF_DTPROF: the kernel's cycles by phase
    tab_lc: u64 = 0, // tf_q35d_dtree_pairs' tables (0: the grow expands on its own)
    tab_place: u64 = 0,
    tab_cut: u64 = 0,
    tab_sc: u64 = 0,
};

/// Every (row, pred) expansion a grow can ask for: row 0's after the pending token, then each row's after each of the
/// previous row's candidates (TF_DTREE_PAIRS=0: the grow expands each taken node itself).
const max_pairs = 1 + (max_rows - 1) * 20;
const tab_lc_at = 0;
const tab_place_at = tab_lc_at + max_pairs * 20 * 8;
const tab_cut_at = tab_place_at + max_pairs * 20;
const tab_sc_at = std.mem.alignForward(usize, tab_cut_at + max_pairs * 4, 16);
const tab_len = tab_sc_at + max_pairs * 20 * 4;
var tabs: ?cuda.DeviceBuffer = null;
var fn_pairs: ?cuda.Function = null;

fn pairsOn() bool {
    const v = @import("sys").c.getenv("TF_DTREE_PAIRS") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The mapped host block: history, the Landing table, and the kernel's outputs.
const hist_at = 0;
const landing_at = hist_cap * 4;
const held_at = landing_at + 16 * 2 * 8;
const parents_at = held_at + 64 * 4;
const own_at = parents_at + 64 * 4;
const grown_at = own_at + 64 * 8;
const prof_at = grown_at + 64;
const block_len = prof_at + 64;

var mode_: ?Mode = null;
var ready = false;
var broken = false;
var fn_grow: ?cuda.Function = null;
var block: ?cuda.HostBuffer = null;
var block_dev: u64 = 0;
var books: ?cuda.HostBuffer = null;
var books_dev: u64 = 0;
var succ_dev: u64 = 0;
var books_device: ?cuda.DeviceBuffer = null;
var launched = false;
var rounds: usize = 0;
var mismatches: usize = 0;
var used: usize = 0;

pub fn mode() Mode {
    if (mode_ == null) {
        const v = @import("sys").c.getenv("TF_DTREE");
        mode_ = if (v) |s| (if (std.mem.eql(u8, std.mem.span(s), "1")) Mode.use else if (std.mem.eql(u8, std.mem.span(s), "2")) Mode.check else Mode.off) else .off;
    }
    return mode_.?;
}

fn setup(d: anytype) bool {
    if (ready) return !broken;
    ready = true;
    broken = true;
    const e = d.e;
    fn_grow = d.mod.function("tf_q35d_dtree_grow") catch return false;
    if (pairsOn()) {
        fn_pairs = d.mod.function("tf_q35d_dtree_pairs") catch null;
        if (fn_pairs != null) tabs = cuda.DeviceBuffer.alloc(e.d, tab_len) catch null;
    }
    var b = cuda.HostBuffer.allocMapped(e.d, block_len) catch return false;
    block_dev = b.device() catch {
        b.free();
        return false;
    };
    const book_bytes = d.w.predecessor.len * 2;
    // the drafter's own host codebooks registered for the device (no copy: no device or host memory added);
    // TF_DTREE_BOOKS=device copies them to device memory, =mapped to a mapped copy
    const how = if (@import("sys").c.getenv("TF_DTREE_BOOKS")) |v| std.mem.span(v) else "register";
    if (std.mem.eql(u8, how, "register")) {
        if (registerBooks(e, std.mem.sliceAsBytes(d.w.predecessor))) |dp| {
            if (registerBooks(e, std.mem.sliceAsBytes(d.w.successor))) |ds| {
                books_dev = dp;
                succ_dev = ds;
                block = b;
                broken = false;
                std.debug.print("info: dtree ({s}): device tree growth, the drafter's codebooks registered (no copy)\n", .{@tagName(mode())});
                return true;
            } else |err| std.debug.print("info: dtree: registering the successor codebook failed ({s})\n", .{@errorName(err)});
        } else |err| std.debug.print("info: dtree: registering the predecessor codebook failed ({s})\n", .{@errorName(err)});
    }
    if (std.mem.eql(u8, how, "device")) if (cuda.DeviceBuffer.alloc(e.d, 2 * book_bytes)) |db| {
        var dev = db;
        dev.upload(0, std.mem.sliceAsBytes(d.w.predecessor)) catch {
            dev.free();
            b.free();
            return false;
        };
        dev.upload(book_bytes, std.mem.sliceAsBytes(d.w.successor)) catch {
            dev.free();
            b.free();
            return false;
        };
        books_dev = dev.ptr;
        succ_dev = dev.ptr + book_bytes;
        books_device = dev;
        block = b;
        broken = false;
        std.debug.print("info: dtree ({s}): device tree growth, codebooks {d} MiB in device memory\n", .{ @tagName(mode()), 2 * book_bytes >> 20 });
        return true;
    } else |_| {};
    var k = cuda.HostBuffer.allocMapped(e.d, 2 * book_bytes) catch {
        b.free();
        return false;
    };
    books_dev = k.device() catch {
        k.free();
        b.free();
        return false;
    };
    succ_dev = books_dev + book_bytes;
    @memcpy(k.bytes[0..book_bytes], std.mem.sliceAsBytes(d.w.predecessor));
    @memcpy(k.bytes[book_bytes..][0 .. d.w.successor.len * 2], std.mem.sliceAsBytes(d.w.successor));
    block = b;
    books = k;
    broken = false;
    std.debug.print("info: dtree ({s}): device tree growth, codebooks {d} MiB mapped\n", .{ @tagName(mode()), 2 * book_bytes >> 20 });
    return true;
}

/// A host slice's whole pages pinned and mapped for the device (no copy); the slice's device address.
fn registerBooks(e: anytype, bytes: []const u8) !u64 {
    const page: usize = 4096;
    const lo = @intFromPtr(bytes.ptr) & ~(page - 1);
    const hi = std.mem.alignForward(usize, @intFromPtr(bytes.ptr) + bytes.len, page);
    const base: *anyopaque = @ptrFromInt(lo);
    const rc = e.d.api.cuMemHostRegister_v2(base, hi - lo, cuda.abi.host_register_devicemap);
    if (rc != cuda.abi.success) {
        std.debug.print("info: dtree: cuMemHostRegister rc {d}\n", .{rc});
        return error.RegisterFailed;
    }
    var dp: cuda.abi.DevicePtr = 0;
    if (e.d.api.cuMemHostGetDevicePointer_v2(&dp, base, 0) != cuda.abi.success) return error.NoDevicePointer;
    const at = dp + (@intFromPtr(bytes.ptr) - lo);
    if (at % 16 != 0) return error.Misaligned;
    return at;
}

/// Whether this frame's single hold is one the kernel grows exactly as the host would.
fn covered(d: anytype, items: []const dr.Hold) bool {
    if (items.len != 1 or d.c.block != 16) return false;
    const it = items[0];
    if (it.lanes != null or it.sampling != null) return false;
    const nodes = tree.nodes();
    if (nodes == 0 or nodes > 31) return false;
    if (d.c.topk > 20 or d.c.block - 1 > max_rows) return false;
    if ((nodes + 1) * tree.branchPub() > max_open) return false;
    if (it.tokens.len > hist_cap) return false;
    // the device reads the Landing table itself: it matches the host only when the host does too (TF_TRUST_FIX=1),
    // or when the trust weighting is off
    if (!@import("cuda_draft.zig").Draft.trustFix() and d.tuneValues()[0] > 0 and @import("sys").c.getenv("TF_DTREE_FORCE") == null) return false;
    return true;
}

/// After the frame's forward, before its words are read back: the device tree for items[0], into the mapped block.
pub fn launch(d: anytype, items: []const dr.Hold, spanned: bool) !void {
    launched = false;
    if (mode() == .off or !covered(d, items) or !setup(d)) return;
    const it = items[0];
    const b = block.?.bytes;
    @memcpy(std.mem.bytesAsSlice(u32, b[hist_at..][0 .. it.tokens.len * 4]), it.tokens);
    const n: usize = if (it.depth == 0) 0 else tree.nodes();
    try run(d, it.pending, it.tokens.len, n, n >= d.c.block - 1 and it.tokens.len >= 4, spanned, 0);
}

/// The history's length the last launch read (the device round appends after it).
pub var hist_len_last: usize = 0;

/// cuda_dround.zig: the tree after a device-fed frame, its pending token and history length from `st`.
pub fn launchDev(d: anytype, spanned: bool, st: u64) !bool {
    launched = false;
    if (mode() == .off or !setup(d) or hist_len_last == 0) return false;
    const n = tree.nodes();
    if (n == 0 or n > 31 or d.c.block != 16 or d.c.topk > 20 or (n + 1) * tree.branchPub() > max_open) return false;
    if (!@import("cuda_draft.zig").Draft.trustFix() and d.tuneValues()[0] > 0) return false;
    try run(d, 0, 0, n, n >= d.c.block - 1, spanned, st);
    return true;
}

pub var launched_most: usize = 0;
var launched_any = false;

fn run(d: anytype, pending: u32, hist_len: usize, n: usize, copy: bool, spanned: bool, st: u64) !void {
    const b = block.?.bytes;
    const tl = std.mem.bytesAsSlice(f64, b[landing_at..][0 .. 16 * 2 * 8]);
    for (d.landing.counts[0], 0..) |x, i| {
        tl[2 * i] = x[0];
        tl[2 * i + 1] = x[1];
    }
    const tl2 = d.tuneValues();
    const prof_on = @import("cuda_dtprof.zig").enabled();
    if (prof_on and launched_any) {
        const pc = std.mem.bytesAsSlice(i64, b[prof_at..][0 .. 7 * 8]);
        std.debug.print("[dtprof] tree cycles stage+edge {d} softmax {d} order {d} pick {d} chain {d} scan {d}\n", .{ pc[0], pc[2], pc[3], pc[4], pc[5], pc[6] });
    }
    launched_any = true;
    const a: Args = .{
        .words = d.words,
        .hp = d.hp,
        .pred = books_dev,
        .succ = succ_dev,
        .landing = block_dev + landing_at,
        .hist = block_dev + hist_at,
        .held = block_dev + held_at,
        .parents = block_dev + parents_at,
        .own = block_dev + own_at,
        .grown = block_dev + grown_at,
        .edge_w = tree.edgeWeightPub(),
        .scale = tree.scalePub(),
        .trust = tl2[0],
        .lam = tl2[1],
        .rank = @intCast(d.c.rank),
        .topk = @intCast(d.c.topk),
        .rows = @intCast(d.c.block - 1),
        .spanned = @intFromBool(spanned),
        .most = @intCast(n),
        .branch = @intCast(tree.branchPub()),
        .hist_len = @intCast(hist_len),
        .copy = @intFromBool(copy),
        .pending = pending,
        .st = st,
        .prof = if (prof_on) block_dev + prof_at else 0,
    };
    var ga = a;
    if (tabs != null and d.c.rank == 256 and d.c.topk <= 20) {
        const t = tabs.?.ptr;
        ga.tab_lc = t + tab_lc_at;
        ga.tab_place = t + tab_place_at;
        ga.tab_cut = t + tab_cut_at;
        ga.tab_sc = t + tab_sc_at;
        const pairs = 1 + (d.c.block - 2) * 20;
        @import("cuda_dtprof.zig").tree(d.e.d, d.e.stream, false);
        try d.e.run(fn_pairs.?, .{ pairs, 1, 1 }, .{ 32, 1, 1 }, .{ga});
        try d.e.run(fn_grow.?, .{ 1, 1, 1 }, .{ 32, 1, 1 }, .{ga});
        @import("cuda_dtprof.zig").tree(d.e.d, d.e.stream, true);
        if (st == 0) hist_len_last = hist_len;
        launched_most = n;
        launched = true;
        return;
    }
    @import("cuda_dtprof.zig").tree(d.e.d, d.e.stream, false);
    try d.e.run(fn_grow.?, .{ 1, 1, 1 }, .{ 32, 1, 1 }, .{a});
    @import("cuda_dtprof.zig").tree(d.e.d, d.e.stream, true);
    if (st == 0) hist_len_last = hist_len;
    launched_most = n;
    launched = true;
}

/// The mapped history's first n tokens on the host side.
pub fn histHost(n: usize) []const u32 {
    return std.mem.bytesAsSlice(u32, block.?.bytes[hist_at..][0 .. n * 4]);
}

pub fn resultPub() @TypeOf(result()) {
    return result();
}

pub const hist_capacity = hist_cap;

/// The mapped history's device address (cuda_dround.zig appends the round's commits to it).
pub fn histDev() u64 {
    return block_dev + hist_at;
}

fn result() struct { held: []const u32, parents: []const i32, own: []const f64, n: usize } {
    const b = block.?.bytes;
    const n: usize = @intCast(std.mem.bytesAsSlice(i32, b[grown_at..][0..4])[0]);
    return .{
        .held = std.mem.bytesAsSlice(u32, b[held_at..][0 .. 64 * 4]),
        .parents = @alignCast(std.mem.bytesAsSlice(i32, b[parents_at..][0 .. 64 * 4])),
        .own = @alignCast(std.mem.bytesAsSlice(f64, b[own_at..][0 .. 64 * 8])),
        .n = n,
    };
}

/// TF_DTREE=1: the hold's tree from the device (true), or false when the host must grow it.
pub fn take(ctx: anytype, most: usize) bool {
    if (mode() != .use or !launched or most != launched_most) return false;
    const r = result();
    if (r.n > ctx.held.len) return false;
    for (0..r.n) |i| {
        ctx.held[i] = r.held[i];
        ctx.parents[i] = r.parents[i];
        ctx.own[i] = r.own[i];
    }
    ctx.grown = r.n;
    ctx.held_n = r.n;
    used += 1;
    if (used == 1 or used % 256 == 0) std.debug.print("[dtree] used {d} device trees\n", .{used});
    return true;
}

/// TF_DTREE=2: the host's tree against the device's, every node's bits.
pub fn check(d: anytype, items: []const dr.Hold, ctx: anytype) void {
    if (mode() != .check or !launched) return;
    const r = result();
    rounds += 1;
    var bad = r.n != ctx.grown;
    if (!bad) for (0..r.n) |i| {
        if (r.held[i] != ctx.held[i] or r.parents[i] != ctx.parents[i] or @as(u64, @bitCast(r.own[i])) != @as(u64, @bitCast(ctx.own[i]))) bad = true;
    };
    if (bad and r.n > 0 and r.own[0] != 0) {
        // the host's grow again with a TrustCtx that outlives the call: which side does the device agree with
        const it = items[0];
        const tv = d.tuneValues();
        const tc = tree.TrustCtx{ .l = &d.landing, .sampled = false };
        const tune: tree.Tune = .{ .trust = tv[0], .lam = tv[1], .ctx = &tc, .chance_fn = if (tv[0] > 0) tree.landingChance else null };
        var hh: [64]u32 = undefined;
        var pp: [64]i32 = undefined;
        var oo: [64]f64 = undefined;
        const rows = d.c.block - 1;
        const n = tree.grow(d.candidates(rows), it.pending, rows, if (it.depth == 0) 0 else tree.nodes(), &hh, &pp, &oo, tune);
        var same = n == r.n;
        if (same) for (0..n) |i| {
            if (hh[i] != r.held[i] or pp[i] != r.parents[i] or @as(u64, @bitCast(oo[i])) != @as(u64, @bitCast(r.own[i]))) same = false;
        };
        std.debug.print("[dtree] scoped-ctx host grow vs device: {s}\n", .{if (same) "SAME" else "DIFFERENT"});
    }
    if (bad) {
        mismatches += 1;
        std.debug.print("[dtree] MISMATCH round {d}: host n={d} device n={d}\n", .{ rounds, ctx.grown, r.n });
        const b = block.?.bytes;
        const tl = std.mem.bytesAsSlice(f64, b[landing_at..][0 .. 16 * 2 * 8]);
        for (0..16) |i| std.debug.print("[dtree]   landing {d}: {d} {d}\n", .{ i, tl[2 * i], tl[2 * i + 1] });
        for (0..@max(r.n, ctx.grown)) |i| {
            const hh: u32 = if (i < ctx.grown) ctx.held[i] else 0;
            const hp: i32 = if (i < ctx.grown) ctx.parents[i] else -9;
            const ho: f64 = if (i < ctx.grown) ctx.own[i] else 0;
            const dh: u32 = if (i < r.n) r.held[i] else 0;
            const dp: i32 = if (i < r.n) r.parents[i] else -9;
            const do_: f64 = if (i < r.n) r.own[i] else 0;
            std.debug.print("[dtree]   {d}: host {d}/{d}/{x} device {d}/{d}/{x}\n", .{ i, hh, hp, @as(u64, @bitCast(ho)), dh, dp, @as(u64, @bitCast(do_)) });
        }
    }
    std.debug.print("[dtree] check rounds {d} mismatches {d}\n", .{ rounds, mismatches });
}
