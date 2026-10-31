//! A stream's state on CUDA: recurrences, conv tails, keys and values, the drafter's tap ring, the last window's log.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const recurrent = @import("cuda_recurrent.zig");

/// Rows a drafter's tap ring holds a stream: its window and a prompt chunk more.
pub const tap_held = 2048;
pub const tap_rows = tap_held + engine.most_verify;

/// qwen35.cuh's Seg and Path.
pub const Seg = extern struct { rec: u64, conv: u64, kv: u64, tap: u64, log: u64, before: u64, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int, taps: c_int, delta: c_int, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0, pn: c_int = 0, plog: u64 = 0 };
pub const Path = extern struct { n: c_int, rows: [engine.most_window]c_int };

/// A cache's keys and values as pages (qwen35.cuh kv_row): page i of the stream's positions at `base` + ids[i] pages.
pub const Paged = struct {
    /// Where page 0 of the arena starts: the core pool's, or the cache's own buffer under TF_QWEN_KV_PAGE_TEST.
    base: u64,
    /// The stream's page table on the device, an int a page, `ids` uploaded.
    table: DeviceBuffer,
    ids: []u32,
    /// Positions a page holds.
    tokens: usize,
    /// The cache owns `ids` (TF_QWEN_KV_PAGE_TEST); a pool's lane maps its lease's.
    own_ids: bool = true,

    /// Elements a page holds (bf16, or bytes for fp8 rows and their scales): every attention layer's of `tokens` rows.
    pub fn elems(p: Paged, g: geometry.Geometry, kv8: bool) usize {
        return if (kv8) g.kvBytes(p.tokens, true) else g.kvBytes(p.tokens, false) / 2;
    }
};

/// TF_QWEN_KV_PAGE_TEST=n: each cache's own buffer read through a table of n-position pages in reverse order (checks).
/// A device block of `len` bytes: a pooled block of exactly that size (from a closed stream), or a new allocation.
/// The caller zeroes or overwrites it before any read, as it does a new one.
fn take(e: *Engine, len: usize) !DeviceBuffer {
    pool_of = e;
    for (e.blocks.items, 0..) |b, i| if (b.len == len) return e.blocks.swapRemove(i);
    return DeviceBuffer.alloc(e.d, len);
}

/// The engine whose pool `give` fills: the one that allocated through `take` (one engine a process).
var pool_of: ?*Engine = null;

/// `take` for the engine's other per-phase device buffers (recurrent scratch, drafter rings): a pooled block of exactly
/// `len` bytes or a new one, which the caller zeroes or overwrites before any read.
pub fn takeBlock(e: *Engine, len: usize) !DeviceBuffer {
    return take(e, len);
}

/// `give` for those buffers: kept for the next phase's same-size buffer, or freed.
pub fn giveBlock(b: *DeviceBuffer) void {
    give(b);
}

/// Stop recycling into `e`'s pool (its deinit).
pub fn forget(e: *Engine) void {
    if (pool_of == e) pool_of = null;
}

/// `b` kept for the next stream (whole driver allocations of at least 1 MiB, up to 32), otherwise freed.
fn give(b: *DeviceBuffer) void {
    const e = pool_of orelse return b.free();
    if (!b.borrowed and b.owner == null and b.len >= 1 << 20 and e.blocks.items.len < 32) {
        if (e.blocks.append(e.gpa, b.*)) |_| {
            b.* = undefined;
            return;
        } else |_| {}
    }
    b.free();
}

pub fn pageTest() !usize {
    const v = std.c.getenv("TF_QWEN_KV_PAGE_TEST") orelse return 0;
    return parsePageTest(std.mem.span(v));
}

pub fn parsePageTest(v: []const u8) !usize {
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadQwenPageTest;
    if (!validPageTokens(n)) return error.BadQwenPageTest;
    return n;
}

pub fn validPageTokens(n: usize) bool {
    return n > 0 and n <= std.math.maxInt(c_int);
}

/// Logical capacity rounded to whole pages, still representable by Seg.cap.
pub fn cacheRows(cap: usize, tokens: usize) !usize {
    if (cap > std.math.maxInt(c_int) or (tokens != 0 and !validPageTokens(tokens))) return error.BadQwenPageTest;
    if (tokens == 0) return cap;
    const rounded = try std.math.add(usize, cap, tokens - 1);
    const rows = rounded / tokens * tokens;
    if (rows > std.math.maxInt(c_int)) return error.BadQwenPageTest;
    return rows;
}

/// Diagnostic paging adds padded cache rows and a device page table to a contiguous cache's budget.
pub fn pageExtra(g: geometry.Geometry, cap: usize, tokens: usize, kv8: bool) !usize {
    if (tokens == 0) return 0;
    const rows = try cacheRows(cap, tokens);
    const padding = try std.math.mul(usize, rows - cap, g.kvBytes(1, kv8));
    return std.math.add(usize, padding, try std.math.mul(usize, rows / tokens, 4));
}

pub const Cache = struct {
    owner: ?*recurrent.Pool = null,
    log_lease: ?recurrent.Lease = null,
    before_lease: ?recurrent.Lease = null,
    /// Bytes this cache adds to its owner's cap while it lives (recurrent.bound).
    scratch: usize = 0,
    untaught: bool = false,
    /// Keys and values through a page table (null: cap contiguous rows a head).
    paged: ?Paged = null,
    /// A pool lane's drafter ring in its run (the drafter borrows it for the lane's stream).
    drafter_ring: ?u64 = null,
    delta: i32 = 0,
    state: DeviceBuffer,
    /// The exact state before a recorded one-row round advances its stable slot in place.
    before: ?DeviceBuffer = null,
    kv: DeviceBuffer,
    /// A drafter's tapped rows (`Engine.taps`), a ring of tap_rows: the row at position p in slot p % tap_rows.
    tap: ?DeviceBuffer = null,
    /// The newest verify window's records (qwen35.cuh's LogRow), held until its keep completes.
    log: DeviceBuffer,
    cap: usize,
    len: usize = 0,
    /// Positions the newest forward wrote up to: a tap ring holds them until the next forward, whatever `keep` lands.
    written: usize = 0,
    logits: []u16,
    /// The newest verify window, until its path is kept: where it started, its rows.
    last: ?struct { start: usize, rows: usize } = null,
    /// A kept prefix of the newest window (rows 0..pend-1) whose recurrence replay the next recorded slot scan folds.
    pend: c_int = 0,
    /// The log half that window wrote (the next window writes the other half of `log`).
    pend_log: u64 = 0,
    /// Which half of `log` the next window writes.
    flip: bool = false,
    /// An image prompt's rotary shift (get_rope_index's rope_delta): rows after its prompt rotate at position + delta.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, cap: usize) !Cache {
        return initFor(gpa, e, cap, true);
    }

    /// `init`, with the drafter's tap ring only when `taps` (a stream that drafts; the forwards skip a missing ring).
    pub fn initFor(gpa: std.mem.Allocator, e: *Engine, cap: usize, taps: bool) !Cache {
        const test_pages = try pageTest();
        const rows = try cacheRows(cap, test_pages);
        var kv = try take(e, try std.math.mul(usize, e.g.kvBytes(1, e.kv8), rows));
        var own = true;
        errdefer if (own) kv.free();
        var paged: ?Paged = null;
        errdefer if (own) {
            if (paged) |*p| freePaged(gpa, p);
        };
        if (test_pages > 0) {
            const n = rows / test_pages;
            const ids = try gpa.alloc(u32, n);
            errdefer gpa.free(ids);
            for (ids, 0..) |*id, i| id.* = @intCast(n - 1 - i); // a table no contiguous layout matches
            var table = try DeviceBuffer.alloc(e.d, n * 4);
            errdefer table.free();
            paged = .{ .base = kv.ptr, .table = table, .ids = ids, .tokens = test_pages };
        }
        var out = try fixed(gpa, e, cap, kv, paged, taps);
        own = false;
        errdefer out.deinit(gpa);
        errdefer cuda.completion.drain(e.d, e.stream, "qwen3_5: failed cache initialization did not drain");
        try kv.fill8(0, e.stream.handle);
        if (paged) |p| try p.table.uploadAsync(0, std.mem.sliceAsBytes(p.ids), e.stream.handle);
        try e.stream.synchronize();
        return out;
    }

    /// A pool lane's cache: a table of `pages` pages at `base`; its state, log and tap ring a run mapped at open.
    pub fn initPaged(gpa: std.mem.Allocator, e: *Engine, pages: usize, tokens: usize, base: u64) !Cache {
        var table = try DeviceBuffer.alloc(e.d, pages * 4);
        errdefer table.free();
        const none = try DeviceBuffer.alloc(e.d, 0);
        return .{ .state = none, .before = none, .kv = none, .paged = .{ .base = base, .table = table, .ids = &.{}, .tokens = tokens, .own_ids = false }, .log = none, .cap = 0, .logits = try gpa.alloc(u16, e.g.vocab) };
    }

    /// Bytes of a pool lane's run: state, before, log, tap ring and a drafter's ring (`ring`), each 256-aligned.
    pub fn runBytes(e: *const Engine, ring: usize) usize {
        var n: usize = 0;
        for (runParts(e, ring)) |len| n += std.mem.alignForward(usize, len, 256);
        return n;
    }

    fn runParts(e: *const Engine, ring: usize) [5]usize {
        return .{ e.g.stateBytes(), e.g.stateBytes(), 2 * e.g.logBytes(engine.most_window), e.tapBytes(), ring };
    }

    /// A lane's state, before, log, taps and drafter ring as views of the run at `base`; a short run is refused.
    pub fn mapRun(self: *Cache, e: *const Engine, base: u64, len: usize, ring: usize) !void {
        if (runBytes(e, ring) > len) return error.QwenRunTooShort;
        var at = base;
        var views: [5]DeviceBuffer = undefined;
        for (runParts(e, ring), &views) |n, *v| {
            v.* = .{ .d = e.d, .ptr = at, .len = n, .borrowed = true };
            at += std.mem.alignForward(usize, n, 256);
        }
        self.state = views[0];
        self.pend = 0;
        self.before = views[1];
        self.log = views[2];
        self.tap = if (e.n_taps > 0) views[3] else null;
        self.drafter_ring = if (ring > 0) views[4].ptr else null;
        for ([_]DeviceBuffer{ views[2], views[3], views[4] }) |v| if (v.len > 0) try v.fill8(0, e.stream.handle);
    }

    /// A closed lane's views dropped: its run went back to the pool.
    pub fn unmapRun(self: *Cache, e: *const Engine) void {
        const none: DeviceBuffer = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
        self.state = none;
        self.pend = 0;
        self.before = none;
        self.log = none;
        self.tap = null;
        self.drafter_ring = null;
    }

    /// The parts every cache holds itself beside `kv` and `paged` (the caller's to free on failure).
    fn fixed(gpa: std.mem.Allocator, e: *Engine, cap: usize, kv: DeviceBuffer, paged: ?Paged, taps: bool) !Cache {
        var state = try take(e, e.g.stateBytes());
        errdefer state.free();
        const adaptive = recurrent.enabled(e);
        var before: ?DeviceBuffer = null;
        errdefer if (before) |*b| b.free();
        if (!adaptive) before = try take(e, e.g.stateBytes());
        var log: DeviceBuffer = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
        if (!adaptive) log = try take(e, 2 * e.g.logBytes(engine.most_window));
        errdefer log.free();
        var tap: ?DeviceBuffer = null;
        errdefer if (tap) |*t| t.free();
        if (e.n_taps > 0 and taps) {
            tap = try take(e, e.tapBytes());
        }
        const logits = try gpa.alloc(u16, e.g.vocab);
        errdefer gpa.free(logits);
        errdefer cuda.completion.drain(e.d, e.stream, "qwen3_5: failed cache state initialization did not drain");
        try state.fill8(0, e.stream.handle);
        if (!adaptive) try log.fill8(0, e.stream.handle);
        if (tap) |t| try t.fill8(0, e.stream.handle);
        try e.stream.synchronize();
        const scratch = if (adaptive) recurrent.bound(e) else 0;
        e.rec_scratch.cap += scratch;
        return .{ .owner = if (adaptive) &e.rec_scratch else null, .scratch = scratch, .state = state, .before = before, .kv = kv, .paged = paged, .tap = tap, .log = log, .cap = cap, .logits = logits };
    }

    /// Publish a completed forward's metadata, including one-row slots that already advanced in place.
    pub fn advanced(self: *Cache, rows: usize, record: bool) void {
        self.last = if (record) .{ .start = self.len, .rows = rows } else null;
        self.len += rows;
        self.written = self.len;
    }

    /// Ready for another stream: no rows, the recurrences and tails zero (on the engine's stream, before its forwards).
    pub fn reset(self: *Cache, e: *const Engine) !void {
        recurrent.release(self);
        self.len = 0;
        self.pend = 0;
        self.written = 0;
        self.last = null;
        self.untaught = false;
        self.delta = 0;
        try self.state.fill8(0, e.stream.handle);
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        recurrent.release(self);
        if (self.owner) |p| p.cap -= self.scratch;
        give(&self.state);
        if (self.before) |*b| give(b);
        if (self.paged) |*p| {
            self.kv.free();
            freePaged(gpa, p);
        } else give(&self.kv);
        give(&self.log);
        if (self.tap) |*t| give(t);
        gpa.free(self.logits);
    }


    /// The stream's window of `rows` rows from forward row `row0`, after `len` cached positions.
    pub fn seg(self: *const Cache, e: *const Engine, len: usize, row0: usize, rows: usize) Seg {
        const g = e.g;
        var sg: Seg = .{ .rec = self.state.ptr, .conv = self.state.ptr + g.deltas() * g.recBytes(), .kv = self.kv.ptr, .tap = if (self.tap) |t| t.ptr else 0, .log = self.log.ptr + if (self.flip) self.log.len / 2 else 0, .before = if (self.before) |b| b.ptr else 0, .cap = @intCast(self.cap), .len = @intCast(len), .row0 = @intCast(row0), .rows = @intCast(rows), .tap_rows = tap_rows, .taps = @intCast(e.n_taps), .delta = self.delta };
        if (self.paged) |p| {
            sg.kv = p.base;
            sg.pt = p.table.ptr;
            sg.ps = p.elems(g, e.kv8);
            sg.ptok = @intCast(p.tokens);
        }
        return sg;
    }

    /// Keeps `path` (window rows, ascending from 0): the state as that chain would leave it, a tree's keys moved down.
    pub fn keep(self: *Cache, e: *Engine, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        try self.flush(e);
        if (path.len == 0) {
            if (last.rows == 1) {
                try recurrent.mark(self, true);
                try self.state.copyFrom(0, (self.before orelse return error.MissingQwenBefore).ptr, e.g.stateBytes(), e.stream.handle);
            }
            self.last = null;
            self.len = last.start;
            return;
        }
        if (path.len > last.rows or path[0] != 0) return error.NothingToKeep;
        var p: Path = .{ .n = @intCast(path.len), .rows = @splat(0) };
        var prefix = true;
        for (path, 0..) |r, i| {
            if (r >= last.rows or (i > 0 and r <= path[i - 1])) return error.NothingToKeep;
            p.rows[i] = @intCast(r);
            prefix = prefix and r == i;
        }
        if (last.rows == 1) {
            self.last = null;
            self.len = last.start + 1;
            return;
        }
        const g = e.g;
        const sg = self.seg(e, last.start, 0, last.rows);
        try recurrent.mark(self, false);
        if (!prefix) try e.run(e.keeps.apply, .{ 1, geometry.dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, p });
        try e.run(e.keeps.conv, .{ g.deltas(), (g.convBytes() / 2 + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ e.geo, sg, @as(c_int, @intCast(path[path.len - 1])) });
        if (!prefix and g.attentions() > 0) try e.run(if (e.kv8) e.fp8.compact else e.keeps.compact, .{ g.attentions(), g.kv_heads, 1 }, .{ geometry.head_dim, 1, 1 }, .{ e.geo, sg, p });
        if (prefix) {
            // a chain's prefix: the next recorded slot scan replays it from this log half while it loads the state
            self.pend = p.n;
            self.pend_log = sg.log;
            self.flip = !self.flip;
        }
        self.last = null;
        self.len = last.start + path.len;
    }

    /// A pending kept prefix replayed into the recurrences now (before anything but a recorded slot scan reads them).
    pub fn flush(self: *Cache, e: *Engine) !void {
        if (self.pend == 0) return;
        var p: Path = .{ .n = self.pend, .rows = @splat(0) };
        const n: usize = @intCast(self.pend);
        for (p.rows[0..n], 0..) |*r, i| r.* = @intCast(i);
        var sg = self.seg(e, self.len, 0, 0);
        sg.log = self.pend_log;
        self.pend = 0;
        const g = e.g;
        try recurrent.mark(self, false);
        try e.run(e.keeps.apply, .{ 1, geometry.dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, p });
    }

    /// The pending prefix handed to this forward's slot scan (`segs`' entry for this cache).
    pub fn fold(self: *Cache, seg_: Seg) Seg {
        var sg = seg_;
        sg.pn = self.pend;
        sg.plog = self.pend_log;
        self.pend = 0;
        return sg;
    }

    /// The newest window kept whole (the round loop keeps nothing when every row landed), before the next forward.
    pub fn settle(self: *Cache, e: *Engine) !void {
        const last = self.last orelse return;
        var all: [engine.most_window]u32 = undefined;
        for (all[0..last.rows], 0..) |*r, i| r.* = @intCast(i);
        try self.keep(e, all[0..last.rows]);
    }
};

fn freePaged(gpa: std.mem.Allocator, p: *Paged) void {
    p.table.free();
    if (p.own_ids) gpa.free(p.ids);
}

test "KV page dimensions: diagnostic rows fit the signed kernel argument" {
    try std.testing.expectEqual(@as(usize, 48), try parsePageTest("48"));
    try std.testing.expectError(error.BadQwenPageTest, parsePageTest("0"));
    try std.testing.expectError(error.BadQwenPageTest, parsePageTest("2147483648"));
    try std.testing.expectError(error.BadQwenPageTest, parsePageTest("18446744073709551615"));
}

test "KV page dimensions: diagnostic admission includes its padded rows and device table" {
    const g: geometry.Geometry = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 64, .linear = @splat(false) };
    try std.testing.expectEqual(@as(usize, 96), try cacheRows(65, 48));
    for ([_]bool{ false, true }) |fp8| try std.testing.expectEqual(31 * g.kvBytes(1, fp8) + 8, try pageExtra(g, 65, 48, fp8));
    try std.testing.expectError(error.BadQwenPageTest, cacheRows(2147483647, 48));
    try std.testing.expectError(error.BadQwenPageTest, cacheRows(std.math.maxInt(usize), 48));
}
