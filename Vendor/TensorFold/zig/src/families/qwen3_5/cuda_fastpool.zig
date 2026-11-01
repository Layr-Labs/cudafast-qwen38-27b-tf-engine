//! The decode weight copies placed on the fastest free memory the driver hands out (TF_FASTPOOL=0 turns it off).
//!
//! On GB10 how fast a committed buffer streams depends on the physical memory behind it, and it stays that way: pieces
//! of 256 MiB committed one after another read anywhere from about 236 to 253 GB/s with the decode kernels' cp.async
//! slab pattern, and a piece reads at the same speed every time it is measured. Before the verify panels and the
//! projection streams are packed, `screen` commits candidate pieces out of the free memory, times reads of each,
//! keeps the fastest pieces that hold the copies and frees the others; the copies are then placed piece by piece
//! (`Pool.alloc`). The copies hold the same words wherever they sit, so no value any kernel computes changes.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const linux = std.os.linux;

/// A candidate's size; the largest copy (one projection's gate or up panel, 89 MB) fits in one.
pub const piece: usize = 96 << 20;
/// Candidates at most (about 112 GiB).
const most = 1200;
/// Free memory left beside the candidates for what the load allocates after the copies (the drafter, the sessions).
const keep_free: usize = 4 << 30;
/// Timed reads of each candidate (after one untimed read); a candidate ranks by its median.
const passes = 5;
const probe_ctas = 136;
const stage_bytes = 128 * 16;
/// Each probe CTA's slab: whole 2 KB stages.
const slab: u64 = piece / probe_ctas / stage_bytes * stage_bytes;
const alignment = 256;

pub const Pool = struct {
    d: *const cuda.Driver,
    ptr: [most]u64 = undefined,
    used: [most]usize = undefined,
    n: usize = 0,

    /// `bytes` out of the first kept piece with room (the pieces are kept fastest first), 256-byte aligned.
    pub fn alloc(p: *Pool, bytes: usize) ?u64 {
        const len = std.mem.alignForward(usize, bytes, alignment);
        for (p.ptr[0..p.n], p.used[0..p.n]) |base, *used| if (used.* + len <= piece) {
            const at = base + used.*;
            used.* += len;
            return at;
        };
        return null;
    }

    pub fn free(p: *Pool) void {
        for (p.ptr[0..p.n]) |base| _ = p.d.api.cuMemFree_v2(base);
        p.n = 0;
    }
};

/// The pieces `alloc` fills when asked for `sizes` in this order (more than `most` when something cannot be placed).
fn piecesFor(sizes: []const usize) usize {
    var used: [most]usize = undefined;
    var n: usize = 0;
    next: for (sizes) |s| {
        const len = std.mem.alignForward(usize, s, alignment);
        if (len > piece) return most + 1;
        for (used[0..n]) |*u| if (u.* + len <= piece) {
            u.* += len;
            continue :next;
        };
        if (n == most) return most + 1;
        used[n] = len;
        n += 1;
    }
    return n;
}

/// The fastest pieces that hold `sizes` (the copies, in the order they will be asked for), or null when the free memory
/// is unreadable or short or a driver call fails; the caller then allocates the copies as before. The candidates come
/// out of /proc/meminfo's MemFree less `keep_free`, are committed on `stream` in turn, read once untimed and then
/// `passes` times by `probe` (affine4_pack.cu's tf_fastpool_read), and every candidate not kept is freed here.
pub fn screen(d: *const cuda.Driver, stream: cuda.Stream, probe: cuda.Function, sizes: []const usize) ?Pool {
    if (comptime builtin.os.tag != .linux) return null;
    const timing = @import("sys").c.getenv("TF_TIMING") != null;
    const want = piecesFor(sizes);
    if (want == 0 or want > most) return null;
    var buf: [16384]u8 = undefined;
    const free_bytes = memFree(read("/proc/meminfo", &buf) orelse return null) orelse return null;
    if (free_bytes <= keep_free) return null;
    const count = @min((free_bytes - keep_free) / piece, most);
    if (count < want) return null;

    var cand: [most]u64 = @splat(0);
    var n: usize = 0;
    defer {
        for (cand[0..n]) |p| {
            if (p != 0) _ = d.api.cuMemFree_v2(p);
        }
    }
    while (n < count) {
        var p: cuda.abi.DevicePtr = 0;
        if (d.api.cuMemAlloc_v2(&p, piece) != cuda.abi.success) break;
        cand[n] = p;
        n += 1;
        // the driver commits on first touch: commit before timing
        if (d.api.cuMemsetD8Async(p, 0, piece, stream.handle) != cuda.abi.success) return null;
    }
    if (n < want) return null;

    var ev: [passes + 1]cuda.abi.Event = @splat(null);
    defer {
        for (ev) |x| {
            if (x != null) _ = d.api.cuEventDestroy_v2(x);
        }
    }
    for (&ev) |*x| {
        if (d.api.cuEventCreate(x, 0) != cuda.abi.success) return null;
    }
    var ms: [most]f32 = undefined;
    for (cand[0..n], ms[0..n]) |p, *m| {
        var args: cuda.Args = .{};
        args.add(p);
        args.add(slab);
        args.add(p); // the sink: written only if the xor of the words read hits a fixed constant
        const cfg: cuda.Config = .{ .grid = .{ .x = probe_ctas, .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } };
        cuda.launch.launch(probe, cfg, stream, &args) catch return null;
        for (0..passes) |i| {
            if (d.api.cuEventRecord(ev[i], stream.handle) != cuda.abi.success) return null;
            cuda.launch.launch(probe, cfg, stream, &args) catch return null;
        }
        if (d.api.cuEventRecord(ev[passes], stream.handle) != cuda.abi.success) return null;
        if (d.api.cuEventSynchronize(ev[passes]) != cuda.abi.success) return null;
        var t: [passes]f32 = undefined;
        for (&t, 0..) |*x, i| {
            if (d.api.cuEventElapsedTime(x, ev[i], ev[i + 1]) != cuda.abi.success) return null;
        }
        std.mem.sort(f32, &t, {}, std.sort.asc(f32));
        m.* = t[passes / 2];
    }

    var order: [most]usize = undefined;
    for (order[0..n], 0..) |*o, i| o.* = i;
    const times: *const [most]f32 = &ms;
    std.mem.sort(usize, order[0..n], times, struct {
        fn faster(t: *const [most]f32, a: usize, b: usize) bool {
            return t[a] < t[b];
        }
    }.faster);
    var pool: Pool = .{ .d = d };
    for (order[0..want]) |i| {
        pool.ptr[pool.n] = cand[i];
        pool.used[pool.n] = 0;
        pool.n += 1;
        cand[i] = 0; // kept: not freed by the defer
    }
    if (timing) {
        const gbs = struct {
            fn of(t: f32) f64 {
                return @as(f64, @floatFromInt(slab * probe_ctas)) / (@as(f64, t) * 1e6);
            }
        }.of;
        var kept: f64 = 0;
        for (order[0..want]) |i| kept += gbs(ms[i]);
        std.debug.print("info: fastpool: {d} pieces of {d} MiB screened (fastest {d:.1}, median {d:.1}, slowest {d:.1} GB/s); kept {d} at {d:.1} GB/s for {d} copies\n", .{ n, piece >> 20, gbs(ms[order[0]]), gbs(ms[order[n / 2]]), gbs(ms[order[n - 1]]), want, kept / @as(f64, @floatFromInt(want)), sizes.len });
    }
    std.log.scoped(.cuda).info("fastpool: the copies on the fastest {d} of {d} screened pieces", .{ want, n });
    return pool;
}

fn read(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = linux.read(fd, buf[n..].ptr, buf.len - n);
        if (linux.errno(got) != .SUCCESS) return null;
        if (got == 0) break;
        n += got;
    }
    return buf[0..n];
}

fn memFree(text: []const u8) ?usize {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "MemFree:")) {
        var it = std.mem.tokenizeAny(u8, line[8..], " \t");
        const kb = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
        return kb *| 1024;
    };
    return null;
}

test "fastpool: pieces counted the way alloc fills them" {
    const t = std.testing;
    const m: usize = 1 << 20;
    try t.expectEqual(@as(usize, 0), piecesFor(&.{}));
    try t.expectEqual(@as(usize, 3), piecesFor(&.{ 89 * m, 89 * m, 44 * m }));
    try t.expectEqual(@as(usize, 3), piecesFor(&.{ 89 * m, 89 * m, 89 * m }));
    try t.expectEqual(@as(usize, 2), piecesFor(&.{ 70 * m, 70 * m, 25 * m })); // the 25 goes back into the first
    try t.expectEqual(@as(usize, most + 1), piecesFor(&.{piece + 1}));
    var p: Pool = .{ .d = undefined };
    p.ptr[0] = 1 << 32;
    p.ptr[1] = 2 << 32;
    p.used[0] = 0;
    p.used[1] = 0;
    p.n = 2;
    try t.expectEqual(@as(?u64, 1 << 32), p.alloc(70 * m));
    try t.expectEqual(@as(?u64, 2 << 32), p.alloc(70 * m));
    try t.expectEqual(@as(?u64, (1 << 32) + 70 * m), p.alloc(25 * m));
    try t.expectEqual(@as(?u64, null), p.alloc(piece));
    try t.expectEqual(@as(?usize, 123 * 1024), memFree("MemTotal: 9 kB\nMemFree:   123 kB\nMemAvailable: 999 kB\n"));
}
