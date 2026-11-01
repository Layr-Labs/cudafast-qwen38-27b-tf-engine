//! Free memory left in small blocks, committed before the weights so the weights commit in the largest blocks.
//!
//! The driver backs a GPU allocation with the kernel's free pages in 64 KiB pieces, taking the smallest free blocks
//! first, and keeps them until the buffer is freed. Each engine run on a boot leaves many GiB of free memory in small
//! blocks, and a buffer committed there streams slower than one committed in large blocks (on GB10 a paired read of two
//! buffers in one process: 64-128 KiB blocks against 2-4 MiB blocks). The decode GEMVs read the whole target every round.
//!
//! From /proc/buddyinfo, the order K is the largest whose free blocks of order K and above still hold everything the
//! load commits while the sponge is held (the arena and the decode views copied from it) with room to spare. A sponge
//! the size of every free block from 64 KiB up to order K-1, committed first, takes exactly those; the buffers then
//! commit in blocks of order K or larger, and the sponge is freed once they are placed. Nothing reads it: where memory
//! sits changes no result.
//!
//! The room matters more than the order. When no free 64 KiB block is left for a piece, the driver commits the whole
//! buffer in 4 KiB pages, which GB10 reads slower (and its random reads at a quarter of the rate). A sponge that leaves
//! the buffers less free memory than they need forces exactly that, so whatever the sponge takes, the blocks of order K
//! and above keep the buffers and an eighth more free.
//!
//! When blocks of 2 MiB and above are fewer than that, a region marked MADV_HUGEPAGE is faulted in first and unmapped
//! (madvise's defrag compacts free memory into 2 MiB pages for it), sized within free memory and the cgroup's limit.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const linux = std.os.linux;

/// The driver's commit piece: free blocks smaller than this are never used for a buffer and are not counted.
const piece: usize = 64 << 10;
/// Memory left available beyond the sponge and the buffers; the sponge never takes more than `most`.
const margin: usize = 6 << 30;
const most: usize = 80 << 30;
const least: usize = 1 << 30;
const granule: usize = 2 << 20;
const max_orders = 24;
/// The transparent huge page: blocks of this size and above are what compaction makes.
const huge: usize = 2 << 20;
/// Free memory and cgroup headroom the compaction region leaves untouched.
const spare: usize = 4 << 30;
const madv_hugepage: u32 = 14;
const madv_populate_write: u32 = 23;

pub const Sponge = struct {
    d: ?*const cuda.Driver = null,
    ptr: u64 = 0,
    len: usize = 0,

    pub fn free(s: *Sponge) void {
        if (s.ptr != 0) {
            if (@import("sys").c.getenv("TF_TIMING") != null) show("sponge: before it is freed");
            _ = s.d.?.api.cuMemFree_v2(s.ptr);
        }
        s.* = .{};
    }
};

/// Commits a sponge over the free blocks below the order the `need` bytes committed while it is held can be placed in,
/// keeping `margin` available; an empty Sponge when the counts are unreadable, nothing can be steered, memory is short,
/// or a driver call fails. Never fails the load. TF_SPONGE=<GiB> caps the sponge below `most`, TF_SPONGE=0 turns it
/// off; TF_COMPACT=0 skips the compaction region.
pub fn take(d: *const cuda.Driver, need: usize) Sponge {
    if (comptime builtin.os.tag != .linux) return .{};
    const page = std.heap.pageSize();
    const c = @import("sys").c;
    const cap: usize = if (c.getenv("TF_SPONGE")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch most >> 30) << 30 else most;
    if (cap == 0) return .{};
    const timing = c.getenv("TF_TIMING") != null;
    var buf: [16384]u8 = undefined;
    var bytes: [max_orders]usize = @splat(0);
    var n = orderBytes(read("/proc/buddyinfo", &buf) orelse return .{}, page, &bytes);
    var mem = memInfo(read("/proc/meminfo", &buf) orelse return .{}) orelse return .{};
    const want = need +| need / 8 +| least;
    const big0 = from(bytes[0..n], page, huge);
    var region: Region = .{};
    if (timing) show("sponge: at start");
    if (big0 < want and !off(c.getenv("TF_COMPACT"))) {
        region = compact(@min(want, room(mem.free, cgroupRoom(&buf))));
        if (region.len > 0) {
            bytes = @splat(0);
            n = orderBytes(read("/proc/buddyinfo", &buf) orelse return .{}, page, &bytes);
            mem = memInfo(read("/proc/meminfo", &buf) orelse return .{}) orelse return .{};
            if (timing) show("sponge: after the compaction region");
        }
    }
    const plan = steer(bytes[0..n], page, need);
    const len = @min(fit(plan.small, mem.avail, need), cap);
    std.log.scoped(.cuda).info("placement: need {d} MiB; free {d} MiB, in blocks >= 2 MiB {d} -> {d} MiB (compaction region {d} MiB, {d} MiB huge); order {d} ({d} KiB blocks) and above hold {d} MiB, below it {d} MiB; sponge {d} MiB", .{
        need >> 20,                                      mem.free >> 20,       big0 >> 20,  from(bytes[0..n], page, huge) >> 20, region.len >> 20, region.huge >> 20, plan.order,
        if (plan.order > 0) (page << @intCast(plan.order)) >> 10 else 0, plan.above >> 20, plan.small >> 20, len >> 20,
    });
    if (len == 0) return .{};
    var p: cuda.abi.DevicePtr = 0;
    if (d.api.cuMemAlloc_v2(&p, len) != cuda.abi.success) return .{};
    // the driver commits on first touch
    if (d.api.cuMemsetD8_v2(p, 0, len) != cuda.abi.success) {
        _ = d.api.cuMemFree_v2(p);
        return .{};
    }
    if (timing) show("sponge: committed");
    return .{ .d = d, .ptr = p, .len = len };
}

const Plan = struct { order: usize = 0, above: usize = 0, small: usize = 0 };

/// The largest order whose free blocks, with every larger one, hold `need` and an eighth more; `small` is the free
/// memory in blocks from the commit piece up to that order. A zero plan when no order above the piece's holds it.
fn steer(bytes: []const usize, page: usize, need: usize) Plan {
    var lo: usize = 0;
    while (lo < bytes.len and (page << @intCast(lo)) < piece) lo += 1;
    const want = need +| need / 8 +| least;
    var above: usize = 0;
    var k = bytes.len;
    while (k > lo + 1) {
        k -= 1;
        above +|= bytes[k];
        if (above >= want) {
            var small: usize = 0;
            for (bytes[lo..k]) |b| small +|= b;
            return .{ .order = k, .above = above, .small = small };
        }
    }
    return .{};
}

fn fit(small: usize, avail: usize, need: usize) usize {
    const keep = need +| margin;
    if (avail <= keep) return 0;
    const len = @min(small, avail - keep, most) / granule * granule;
    return if (len < least) 0 else len;
}

/// Free bytes in blocks of at least `size`.
fn from(bytes: []const usize, page: usize, size: usize) usize {
    var t: usize = 0;
    for (bytes, 0..) |b, i| {
        if ((page << @intCast(i)) >= size) t +|= b;
    }
    return t;
}

/// What a compaction region may fault in: free memory and cgroup headroom less `spare` each, at most `most`.
fn room(free: usize, cgroup: usize) usize {
    const r = @min(free -| spare, cgroup -| spare, most) / huge * huge;
    return if (r < least) 0 else r;
}

const Region = struct { len: usize = 0, huge: usize = 0 };

/// Faults in `len` bytes of anonymous memory marked MADV_HUGEPAGE and unmaps them: each fault that gets a 2 MiB page
/// compacts free memory for it, and the pages return to the free lists as blocks of 2 MiB and above. A zero Region when
/// `len` is under `least` or a call fails (MADV_POPULATE_WRITE needs Linux 5.14).
fn compact(len: usize) Region {
    if (len < least) return .{};
    const span = len + huge;
    const addr = linux.mmap(null, span, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (linux.errno(addr) != .SUCCESS) return .{};
    const base: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(addr);
    defer _ = linux.munmap(base, span);
    const at: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(std.mem.alignForward(usize, addr, huge));
    if (linux.errno(linux.madvise(at, len, madv_hugepage)) != .SUCCESS) return .{};
    if (linux.errno(linux.madvise(at, len, madv_populate_write)) != .SUCCESS) return .{};
    var buf: [4096]u8 = undefined;
    return .{ .len = len, .huge = anonHuge(read("/proc/self/smaps_rollup", &buf) orelse "") };
}

/// The cgroup v2 memory headroom (memory.max less memory.current); no limit when unreadable or "max".
fn cgroupRoom(buf: []u8) usize {
    const limit = std.mem.trim(u8, read("/sys/fs/cgroup/memory.max", buf) orelse return std.math.maxInt(usize), " \n");
    const max = std.fmt.parseInt(usize, limit, 10) catch return std.math.maxInt(usize);
    const used = std.mem.trim(u8, read("/sys/fs/cgroup/memory.current", buf) orelse return 0, " \n");
    return max -| (std.fmt.parseInt(usize, used, 10) catch return 0);
}

fn off(v: ?[*:0]const u8) bool {
    return if (v) |s| std.mem.eql(u8, std.mem.span(s), "0") else false;
}

/// Free MiB by order and MemFree, for TF_TIMING.
fn show(tag: []const u8) void {
    const page = std.heap.pageSize();
    var buf: [16384]u8 = undefined;
    var bytes: [max_orders]usize = @splat(0);
    const n = orderBytes(read("/proc/buddyinfo", &buf) orelse return, page, &bytes);
    const mem = memInfo(read("/proc/meminfo", &buf) orelse return) orelse return;
    var line: [512]u8 = undefined;
    var at: usize = 0;
    for (bytes[0..n], 0..) |b, i| at += (std.fmt.bufPrint(line[at..], " {d}:{d}", .{ i, b >> 20 }) catch break).len;
    std.debug.print("info: {s}: free {d} MiB, available {d} MiB; free MiB by order{s}\n", .{ tag, mem.free >> 20, mem.avail >> 20, line[0..at] });
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

/// Free bytes per order summed over every zone: /proc/buddyinfo's "Node 0, zone Normal c0 c1 ..." counts by order.
/// Returns the number of orders seen.
fn orderBytes(text: []const u8, page: usize, out: *[max_orders]usize) usize {
    var orders: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t,");
        var i: usize = 0;
        var order: usize = 0;
        while (it.next()) |tok| : (i += 1) {
            if (i < 4) continue; // Node, its number, zone, its name
            const cnt = std.fmt.parseInt(usize, tok, 10) catch break;
            if (order == max_orders) break;
            const block = std.math.shlExact(usize, page, @intCast(order)) catch break;
            out[order] +|= cnt *| block;
            order += 1;
        }
        orders = @max(orders, order);
    }
    return orders;
}

const Mem = struct { free: usize, avail: usize };

fn memInfo(text: []const u8) ?Mem {
    return .{ .free = field(text, "MemFree:") orelse return null, .avail = field(text, "MemAvailable:") orelse return null };
}

fn field(text: []const u8, key: []const u8) ?usize {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, key)) {
        var it = std.mem.tokenizeAny(u8, line[key.len..], " \t");
        const kb = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
        return kb *| 1024;
    };
    return null;
}

/// AnonHugePages from /proc/self/smaps_rollup, in bytes; 0 when absent.
fn anonHuge(text: []const u8) usize {
    return field(text, "AnonHugePages:") orelse 0;
}

test "sponge: free bytes by order over every zone" {
    const info =
        \\Node 0, zone      DMA      1      0      0      0      0      0      0      0      0      0      0      0      0      1
        \\Node 0, zone   Normal      3      0      0      0      0      0      0      0      0      1      0      0      1      2
    ;
    var b: [max_orders]usize = @splat(0);
    try std.testing.expectEqual(@as(usize, 14), orderBytes(info, 4096, &b));
    try std.testing.expectEqual(@as(usize, 4 * 4096), b[0]);
    try std.testing.expectEqual(@as(usize, 2 << 20), b[9]);
    try std.testing.expectEqual(@as(usize, 16 << 20), b[12]);
    try std.testing.expectEqual(@as(usize, 3 * (32 << 20)), b[13]);
    try std.testing.expectEqual(@as(usize, (2 << 20) + (16 << 20) + 3 * (32 << 20)), from(b[0..14], 4096, huge));
    const m = memInfo("MemTotal:  999 kB\nMemFree:  777 kB\nMemAvailable:   123456 kB\n").?;
    try std.testing.expectEqual(@as(usize, 123456 * 1024), m.avail);
    try std.testing.expectEqual(@as(usize, 777 * 1024), m.free);
    try std.testing.expectEqual(@as(?Mem, null), memInfo("MemTotal: 1 kB\n"));
    try std.testing.expectEqual(@as(usize, 2048 * 1024), anonHuge("Rss: 5 kB\nAnonHugePages:      2048 kB\n"));
}

test "sponge: the buffer steered to the largest order that holds it" {
    const t = std.testing;
    const g: usize = 1 << 30;
    // 4 KiB pages: orders 0-3 are below the 64 KiB piece and never counted
    var b: [14]usize = @splat(0);
    b[2] = 5 * g;
    b[4] = 10 * g;
    b[8] = 20 * g;
    b[9] = 8 * g;
    b[10] = 6 * g;
    b[12] = 2 * g;
    // need 12 GiB: want 14.5; orders >= 10 hold 8, >= 9 hold 16: order 9, sponge over orders 4..8 = 30 GiB
    const p = steer(&b, 4096, 12 * g);
    try t.expectEqual(@as(usize, 9), p.order);
    try t.expectEqual(@as(usize, 16 * g), p.above);
    try t.expectEqual(@as(usize, 30 * g), p.small);
    // nothing above the piece holds 100 GiB
    try t.expectEqual(@as(usize, 0), steer(&b, 4096, 100 * g).small);
    try t.expectEqual(@as(usize, 30 * g), fit(30 * g, 100 * g, 12 * g));
    try t.expectEqual(@as(usize, 80 * g), fit(90 * g, 200 * g, 12 * g)); // at most `most`
    try t.expectEqual(@as(usize, 2 * g), fit(30 * g, 20 * g, 12 * g)); // the kept memory first
    try t.expectEqual(@as(usize, 0), fit(30 * g, 18 * g, 12 * g));
    // the compaction region: free and cgroup headroom less 4 GiB each
    try t.expectEqual(@as(usize, 36 * g), room(40 * g, std.math.maxInt(usize)));
    try t.expectEqual(@as(usize, 20 * g), room(100 * g, 24 * g));
    try t.expectEqual(@as(usize, 0), room(4 * g, 64 * g));
}
