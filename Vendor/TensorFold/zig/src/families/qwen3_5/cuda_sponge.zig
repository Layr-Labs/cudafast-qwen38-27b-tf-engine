//! Free memory left in small blocks, committed before the weights so the weights commit in large ones.
//!
//! The driver backs a GPU allocation with the kernel's free pages, smallest blocks first, and keeps them until the
//! buffer is freed. Each engine run on a boot leaves many GiB of free memory in small blocks that cannot be compacted,
//! and a buffer the next engine commits there streams slower: on GB10, reads from 2 MiB blocks run about 237 GB/s,
//! from 32 MiB blocks about 258 GB/s. The decode GEMVs read the whole target every round at that rate. A buffer sized
//! from /proc/buddyinfo, committed first, takes the blocks the driver would hand out first; the weights then commit in
//! the blocks it leaves, and the buffer is freed once they are placed. Nothing reads it: where memory sits changes no
//! result.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const linux = std.os.linux;

/// Free blocks of at least `chunk` are what the buffer takes, all but `need` and `headroom` of them; smaller ones it
/// leaves. The buffer never takes more than `most`.
const chunk: usize = 2 << 20;
const headroom: usize = 16 << 30;
const most: usize = 96 << 30;
const least: usize = 1 << 30;
const granule: usize = 2 << 20;

pub const Sponge = struct {
    d: ?*const cuda.Driver = null,
    ptr: u64 = 0,
    len: usize = 0,

    pub fn free(s: *Sponge) void {
        if (s.ptr != 0) _ = s.d.?.api.cuMemFree_v2(s.ptr);
        s.* = .{};
    }
};

/// Commits a buffer over the free blocks of at least `chunk`, keeping `need` bytes and `headroom` of them; an empty
/// Sponge when the counts are unreadable, memory is short, or a driver call fails. Never fails the load.
/// The page size is no reason to decline: GB10's 4 KiB-page kernel has buddy orders up to 13, so blocks up to 32 MiB.
/// TF_SPONGE=<GiB> caps the buffer below `most`, TF_SPONGE=0 turns it off.
pub fn take(d: *const cuda.Driver, need: usize) Sponge {
    if (comptime builtin.os.tag != .linux) return .{};
    const page = std.heap.pageSize();
    const c = @import("sys").c;
    const cap: usize = if (c.getenv("TF_SPONGE")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch most >> 30) << 30 else most;
    if (cap == 0) return .{};
    var buf: [16384]u8 = undefined;
    const usable = usableFree(read("/proc/buddyinfo", &buf) orelse return .{}, page);
    const len = @min(size(usable, need), cap);
    if (c.getenv("TF_TIMING") != null)
        std.debug.print("info: sponge: usable {d} MiB, need {d} MiB, cap {d} MiB, taking {d} MiB\n", .{ usable >> 20, need >> 20, cap >> 20, len >> 20 });
    if (len == 0) return .{};
    var p: cuda.abi.DevicePtr = 0;
    if (d.api.cuMemAlloc_v2(&p, len) != cuda.abi.success) return .{};
    // the driver commits on first touch
    if (d.api.cuMemsetD8_v2(p, 0, len) != cuda.abi.success) {
        _ = d.api.cuMemFree_v2(p);
        return .{};
    }
    std.log.scoped(.cuda).info("sponge: {d} MiB of free blocks held while the weights commit", .{len >> 20});
    return .{ .d = d, .ptr = p, .len = len };
}

fn size(usable: usize, need: usize) usize {
    const keep = need +| headroom;
    if (usable <= keep) return 0;
    const len = @min(usable - keep, most) / granule * granule;
    return if (len < least) 0 else len;
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

/// Bytes free in blocks of at least `chunk`, every zone: /proc/buddyinfo's "Node 0, zone Normal c0 c1 ..." counts by order.
fn usableFree(text: []const u8, page: usize) usize {
    var total: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t,");
        var i: usize = 0;
        var order: u6 = 0;
        while (it.next()) |tok| : (i += 1) {
            if (i < 4) continue; // Node, its number, zone, its name
            const n = std.fmt.parseInt(usize, tok, 10) catch break;
            const block = std.math.shlExact(usize, page, order) catch break;
            if (block >= chunk) total +|= n *| block;
            order += 1;
            if (order == 63) break;
        }
    }
    return total;
}

test "sponge: 4 KiB pages, orders 0-8 are not counted, orders 9 and up are" {
    const info = "Node 0, zone   Normal      3      0      0      0      0      0      0      0      0      1      0      0      1      2\n";
    try std.testing.expectEqual(@as(usize, (2 << 20) + (16 << 20) + 2 * (32 << 20)), usableFree(info, 4096));
}

test "sponge: free blocks counted by order, sized within the blocks kept free" {
    const t = std.testing;
    const info =
        \\Node 0, zone      DMA      3      2      1      0      0      0      0      0      0      0      0      0      0      0
        \\Node 0, zone   Normal   1000    500    100     10      4      2      1      0      0      1      0      0      0     50
    ;
    // 64 KiB pages: orders 5 and up are 2 MiB and larger; orders 0-4 are not counted
    const k: usize = 64 << 10;
    try t.expectEqual(2 * 32 * k + 1 * 64 * k + 1 * 512 * k + 50 * 8192 * k, usableFree(info, k));
    const g: usize = 1 << 30;
    try t.expectEqual(@as(usize, 54 * g), size(100 * g, 30 * g));
    try t.expectEqual(@as(usize, 96 * g), size(200 * g, 30 * g)); // at most `most`
    try t.expectEqual(@as(usize, 6 * g), size(52 * g, 30 * g)); // the kept blocks first
    try t.expectEqual(@as(usize, 0), size(46 * g, 30 * g));
    try t.expectEqual(@as(usize, 0), size(46 * g + g / 2, 30 * g)); // below `least`
}
