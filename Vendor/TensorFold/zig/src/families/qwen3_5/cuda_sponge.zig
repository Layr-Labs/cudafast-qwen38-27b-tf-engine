//! Content-free setup buffer held while weights commit, then freed before any measured phase.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const linux = std.os.linux;

const chunk: usize = 2 << 20;
const below: usize = 32 << 20;
const headroom: usize = 16 << 30;
const most: usize = 60 << 30;
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

/// Reserves blocks at least 2 MiB and below 32 MiB, leaving `need` and `headroom` in blocks of at least 2 MiB.
/// File read failures, insufficient memory and driver failures leave the normal load unchanged.
pub fn take(d: *const cuda.Driver, need: usize) Sponge {
    if (comptime builtin.os.tag != .linux) return .{};
    const page = std.heap.pageSize();
    var buf: [16384]u8 = undefined;
    const free = freeBlocks(read("/proc/buddyinfo", &buf) orelse return .{}, page);
    const len = size(free.small, free.usable, need);
    if (len == 0) return .{};
    var p: cuda.abi.DevicePtr = 0;
    if (d.api.cuMemAlloc_v2(&p, len) != cuda.abi.success) return .{};
    // the driver commits on first touch
    if (d.api.cuMemsetD8_v2(p, 0, len) != cuda.abi.success) {
        _ = d.api.cuMemFree_v2(p);
        return .{};
    }
    std.log.scoped(.cuda).info("sponge: {d} MiB of small free blocks held while the weights commit", .{len >> 20});
    return .{ .d = d, .ptr = p, .len = len };
}

fn size(small: usize, usable: usize, need: usize) usize {
    const keep = need +| headroom;
    if (usable <= keep) return 0;
    const len = @min(small, usable - keep, most) / granule * granule;
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

const Free = struct { small: usize = 0, usable: usize = 0 };

/// Counts every zone's buddy orders: `small` has blocks below 32 MiB; both totals exclude blocks below 2 MiB.
fn freeBlocks(text: []const u8, page: usize) Free {
    var free: Free = .{};
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeAny(u8, line, " \t,");
        var i: usize = 0;
        var order: u6 = 0;
        while (it.next()) |tok| : (i += 1) {
            if (i < 4) continue;
            const n = std.fmt.parseInt(usize, tok, 10) catch break;
            const block = std.math.shlExact(usize, page, order) catch break;
            if (block >= chunk) {
                free.usable +|= n *| block;
                if (block < below) free.small +|= n *| block;
            }
            order += 1;
            if (order == 63) break;
        }
    }
    return free;
}

test "sponge: 64 KiB buddy counts and bounded placement reserve" {
    const t = std.testing;
    const info =
        \\Node 0, zone DMA 3 2 1 0 0 0 0 0 0 0 0 0 0 0
        \\Node 0, zone Normal 1000 500 100 10 4 2 1 0 0 1 0 0 0 50
    ;
    const k: usize = 64 << 10;
    const free = freeBlocks(info, k);
    try t.expectEqual(@as(usize, 2 * 32 * k + 64 * k), free.small);
    try t.expectEqual(@as(usize, 2 * 32 * k + 64 * k + 512 * k + 50 * 8192 * k), free.usable);
    const g: usize = 1 << 30;
    try t.expectEqual(@as(usize, 20 * g), size(20 * g, 100 * g, 30 * g));
    try t.expectEqual(@as(usize, 60 * g), size(90 * g, 200 * g, 30 * g));
    try t.expectEqual(@as(usize, 6 * g), size(20 * g, 52 * g, 30 * g));
    try t.expectEqual(@as(usize, 0), size(20 * g, 46 * g, 30 * g));
    try t.expectEqual(@as(usize, 0), size(g / 2, 100 * g, 30 * g));
    try t.expectEqual(g, size(g + 4096, 100 * g, 30 * g));
    try t.expectEqual(@as(usize, 0), size(g, std.math.maxInt(usize), std.math.maxInt(usize)));
}

test "sponge: 4 KiB placement budget excludes sub-2 MiB blocks" {
    const info = "Node 0, zone Normal 3 0 0 0 0 0 0 0 0 1 0 0 1 2\n";
    const free = freeBlocks(info, 4096);
    try std.testing.expectEqual(@as(usize, (2 << 20) + (16 << 20)), free.small);
    try std.testing.expectEqual(@as(usize, (2 << 20) + (16 << 20) + 2 * (32 << 20)), free.usable);
}
