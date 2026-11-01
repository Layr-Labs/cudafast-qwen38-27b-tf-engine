//! Content-free setup reservation held while weights commit, then freed before phase work.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const linux = std.os.linux;

/// Free blocks of at least this size contribute to the setup reservation.
const chunk: usize = 2 << 20;
/// Memory left free beyond the engine's own buffers; the buffer never takes more than `most`.
const margin: usize = 24 << 30;
const most: usize = 40 << 30;
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

/// Reserves usable buddy blocks while retaining the engine's need and the existing memory margin.
/// Unreadable counts, insufficient memory and driver failures leave the normal load unchanged.
pub fn take(d: *const cuda.Driver, need: usize) Sponge {
    if (comptime builtin.os.tag != .linux) return .{};
    const page = std.heap.pageSize();
    var buf: [16384]u8 = undefined;
    const usable = usableFree(read("/proc/buddyinfo", &buf) orelse return .{}, page);
    const avail = memAvailable(read("/proc/meminfo", &buf) orelse return .{}) orelse return .{};
    const len = size(usable, avail, need);
    if (len == 0) return .{};
    var p: cuda.abi.DevicePtr = 0;
    if (d.api.cuMemAlloc_v2(&p, len) != cuda.abi.success) return .{};
    if (d.api.cuMemsetD8_v2(p, 0, len) != cuda.abi.success) {
        _ = d.api.cuMemFree_v2(p);
        return .{};
    }
    std.log.scoped(.cuda).info("sponge: {d} MiB of free blocks held while the weights commit", .{len >> 20});
    return .{ .d = d, .ptr = p, .len = len };
}

fn size(usable: usize, avail: usize, need: usize) usize {
    const keep = need +| margin;
    if (avail <= keep) return 0;
    const len = @min(usable, avail - keep, most) / granule * granule;
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

/// Bytes free in buddy blocks of at least `chunk`, across the reported zones.
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

fn memAvailable(text: []const u8) ?usize {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var it = std.mem.tokenizeAny(u8, line[13..], " \t");
        const kb = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
        return kb *| 1024;
    };
    return null;
}

test "sponge: usable buddy blocks retain the memory reserve" {
    const t = std.testing;
    const info =
        \\Node 0, zone      DMA      3      2      1      0      0      0      0      0      0      0      0      0      0      0
        \\Node 0, zone   Normal   1000    500    100     10      4      2      1      0      0      1      0      0      0     50
    ;
    const k: usize = 64 << 10;
    const want = 2 * 32 * k + 64 * k + 512 * k + 50 * 8192 * k;
    try t.expectEqual(want, usableFree(info, k));
    try t.expectEqual(@as(?usize, 123456 * 1024), memAvailable("MemTotal:  999 kB\nMemAvailable:   123456 kB\n"));
    try t.expectEqual(@as(?usize, null), memAvailable("MemTotal: 1 kB\n"));
    const g: usize = 1 << 30;
    try t.expectEqual(@as(usize, 20 * g), size(20 * g, 100 * g, 30 * g));
    try t.expectEqual(@as(usize, 40 * g), size(90 * g, 200 * g, 30 * g)); // at most `most`
    try t.expectEqual(@as(usize, 6 * g), size(20 * g, 60 * g, 30 * g)); // the kept memory first
    try t.expectEqual(@as(usize, 0), size(20 * g, 50 * g, 30 * g));
    try t.expectEqual(@as(usize, 0), size(g / 2, 100 * g, 30 * g));
}

test "sponge: four KiB pages retain large buddy orders and the existing reserve limits" {
    const t = std.testing;
    const info = "Node 0, zone Normal 3 0 0 0 0 0 0 0 0 1 0 0 1 2\n";
    try t.expectEqual(@as(usize, (2 << 20) + (16 << 20) + 2 * (32 << 20)), usableFree(info, 4096));
    const g: usize = 1 << 30;
    try t.expectEqual(@as(usize, 40 * g), size(100 * g, 200 * g, 30 * g));
    try t.expectEqual(@as(usize, 0), size(g, std.math.maxInt(usize), std.math.maxInt(usize)));
}
