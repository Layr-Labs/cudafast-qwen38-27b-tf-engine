//! Prefer the high-capacity cores for the native CUDA launch thread, within its existing allowed CPU set.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

threadlocal var attempted = false;

pub fn preferPerformanceCores() void {
    if (comptime builtin.os.tag != .linux) return;
    if (attempted) return;
    attempted = true;
    var allowed: linux.cpu_set_t = undefined;
    if (linux.errno(linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &allowed)) != .SUCCESS) return;
    const chosen = preferred(allowed, Capacity{}) orelse return;
    linux.sched_setaffinity(0, &chosen) catch {};
}

fn preferred(allowed: linux.cpu_set_t, source: anytype) ?linux.cpu_set_t {
    var chosen: linux.cpu_set_t = @splat(0);
    var highest: u32 = 0;
    var lowest: u32 = std.math.maxInt(u32);
    for (allowed, 0..) |word, wi| {
        var bits = word;
        while (bits != 0) : (bits &= bits - 1) {
            const bit = @ctz(bits);
            const cpu = wi * @bitSizeOf(usize) + bit;
            // Incomplete topology must not accidentally exclude an unknown, faster core.
            const value = source.capacity(cpu) orelse return null;
            lowest = @min(lowest, value);
            if (value > highest) {
                @memset(&chosen, 0);
                highest = value;
            }
            if (value == highest) chosen[wi] |= @as(usize, 1) << @intCast(bit);
        }
    }
    return if (highest > lowest) chosen else null;
}

const Capacity = struct {
    fn capacity(_: Capacity, cpu: usize) ?u32 {
        var path_buf: [96]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&path_buf, "/sys/devices/system/cpu/cpu{d}/cpu_capacity", .{cpu}, 0) catch return null;
        const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (linux.errno(rc) != .SUCCESS) return null;
        const fd: i32 = @intCast(rc);
        defer _ = linux.close(fd);
        var buf: [32]u8 = undefined;
        const n = linux.read(fd, &buf, buf.len);
        if (linux.errno(n) != .SUCCESS) return null;
        return std.fmt.parseInt(u32, std.mem.trim(u8, buf[0..n], " \n"), 10) catch null;
    }
};

test "CUDA affinity respects the allowed CPUs and leaves incomplete or uniform topology alone" {
    const Source = struct {
        values: []const ?u32,
        fn capacity(s: @This(), cpu: usize) ?u32 {
            return s.values[cpu];
        }
    };
    var allowed: linux.cpu_set_t = @splat(0);
    allowed[0] = 0b1011;
    const chosen = preferred(allowed, Source{ .values = &.{ 512, 1024, 2048, 1024 } }).?;
    try std.testing.expectEqual(@as(usize, 0b1010), chosen[0]);
    for (chosen[1..]) |word| try std.testing.expectEqual(@as(usize, 0), word);
    try std.testing.expect(preferred(allowed, Source{ .values = &.{ 1024, 1024, 2048, 1024 } }) == null);
    try std.testing.expect(preferred(allowed, Source{ .values = &.{ 512, null, 2048, 1024 } }) == null);
    @memset(&allowed, 0);
    try std.testing.expect(preferred(allowed, Source{ .values = &.{} }) == null);
}
