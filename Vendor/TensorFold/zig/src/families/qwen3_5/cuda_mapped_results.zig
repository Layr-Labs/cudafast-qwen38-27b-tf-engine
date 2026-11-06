//! Pinned result storage. A failed mapping keeps the existing explicit-copy path.
const std = @import("std");
const cuda = @import("cuda");

pub fn alloc(d: *const cuda.Driver, len: usize, enabled: bool, device: *u64) !cuda.HostBuffer {
    return withBuffer(d, len, enabled, device, cuda.HostBuffer);
}

fn withBuffer(d: anytype, len: usize, enabled: bool, device: *u64, comptime Buffer: type) !Buffer {
    device.* = 0;
    if (enabled) {
        if (Buffer.allocMapped(d, len)) |mapped_| {
            var mapped = mapped_;
            const address = mapped.device() catch 0;
            // Zero or an unexpectedly unaligned device view cannot replace vector stores.
            if (address != 0 and address % 256 == 0) {
                device.* = address;
                return mapped;
            }
            mapped.free();
        } else |_| {}
    }
    return Buffer.alloc(d, len);
}

const Fake = struct {
    const Fault = enum { none, allocate, map, zero, unaligned, fallback };
    var fault: Fault = .none;
    var releases: usize = 0;
    var plain: usize = 0;
    pub fn allocMapped(_: void, _: usize) !Fake {
        if (fault == .allocate or fault == .fallback) return error.MappingUnsupported;
        return .{};
    }
    pub fn device(_: Fake) !u64 {
        return switch (fault) {
            .map => error.MappingUnsupported,
            .zero => 0,
            .unaligned => 257,
            else => 256,
        };
    }
    pub fn alloc(_: void, _: usize) !Fake {
        plain += 1;
        if (fault == .fallback) return error.OutOfMemory;
        return .{};
    }
    pub fn free(_: *Fake) void { releases += 1; }
};

test "mapped results: mapping failures free the view and retain explicit copies" {
    for ([_]Fake.Fault{ .none, .allocate, .map, .zero, .unaligned, .fallback }) |fault| {
        Fake.fault = fault;
        Fake.releases = 0;
        Fake.plain = 0;
        var address: u64 = 999;
        const result = withBuffer({}, 512, true, &address, Fake);
        if (fault == .fallback) try std.testing.expectError(error.OutOfMemory, result) else _ = try result;
        try std.testing.expectEqual(@as(u64, if (fault == .none) 256 else 0), address);
        try std.testing.expectEqual(@as(usize, @intFromBool(fault == .map or fault == .zero or fault == .unaligned)), Fake.releases);
        try std.testing.expectEqual(@as(usize, @intFromBool(fault != .none)), Fake.plain);
    }
    Fake.fault = .none;
    Fake.plain = 0;
    var address: u64 = 999;
    _ = try withBuffer({}, 512, false, &address, Fake);
    try std.testing.expectEqual(@as(u64, 0), address);
    try std.testing.expectEqual(@as(usize, 1), Fake.plain);
}
