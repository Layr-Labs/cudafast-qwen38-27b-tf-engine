//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,

    /// How the host waits for the device (CU_CTX_SCHED_*); host scheduling only, no launch or value changes.
    pub const Schedule = enum(c_uint) { auto = 0, spin = 1, yield = 2, blocking = 4 };

    /// The device's primary context's schedule, before any `init` of it (the driver ignores it on an active context).
    pub fn setSchedule(d: *const Driver, ordinal: c_int, s: Schedule) Error!void {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        try d.check(d.api.cuDevicePrimaryCtxSetFlags_v2(dev, @backingInt(s)), "cuDevicePrimaryCtxSetFlags");
    }

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try retainPrimary(d, &ctx, dev);
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        try @import("context_health.zig").retain(d, ctx);
        @import("copy_source.zig").setCurrent(ctx);
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// For a library in a host's process: the current device's primary context (device 0's if none), or error.Invalid.
    pub fn shared(d: *const Driver) Error!Context {
        var current: abi.Context = null;
        try d.check(d.api.cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (current == null) return init(d, 0);
        var dev: abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        if (ctx != current) {
            _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
            return error.Invalid;
        }
        @import("context_health.zig").retain(d, ctx) catch |err| {
            _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
            return err;
        };
        @import("copy_source.zig").setCurrent(ctx); // the host made it current: copy sources are tagged with it
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        self.d.synchronize() catch {};
        @import("copy_source.zig").dropSpares(self.d, self.handle); // their pinned bytes go with the context
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        @import("context_health.zig").release(self.d, self.handle);
        self.* = undefined;
    }

    /// Makes the context current on another thread before it calls the driver.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent");
        @import("copy_source.zig").setCurrent(self.handle);
    }

    /// Readiness can be read from HTTP threads without making the context current.
    pub fn healthy(self: *const Context) bool {
        return @import("context_health.zig").healthy(self.d, self.handle);
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.synchronize();
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuDeviceGetAttribute(&v, a, self.device), "cuDeviceGetAttribute");
        return v;
    }

    /// Compute capability as 10 * major + minor (GB10: 121).
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "cuDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.cuMemGetInfo_v2(&m.free, &m.total), "cuMemGetInfo");
        return m;
    }
};

/// Creating a primary context can fail with CUDA_ERROR_OUT_OF_MEMORY while most of the host's memory is free: the
/// driver needs a few large physically contiguous blocks, and on a host whose free memory is fragmented the kernel's
/// compaction and reclaim make them only some time later. So that one error is retried, `retain_tries` attempts in all,
/// `retain_nap_s` seconds apart, each retry logged; any other error, or the last attempt's, fails as it did before.
const retain_tries = 10;
const retain_nap_s = if (builtin.is_test) 0 else 1;

fn retainPrimary(d: *const Driver, ctx: *abi.Context, dev: abi.Device) Error!void {
    var attempt: u32 = 1;
    while (true) : (attempt += 1) {
        const res = d.api.cuDevicePrimaryCtxRetain(ctx, dev);
        if (res != abi.error_out_of_memory or attempt == retain_tries) return d.check(res, "cuDevicePrimaryCtxRetain");
        std.log.warn("cuDevicePrimaryCtxRetain: out of memory on attempt {d} of {d}; retrying in {d} s", .{
            attempt, retain_tries, retain_nap_s,
        });
        nap(retain_nap_s);
    }
}

fn nap(seconds: u32) void {
    if (seconds == 0) return;
    switch (builtin.os.tag) {
        .windows => {},
        else => {
            const t: std.c.timespec = .{ .sec = seconds, .nsec = 0 };
            _ = std.c.nanosleep(&t, null);
        },
    }
}
