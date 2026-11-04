//! Optional, lossless immutable gate/up views. Exact sizes are measured before screened placement;
//! the pool and all views are published together only after packing proves completion.
const std = @import("std");
const cuda = @import("cuda");
const Buffer = cuda.DeviceBuffer;
const placement = @import("cuda_fastpool.zig");
const table_bytes = 131072 + 16;

pub const Storage = struct {
    pool: ?placement.Pool = null,
    pub fn deinit(self: *Storage, gpa: std.mem.Allocator) void {
        _ = gpa;
        if (self.pool) |*p| p.free();
        self.* = .{};
    }
};

const Plan = struct {
    q: *cuda.qlinear.Affine4,
    offsets: []u32,
    bytes: usize,
    index: u64 = 0,
    blob: u64 = 0,
};

fn eligible(q: cuda.qlinear.Affine4) bool {
    return q.n == 17408 and q.k == 5120 and q.verify_w != 0 and
        q.code & cuda.qlinear.dict_kmajor != 0 and cuda.qlinear.pairCode(q.code);
}

/// Rewrites sizes into exact u32 offsets. Decline invalid, growing and overflowing tensors.
fn offsetsFor(sizes: []u32, raw_bytes: usize) ?usize {
    var total: usize = 0;
    for (sizes) |*h| {
        const n = h.*;
        if (n < 1556 or n > 2580 or n & 3 != 0) return null;
        h.* = std.math.cast(u32, total) orelse return null;
        total = std.math.add(usize, total, n) catch return null;
    }
    const indices = std.math.mul(usize, sizes.len, 4) catch return null;
    const bytes = std.math.add(usize, total, indices) catch return null;
    if (bytes >= raw_bytes or total > std.math.maxInt(u32)) return null;
    return total;
}

fn measure(e: anytype, pack: cuda.Function, q: *cuda.qlinear.Affine4) !?Plan {
    const raw_bytes = q.npad * q.k / 2;
    if (raw_bytes % 2048 != 0) return null;
    const count = raw_bytes / 2048;
    var sizes = try Buffer.alloc(e.d, count * 4);
    // This scope owns the GPU destination: drain BEFORE freeing it even when launch/sync fails.
    defer {
        cuda.completion.drain(e.d, e.stream, "bitplane measurement did not drain");
        sizes.free();
    }
    const host = try e.gpa.alloc(u32, count);
    var retained = false;
    defer if (!retained) e.gpa.free(host);
    try e.run(pack, .{ count, 1, 1 }, .{ 32, 1, 1 }, .{ q.verify_w, sizes.ptr, @as(u64, 0), @as(u64, 0), @as(c_int, @intCast(count)) });
    try e.stream.sync();
    try sizes.download(0, std.mem.sliceAsBytes(host));
    const total = offsetsFor(host, raw_bytes) orelse return null;
    retained = true;
    return .{ .q = q, .offsets = host, .bytes = total };
}

pub fn prepare(e: anytype) !void {
    // Optional copies must never turn a memory-constrained baseline into an engine-open OOM.
    try prepareWithScreen(e, placement.screen);
}

/// Injecting the weight-only screen also permits deterministic host failure-path tests.
pub fn prepareWithScreen(e: anytype, screen: anytype) !void {
    prepareScreened(e, screen) catch |err| switch (err) {
        error.OutOfMemory, error.OutOfDeviceMemory => std.log.scoped(.cuda).info("bitplane: optional copies declined (memory)", .{}),
        else => return err,
    };
}

fn prepareScreened(e: anytype, screen: anytype) !void {
    if (e.mlp_pair.bitplane_fn == null or e.mlp_pair.kmajor_fn == null or !e.mlp_pair.kmajor_single or
        e.mlp_pair.single_dict_fn == null or e.mlp_pair.compact_pair_fn == null) return;
    if (@import("sys").c.getenv("TF_FASTPOOL")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return;
    std.debug.assert(e.bitplane_weights.pool == null);
    const pack = try e.mods[0].function("tf_bp54_pack");
    var plans: std.ArrayListUnmanaged(Plan) = .empty;
    defer {
        for (plans.items) |p| e.gpa.free(p.offsets);
        plans.deinit(e.gpa);
    }
    for (e.layers) |*layer| switch (layer.mlp) {
        .dense => |*m| {
            if (m.gate != .a4 or m.up != .a4 or !eligible(m.gate.a4) or !eligible(m.up.a4) or
                m.gate.a4.npad != m.up.a4.npad) continue;
            const gate = (try measure(e, pack, &m.gate.a4)) orelse continue;
            var gate_owned = true;
            defer if (gate_owned) e.gpa.free(gate.offsets);
            const up = (try measure(e, pack, &m.up.a4)) orelse continue;
            var up_owned = true;
            defer if (up_owned) e.gpa.free(up.offsets);
            try plans.append(e.gpa, gate);
            gate_owned = false;
            try plans.append(e.gpa, up);
            up_owned = false;
        },
        else => {},
    };
    if (plans.items.len == 0) return;
    // Exactly the order and alignment Pool.alloc will use: shared table, then each index and blob.
    const sizes = try e.gpa.alloc(usize, 1 + 2 * plans.items.len);
    defer e.gpa.free(sizes);
    sizes[0] = table_bytes;
    for (plans.items, 0..) |p, i| {
        sizes[1 + 2 * i] = p.offsets.len * 4;
        sizes[2 + 2 * i] = p.bytes;
    }
    const probe = try e.mods[3].function("tf_fastpool_read");
    var pool = screen(e.d, e.stream, probe, sizes) orelse return; // reserve unavailable: baseline only
    var committed = false;
    defer {
        cuda.completion.drain(e.d, e.stream, "bitplane packing did not drain");
        if (!committed) pool.free();
    }
    const table = pool.alloc(table_bytes) orelse return error.FastPoolShort;
    const init = try e.mods[0].function("tf_bp54_init");
    try e.run(init, .{ 256, 1, 1 }, .{ 256, 1, 1 }, .{ table, table + 131072 });
    for (plans.items) |*p| {
        p.index = pool.alloc(p.offsets.len * 4) orelse return error.FastPoolShort;
        p.blob = pool.alloc(p.bytes) orelse return error.FastPoolShort;
        const index: Buffer = .{ .d = e.d, .ptr = p.index, .len = p.offsets.len * 4, .borrowed = true };
        index.upload(0, std.mem.sliceAsBytes(p.offsets)) catch |err| {
            // A failed blocking upload can retain an in-flight default-stream write. The engine stream is
            // non-blocking, so draining only it does not prove this borrowed destination can be freed.
            cuda.completion.drain(e.d, .{ .d = e.d, .handle = null }, "bitplane index upload did not drain");
            return err;
        };
        // Sizes is unused in the write pass; the count and measured offsets fix every output extent.
        try e.run(pack, .{ p.offsets.len, 1, 1 }, .{ 32, 1, 1 }, .{ p.q.verify_w, @as(u64, 0), p.index, p.blob, @as(c_int, @intCast(p.offsets.len)) });
    }
    try e.stream.sync();
    // No fallible work past this boundary. Raw verify panels remain owned by their original pool/buffer.
    e.bitplane_weights.pool = pool;
    for (plans.items) |p| p.q.bitplane = .{ .blob = p.blob, .offsets = p.index, .table = table, .mapping = table + 131072 };
    committed = true;
    std.log.scoped(.cuda).info("bitplane: {d} immutable projections in a separately screened pool", .{plans.items.len});
}

test "bitplane exact offsets reject bad metadata and growth" {
    const t = std.testing;
    var valid = [_]u32{ 1556, 2040, 2000 };
    try t.expectEqual(@as(?usize, 5596), offsetsFor(&valid, 3 * 2048));
    try t.expectEqualSlices(u32, &.{ 0, 1556, 3596 }, &valid);
    var unaligned = [_]u32{1557};
    var short = [_]u32{1552};
    var long = [_]u32{2584};
    var growing = [_]u32{2048};
    var equal = [_]u32{2044};
    try t.expectEqual(@as(?usize, null), offsetsFor(&unaligned, 2048));
    try t.expectEqual(@as(?usize, null), offsetsFor(&short, 2048));
    try t.expectEqual(@as(?usize, null), offsetsFor(&long, 4096));
    try t.expectEqual(@as(?usize, null), offsetsFor(&growing, 2048));
    try t.expectEqual(@as(?usize, null), offsetsFor(&equal, 2048));
}
