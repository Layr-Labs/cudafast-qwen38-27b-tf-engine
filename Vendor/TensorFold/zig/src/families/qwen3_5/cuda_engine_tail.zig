//! Fallible post-load allocations, synchronization and warmup before the engine takes ownership.
const std = @import("std");
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;
const tables_len = @import("cuda_round.zig").bytes;

pub fn finish(e: anytype, io: std.Io, started: i96) !void {
    errdefer if (e.kit) |k| k.deinit(e.gpa);
    _ = try e.scratch(.allocate);
    // the fused gdn_post's arrival counts start at zero; each head's last CTA returns its count to zero
    try e.d.check(e.d.api.cuMemsetD32Async(e.gdn_count, 0, @import("cuda_round.zig").max_segs * e.g.v_heads, e.stream.handle), "cuMemsetD32Async");
    try e.d.check(e.d.api.cuMemsetD32Async(e.merge_count, 0, (@import("cuda_engine.zig").most_verify + 1) / 2 * e.g.kv_heads, e.stream.handle), "cuMemsetD32Async");
    e.tables = try DeviceBuffer.alloc(e.d, tables_len);
    errdefer e.tables.free();
    e.staged = try cuda.HostBuffer.alloc(e.d, tables_len);
    errdefer e.staged.free();
    e.host_logits = try cuda.HostBuffer.alloc(e.d, e.verify_rows * e.g.vocab * 2);
    errdefer e.host_logits.free();
    try e.stream.synchronize();
    try e.finishWarm(io, started);
}
