//! A streamed buffer committed where it writes fastest, out of a few candidates.
//!
//! On GB10 the GPU reads the same DRAM at very different rates depending on where a buffer's physical pages sit: reads
//! from memory left in 2 MiB blocks run about 237 GB/s, from 32 MiB blocks about 258, and a freshly booted box reads
//! about 262. The decode GEMVs stream the whole target from the weight arena and its packed copies every round at that
//! rate. Where a buffer lands is the driver's choice, so this commits up to `tries` candidates of the buffer's size,
//! times one full write pass over each (a memset, first touch already done), keeps the fastest and frees the rest.
//! Nothing is read from the candidates: where memory sits changes no result.
const std = @import("std");
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;

/// Candidates committed at most.
const tries: usize = 3;
/// Free memory kept beyond the candidates.
const margin: usize = 24 << 30;
/// Buffers smaller than this are allocated once.
const least: usize = 1 << 30;

/// `len` bytes, zero-filled (committed on first touch), placed in the fastest of up to `tries` candidates. One plain
/// allocation when TF_PLACEMENT=0, the buffer is small, memory is short, or a timing call fails.
pub fn alloc(d: *const cuda.Driver, len: usize) !DeviceBuffer {
    var free_bytes: usize = 0;
    var total: usize = 0;
    const room = d.api.cuMemGetInfo_v2(&free_bytes, &total) == cuda.abi.success;
    const n = if (!on() or len < least or !room) 1 else @min(tries, (free_bytes -| margin) / len);
    if (n < 2) return plain(d, len);
    var cands: [tries]DeviceBuffer = undefined;
    var ms: [tries]f32 = @splat(std.math.inf(f32));
    var k: usize = 0;
    defer for (cands[0..k]) |*c| if (c.ptr != 0) c.free();
    while (k < n) {
        cands[k] = DeviceBuffer.alloc(d, len) catch break;
        k += 1;
        cands[k - 1].fill8(0, null) catch break; // first touch: the driver commits the pages now
        ms[k - 1] = timed(d, cands[k - 1]) orelse break;
    }
    if (k == 0) return plain(d, len);
    var best: usize = 0;
    for (ms[0..k], 0..) |t, i| if (t < ms[best]) {
        best = i;
    };
    std.log.scoped(.cuda).info("placement: {d} MiB, {d} candidates, best {d:.2} ms (first {d:.2} ms)", .{ len >> 20, k, ms[best], ms[0] });
    const keep = cands[best];
    cands[best].ptr = 0; // the deferred free skips it
    return keep;
}

fn plain(d: *const cuda.Driver, len: usize) !DeviceBuffer {
    var b = try DeviceBuffer.alloc(d, len);
    errdefer b.free();
    try b.fill8(0, null);
    return b;
}

/// One memset pass over `b` timed with events on the legacy stream; null if any call fails.
fn timed(d: *const cuda.Driver, b: DeviceBuffer) ?f32 {
    var start: cuda.abi.Event = null;
    var end: cuda.abi.Event = null;
    if (d.api.cuEventCreate(&start, 0) != cuda.abi.success) return null;
    defer _ = d.api.cuEventDestroy_v2(start);
    if (d.api.cuEventCreate(&end, 0) != cuda.abi.success) return null;
    defer _ = d.api.cuEventDestroy_v2(end);
    if (d.api.cuEventRecord(start, null) != cuda.abi.success) return null;
    if (d.api.cuMemsetD8Async(b.ptr, 0, b.len, null) != cuda.abi.success) return null;
    if (d.api.cuEventRecord(end, null) != cuda.abi.success) return null;
    if (d.api.cuEventSynchronize(end) != cuda.abi.success) return null;
    var t: f32 = 0;
    if (d.api.cuEventElapsedTime(&t, start, end) != cuda.abi.success) return null;
    return t;
}

fn on() bool {
    const v = @import("sys").c.getenv("TF_PLACEMENT") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}
