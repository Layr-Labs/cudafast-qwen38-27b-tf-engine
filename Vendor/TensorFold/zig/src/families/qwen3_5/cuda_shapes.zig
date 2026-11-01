//! Captured launch geometry, separate from the live counts read through the staged device tables.
const std = @import("std");
const Round = @import("cuda_round.zig");

pub const Shape = struct {
    total: u16, // a prompt chunk's rows (up to Round.max_rows, never captured), or a window bucket
    nseg: u8,
    record: bool,
    prompt: bool,
    taps: bool,
    slots: u8,
    hidden: bool = false,
    paged: bool = false,
    short: bool = false, // every stream's cache holds fewer than GROUP_LEN positions: no row of attn_rows's
    amax: bool = false,

    pub fn init(rows: usize, segments: usize, record: bool, taps: bool, slots: u8) Shape {
        std.debug.assert(rows > 0 and rows <= @as(usize, if (record) Round.most_verify else Round.max_rows) and segments > 0 and segments <= rows);
        const bucket: usize = if (rows == 1) 1 else std.mem.alignForward(usize, rows, 16);
        const windows = std.math.ceilPowerOfTwo(usize, segments) catch unreachable;
        return .{ .total = @intCast(if (record) bucket else rows), .nseg = @intCast(if (record) windows else segments), .record = record, .prompt = !record, .taps = taps, .slots = slots };
    }

    pub fn key(s: Shape) u32 {
        // captured shapes only: a window bucket's rows fit the low byte
        return @as(u32, @as(u8, @intCast(s.total))) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, s.slots) << 19 | @as(u32, @intFromBool(s.hidden)) << 27 | @as(u32, @intFromBool(s.paged)) << 28 | @as(u32, @intFromBool(s.short)) << 29 | @as(u32, @intFromBool(s.amax)) << 30;
    }
};

test "greedy token and full-logit graphs coexist for the same window" {
    const logits = Shape.init(15, 1, true, false, 1);
    var tokens = logits;
    tokens.amax = true;
    var graphs: std.AutoHashMapUnmanaged(u32, enum { logits, tokens }) = .empty;
    defer graphs.deinit(std.testing.allocator);
    try graphs.put(std.testing.allocator, logits.key(), .logits);
    try graphs.put(std.testing.allocator, tokens.key(), .tokens);
    try std.testing.expectEqual(.logits, graphs.get(logits.key()).?);
    try std.testing.expectEqual(.tokens, graphs.get(tokens.key()).?);
}
