//! Captured launch geometry, separate from the live counts read through the staged device tables.
const std = @import("std");
const Round = @import("cuda_round.zig");

pub const Shape = struct {
    pub const max_partial_chunks: u8 = 12;
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
    partial_chunks: u8 = max_partial_chunks,

    pub fn init(rows: usize, segments: usize, record: bool, taps: bool, slots: u8) Shape {
        std.debug.assert(rows > 0 and rows <= @as(usize, if (record) Round.most_verify else Round.max_rows) and segments > 0 and segments <= rows);
        const bucket: usize = if (rows == 1) 1 else std.mem.alignForward(usize, rows, 16);
        const windows = std.math.ceilPowerOfTwo(usize, segments) catch unreachable;
        return .{ .total = @intCast(if (record) bucket else rows), .nseg = @intCast(if (record) windows else segments), .record = record, .prompt = !record, .taps = taps, .slots = slots };
    }

    pub fn attentionChunks(keys: usize) u8 {
        return @intCast(@max(1, @min(max_partial_chunks, keys / 128 + @intFromBool(keys % 128 != 0))));
    }

    pub fn key(s: Shape) u64 {
        // captured shapes only: a window bucket's rows fit the low byte
        const low = @as(u32, @as(u8, @intCast(s.total))) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, s.slots) << 19 | @as(u32, @intFromBool(s.hidden)) << 27 | @as(u32, @intFromBool(s.paged)) << 28 | @as(u32, @intFromBool(s.short)) << 29 | @as(u32, @intFromBool(s.amax)) << 30;
        return low | @as(u64, s.partial_chunks) << 32;
    }
};

test "greedy token and full-logit graphs coexist for the same window" {
    const logits = Shape.init(15, 1, true, false, 1);
    var tokens = logits;
    tokens.amax = true;
    var graphs: std.AutoHashMapUnmanaged(u64, enum { logits, tokens }) = .empty;
    defer graphs.deinit(std.testing.allocator);
    try graphs.put(std.testing.allocator, logits.key(), .logits);
    try graphs.put(std.testing.allocator, tokens.key(), .tokens);
    try std.testing.expectEqual(.logits, graphs.get(logits.key()).?);
    try std.testing.expectEqual(.tokens, graphs.get(tokens.key()).?);
}

test "native partial chunks retain context boundaries without overflow" {
    const t = std.testing;
    try t.expectEqual(@as(u8, 1), Shape.attentionChunks(0));
    try t.expectEqual(@as(u8, 1), Shape.attentionChunks(128));
    try t.expectEqual(@as(u8, 2), Shape.attentionChunks(129));
    try t.expectEqual(@as(u8, 11), Shape.attentionChunks(1408));
    try t.expectEqual(@as(u8, 12), Shape.attentionChunks(1409));
    try t.expectEqual(@as(u8, 12), Shape.attentionChunks(std.math.maxInt(usize)));
    var graphs: std.AutoHashMapUnmanaged(u64, u8) = .empty;
    defer graphs.deinit(t.allocator);
    var shape = Shape.init(16, 1, true, false, 37);
    for (1..13) |chunks| {
        shape.partial_chunks = @intCast(chunks);
        try graphs.put(t.allocator, shape.key(), @intCast(chunks));
    }
    for (1..13) |chunks| {
        shape.partial_chunks = @intCast(chunks);
        try t.expectEqual(@as(u8, @intCast(chunks)), graphs.get(shape.key()).?);
    }
}
