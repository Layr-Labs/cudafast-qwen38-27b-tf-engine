//! Four completed greedy verification rounds, local to one stream; no token or prompt storage.
const std = @import("std");

pub const default_nodes = 15;
pub const neutral_scale: f64 = 1.5;

pub const History = struct {
    depths: [4]u8 = @splat(0),
    next: u8 = 0,
    used: u8 = 0,
    total: u16 = 0,

    pub fn reset(h: *History) void { h.* = .{}; }

    pub fn observe(h: *History, depth: u8) void {
        // Only the bounded default tree feeds the controller. Invalid observations change nothing.
        if (depth > default_nodes) return;
        if (h.used == h.depths.len) h.total -= h.depths[h.next] else h.used += 1;
        h.depths[h.next] = depth;
        h.total += depth;
        h.next = (h.next + 1) % h.depths.len;
    }

    pub fn scale(h: *const History) f64 {
        if (h.used < h.depths.len) return neutral_scale;
        // Short acceptance flattens the proposal (more siblings); long acceptance sharpens it (more depth).
        if (h.total <= 4) return 1.8;
        if (h.total >= 16) return 1.3;
        return neutral_scale;
    }
};

/// Decode kept row indices, excluding pending row 0. Skip censored budgets and non-path observations.
pub fn acceptedDepth(parents: []const i32, offered: usize, kept: []const u32) ?u8 {
    if (parents.len != default_nodes or offered != parents.len or kept.len == 0 or
        kept.len > offered + 1 or kept[0] != 0) return null;
    var previous: u32 = 0;
    for (kept[1..]) |row| {
        if (row <= previous or row > offered) return null;
        const expected: i32 = if (previous == 0) -1 else @intCast(previous - 1);
        if (parents[row - 1] != expected) return null;
        previous = row;
    }
    return @intCast(kept.len - 1);
}

test "history: neutral cold start, four-round eviction, threshold edges and reset" {
    var h: History = .{};
    for (0..3) |_| {
        h.observe(0);
        try std.testing.expectEqual(neutral_scale, h.scale());
    }
    h.observe(4);
    try std.testing.expectEqual(@as(f64, 1.8), h.scale()); // total 4
    h.observe(1);
    try std.testing.expectEqual(neutral_scale, h.scale()); // total 5
    h.observe(5);
    h.observe(5);
    try std.testing.expectEqual(neutral_scale, h.scale()); // total 15
    h.observe(5);
    try std.testing.expectEqual(@as(f64, 1.3), h.scale()); // total 16
    const before = h;
    h.observe(255);
    try std.testing.expectEqualDeep(before, h);
    h.reset();
    try std.testing.expectEqualDeep(History{}, h);
}

test "history: sparse kept rows count depth; branches, duplicates and clipped offers do not train" {
    const parents = [_]i32{ -1, -1, 0, 0, 2, 2, 1, 1, 4, 4, 6, 6, 8, 8, 12 };
    try std.testing.expectEqual(@as(?u8, 0), acceptedDepth(&parents, 15, &.{0}));
    try std.testing.expectEqual(@as(?u8, 4), acceptedDepth(&parents, 15, &.{ 0, 1, 3, 5, 9 }));
    try std.testing.expectEqual(@as(?u8, null), acceptedDepth(&parents, 14, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(?u8, null), acceptedDepth(&parents, 15, &.{ 0, 1, 2 }));
    try std.testing.expectEqual(@as(?u8, null), acceptedDepth(&parents, 15, &.{ 0, 1, 1 }));
    try std.testing.expectEqual(@as(?u8, null), acceptedDepth(&parents, 15, &.{16}));
    try std.testing.expectEqual(@as(?u8, null), acceptedDepth(&parents, 15, &.{}));
}
