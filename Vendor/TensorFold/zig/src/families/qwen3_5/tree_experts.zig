//! Choose a tree policy from this stream's observed, censored match differences.
const std = @import("std");
pub fn choose(scores: [3]i32, margin: i32) usize {
    var selected: usize = 1;
    for ([_]usize{0, 2}) |i| {
        if (scores[i] >= scores[1] + margin and scores[i] > scores[selected]) selected = i;
    }
    return selected;
}
test "baseline until evidence exceeds margin and deterministic ties" {
    try std.testing.expectEqual(@as(usize, 1), choose(.{ 1, 0, 1 }, 2));
    try std.testing.expectEqual(@as(usize, 0), choose(.{ 2, 0, 2 }, 2));
    try std.testing.expectEqual(@as(usize, 2), choose(.{ 3, 0, 4 }, 2));
}
