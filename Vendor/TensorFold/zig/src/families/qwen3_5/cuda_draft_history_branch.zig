//! Prefix-conditioned request-history proposals under non-greedy DFlash branches.
//! No target decision: every node still goes through ordinary greedy verification.
const std = @import("std");

/// Keep the best-first prefix through a matching node and replace only its tail
/// with the observed historical continuation. Return the first retrieved node.
/// Requirements mirror the inherited greedy-history path: >=3 proposed matches,
/// >=6 history+proposal matches. Search is bounded to this request's last 4096 tokens.
/// Input nodes must be unique by (parent,token), parent-closed and best-first ordered.
/// No allocation, model access, persistence, extra rows or additional draft forward.
pub fn graft(hist: []const u32, pending: u32, tokens: []u32, parents: []i32, own: []f64) ?usize {
    if (hist.len < 4 or tokens.len < 4 or hist[hist.len - 1] != pending) return null;
    std.debug.assert(tokens.len == parents.len and tokens.len == own.len);
    const start = hist.len -| 4096;
    var best_node: usize = 0;
    var best_pos: usize = 0;
    var best_match: usize = 0;
    var best_depth: usize = 0;
    var idx = hist.len - 1;
    while (idx > start) {
        idx -= 1;
        if (hist[idx] != pending) continue;
        var hm: usize = 1;
        while (hm <= idx and hm < 64 and hist[idx - hm] == hist[hist.len - 1 - hm]) hm += 1;
        var parent: i32 = -1;
        var depth: usize = 0;
        while (idx + depth + 1 < hist.len) {
            const next = hist[idx + depth + 1];
            var found: ?usize = null;
            for (tokens, parents, 0..) |token, p, i| {
                if (p == parent and token == next) {
                    found = i;
                    break;
                }
            }
            const node = found orelse break;
            depth += 1;
            parent = @intCast(node);
            if (depth < 3 or hm + depth < 6 or node + 1 >= tokens.len) continue;
            // Require an actual continuation. Matching an old suffix alone supplies no proposal.
            if (idx + depth + 1 >= hist.len) continue;
            if (hm + depth > best_match or (hm + depth == best_match and depth > best_depth)) {
                best_node = node;
                best_pos = idx + depth + 1;
                best_match = hm + depth;
                best_depth = depth;
            }
        }
    }
    if (best_depth == 0) return null;
    const retained = best_node + 1;
    const copied = @min(tokens.len - retained, hist.len - best_pos);
    // A short continuation cannot justify discarding existing alternatives. Leave it untouched.
    if (copied != tokens.len - retained) return null;
    for (tokens[retained..], parents[retained..], own[retained..], 0..) |*token, *parent, *chance, j| {
        token.* = hist[best_pos + j];
        parent.* = @intCast(best_node + j);
        chance.* = 0; // inherited history proposal prior, never observed as a neural edge
    }
    return retained;
}

test "history branch: graft beneath a non-greedy prefix, preserve budget and competing roots" {
    // The first root 90 is greedy; root 10, not on that chain, matches old text.
    const hist = [_]u32{ 7, 8, 9, 10, 11, 12, 13, 14, 15, 6, 7, 8, 9 };
    var tokens = [_]u32{ 90, 10, 11, 12, 70, 71, 72 };
    var parents = [_]i32{ -1, -1, 1, 2, 0, 4, 5 };
    var own: [7]f64 = @splat(-1);
    const retained = graft(&hist, 9, &tokens, &parents, &own).?;
    try std.testing.expectEqual(@as(usize, 4), retained);
    try std.testing.expectEqualSlices(u32, &.{ 90, 10, 11, 12, 13, 14, 15 }, &tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 1, 2, 3, 4, 5 }, &parents);
    try std.testing.expectEqualSlices(f64, &.{ -1, -1, -1, -1, 0, 0, 0 }, &own);
}

test "history branch: weak evidence and missing continuation leave neural tree untouched" {
    var tokens = [_]u32{ 10, 11, 12, 70, 71, 72, 73 };
    const original = tokens;
    var parents = [_]i32{ -1, 0, 1, -1, 3, 4, 5 };
    var own: [7]f64 = @splat(-1);
    try std.testing.expect(graft(&.{ 9, 10, 11, 12, 13, 14, 15, 9 }, 9, &tokens, &parents, &own) == null);
    try std.testing.expect(graft(&.{ 7, 8, 9, 10, 11, 12, 7, 8, 9 }, 9, &tokens, &parents, &own) == null);
    try std.testing.expectEqualSlices(u32, &original, &tokens);
}

test "history branch: empty holds and foreign pending tokens are inert" {
    try std.testing.expect(graft(&.{}, 9, &.{}, &.{}, &.{}) == null);
    var tokens = [_]u32{ 10, 11, 12, 13 };
    var parents = [_]i32{ -1, 0, 1, 2 };
    var own: [4]f64 = @splat(-1);
    try std.testing.expect(graft(&.{ 7, 8, 9, 10, 11, 12, 7, 8, 9 }, 8, &tokens, &parents, &own) == null);
}
