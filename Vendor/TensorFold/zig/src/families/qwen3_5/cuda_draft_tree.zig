//! A hold's drafts from its rows' candidates: a chain, the planner's lanes, or a best-first tree as the 27B engine's.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

/// Children a node, and the nodes a tree takes by default.
const branch = 12;
/// A sampled stream's candidates carry the draw's own Gumbel noise at this weight (the 27B engine's draft_tree).
const noise_weight = 0.7;
const default_nodes = 15;
const lp_top = 20;
const lp_words = 3 + 2 * lp_top;

/// How a tree ranks a row's candidates: softmax of (unary + edge_weight x selector) / scale.
pub const Policy = struct { edge_weight: f64, scale: f64 };

/// The rankings a stream's replay compares. The first is the 27B engine's, every node's `own` chance is under it (the
/// planner's landing table keeps one meaning), and a sampled stream's trees always take it.
pub const policies = [_]Policy{
    .{ .edge_weight = 0.6, .scale = 1.5 },
    .{ .edge_weight = 0.6, .scale = 1.25 },
    .{ .edge_weight = 0.6, .scale = 2.0 },
    .{ .edge_weight = 0.6, .scale = 2.75 },
    .{ .edge_weight = 1.0, .scale = 1.5 },
    .{ .edge_weight = 1.0, .scale = 2.0 },
};

/// A hold's rows: each row's logprob words (ids, then logits at lp_top), its selector projection, the codebooks.
pub const Candidates = struct {
    words: []const u8,
    projected: []const u8,
    topk: usize,
    rank: usize,
    predecessor: []const u16, // (vocab, rank) bf16
    successor: []const u16,
    // a sampled stream's rule and its first draft's position: row r's candidates scored as the target's draw at r
    sampling: ?lanes.Sampling = null,
    position: u64 = 0,
};

/// A candidate's score as the target's draw sees it: its logit at the stream's temperature plus the draw's noise.
fn drawn(c: Candidates, r: usize, token: u32, logit: f64) f64 {
    const s = c.sampling orelse return logit;
    const u = @min(lanes.gpu_rule.uniform24(s.seed, @intCast(c.position + r), token), @as(f32, @bitCast(@as(u32, 0x3F7FFFFF))));
    return logit / @max(s.temperature, 1e-6) + noise_weight * -@log(-@log(@as(f64, u)));
}

/// A grown tree's nodes at most: 0 when the drafter drafts chains and lane trees (TF_NO_TREES, TF_DRAFT_NODES=0).
pub fn nodes() usize {
    if (platform.c.getenv("TF_NO_TREES") != null) return 0;
    const n = if (platform.c.getenv("TF_DRAFT_NODES")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_nodes else default_nodes;
    return @min(n, engine.most_window - 1);
}

/// Offered tree nodes (parent row kept) and landed ones by own log-chance in half-nat bins, greedy then every round's.
pub const Landing = struct {
    counts: [2][16][2]f64 = @splat(@splat(.{ 0, 0 })),

    fn bin(own: f64) usize {
        return @min(15, @as(usize, @intFromFloat(@max(0, @floor(-own / 0.5)))));
    }

    pub fn observe(l: *Landing, sampled: bool, own: f64, landed: bool) void {
        for (l.counts[@intFromBool(sampled)..]) |*table| {
            const x = &table[bin(own)];
            x[0] += 1;
            if (landed) x[1] += 1;
        }
    }

    pub fn chance(l: *const Landing, sampled: bool, own: f64) f64 {
        const x = l.counts[@intFromBool(sampled)][bin(own)];
        return (x[1] + 2 * @exp(own)) / (x[0] + 2);
    }

    /// A verified window's offered nodes (the first `offered`, node i at row i + 1) against the rows it kept.
    pub fn observeTree(l: *Landing, sampled: bool, parents: []const i32, own: []const f64, offered: usize, kept: []const u32) void {
        for (parents[0..@min(offered, parents.len)], own[0..@min(offered, own.len)], 0..) |p, o, i| {
            const parent_row: u32 = if (p < 0) 0 else @intCast(p + 1);
            if (std.mem.indexOfScalar(u32, kept, parent_row) == null) continue;
            l.observe(sampled, o, std.mem.indexOfScalar(u32, kept, @intCast(i + 1)) != null);
        }
    }
};

fn bf16(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn bf32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Row `r`'s `rank`th candidate after `pred`: unary logit + pred . projected . successor, ties to the earlier one.
fn choice(c: Candidates, r: usize, pred: u32, rank: usize) u32 {
    const w = ids(c, r);
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    var scores: [lp_top]f32 = undefined;
    for (0..c.topk) |j| {
        var edge: f32 = 0;
        for (0..c.rank) |k| edge += bf32(c.predecessor[pred * c.rank + k]) * bf32(hp[k]) * bf32(c.successor[w[2 + j] * c.rank + k]);
        scores[j] = @floatCast(drawn(c, r, w[2 + j], @as(f32, @bitCast(w[2 + lp_top + j])) + edge));
    }
    var taken: [lp_top]bool = @splat(false);
    var pick: usize = 0;
    for (0..rank + 1) |_| {
        var best: f32 = -std.math.inf(f32);
        pick = 0;
        for (0..c.topk) |j| if (!taken[j] and scores[j] > best) {
            best = scores[j];
            pick = j;
        };
        taken[pick] = true;
    }
    return w[2 + pick];
}

/// The path through the rows' candidates: each row's best after the one before.
pub fn chain(c: Candidates, pending: u32, out: []u32) void {
    var pred = pending;
    for (out, 0..) |*o, r| {
        o.* = choice(c, r, pred, 0);
        pred = o.*;
    }
}

/// The planner's lanes in order: lane i the rank-r candidate at its depth's row after its parent lane's token.
pub fn lanesOf(c: Candidates, pending: u32, rows: usize, l: *const lanes.shape.Shape, out: []u32) !void {
    for (l.parents, l.depths, l.ranks, 0..) |par, depth, rank, i| {
        if (depth >= rows or rank >= c.topk or par >= @as(i32, @intCast(i))) return error.BadLanes;
        out[i] = choice(c, depth, if (par < 0) pending else out[@intCast(par)], rank);
    }
}

fn ids(c: Candidates, r: usize) []align(1) const u32 {
    return std.mem.bytesAsSlice(u32, c.words[r * lp_words * 4 ..][0 .. lp_words * 4]);
}

/// Row `r`'s candidates after `pred`: each one's unary logit and its selector edge (pred . projected . successor).
fn scoresAfter(c: Candidates, r: usize, pred: u32, unary: *[lp_top]f64, edge: *[lp_top]f64) void {
    const w = ids(c, r);
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    for (0..c.topk) |j| {
        var e: f64 = 0;
        for (0..c.rank) |k| e += bf16(c.predecessor[pred * c.rank + k]) * bf16(hp[k]) * bf16(c.successor[w[2 + j] * c.rank + k]);
        edge[j] = e;
        unary[j] = @as(f32, @bitCast(w[2 + lp_top + j]));
    }
}

/// Row `r`'s candidates as `p` ranks them, as log-chances: softmax of (unary + edge_weight x edge) / scale.
fn chancesUnder(c: Candidates, r: usize, p: Policy, unary: *const [lp_top]f64, edge: *const [lp_top]f64, out: *[lp_top]f64) void {
    const w = ids(c, r);
    var most: f64 = -std.math.inf(f64);
    for (0..c.topk) |j| {
        out[j] = drawn(c, r, w[2 + j], unary[j] + p.edge_weight * edge[j]) / p.scale;
        most = @max(most, out[j]);
    }
    var total: f64 = 0;
    for (out[0..c.topk]) |v| total += @exp(v - most);
    const log_total = most + @log(total);
    for (out[0..c.topk]) |*v| v.* -= log_total;
}

/// Row `r`'s candidates after `pred` as the default policy's log-chances.
fn chancesAfter(c: Candidates, r: usize, pred: u32, out: *[lp_top]f64) void {
    var unary: [lp_top]f64 = undefined;
    var edge: [lp_top]f64 = undefined;
    scoresAfter(c, r, pred, &unary, &edge);
    chancesUnder(c, r, policies[0], &unary, &edge, out);
}

/// Up to `most` nodes best first by `p`'s path cost: the open node of least cost next (ties by parent, token, row); the
/// count grown. Each node's `own` is its log-chance under the default policy, whatever `p` ranks by.
pub fn grow(c: Candidates, pending: u32, rows: usize, most: usize, p: Policy, tokens: []u32, parents: []i32, own: []f64) usize {
    const Open = struct { cost: f64, parent: i32, token: u32, row: usize, own: f64 };
    var open: [branch * (engine.most_window + 1)]Open = undefined;
    var n_open: usize = 0;
    var count: usize = 0;
    var pred = pending;
    var parent: i32 = -1;
    var row: usize = 0;
    var path: f64 = 0;
    while (true) {
        // the node just taken (or the pending token) opens its best children on the next row
        if (row < rows) {
            var unary: [lp_top]f64 = undefined;
            var edge: [lp_top]f64 = undefined;
            scoresAfter(c, row, pred, &unary, &edge);
            var lc: [lp_top]f64 = undefined;
            chancesUnder(c, row, p, &unary, &edge, &lc);
            var base: [lp_top]f64 = undefined;
            if (std.meta.eql(p, policies[0])) base = lc else chancesUnder(c, row, policies[0], &unary, &edge, &base);
            var order: [lp_top]usize = undefined;
            for (order[0..c.topk], 0..) |*o, i| o.* = i;
            std.mem.sort(usize, order[0..c.topk], &lc, struct {
                fn more(v: *const [lp_top]f64, a: usize, b: usize) bool {
                    return v[a] > v[b] or (v[a] == v[b] and a < b);
                }
            }.more);
            const w = ids(c, row);
            for (order[0..@min(branch, c.topk)]) |j| {
                if (!std.math.isFinite(lc[j]) or n_open == open.len) break;
                open[n_open] = .{ .cost = path - lc[j], .parent = parent, .token = w[2 + j], .row = row, .own = base[j] };
                n_open += 1;
            }
        }
        if (count == @min(most, tokens.len) or n_open == 0) break;
        var best: usize = 0;
        for (open[1..n_open], 1..) |o, i| {
            const b = open[best];
            if (o.cost < b.cost or (o.cost == b.cost and (o.parent < b.parent or (o.parent == b.parent and (o.token < b.token or (o.token == b.token and o.row < b.row)))))) best = i;
        }
        const node = open[best];
        open[best] = open[n_open - 1];
        n_open -= 1;
        tokens[count] = node.token;
        parents[count] = node.parent;
        own[count] = node.own;
        pred = node.token;
        parent = @intCast(count);
        row = node.row + 1;
        path = node.cost;
        count += 1;
    }
    return count;
}

/// TF_NO_REPLAY=1: every tree ranks by the default policy (for comparing).
pub fn replaying() bool {
    return platform.c.getenv("TF_NO_REPLAY") == null;
}

/// The drafts a tree keeps along `follow`, the tokens the target chose after the tree's pending one.
fn walk(tokens: []const u32, parents: []const i32, follow: []const u32) usize {
    var at: i32 = -1;
    var kept: usize = 0;
    next: while (kept < follow.len) {
        for (tokens, parents, 0..) |t, p, i| {
            if (p == at and t == follow[kept]) {
                at = @intCast(i);
                kept += 1;
                continue :next;
            }
        }
        break;
    }
    return kept;
}

/// A greedy stream's last rounds replayed against the reply as it lands. Each policy's tree, grown from a round's own
/// candidates, is walked along the tokens the target went on to choose, and the policy whose trees would have kept
/// the most drafts ranks the stream's next trees. The target chooses the same tokens whatever tree carried them, so
/// every policy is scored on the same reply. A stream's open resets it: nothing in it outlives the stream.
pub const Replay = struct {
    pub const rounds = 16;
    pub const max_rank = 256;
    const max_rows = 15;
    /// Rounds replayed before another policy can rank a tree, and the drafts by which it must lead the default.
    const settle = 3;
    const lead = 0.5;
    /// Rounds a call replays at most: the host's replay stays inside the frame it overlaps.
    const per_call = 2;

    position: [rounds]u64 = undefined, // each round's first draft's position: its pending token sits just before
    pending: [rounds]u32 = undefined,
    rows: [rounds]usize = undefined,
    words: [rounds][max_rows * lp_words]u32 = undefined,
    projected: [rounds][max_rows * max_rank]u16 = undefined,
    recorded: usize = 0,
    replayed: usize = 0, // rounds walked or passed over; the oldest round still held is this one
    walked: usize = 0,
    kept: [policies.len]f64 = @splat(0),

    pub fn reset(r: *Replay) void {
        r.recorded = 0;
        r.replayed = 0;
        r.walked = 0;
        r.kept = @splat(0);
    }

    /// A round's candidates kept for its replay; a round still unreplayed after `rounds` more is passed over.
    pub fn record(r: *Replay, c: Candidates, pending: u32, position: u64, rows: usize) void {
        if (c.rank > max_rank or c.topk > lp_top or rows > max_rows or position == 0) return;
        if (r.recorded - r.replayed == rounds) r.replayed += 1;
        const slot = r.recorded % rounds;
        r.position[slot] = position;
        r.pending[slot] = pending;
        r.rows[slot] = rows;
        @memcpy(std.mem.sliceAsBytes(r.words[slot][0 .. rows * lp_words]), c.words[0 .. rows * lp_words * 4]);
        @memcpy(std.mem.sliceAsBytes(r.projected[slot][0 .. rows * c.rank]), c.projected[0 .. rows * c.rank * 2]);
        r.recorded += 1;
    }

    /// The oldest held rounds whose rows' tokens have all landed in `tokens` (the stream's context), up to `per_call`,
    /// through each policy's tree of up to `most` nodes. `base` gives the selector's codebooks and widths.
    pub fn replay(r: *Replay, base: Candidates, tokens: []const u32, most: usize) void {
        const stop = r.walked + per_call;
        while (r.replayed < r.recorded and r.walked < stop) {
            const slot = r.replayed % rounds;
            const at: usize = @intCast(r.position[slot]);
            const rows = r.rows[slot];
            if (tokens.len < at + rows) return;
            r.replayed += 1;
            if (tokens[at - 1] != r.pending[slot]) continue;
            var c = base;
            c.words = std.mem.sliceAsBytes(r.words[slot][0 .. rows * lp_words]);
            c.projected = std.mem.sliceAsBytes(r.projected[slot][0 .. rows * base.rank]);
            c.sampling = null;
            c.position = at;
            for (policies, &r.kept) |p, *k| {
                var held: [engine.most_window]u32 = undefined;
                var parents: [engine.most_window]i32 = undefined;
                var own: [engine.most_window]f64 = undefined;
                const n = grow(c, r.pending[slot], rows, @min(most, held.len), p, &held, &parents, &own);
                k.* += @floatFromInt(walk(held[0..n], parents[0..n], tokens[at..][0..rows]));
            }
            r.walked += 1;
        }
    }

    /// The policy the stream's next tree ranks by: the default until `settle` rounds have replayed, then the policy
    /// whose trees kept the most, when that leads the default's by `lead` drafts.
    pub fn choice(r: *const Replay) Policy {
        if (r.walked < settle) return policies[0];
        var best: usize = 0;
        for (r.kept[1..], 1..) |k, i| {
            if (k > r.kept[best] and k >= r.kept[0] + lead) best = i;
        }
        return policies[best];
    }
};

test "landing: a node is offered when its parent's row is kept, and landed when its own is" {
    var l: Landing = .{};
    // nodes 0 and 1 under the pending row, node 2 under node 0; the window kept rows 0, 1 (node 0) and 3 (node 2)
    l.observeTree(false, &.{ -1, -1, 0 }, &.{ -0.1, -1.2, -0.3 }, 3, &.{ 0, 1, 3 });
    try std.testing.expectEqual(@as(f64, 2), l.counts[0][0][0]); // nodes 0 and 2, both landed
    try std.testing.expectEqual(@as(f64, 2), l.counts[0][0][1]);
    try std.testing.expectEqual(@as(f64, 1), l.counts[0][2][0]); // node 1 offered, not landed
    try std.testing.expectEqual(@as(f64, 0), l.counts[0][2][1]);
    try std.testing.expectEqual(l.counts[0], l.counts[1]); // greedy rounds count for sampled streams too
    l.observeTree(true, &.{-1}, &.{-0.1}, 1, &.{0});
    try std.testing.expect(l.chance(true, -0.1) < l.chance(false, -0.1));
    try std.testing.expectApproxEqAbs((2 + 2 * @exp(@as(f64, -0.1))) / 4, l.chance(false, -0.1), 1e-12);
}

test "best first: a node's children follow it, the cheapest open node first, chances that sum to one a row" {
    // two rows, four candidates each, no selector (rank 1, zero codebooks): unary logits alone order the tree
    const rank = 1;
    var words: [2 * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    const unary = [2][4]f32{ .{ 3, 1, 0, -1 }, .{ 2, 2, -5, -5 } };
    for (0..2) |r| for (0..4) |j| {
        w[r * lp_words + 2 + j] = @intCast(10 * r + j);
        w[r * lp_words + 2 + lp_top + j] = @bitCast(unary[r][j]);
    };
    const zeros: [32]u16 = @splat(0);
    const c: Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..2]), .topk = 4, .rank = rank, .predecessor = &zeros, .successor = &zeros };
    var lc: [lp_top]f64 = undefined;
    chancesAfter(c, 0, 0, &lc);
    var total: f64 = 0;
    for (lc[0..4]) |v| total += @exp(v);
    try std.testing.expectApproxEqAbs(@as(f64, 1), total, 1e-12);
    var tokens: [6]u32 = undefined;
    var parents: [6]i32 = undefined;
    var own: [6]f64 = undefined;
    const n = grow(c, 5, 2, 6, policies[0], &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, 6), n);
    try std.testing.expectEqual(@as(u32, 0), tokens[0]); // row 0's best first
    try std.testing.expectEqual(@as(i32, -1), parents[0]);
    for (parents[0..n], 0..) |p, i| try std.testing.expect(p < @as(i32, @intCast(i)));
    for (tokens[0..n], parents[0..n]) |t, p| try std.testing.expectEqual(p >= 0, t >= 10); // row 1's ids under row 0's
    // another policy ranks by its own scale, yet each node's own chance stays the default policy's
    var flat_tokens: [6]u32 = undefined;
    var flat_parents: [6]i32 = undefined;
    var flat_own: [6]f64 = undefined;
    _ = grow(c, 5, 2, 6, .{ .edge_weight = 0.6, .scale = 4.0 }, &flat_tokens, &flat_parents, &flat_own);
    try std.testing.expectEqual(lc[0], flat_own[0]);
}

test "replay: a round is walked once its rows' tokens land, through every policy's tree" {
    const rank = 1;
    var words: [2 * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    const unary = [2][4]f32{ .{ 3, 1, 0, -1 }, .{ 2, 2, -5, -5 } };
    for (0..2) |r| for (0..4) |j| {
        w[r * lp_words + 2 + j] = @intCast(10 * r + j);
        w[r * lp_words + 2 + lp_top + j] = @bitCast(unary[r][j]);
    };
    const zeros: [32]u16 = @splat(0);
    const c: Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..2]), .topk = 4, .rank = rank, .predecessor = &zeros, .successor = &zeros };
    const r = try std.testing.allocator.create(Replay);
    defer std.testing.allocator.destroy(r);
    r.* = .{};
    r.record(c, 5, 3, 2); // pending 5 at position 2, its drafts' rows at 3 and 4
    r.replay(c, &.{ 7, 7, 5, 0 }, 6); // the second row's token has not landed
    try std.testing.expectEqual(@as(usize, 0), r.walked);
    r.replay(c, &.{ 7, 7, 5, 0, 10 }, 6);
    try std.testing.expectEqual(@as(usize, 1), r.walked);
    for (r.kept) |k| try std.testing.expectEqual(@as(f64, 2), k); // every tree holds 0, then 10 under it
    try std.testing.expect(std.meta.eql(policies[0], r.choice())); // not settled
    r.record(c, 9, 3, 2);
    r.replay(c, &.{ 7, 7, 5, 0, 10 }, 6); // a context that no longer holds the round's pending token: passed over
    try std.testing.expectEqual(@as(usize, 1), r.walked);
    try std.testing.expectEqual(r.recorded, r.replayed);
    r.reset();
    try std.testing.expectEqual(@as(usize, 0), r.recorded);
}
