//! A hold's drafts from its rows' candidates: a chain, the planner's lanes, or a best-first tree as the 27B engine's.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

/// Children a node, the nodes the record's tree takes (a 16-row window), and the most a tree takes (32 rows).
const branch = 12;
/// A sampled stream's candidates carry the draw's own Gumbel noise at this weight (the 27B engine's draft_tree).
const noise_weight = 0.7;
const record_nodes = 15;
const most_nodes = 31;
const lp_top = 20;
const lp_words = 3 + 2 * lp_top;
/// Selector ranks whose pred x selector products `scoresAfter` keeps on the stack (larger ranks: the plain loop).
const max_hoisted_rank = 1024;

/// How a tree ranks a row's candidates: softmax of (unary + edge_weight x selector) / scale, and the nodes it takes.
pub const Policy = struct { edge_weight: f64, scale: f64, nodes: usize };

/// The trees a stream's replay compares: the 27B engine's ranking at 31 nodes (the default: sampled streams and each
/// stream's first rounds take it) and at the record's 15. The 15-node tree is the 31-node tree's first 15 nodes, so
/// the replay weighs only what the last 16 nodes keep against what their wider window costs. Every node's `own`
/// chance is under this ranking, so the planner's landing table keeps one meaning.
pub const policies = [_]Policy{
    .{ .edge_weight = 0.6, .scale = 1.5, .nodes = most_nodes },
    .{ .edge_weight = 0.6, .scale = 1.5, .nodes = record_nodes },
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

/// A grown tree's nodes at most, whichever policy grows it: 0 when the drafter drafts chains and lane trees
/// (TF_NO_TREES, TF_DRAFT_NODES=0).
pub fn nodes() usize {
    if (platform.c.getenv("TF_NO_TREES") != null) return 0;
    const n = if (platform.c.getenv("TF_DRAFT_NODES")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch most_nodes else most_nodes;
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

/// One round's selector edges by (row, predecessor): the trees a replay grows from one round's candidates share them.
pub const Edges = struct {
    const room = 96;
    keys: [room]u64 = undefined,
    values: [room][lp_top]f64 = undefined,
    n: usize = 0,
};

/// Row `r`'s candidates after `pred`: each one's unary logit and its selector edge (pred . projected . successor).
fn scoresAfter(c: Candidates, r: usize, pred: u32, unary: *[lp_top]f64, edge: *[lp_top]f64, edges: ?*Edges) void {
    const w = ids(c, r);
    for (0..c.topk) |j| unary[j] = @as(f32, @bitCast(w[2 + lp_top + j]));
    const key = @as(u64, r) << 32 | pred;
    if (edges) |x| for (x.keys[0..x.n], x.values[0..x.n]) |k, v| {
        if (k == key) {
            edge.* = v;
            return;
        }
    };
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    // pred x selector once a row, each product as the candidates' loop formed it ((pred * hp) * successor), so a
    // candidate costs rank multiplies, not 2 x rank
    var ph_buf: [max_hoisted_rank]f64 = undefined;
    const ph = ph_buf[0..@min(c.rank, max_hoisted_rank)];
    if (c.rank <= max_hoisted_rank) for (ph, 0..) |*v, k| {
        v.* = bf16(c.predecessor[pred * c.rank + k]) * bf16(hp[k]);
    };
    for (0..c.topk) |j| {
        var e: f64 = 0;
        const succ = c.successor[w[2 + j] * c.rank ..][0..c.rank];
        if (c.rank <= max_hoisted_rank) {
            for (ph, succ) |v, x| e += v * bf16(x);
        } else for (0..c.rank) |k| {
            e += bf16(c.predecessor[pred * c.rank + k]) * bf16(hp[k]) * bf16(succ[k]);
        }
        edge[j] = e;
    }
    if (edges) |x| if (x.n < Edges.room) {
        x.keys[x.n] = key;
        x.values[x.n] = edge.*;
        x.n += 1;
    };
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
    scoresAfter(c, r, pred, &unary, &edge, null);
    chancesUnder(c, r, policies[0], &unary, &edge, out);
}

/// Up to `most` nodes best first by `p`'s path cost: the open node of least cost next (ties by parent, token, row); the
/// count grown. Each node's `own` is its log-chance under the default policy, whatever `p` ranks by. `edges` (null:
/// none) shares the selector's edges among trees grown from the same candidates.
pub fn grow(c: Candidates, pending: u32, rows: usize, most: usize, p: Policy, edges: ?*Edges, tokens: []u32, parents: []i32, own: []f64) usize {
    const default_rank = p.edge_weight == policies[0].edge_weight and p.scale == policies[0].scale;
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
            scoresAfter(c, row, pred, &unary, &edge, edges);
            var lc: [lp_top]f64 = undefined;
            chancesUnder(c, row, p, &unary, &edge, &lc);
            var base: [lp_top]f64 = undefined;
            if (default_rank) base = lc else chancesUnder(c, row, policies[0], &unary, &edge, &base);
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
/// candidates, is walked along the tokens the target went on to choose and credited with the drafts it would have
/// kept; the policy with the most kept tokens a millisecond ranks the stream's next trees. The target chooses the same
/// tokens whatever tree carried them, so every policy is scored on the same reply. A round is scored once every
/// policy's walk has ended inside the tokens landed so far (or all its rows' tokens have landed), so no walk is cut
/// short. A stream's open resets it: nothing in it outlives the stream.
pub const Replay = struct {
    pub const rounds = 16;
    pub const max_rank = 256;
    const max_rows = 15;
    /// Rounds scored before another policy can take a tree, and the kept drafts by which it must beat the default.
    const settle = 4;
    const lead = 0.5;
    /// Rounds whose trees a call grows at most: the host's replay stays inside the frame it overlaps.
    const per_call = 2;

    position: [rounds]u64 = undefined, // each round's first draft's position: its pending token sits just before
    pending: [rounds]u32 = undefined,
    rows: [rounds]usize = undefined,
    words: [rounds][max_rows * lp_words]u32 = undefined,
    projected: [rounds][max_rows * max_rank]u16 = undefined,
    // each round's trees, grown at its first replay
    grown: [rounds]bool = undefined,
    scored: [rounds]bool = undefined,
    count: [rounds][policies.len]usize = undefined,
    tokens: [rounds][policies.len][most_nodes]u32 = undefined,
    parents: [rounds][policies.len][most_nodes]i32 = undefined,
    recorded: usize = 0,
    oldest: usize = 0, // the oldest round still held
    walked: usize = 0,
    kept: [policies.len]f64 = @splat(0),
    // each hold's round as the drafter sees it, hold to hold (the target's verify of its tree, the keep, the next
    // frame), timed by the policy that grew the tree
    held_at: ?u64 = null,
    held_by: usize = 0,
    holds: usize = 0,
    round_ms: [policies.len]f64 = @splat(0),
    timed: [policies.len]usize = @splat(0),

    pub fn reset(r: *Replay) void {
        r.recorded = 0;
        r.oldest = 0;
        r.walked = 0;
        r.kept = @splat(0);
        r.held_at = null;
        r.held_by = 0;
        r.holds = 0;
        r.round_ms = @splat(0);
        r.timed = @splat(0);
    }

    /// The hold from which each other policy grows `probes` trees in turn, so that its round is timed on this stream
    /// rather than priced from the load-time sweep (which times chain windows on an empty cache).
    const probe_from = 3;
    const probes = 2;

    /// The policy (an index into `policies`) the hold at `now_ns` grows. The hold closes the previous hold's round,
    /// timed to the policy that grew it (the stream's first round is not: it follows the prompt). `swept` prices each
    /// policy's round from the load-time sweep until the stream has timed two of its rounds.
    pub fn take(r: *Replay, now_ns: u64, swept: *const [policies.len]f64) usize {
        if (r.held_at) |t| if (r.holds >= 2 and now_ns > t) {
            r.round_ms[r.held_by] += @as(f64, @floatFromInt(now_ns - t)) / 1e6;
            r.timed[r.held_by] += 1;
        };
        var ms: [policies.len]f64 = undefined;
        for (&ms, swept, r.round_ms, r.timed) |*m, s, total, n| m.* = if (n >= 2) total / @as(f64, @floatFromInt(n)) else s;
        const probe = if (r.holds >= probe_from) (r.holds - probe_from) / probes + 1 else policies.len;
        const pick = if (probe < policies.len) probe else r.choice(&ms);
        r.held_at = now_ns;
        r.held_by = pick;
        r.holds += 1;
        return pick;
    }

    /// A round's candidates kept for its replay; a round still unscored after `rounds` more is passed over.
    pub fn record(r: *Replay, c: Candidates, pending: u32, position: u64, rows: usize) void {
        if (c.rank > max_rank or c.topk > lp_top or rows > max_rows or position == 0) return;
        if (r.recorded - r.oldest == rounds) r.oldest += 1;
        const slot = r.recorded % rounds;
        r.position[slot] = position;
        r.pending[slot] = pending;
        r.rows[slot] = rows;
        r.grown[slot] = false;
        r.scored[slot] = false;
        @memcpy(std.mem.sliceAsBytes(r.words[slot][0 .. rows * lp_words]), c.words[0 .. rows * lp_words * 4]);
        @memcpy(std.mem.sliceAsBytes(r.projected[slot][0 .. rows * c.rank]), c.projected[0 .. rows * c.rank * 2]);
        r.recorded += 1;
        r.advance();
    }

    /// The held rounds walked along the tokens that have landed in `tokens` (the stream's context), each policy's tree
    /// up to `most` nodes; at most `per_call` rounds' trees are grown a call. `base` gives the codebooks and widths.
    pub fn replay(r: *Replay, base: Candidates, tokens: []const u32, most: usize) void {
        var grew: usize = 0;
        var i = r.oldest;
        while (i < r.recorded) : (i += 1) {
            const slot = i % rounds;
            if (r.scored[slot]) continue;
            const at: usize = @intCast(r.position[slot]);
            if (tokens.len <= at) break; // neither this round's first draft nor any later round's has landed
            if (tokens[at - 1] != r.pending[slot]) { // a context that no longer holds the round: passed over
                r.scored[slot] = true;
                continue;
            }
            if (!r.grown[slot]) {
                if (grew == per_call) break;
                r.growTrees(slot, base, most);
                grew += 1;
            }
            const follow = tokens[at..][0..@min(tokens.len - at, r.rows[slot])];
            var kept: [policies.len]usize = undefined;
            var going = false;
            for (&kept, r.count[slot], &r.tokens[slot], &r.parents[slot]) |*k, n, *t, *p| {
                k.* = walk(t[0..n], p[0..n], follow);
                going = going or k.* == follow.len;
            }
            if (going and follow.len < r.rows[slot]) continue; // a walk may go on past the landed tokens
            for (&r.kept, kept) |*total, k| total.* += @floatFromInt(k);
            r.walked += 1;
            r.scored[slot] = true;
        }
        r.advance();
    }

    fn advance(r: *Replay) void {
        while (r.oldest < r.recorded and r.scored[r.oldest % rounds]) r.oldest += 1;
    }

    fn growTrees(r: *Replay, slot: usize, base: Candidates, most: usize) void {
        const rows = r.rows[slot];
        var c = base;
        c.words = std.mem.sliceAsBytes(r.words[slot][0 .. rows * lp_words]);
        c.projected = std.mem.sliceAsBytes(r.projected[slot][0 .. rows * base.rank]);
        c.sampling = null;
        c.position = r.position[slot];
        var edges: Edges = .{};
        for (policies, &r.count[slot], &r.tokens[slot], &r.parents[slot]) |p, *n, *t, *q| {
            var own: [most_nodes]f64 = undefined;
            n.* = grow(c, r.pending[slot], rows, @min(p.nodes, most), p, &edges, t, q, &own);
        }
        r.grown[slot] = true;
    }

    /// The policy (an index into `policies`) the stream's next tree takes: the default until `settle` rounds have
    /// scored, then the one with the most kept tokens a millisecond, if that beats the default's by `lead` drafts.
    /// `ms` is each policy's round (its verify window and the frame); a policy whose `ms` is not finite is never taken.
    pub fn choice(r: *const Replay, ms: *const [policies.len]f64) usize {
        if (r.walked < settle or !(ms[0] > 0 and std.math.isFinite(ms[0]))) return 0;
        const w: f64 = @floatFromInt(r.walked);
        const bar = (r.kept[0] + w + lead) / ms[0];
        var best: usize = 0;
        var rate = (r.kept[0] + w) / ms[0];
        for (r.kept[1..], ms[1..], 1..) |k, m, i| {
            if (!(m > 0 and std.math.isFinite(m))) continue;
            const v = (k + w) / m;
            if (v > rate and v >= bar) {
                best = i;
                rate = v;
            }
        }
        return best;
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
    const n = grow(c, 5, 2, 6, policies[0], null, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, 6), n);
    try std.testing.expectEqual(@as(u32, 0), tokens[0]); // row 0's best first
    try std.testing.expectEqual(@as(i32, -1), parents[0]);
    for (parents[0..n], 0..) |p, i| try std.testing.expect(p < @as(i32, @intCast(i)));
    for (tokens[0..n], parents[0..n]) |t, p| try std.testing.expectEqual(p >= 0, t >= 10); // row 1's ids under row 0's
    // another policy ranks by its own scale, yet each node's own chance stays the default policy's
    var flat_tokens: [6]u32 = undefined;
    var flat_parents: [6]i32 = undefined;
    var flat_own: [6]f64 = undefined;
    _ = grow(c, 5, 2, 6, .{ .edge_weight = 0.6, .scale = 4.0, .nodes = 6 }, null, &flat_tokens, &flat_parents, &flat_own);
    try std.testing.expectEqual(lc[0], flat_own[0]);
}

test "replay: a round scores once every policy's walk ends inside the landed tokens" {
    // two rows, four candidates each, no selector: the 31-node tree holds all 20 nodes, the 15-node tree the best 15
    // (3 then 12 is the 31-node tree's alone)
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
    r.replay(c, &.{ 7, 7, 5, 3 }, most_nodes); // both trees keep 3 and may keep a second draft: not scored yet
    try std.testing.expectEqual(@as(usize, 0), r.walked);
    r.replay(c, &.{ 7, 7, 5, 3, 12 }, most_nodes);
    try std.testing.expectEqual(@as(usize, 1), r.walked);
    try std.testing.expectEqualSlices(f64, &.{ 2, 1 }, &r.kept);
    // pending 12 at position 4, then 0 and 10: both trees keep both
    r.record(c, 12, 5, 2);
    r.replay(c, &.{ 7, 7, 5, 3, 12, 0 }, most_nodes);
    try std.testing.expectEqual(@as(usize, 1), r.walked);
    r.replay(c, &.{ 7, 7, 5, 3, 12, 0, 10 }, most_nodes);
    try std.testing.expectEqual(@as(usize, 2), r.walked);
    try std.testing.expectEqualSlices(f64, &.{ 4, 3 }, &r.kept);
    // a round recorded with pending 9 before position 7, where the context holds 10: passed over
    r.record(c, 9, 7, 2);
    r.replay(c, &.{ 7, 7, 5, 3, 12, 0, 10, 2, 9 }, most_nodes);
    try std.testing.expectEqual(@as(usize, 2), r.walked);
    try std.testing.expectEqual(r.recorded, r.oldest);
    var ms = [policies.len]f64{ 1.1, 1.0 };
    try std.testing.expectEqual(@as(usize, 0), r.choice(&ms)); // not settled
    r.walked = 6; // settled: 15 nodes when their cheaper round beats 31 nodes' kept tokens a ms by `lead`
    try std.testing.expectEqual(@as(usize, 0), r.choice(&ms)); // (4 + 6) / 1.1 against (3 + 6) / 1.0
    ms[0] = 1.3;
    try std.testing.expectEqual(@as(usize, 1), r.choice(&ms));
    ms[1] = std.math.inf(f64); // a window never measured is never taken
    try std.testing.expectEqual(@as(usize, 0), r.choice(&ms));
    r.reset();
    try std.testing.expectEqual(@as(usize, 0), r.recorded);
    // timed rounds: holds 0-2 take the default, 3 and 4 probe 15 nodes, then the choice prices both by their timed
    // rounds (the first round, after the prompt, is not timed): 31 nodes 100 ms a round, 15 nodes 80 ms
    const swept = [policies.len]f64{ 1.0, 1.0 };
    const ms_ns: u64 = 1_000_000;
    for ([_]u64{ 0, 100, 200, 300, 380 }, [_]usize{ 0, 0, 0, 1, 1 }) |t, want| try std.testing.expectEqual(want, r.take(t * ms_ns, &swept));
    r.walked = 6;
    r.kept = .{ 4, 3 }; // (4 + 6) / 100 against (3 + 6) / 80: 15 nodes, past the default's lead
    try std.testing.expectEqual(@as(usize, 1), r.take(460 * ms_ns, &swept));
    try std.testing.expectEqualSlices(f64, &.{ 200, 160 }, &r.round_ms);
}

test "hoisted selector: chancesAfter keeps the per-candidate formula's bits" {
    // five rows of six candidates over a 64-token vocabulary, rank-4 codebooks: every edge term non-zero
    const rank = 4;
    const rows = 5;
    const topk = 6;
    var words: [rows * lp_words * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (0..rows) |r| for (0..topk) |j| {
        w[r * lp_words + 2 + j] = @intCast((r * 11 + j * 7) % 64);
        w[r * lp_words + 2 + lp_top + j] = @bitCast(@as(f32, 3.0 - @as(f32, @floatFromInt(j)) * 0.5 + rnd.float(f32)));
    };
    var book: [2][64 * rank]u16 = undefined;
    for (&book) |*b| for (b) |*x| {
        x.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
    };
    var hp: [rows * rank]u16 = undefined;
    for (&hp) |*x| x.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
    const c: Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(&hp), .topk = topk, .rank = rank, .predecessor = &book[0], .successor = &book[1] };
    for (0..rows) |r| for ([_]u32{ 0, 3, 17, 63 }) |pred| {
        var got: [lp_top]f64 = undefined;
        chancesAfter(c, r, pred, &got);
        // the original loop, verbatim
        const wr = ids(c, r);
        var want: [lp_top]f64 = undefined;
        var most: f64 = -std.math.inf(f64);
        for (0..topk) |j| {
            var edge: f64 = 0;
            for (0..rank) |k| edge += bf16(c.predecessor[pred * rank + k]) * bf16(hp[r * rank + k]) * bf16(c.successor[wr[2 + j] * rank + k]);
            want[j] = drawn(c, r, wr[2 + j], @as(f64, @as(f32, @bitCast(wr[2 + lp_top + j]))) + policies[0].edge_weight * edge) / policies[0].scale;
            most = @max(most, want[j]);
        }
        var total: f64 = 0;
        for (want[0..topk]) |v| total += @exp(v - most);
        const log_total = most + @log(total);
        for (want[0..topk], got[0..topk]) |a, b| try std.testing.expectEqual(@as(u64, @bitCast(a - log_total)), @as(u64, @bitCast(b)));
    };
}
