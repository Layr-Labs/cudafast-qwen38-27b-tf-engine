// draw 51
//! A hold's drafts from its rows' candidates: a chain, the planner's lanes, or a best-first tree as the 27B engine's.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

/// Children a node at most, the selector's weight, the softmax scale, and the nodes a tree takes by default.
/// Each is env-tunable once (errors fall back): TF_TREE_BRANCH clamped to 1..max_branch, TF_EDGE_WEIGHT,
/// TF_SCALE, TF_NOISE_WEIGHT.
const default_branch = 12;
const max_branch = 24;
const default_edge = 0.6;
const default_scale = 1.5;
/// A sampled stream's candidates carry the draw's own Gumbel noise at this weight (the 27B engine's draft_tree).
const default_noise = 0.7;
const default_nodes = 15;

/// Children a node at most: TF_TREE_BRANCH, 1..max_branch (a node's row offers at most lp_top anyway).
fn branchN() usize {
    const S = struct {
        var v: ?usize = null;
    };
    if (S.v == null) {
        const n = if (platform.c.getenv("TF_TREE_BRANCH")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_branch else default_branch;
        S.v = @min(@max(n, 1), max_branch);
    }
    return S.v.?;
}

/// The device tree's (cuda_dtree.zig) view of the env-tuned values, the same ones grow() reads.
pub fn branchPub() usize {
    return branchN();
}
pub fn edgeWeightPub() f64 {
    return edgeWeight();
}
pub fn scalePub() f64 {
    return scaleVal();
}

/// TF_EDGE_WEIGHT: the selector projection's weight in a candidate's score.
fn edgeWeight() f64 {
    const S = struct {
        var v: ?f64 = null;
    };
    if (S.v == null) S.v = if (platform.c.getenv("TF_EDGE_WEIGHT")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch default_edge else default_edge;
    return S.v.?;
}

/// TF_SCALE: the softmax scale of a candidate's score.
fn scaleVal() f64 {
    const S = struct {
        var v: ?f64 = null;
    };
    if (S.v == null) S.v = if (platform.c.getenv("TF_SCALE")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch default_scale else default_scale;
    return S.v.?;
}

/// TF_NOISE_WEIGHT: the draw's Gumbel noise weight in a sampled stream's candidates' scores.
fn noiseWeight() f64 {
    const S = struct {
        var v: ?f64 = null;
    };
    if (S.v == null) S.v = if (platform.c.getenv("TF_NOISE_WEIGHT")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch default_noise else default_noise;
    return S.v.?;
}
const lp_top = 20;
const lp_words = 3 + 2 * lp_top;

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
    return logit / @max(s.temperature, 1e-6) + noiseWeight() * -@log(-@log(@as(f64, u)));
}

/// TF_R4LOG=1: per-visit candidate-distribution telemetry on stderr (default off; offline tree search's data).
pub fn r4log() bool {
    const S = struct {
        var on: ?bool = null;
    };
    if (S.on == null) S.on = platform.c.getenv("TF_R4LOG") != null;
    return S.on.?;
}

/// One visit's row header, once per row per grow() sweep: the row's candidate ids and unary logits.
var r4_seen: [fast_rows]bool = @splat(false);

fn hexList(buf: []u8, i: *usize, first: bool, bits: u64) void {
    const digits = "0123456789abcdef";
    if (!first and i.* < buf.len) {
        buf[i.*] = ',';
        i.* += 1;
    }
    var tmp: [16]u8 = undefined;
    var n: usize = 0;
    var v = bits;
    if (v == 0) {
        tmp[0] = '0';
        n = 1;
    }
    while (v > 0) : (v /= 16) {
        tmp[n] = digits[@intCast(v % 16)];
        n += 1;
    }
    var k: usize = 0;
    while (k < n and i.* < buf.len) : (k += 1) {
        buf[i.*] = tmp[n - 1 - k];
        i.* += 1;
    }
}

fn decList(buf: []u8, i: *usize, first: bool, v: i64) void {
    var tmp: [20]u8 = undefined;
    var n: usize = 0;
    var x = v;
    const neg = x < 0;
    if (neg) x = -x;
    if (x == 0) {
        tmp[0] = '0';
        n = 1;
    }
    while (x > 0) : (x = @divTrunc(x, 10)) {
        tmp[n] = @intCast('0' + @as(u8, @intCast(@rem(x, 10))));
        n += 1;
    }
    if (!first and i.* < buf.len) {
        buf[i.*] = ',';
        i.* += 1;
    }
    if (neg and i.* < buf.len) {
        buf[i.*] = '-';
        i.* += 1;
    }
    var k: usize = 0;
    while (k < n and i.* < buf.len) : (k += 1) {
        buf[i.*] = tmp[n - 1 - k];
        i.* += 1;
    }
}

/// R4R: a row's candidate ids and unary logits (the pred-independent halves of the scores).
fn dumpRowBits(c: Candidates, r: usize) void {
    const w = ids(c, r);
    var buf: [1536]u8 = undefined;
    var i: usize = 0;
    const head = std.fmt.bufPrint(buf[i..], "R4R r={d} k={d} ids=", .{ r, c.topk }) catch return;
    i += head.len;
    for (w[2..][0..c.topk], 0..) |t, j| decList(&buf, &i, j == 0, t);
    const mid = std.fmt.bufPrint(buf[i..], " u=", .{}) catch return;
    i += mid.len;
    for (w[2 + lp_top ..][0..c.topk], 0..) |u, j| {
        const bits: u32 = @bitCast(u);
        hexList(&buf, &i, j == 0, bits);
    }
    buf[i] = '\n';
    i += 1;
    std.debug.print("{s}", .{buf[0..i]});
}

/// R4V: one (row, pred) visit: the selector-edge vector before the weight, the final log-chances after.
fn dumpVisitBits(c: Candidates, r: usize, pred: u32, edge: *const [lp_top]f64, out: *const [lp_top]f64) void {
    if (!r4_seen[r]) {
        r4_seen[r] = true;
        dumpRowBits(c, r);
    }
    var buf: [2560]u8 = undefined;
    var i: usize = 0;
    const head = std.fmt.bufPrint(buf[i..], "R4V r={d} p={d} e=", .{ r, pred }) catch return;
    i += head.len;
    for (edge[0..c.topk], 0..) |v, j| hexList(&buf, &i, j == 0, @bitCast(v));
    const mid = std.fmt.bufPrint(buf[i..], " l=", .{}) catch return;
    i += mid.len;
    for (out[0..c.topk], 0..) |v, j| hexList(&buf, &i, j == 0, @bitCast(v));
    buf[i] = '\n';
    i += 1;
    std.debug.print("{s}", .{buf[0..i]});
}

/// R4T: a verified round's offered tree (parents in node order, own log-chances, tokens) and the rows it kept
/// (row 0 is the pending token; node i sits at row i + 1). The offline tree search's outcome labels.
pub fn dumpTree(parents: []const i32, own: []const f64, tokens: []const u32, offered: usize, kept: []const u32, sampled: bool) void {
    var buf: [4096]u8 = undefined;
    var i: usize = 0;
    const head = std.fmt.bufPrint(buf[i..], "R4T n={d} off={d} sp={d} par=", .{ parents.len, offered, @intFromBool(sampled) }) catch return;
    i += head.len;
    for (parents, 0..) |p, j| decList(&buf, &i, j == 0, p);
    const mid = std.fmt.bufPrint(buf[i..], " own=", .{}) catch return;
    i += mid.len;
    for (own, 0..) |o, j| hexList(&buf, &i, j == 0, @bitCast(o));
    const mid2 = std.fmt.bufPrint(buf[i..], " tok=", .{}) catch return;
    i += mid2.len;
    for (tokens, 0..) |t, j| decList(&buf, &i, j == 0, t);
    const mid3 = std.fmt.bufPrint(buf[i..], " kept=", .{}) catch return;
    i += mid3.len;
    for (kept, 0..) |k, j| decList(&buf, &i, j == 0, k);
    buf[i] = '\n';
    i += 1;
    std.debug.print("{s}", .{buf[0..i]});
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

/// TF_TREE_FAST=0: grow() converts each candidate's successor row anew for every expansion (the original path).
fn treeFast() bool {
    const S = struct {
        var on: ?bool = null;
    };
    if (S.on == null) S.on = if (platform.c.getenv("TF_TREE_FAST")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    return S.on.?;
}

/// TF_TREE_CHECK=1: every fast expansion also runs chancesAfter and reports any bit that differs.
fn treeCheck() bool {
    const S = struct {
        var on: ?bool = null;
    };
    if (S.on == null) S.on = platform.c.getenv("TF_TREE_CHECK") != null;
    return S.on.?;
}

const fast_rank = 256;
const fast_rows = 32;
const lanes_f64 = 4;
const vec_n = lp_top / lanes_f64;
const V = @Vector(lanes_f64, f64);

/// A grow() call's rows in f64, converted once a row: each row's projection, and its candidates' successor rows laid
/// out k-major (the lp_top candidates of one k side by side), so an expansion sums every candidate's edge at once,
/// each candidate's own sum still over k in order. The products are exact in f64 (bf16 x bf16 x bf16), so the edges
/// are chancesAfter's bits whether the adds fuse or not.
const RowCache = struct {
    filled: [fast_rows]bool = @splat(false),
    hp: [fast_rows][fast_rank]f64 = undefined,
    succ: [fast_rows][fast_rank][vec_n]V = undefined,
};
var row_cache: RowCache = .{};

fn cacheRow(c: Candidates, r: usize) void {
    if (row_cache.filled[r]) return;
    const w = ids(c, r);
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    for (0..c.rank) |k| row_cache.hp[r][k] = bf16(hp[k]);
    var cols: [lp_top]f64 = undefined;
    for (0..c.rank) |k| {
        for (&cols, 0..) |*x, j| x.* = if (j < c.topk) bf16(c.successor[w[2 + j] * c.rank + k]) else 0;
        for (0..vec_n) |v| row_cache.succ[r][k][v] = cols[v * lanes_f64 ..][0..lanes_f64].*;
    }
    row_cache.filled[r] = true;
}

/// chancesAfter through the row cache: the same edges, the same unary and softmax arithmetic.
fn chancesAfterFast(c: Candidates, r: usize, pred: u32, out: *[lp_top]f64) void {
    cacheRow(c, r);
    const w = ids(c, r);
    const pr = c.predecessor[pred * c.rank ..][0..c.rank];
    var acc: [vec_n]V = @splat(@splat(0));
    for (0..c.rank) |k| {
        const ph: V = @splat(bf16(pr[k]) * row_cache.hp[r][k]);
        for (&acc, row_cache.succ[r][k]) |*a, sv| a.* += ph * sv;
    }
    var edge: [lp_top]f64 = undefined;
    for (0..vec_n) |v| edge[v * lanes_f64 ..][0..lanes_f64].* = acc[v];
    var most: f64 = -std.math.inf(f64);
    for (0..c.topk) |j| {
        out[j] = drawn(c, r, w[2 + j], @as(f64, @as(f32, @bitCast(w[2 + lp_top + j]))) + edgeWeight() * edge[j]) / scaleVal();
        most = @max(most, out[j]);
    }
    var total: f64 = 0;
    for (out[0..c.topk]) |v| total += @exp(v - most);
    const log_total = most + @log(total);
    for (out[0..c.topk]) |*v| v.* -= log_total;
    if (r4log()) dumpVisitBits(c, r, pred, &edge, out);
}

/// Row `r`'s candidates after `pred` as log-chances: softmax of (unary + edge x selector) / scale.
fn chancesAfter(c: Candidates, r: usize, pred: u32, out: *[lp_top]f64) void {
    const w = ids(c, r);
    const hp = std.mem.bytesAsSlice(u16, c.projected[r * c.rank * 2 ..][0 .. c.rank * 2]);
    var most: f64 = -std.math.inf(f64);
    // every candidate's edge sum at once, each still over k in order (its own chain): the same f64 bits as one by one
    var edge: [lp_top]f64 = @splat(0);
    const pr = c.predecessor[pred * c.rank ..][0..c.rank];
    var succ: [lp_top][]const u16 = undefined;
    for (succ[0..c.topk], 0..) |*row, j| row.* = c.successor[w[2 + j] * c.rank ..][0..c.rank];
    for (0..c.rank) |k| {
        const ph = bf16(pr[k]) * bf16(hp[k]);
        for (edge[0..c.topk], succ[0..c.topk]) |*e, row| e.* += ph * bf16(row[k]);
    }
    for (0..c.topk) |j| {
        out[j] = drawn(c, r, w[2 + j], @as(f64, @as(f32, @bitCast(w[2 + lp_top + j]))) + edgeWeight() * edge[j]) / scaleVal();
        most = @max(most, out[j]);
    }
    var total: f64 = 0;
    for (out[0..c.topk]) |v| total += @exp(v - most);
    const log_total = most + @log(total);
    for (out[0..c.topk]) |*v| v.* -= log_total;
    if (r4log()) dumpVisitBits(c, r, pred, &edge, out);
}

/// Up to `most` nodes best first: the open node of least path cost next (ties by parent, token, row); the count grown.
/// `tune` (default zeroed = the crown behavior bit for bit): `trust` blends each candidate's path term from
/// its own log-chance toward the Landing tables' observed landing chance (offline search's tau), and `lam`
/// adds lam x row to a node's cost (a shallow bias; the offline search's depth-vs-breadth knob).
pub const Tune = struct {
    trust: f64 = 0,
    lam: f64 = 0,
    ctx: ?*const anyopaque = null,
    chance_fn: ?*const fn (?*const anyopaque, f64) f64 = null,
};

/// The Landing tables behind a trust tune: the caller pins the stream's sampled flag.
pub const TrustCtx = struct { l: *const Landing, sampled: bool };

pub fn landingChance(ctx: ?*const anyopaque, own: f64) f64 {
    const t: *const TrustCtx = @ptrCast(@alignCast(ctx.?));
    return t.l.chance(t.sampled, own);
}

pub fn grow(c: Candidates, pending: u32, rows: usize, most: usize, tokens: []u32, parents: []i32, own: []f64, tune: Tune) usize {
    const Open = struct { cost: f64, parent: i32, token: u32, row: usize, own: f64 };
    const branch = branchN();
    // the open list is sized for the largest branch the env may select; growth checks n_open == open.len
    var open: [max_branch * (engine.most_window + 1)]Open = undefined;
    var n_open: usize = 0;
    var count: usize = 0;
    var pred = pending;
    var parent: i32 = -1;
    var row: usize = 0;
    var path: f64 = 0;
    const fast = treeFast() and rows <= fast_rows and c.rank == fast_rank and c.topk <= lp_top;
    if (fast) row_cache.filled = @splat(false);
    if (r4log()) r4_seen = @splat(false);
    while (true) {
        // the node just taken (or the pending token) opens its best children on the next row
        if (row < rows) {
            var lc: [lp_top]f64 = undefined;
            if (fast) chancesAfterFast(c, row, pred, &lc) else chancesAfter(c, row, pred, &lc);
            if (fast and treeCheck()) {
                var ref: [lp_top]f64 = undefined;
                chancesAfter(c, row, pred, &ref);
                for (lc[0..c.topk], ref[0..c.topk]) |x, y| if (@as(u64, @bitCast(x)) != @as(u64, @bitCast(y))) {
                    std.debug.print("TF_TREE_CHECK mismatch row {d}: {e} vs {e}\n", .{ row, x, y });
                };
            }
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
                var cost = path - lc[j];
                if (tune.trust > 0 and tune.chance_fn != null) {
                    const pm = (1 - tune.trust) * @exp(lc[j]) + tune.trust * @max(tune.chance_fn.?(tune.ctx, lc[j]), 1e-300);
                    cost = path - @log(@max(pm, 1e-300));
                }
                if (tune.lam != 0) cost += tune.lam * @as(f64, @floatFromInt(row));
                open[n_open] = .{ .cost = cost, .parent = parent, .token = w[2 + j], .row = row, .own = lc[j] };
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
    const n = grow(c, 5, 2, 6, &tokens, &parents, &own, .{});
    try std.testing.expectEqual(@as(usize, 6), n);
    try std.testing.expectEqual(@as(u32, 0), tokens[0]); // row 0's best first
    try std.testing.expectEqual(@as(i32, -1), parents[0]);
    for (parents[0..n], 0..) |p, i| try std.testing.expect(p < @as(i32, @intCast(i)));
    for (tokens[0..n], parents[0..n]) |t, p| try std.testing.expectEqual(p >= 0, t >= 10); // row 1's ids under row 0's
}
