//! Warm each captured launch shape, then measure every live width with a freshly reset cache.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
pub const profile = @import("cuda_startup_profile.zig");

pub const Seen = struct {
    keys: [E.most_verify + E.most_window]u32 = undefined,
    len: usize = 0,
    counts: [E.most_verify + E.most_window]u8 = @splat(0),

    pub fn contains(self: *const Seen, shape: E.Engine.Shape) bool {
        return std.mem.indexOfScalar(u32, self.keys[0..self.len], shape.key()) != null;
    }

    pub fn passes(self: *const Seen, shape: E.Engine.Shape) u8 {
        for (self.keys[0..self.len], 0..) |key, i| if (key == shape.key()) return self.counts[i];
        return 0;
    }

    pub fn mark(self: *Seen, shape: E.Engine.Shape) void {
        for (self.keys[0..self.len], 0..) |key, i| if (key == shape.key()) {
            self.counts[i] +|= 1;
            return;
        };
        std.debug.assert(self.len < self.keys.len);
        self.keys[self.len] = shape.key();
        self.counts[self.len] = 1;
        self.len += 1;
    }
};

/// `parents` (a window's rows, row 0 the pending one) as a tree whose depth-first register-slot plan (core/tree_slots)
/// peaks at `peak` live slots: a node whose light child is such a tree one slot shallower and whose heavy child is a
/// chain as long, 2^peak - 1 rows in all; the rows past it extend the last chain.
pub fn treeOfPeak(parents: []i32, peak: usize) void {
    var at: usize = 0;
    nest(parents, &at, -1, peak);
    while (at < parents.len) : (at += 1) parents[at] = @intCast(at - 1);
}

fn nest(parents: []i32, at: *usize, parent: i32, peak: usize) void {
    const node = at.*;
    parents[node] = parent;
    at.* += 1;
    if (peak <= 1) return;
    nest(parents, at, @intCast(node), peak - 1);
    var prev: i32 = @intCast(node);
    for (0..(@as(usize, 1) << @intCast(peak - 1)) - 1) |_| {
        parents[at.*] = prev;
        prev = @intCast(at.*);
        at.* += 1;
    }
}

/// Every register-slot variant a 16- and a 32-row tree window can take, captured once at load: a stream whose tree
/// changes shape mid-reply (a 31-node tree after 15-node ones) meets no forward capture inside its timed decode.
fn warmTrees(e: anytype, cache: anytype, ids: []const u32, picks: []const E.Draw) !void {
    if (comptime !@hasDecl(@TypeOf(e.*), "deltaScan")) return; // a calibration fixture has no tree windows
    for ([_]usize{ 16, 32 }) |rows| {
        if (rows > e.window_rows) continue;
        var peak: usize = 2;
        while ((@as(usize, 1) << @intCast(peak)) - 1 <= rows) : (peak += 1) {
            var parents: [E.most_window]i32 = undefined;
            treeOfPeak(parents[0..rows], peak);
            try cache.reset(e);
            try e.forward(&.{.{ .cache = cache, .rows = rows, .parents = parents[0..rows] }}, ids[0..rows], .{ .draws = picks[0..rows] });
        }
    }
}

pub fn time(io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen) !void {
    var phases: profile.Times = .{};
    return timed(io, e, d, cache, warmed, &phases);
}

fn timed(io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen, phases: *profile.Times) !void {
    const started = profile.now(io);
    var warm_ns: i96 = 0;
    const ids: [E.most_window]u32 = @splat(0);
    var picks: [E.most_window]E.Draw = undefined;
    for (&picks) |*pk| pk.* = .{ .rule = null, .position = 1 };
    var measured: [E.most_window]struct { rows: u32, ms: f64 } = undefined;
    for (1..e.window_rows + 1) |rows| {
        const shape = E.Engine.Shape.init(rows, 1, true, e.n_taps > 0, 1);
        while (warmed.passes(shape) < 2) {
            const warm_start = profile.now(io);
            try cache.reset(e);
            try e.forward(&.{.{ .cache = cache, .rows = rows }}, ids[0..rows], .{ .draws = picks[0..rows] });
            warmed.mark(shape);
            warm_ns += profile.elapsed(io, warm_start);
        }
        measured[rows - 1] = .{ .rows = @intCast(rows), .ms = try timeWidth(io, e, cache, rows, &ids, &picks) };
    }
    const trees_start = profile.now(io);
    try warmTrees(e, cache, &ids, &picks);
    warm_ns += profile.elapsed(io, trees_start);
    try cache.reset(e);
    for (measured[0..e.window_rows]) |m| d.setCost(m.rows, m.ms);
    phases.warmups += warm_ns;
    phases.calibration += profile.elapsed(io, started) - warm_ns;
}

/// One width's cost: the fastest of its timed runs, so one slow run cannot set the table.
fn timeWidth(io: std.Io, e: anytype, cache: anytype, rows: usize, ids: []const u32, picks: []const E.Draw) !f64 {
    var runs: [lanes.cost_rule.reps]f64 = undefined;
    for (&runs) |*entry| {
        try cache.reset(e);
        const t0 = std.Io.Clock.awake.now(io);
        try e.forward(&.{.{ .cache = cache, .rows = rows }}, ids[0..rows], .{ .draws = picks[0..rows] });
        const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        if (ms <= 0 or !std.math.isFinite(ms)) return error.InvalidWindowMeasurement;
        entry.* = ms;
    }
    return lanes.cost_rule.keepFastest(&runs);
}

/// One width's quick check against a kept table: the middle width, whose cost the window choice turns on.
pub fn probeWidth(io: std.Io, e: anytype, cache: anytype, rows: usize) !f64 {
    var ids: [E.most_window]u32 = @splat(0);
    var picks: [E.most_window]E.Draw = undefined;
    for (&picks) |*pk| pk.* = .{ .rule = null, .position = 1 };
    return timeWidth(io, e, cache, rows, ids[0..rows], picks[0..rows]);
}

pub fn prepare(gpa: std.mem.Allocator, io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen) !void {
    _ = try prepareTimed(gpa, io, e, d, cache, warmed);
}

pub fn prepareTimed(gpa: std.mem.Allocator, io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen) !profile.Times {
    return prepareMode(gpa, io, e, d, cache, warmed, true);
}

pub fn prepareMode(gpa: std.mem.Allocator, io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen, comptime measure: bool) !profile.Times {
    var phases: profile.Times = .{};
    if (cache.cap >= e.window_rows) {
        try run(io, e, d, cache, warmed, &phases, measure);
        return phases;
    }
    const started = profile.now(io);
    var temporary = try @TypeOf(cache.*).init(gpa, e, e.window_rows);
    errdefer temporary.deinit(gpa);
    phases.reserve += profile.elapsed(io, started);
    try run(io, e, d, &temporary, warmed, &phases, measure);
    const freeing = profile.now(io);
    temporary.deinit(gpa);
    phases.reserve += profile.elapsed(io, freeing);
    return phases;
}

fn run(io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen, phases: *profile.Times, comptime measure: bool) !void {
    if (measure) return timed(io, e, d, cache, warmed, phases);
    const started = profile.now(io);
    const ids: [E.most_window]u32 = @splat(0);
    var picks: [E.most_window]E.Draw = undefined;
    for (&picks) |*pk| pk.* = .{ .rule = null, .position = 1 };
    for (1..e.window_rows + 1) |rows| {
        const shape = E.Engine.Shape.init(rows, 1, true, e.n_taps > 0, 1);
        while (warmed.passes(shape) < 2) {
            try cache.reset(e);
            try e.forward(&.{.{ .cache = cache, .rows = rows }}, ids[0..rows], .{ .draws = picks[0..rows] });
            warmed.mark(shape);
        }
    }
    try warmTrees(e, cache, &ids, &picks);
    try cache.reset(e);
    phases.warmups += profile.elapsed(io, started);
}

pub const Test = if (@import("builtin").is_test) @import("cuda_warm_test.zig") else struct {};

test "a tree of each register-slot peak, as the slot plan the forward stages counts it" {
    const slots = @import("core").tree_slots;
    for ([_]usize{ 16, 32 }) |rows| {
        var peak: usize = 2;
        while ((@as(usize, 1) << @intCast(peak)) - 1 <= rows) : (peak += 1) {
            var parents: [64]i32 = undefined;
            treeOfPeak(parents[0..rows], peak);
            try std.testing.expectEqual(@as(i32, -1), parents[0]);
            for (parents[1..rows], 1..) |p, row| try std.testing.expect(p >= 0 and p < @as(i32, @intCast(row)));
            var steps: [64]slots.Step = undefined;
            try std.testing.expectEqual(peak, try slots.build(std.testing.allocator, parents[0..rows], steps[0..rows], 0, slots.max_slots));
        }
    }
}
