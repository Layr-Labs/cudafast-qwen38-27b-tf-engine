//! Warm each captured launch shape, then measure every live width with a freshly reset cache.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const Round = @import("cuda_round.zig");
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
    try cache.reset(e);
    for (measured[0..e.window_rows]) |m| d.setCost(m.rows, m.ms);
    const trees_start = profile.now(io);
    try trees(e, cache, warmed);
    warm_ns += profile.elapsed(io, trees_start);
    phases.warmups += warm_ns;
    phases.calibration += profile.elapsed(io, started) - warm_ns;
}

/// A drafting engine's tree windows: each 16-row bucket at every GDN slot variant a tree reaches, captured and run
/// twice now (content-free: zero ids on a reset cache), as the chain widths are above. A tree's shape is otherwise
/// captured, uploaded and its kernels loaded inside the first timed round that meets it.
fn trees(e: anytype, cache: anytype, warmed: *Seen) !void {
    if (e.n_taps == 0) return;
    const ids: [E.most_window]u32 = @splat(0);
    var picks: [E.most_window]E.Draw = undefined;
    for (&picks) |*pk| pk.* = .{ .rule = null, .position = 1 };
    const staging = try std.heap.page_allocator.alignedAlloc(u8, .@"16", Round.bytes);
    defer std.heap.page_allocator.free(staging);
    var parents: [E.most_window]i32 = undefined;
    var bucket: usize = 16;
    while (bucket <= e.window_rows) : (bucket += 16) {
        // three families whose slot peaks cover every variant a drafted tree reaches: a root's k children with a chain
        // under the last, k chains interleaved row by row (as a best-first tree's rows interleave its branches), and
        // k-ary heaps
        for (0..3) |family| for (2..bucket) |k| {
            parents[0] = -1;
            for (1..bucket) |r| parents[r] = switch (family) {
                0 => if (r <= k) 0 else @intCast(r - 1),
                1 => if (r <= k) 0 else @intCast(r - k),
                else => @intCast((r - 1) / k),
            };
            const part: E.Part = .{ .cache = cache, .rows = bucket, .parents = parents[0..bucket] };
            const peak = Round.build(staging, &.{part}, true) catch continue;
            const shape = E.Engine.Shape.init(bucket, 1, true, true, @intCast(Round.variant(peak)));
            while (warmed.passes(shape) < 2) {
                try cache.reset(e);
                try e.forward(&.{part}, ids[0..bucket], .{ .draws = picks[0..bucket] });
                warmed.mark(shape);
            }
        };
    }
    try cache.reset(e);
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
    try trees(e, cache, warmed);
    phases.warmups += profile.elapsed(io, started);
}

pub const Test = if (@import("builtin").is_test) @import("cuda_warm_test.zig") else struct {};
