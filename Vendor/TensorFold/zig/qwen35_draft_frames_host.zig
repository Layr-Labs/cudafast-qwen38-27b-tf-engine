//! Fixed frame schedules, complete byte records and actual prompt-control routing without device execution.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const lanes = @import("lanes");
const fixture = @import("qwen35_verify_host.zig");
const plan = @import("tests/cuda/qwen_draft_frames_plan.zig");
const control = @import("tests/cuda/qwen_draft_frames_control.zig");
const t = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "Draft frames: the configured tree budget reaches both facts and frame admission" {
    _ = unsetenv("TF_DRAFT_NODES");
    _ = unsetenv("TF_NO_TREES");
    defer _ = unsetenv("TF_DRAFT_NODES");
    var c: q.draft.Config = .{ .default_tree_nodes = 31, .hidden = 256, .intermediate = 256, .layers = 1, .vocab = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .block = 16, .mask = 255, .window = 2048, .group = 16, .rank = 2, .topk = 2, .eps = 1e-6, .theta = 10000000, .taps = @splat(0), .n_taps = 1 };
    var d: q.draft.Draft = undefined;
    d.projection_role = .proposal;
    d.frame_streams = 1;
    d.scratch_one = 0;
    d.target_taps = 0;
    d.step_ms = 1;
    d.n_costs = 0;
    d.c = c;
    try t.expectEqual(@as(u32, 31), d.drafter().facts().tree_nodes);
    const F = q.draft.frame;
    var storage: [F.stagedLen(2)]u8 align(8) = undefined;
    var ctx: q.draft.Ctx = .{ .ring = .{ .d = undefined, .ptr = 0x1000, .len = 1 }, .begin = 0, .end = 100 };
    var refs = [_]*q.draft.Ctx{&ctx};
    const items = [_]lanes.drafter.Hold{.{ .stream = @ptrFromInt(0x1000), .pending = 1, .position = 101, .depth = 31 }};
    try F.stage(c, &items, &refs, &storage);
    _ = setenv("TF_DRAFT_NODES", "7", 1);
    try t.expectEqual(@as(u32, 7), d.drafter().facts().tree_nodes);
    try t.expectError(error.BadDraftFrame, F.stage(c, &items, &refs, &storage));
    _ = unsetenv("TF_DRAFT_NODES");
    c.default_tree_nodes = 15;
    d.c = c;
    try t.expectEqual(@as(u32, 15), d.drafter().facts().tree_nodes);
    try t.expectError(error.BadDraftFrame, F.stage(c, &items, &refs, &storage));
}

fn fill(c: *plan.Capture, partition: plan.Partition) !void {
    const Fake = struct {
        capture: *plan.Capture,
        count: [plan.streams]usize = @splat(0),
        pub fn frame(f: *@This(), round: usize, group: []const usize) !void {
            var words: [plan.word_bytes]u8 = undefined;
            var selector: [plan.depth * 2]u8 = undefined;
            for (group) |id| {
                try t.expectEqual(round, f.count[id]);
                f.count[id] += 1;
                @memset(&words, @intCast(id));
                @memset(&selector, @intCast(round));
                const proposals: [plan.depth]u32 = @splat(@intCast(100 * id + round));
                try f.capture.write(round, id, &proposals, &words, &selector, (id + round) % plan.block);
            }
        }
    };
    var fake: Fake = .{ .capture = c };
    try plan.schedule(&fake, partition);
    try t.expectEqual(@as([plan.streams]usize, @splat(plan.rounds)), fake.count);
}

test "Draft frames: all partitions execute every stream once a round and canonical records ignore frame order" {
    var base = try plan.Capture.init(t.allocator, 1);
    defer base.deinit();
    try fill(&base, plan.partitions[0]);
    for (plan.partitions[1..]) |partition| {
        var candidate = try plan.Capture.init(t.allocator, 1);
        defer candidate.deinit();
        try fill(&candidate, partition);
        try t.expectEqual(@as(?plan.Difference, null), try plan.compare(base, candidate, false));
    }
    try t.expectEqualSlices(usize, &.{ 37, 64, 129, 200, 300, 511, 700, 1000 }, &plan.lengths);
    try t.expectError(error.BadFramePartition, plan.validate(.{ .mode = .mixed_a, .groups = &.{ &.{ 0, 1 }, &.{ 1, 2, 3, 4, 5, 6, 7 } } }));
    try t.expectError(error.BadFramePartition, plan.validate(.{ .mode = .mixed_a, .groups = &.{&.{ 0, 1, 2, 3, 4, 5, 6, 8 }} }));
    try t.expectError(error.BadFramePartition, plan.validate(.{ .mode = .mixed_a, .groups = &.{&.{ 0, 1, 2 }} }));
}

test "Draft frames: proposals, staged words, selector bytes and acceptance are all compared, with no label oracle" {
    var base = try plan.Capture.init(t.allocator, 1);
    defer base.deinit();
    var candidate = try plan.Capture.init(t.allocator, 1);
    defer candidate.deinit();
    try fill(&base, plan.partitions[0]);
    try fill(&candidate, plan.partitions[1]);
    for ([_]usize{ 0, plan.depth * 4, plan.depth * 4 + plan.word_bytes, base.stride - 1 }) |offset| {
        const record = try candidate.slot(5, 7);
        record[offset] ^= 1;
        try t.expectEqual(plan.Difference{ .round = 5, .stream = 7, .byte = offset }, (try base.difference(candidate)).?);
        try t.expectError(error.DraftFrameDependent, plan.compare(base, candidate, false));
        try t.expect((try plan.compare(base, candidate, true)) != null);
        record[offset] ^= 1;
    }
    try t.expectError(error.FrameControlNotDetected, plan.compare(base, candidate, true));
    try candidate.swapStreams(0, 1);
    try t.expectEqual(@as(usize, 0), (try plan.compare(base, candidate, true)).?.stream);
    try candidate.swapStreams(0, 1);
    try t.expectEqual(@as(?plan.Difference, null), try base.difference(candidate));
    candidate.seen[47] = false;
    try t.expectError(error.MissingFrameRecord, plan.compare(base, candidate, true));
}

test "Draft frames: record writes reject duplicates, invalid sizes and out-of-range identities" {
    var c = try plan.Capture.init(t.allocator, 1);
    defer c.deinit();
    const proposals: [plan.depth]u32 = @splat(3);
    const words: [plan.word_bytes]u8 = @splat(0xa5);
    const selector: [plan.depth * 2]u8 = @splat(0xff);
    try t.expectError(error.BadFrameRecord, c.write(6, 0, &proposals, &words, &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 8, &proposals, &words, &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, proposals[1..], &words, &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, words[1..], &selector, 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, &words, selector[1..], 0));
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, &words, &selector, 16));
    try c.write(0, 0, &proposals, &words, &selector, 15);
    try t.expectError(error.BadFrameRecord, c.write(0, 0, &proposals, &words, &selector, 0));
}

test "Draft frames: greedy acceptance stops at the first mismatch and retains a pending or bonus row" {
    var proposals: [plan.depth]u32 = undefined;
    var sampled: [plan.block]u32 = undefined;
    for (&sampled, 0..) |*s, i| s.* = @intCast(i);
    @memcpy(&proposals, sampled[0..plan.depth]);
    try t.expectEqual(@as(usize, plan.depth), try plan.accepted(&proposals, &sampled));
    for (0..plan.depth) |at| {
        proposals[at] += 1;
        try t.expectEqual(at, try plan.accepted(&proposals, &sampled));
        proposals[at] -= 1;
    }
}

test "Draft frames: retained JSON records preserve raw bytes and canonical identity" {
    var c = try plan.Capture.init(t.allocator, 1);
    defer c.deinit();
    const proposals: [plan.depth]u32 = @splat(248319);
    const words: [plan.word_bytes]u8 = @splat(0xff);
    const selector: [plan.depth * 2]u8 = @splat('a');
    try c.write(3, 5, &proposals, &words, &selector, 7);
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    try c.json(&out.writer, .mixed_b);
    const doc = try std.json.parseFromSlice(std.json.Value, t.allocator, out.written(), .{});
    defer doc.deinit();
    const obj = doc.value.object;
    try t.expectEqualStrings("mixed_b", obj.get("mode").?.string);
    try t.expectEqual(@as(i64, 3), obj.get("round").?.integer);
    try t.expectEqual(@as(i64, 5), obj.get("stream").?.integer);
    try t.expectEqual(@as(i64, 7), obj.get("accepted").?.integer);
    for (obj.get("proposals").?.array.items) |v| try t.expectEqual(@as(i64, 248319), v.integer);
    const w = obj.get("words").?.array.items;
    try t.expectEqual(@as(usize, plan.word_bytes), w.len);
    for (w) |v| try t.expectEqual(@as(i64, 255), v.integer);
    const s = obj.get("selector").?.array.items;
    try t.expectEqual(@as(usize, plan.depth * 2), s.len);
    for (s) |v| try t.expectEqual(@as(i64, 'a'), v.integer);
}

const Witness = struct {
    c: struct { block: usize = plan.block } = .{},
    projection_role: q.draft.precision.Role = .proposal,
    e: *q.engine.Engine,
    fail: bool = false,
    pub fn drafter(w: *Witness) struct {
        w: *Witness,
        pub fn hold(x: @This(), items: []const lanes.drafter.Hold) !void {
            for (items) |item| {
                const shape = item.lanes.?;
                try t.expectEqual(@as(usize, plan.depth), shape.parents.len);
                for (shape.parents, shape.ranks, shape.depths, 0..) |parent, rank, depth, i| {
                    try t.expectEqual(@as(i32, @intCast(i)) - 1, parent);
                    try t.expectEqual(@as(u8, 0), rank);
                    try t.expectEqual(@as(u8, @intCast(i)), depth);
                }
            }
            if (x.w.fail) return error.PlantedHoldFailure;
            const weight: cuda.qlinear.Affine4 = .{ .w = 10, .s = 20, .b = 30, .k = 8192, .n = 4096, .npad = 4096 };
            try q.draft.precision.affine(x.w.e, weight, 0x10000, 0x20000, 0x30000, items.len * plan.block, false, x.w.projection_role);
        }
    } {
        return .{ .w = w };
    }
};

test "Draft frames: the prompt control reaches real projection dispatch only for shared proposal frames" {
    var driver: cuda.Driver = undefined;
    var e = fixture.engine(&driver, true);
    var witness: Witness = .{ .e = &e };
    var items: [plan.streams]lanes.drafter.Hold = @splat(.{ .stream = undefined, .pending = 1, .position = 2, .depth = plan.depth });
    for ([_]bool{ false, true }) |negative| for ([_]usize{ 1, 2, 4, 8 }) |n| {
        fixture.Trace.count = 0;
        fixture.Trace.prompts = 0;
        var stats: control.Stats = .{};
        try control.hold(&witness, items[0..n], negative, &stats);
        const prompt = negative and n > 1;
        try t.expectEqual(@as(usize, @intFromBool(prompt)), fixture.Trace.prompts);
        try t.expectEqual(@as(usize, @intFromBool(prompt)), stats.prompt_frames);
        try t.expectEqual(@as(usize, 1), stats.frames);
        try t.expectEqual(n * plan.block, stats.max_rows);
        try t.expectEqual(q.draft.precision.Role.proposal, witness.projection_role);
    };
    witness.fail = true;
    var stats: control.Stats = .{};
    try t.expectError(error.PlantedHoldFailure, control.hold(&witness, &items, true, &stats));
    try t.expectEqual(q.draft.precision.Role.proposal, witness.projection_role);
    try t.expectEqual(@as(usize, 0), stats.frames);
}
