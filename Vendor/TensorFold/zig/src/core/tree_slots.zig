//! Depth-first trees recycle a parent's register slot at its last child and preserve original output rows.
const std = @import("std");

pub const max_slots = 16;
pub const Step = extern struct { row: u32, read_slot: i32, write_slot: i32 };

pub fn build(gpa: std.mem.Allocator, parents: []const i32, steps: []Step, first: u32, limit: usize) !usize {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots or parents.len > std.math.maxInt(u32) - first) return error.BadSlotPlan;
    const sizes = try gpa.alloc(usize, parents.len);
    defer gpa.free(sizes);
    @memset(sizes, 1);
    for (parents, 0..) |parent, row| if (parent < -1 or parent >= row) return error.BadSlotParents;
    var remaining = parents.len;
    while (remaining > 0) {
        remaining -= 1;
        if (parents[remaining] >= 0) sizes[@intCast(parents[remaining])] += sizes[remaining];
    }
    var planner = Builder{ .parents = parents, .sizes = sizes, .steps = steps, .first = first, .limit = limit };
    for (parents, 0..) |parent, row| if (parent == -1) try planner.visit(row, -1, false);
    std.debug.assert(planner.at == steps.len and planner.live == 0);
    return planner.peak;
}

const Builder = struct {
    parents: []const i32,
    sizes: []const usize,
    steps: []Step,
    first: u32,
    limit: usize,
    used: [max_slots]bool = @splat(false),
    live: usize = 0,
    peak: usize = 0,
    at: usize = 0,

    fn take(b: *Builder) !i32 {
        for (b.used[0..b.limit], 0..) |*used, slot| if (!used.*) {
            used.* = true;
            b.live += 1;
            b.peak = @max(b.peak, b.live);
            return @intCast(slot);
        };
        return error.SlotOverflow;
    }

    fn visit(b: *Builder, row: usize, read: i32, reuse: bool) error{SlotOverflow}!void {
        const write = if (reuse) read else try b.take();
        b.steps[b.at] = .{ .row = b.first + @as(u32, @intCast(row)), .read_slot = read, .write_slot = write };
        b.at += 1;
        var last: ?usize = null;
        for (b.parents, 0..) |parent, child| if (parent == row) {
            if (last == null or b.sizes[child] >= b.sizes[last.?]) last = child;
        };
        if (last) |heavy| {
            for (b.parents, 0..) |parent, child| if (parent == row and child != heavy) try b.visit(child, write, false);
            try b.visit(heavy, write, true);
        } else {
            b.used[@intCast(write)] = false;
            b.live -= 1;
        }
    }
};

/// A carried step's parent state is still in the registers (the step before it was its parent's): it reads nothing.
/// Only the scans that tell it from -1 (the root) take carried plans: qwen35_slots.cuh's q35_slots and
/// qwen35_slots16.cuh's q35_slots16_ready, each reading the root for -1 alone.
pub const in_registers: i32 = -2;

/// `build`'s depth-first order (a node's lighter children first, the heaviest last) with the registers as a state: a row
/// whose parent is the step just before reads nothing (`in_registers`), and a row's state goes to a slot only when a
/// later child needs it after the first child's subtree (a node of two or more children, until its last child starts;
/// write_slot -1: no step reads it again). Every row's parent state and arithmetic are `build`'s; the peak of slots held
/// at once is one below `build`'s for any tree with a branch, at least 1.
pub fn carried(gpa: std.mem.Allocator, parents: []const i32, steps: []Step, first: u32, limit: usize) !usize {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots or parents.len > std.math.maxInt(u32) - first) return error.BadSlotPlan;
    const sizes = try gpa.alloc(usize, parents.len);
    defer gpa.free(sizes);
    @memset(sizes, 1);
    for (parents, 0..) |parent, row| if (parent < -1 or parent >= row) return error.BadSlotParents;
    var remaining = parents.len;
    while (remaining > 0) {
        remaining -= 1;
        if (parents[remaining] >= 0) sizes[@intCast(parents[remaining])] += sizes[remaining];
    }
    var planner = Carrier{ .parents = parents, .sizes = sizes, .steps = steps, .first = first, .limit = limit };
    for (parents, 0..) |parent, row| if (parent == -1) try planner.visit(row, -1);
    std.debug.assert(planner.at == steps.len and planner.live == 0);
    return @max(1, planner.peak);
}

const Carrier = struct {
    parents: []const i32,
    sizes: []const usize,
    steps: []Step,
    first: u32,
    limit: usize,
    used: [max_slots]bool = @splat(false),
    live: usize = 0,
    peak: usize = 0,
    at: usize = 0,

    fn take(b: *Carrier) !i32 {
        for (b.used[0..b.limit], 0..) |*used, slot| if (!used.*) {
            used.* = true;
            b.live += 1;
            b.peak = @max(b.peak, b.live);
            return @intCast(slot);
        };
        return error.SlotOverflow;
    }

    fn visit(b: *Carrier, row: usize, read: i32) error{SlotOverflow}!void {
        var children: usize = 0;
        var last: ?usize = null;
        for (b.parents, 0..) |parent, child| if (parent == row) {
            children += 1;
            if (last == null or b.sizes[child] >= b.sizes[last.?]) last = child;
        };
        const write: i32 = if (children >= 2) try b.take() else -1;
        b.steps[b.at] = .{ .row = b.first + @as(u32, @intCast(row)), .read_slot = read, .write_slot = write };
        b.at += 1;
        const heavy = last orelse return;
        var source = in_registers;
        for (b.parents, 0..) |parent, child| if (parent == row and child != heavy) {
            try b.visit(child, source);
            source = write;
        };
        // the heaviest child reads the slot first; its own step may take the slot again for its children
        if (write >= 0) {
            b.used[@intCast(write)] = false;
            b.live -= 1;
        }
        try b.visit(heavy, source);
    }
};

/// `validate` for `carried` plans: a step reading in_registers follows its parent's step, -1 reads the root (a row
/// without a parent), write_slot -1 keeps nothing, and a slot read still holds the parent's state.
pub fn validateCarried(parents: []const i32, steps: []const Step, first: u32, limit: usize) !void {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots) return error.BadSlotPlan;
    var owners: [max_slots]i32 = @splat(-2);
    var registers: i32 = -2;
    for (steps, 0..) |step, at| {
        if (step.row < first or step.row - first >= parents.len or step.read_slot < in_registers or step.read_slot >= limit or step.write_slot < -1 or step.write_slot >= limit) return error.BadSlotPlan;
        const row = step.row - first;
        const parent = parents[row];
        if (parent < -1 or parent >= row) return error.BadSlotParents;
        if (step.read_slot == in_registers) {
            if (parent < 0 or registers != parent) return error.BadSlotPlan;
        } else if (step.read_slot < 0) {
            if (parent != -1) return error.BadSlotPlan;
        } else if (owners[@intCast(step.read_slot)] != parent) return error.BadSlotPlan;
        for (steps[0..at]) |prior| if (prior.row == step.row) return error.BadSlotPlan;
        if (step.write_slot >= 0) owners[@intCast(step.write_slot)] = @intCast(row);
        registers = @intCast(row);
    }
}

pub fn validate(parents: []const i32, steps: []const Step, first: u32, limit: usize) !void {
    if (parents.len == 0 or parents.len != steps.len or limit == 0 or limit > max_slots) return error.BadSlotPlan;
    var owners: [max_slots]i32 = @splat(-2);
    for (steps, 0..) |step, at| {
        if (step.row < first or step.row - first >= parents.len or step.read_slot < -1 or step.read_slot >= limit or step.write_slot < 0 or step.write_slot >= limit) return error.BadSlotPlan;
        const row = step.row - first;
        const parent = parents[row];
        if (parent < -1 or parent >= row) return error.BadSlotParents;
        if (step.read_slot < 0) {
            if (parent != -1) return error.BadSlotPlan;
        } else if (owners[@intCast(step.read_slot)] != parent) return error.BadSlotPlan;
        for (steps[0..at]) |prior| if (prior.row == step.row) return error.BadSlotPlan;
        owners[@intCast(step.write_slot)] = @intCast(row);
    }
}

test "slot schedules retain each parent's state while moving the largest child last" {
    const parents = [_]i32{ -1, 0, 0, 1, 1, 3, 2, 6, 6 };
    var steps: [parents.len]Step = undefined;
    try std.testing.expectEqual(@as(usize, 3), try build(std.testing.allocator, &parents, &steps, 23, 16));
    try validate(&parents, &steps, 23, 16);
    try std.testing.expectEqualSlices(u32, &.{ 23, 24, 27, 26, 28, 25, 29, 30, 31 }, &rowsOf(steps));
    steps[2].read_slot = 15;
    try std.testing.expectError(error.BadSlotPlan, validate(&parents, &steps, 23, 16));
}

fn rowsOf(steps: anytype) [steps.len]u32 {
    var rows: [steps.len]u32 = undefined;
    for (steps, &rows) |step, *row| row.* = step.row;
    return rows;
}

test "long chains need one slot and stars need two independent of depth or width" {
    var parents: [512]i32 = undefined;
    var steps: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast(row - 1);
    try std.testing.expectEqual(@as(usize, 1), try build(std.testing.allocator, &parents, &steps, 0, 1));
    try validate(&parents, &steps, 0, 1);
    @memset(parents[1..], 0);
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 0, 1));
    try std.testing.expectEqual(@as(usize, 2), try build(std.testing.allocator, &parents, &steps, 0, 2));
    try validate(&parents, &steps, 0, 2);
}

test "balanced trees reuse ancestor slots and an explicit slot budget refuses overflow" {
    var parents: [127]i32 = undefined;
    var steps: [parents.len]Step = undefined;
    for (&parents, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast((row - 1) / 2);
    try std.testing.expectEqual(@as(usize, 7), try build(std.testing.allocator, &parents, &steps, 0, 16));
    try validate(&parents, &steps, 0, 16);
    try std.testing.expectError(error.SlotOverflow, build(std.testing.allocator, &parents, &steps, 0, 6));
}

test "forests reset to committed state and bad parents never make a schedule" {
    const parents = [_]i32{ -1, 0, -1, 2, 2, -1 };
    var steps: [parents.len]Step = undefined;
    _ = try build(std.testing.allocator, &parents, &steps, 73, 16);
    try validate(&parents, &steps, 73, 16);
    for ([_][]const i32{ &.{ -1, 1 }, &.{ -2, 0 }, &.{ -1, 2 } }) |bad| {
        try std.testing.expectError(error.BadSlotParents, build(std.testing.allocator, bad, steps[0..bad.len], 0, 16));
    }
}

test "irregular trees compute every row from the same parent as an independent topological replay" {
    var random: u64 = 1729;
    for ([_]usize{ 17, 32, 64, 128 }) |width| {
        var parents: [128]i32 = undefined;
        var steps: [128]Step = undefined;
        var expected: [128]u64 = undefined;
        var actual: [128]u64 = undefined;
        for (0..32) |_| {
            for (parents[0..width], 0..) |*parent, row| {
                random = random *% 6364136223846793005 +% 1442695040888963407;
                parent.* = if (row == 0) -1 else @intCast(random % row);
                const value = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
                expected[row] = value *% 65537 +% row +% 1;
            }
            _ = try build(std.testing.allocator, parents[0..width], steps[0..width], 11, 16);
            try validate(parents[0..width], steps[0..width], 11, 16);
            var registers: [max_slots]u64 = @splat(0);
            for (steps[0..width]) |step| {
                const row = step.row - 11;
                const value = if (step.read_slot < 0) @as(u64, 23) else registers[@intCast(step.read_slot)];
                actual[row] = value *% 65537 +% row +% 1;
                registers[@intCast(step.write_slot)] = actual[row];
            }
            try std.testing.expectEqualSlices(u64, expected[0..width], actual[0..width]);
        }
    }
}

test "carried plans keep a state in a slot only for a later child, and chains in the registers" {
    const parents = [_]i32{ -1, 0, 0, 1, 1, 3, 2, 6, 6 };
    var steps: [parents.len]Step = undefined;
    const peak = try carried(std.testing.allocator, &parents, &steps, 23, 16);
    try std.testing.expect(peak <= 2);
    try validateCarried(&parents, &steps, 23, 16);
    try std.testing.expectEqualSlices(u32, &.{ 23, 24, 27, 26, 28, 25, 29, 30, 31 }, &rowsOf(steps));
    var chain: [64]i32 = undefined;
    var chain_steps: [chain.len]Step = undefined;
    for (&chain, 0..) |*parent, row| parent.* = if (row == 0) -1 else @intCast(row - 1);
    try std.testing.expectEqual(@as(usize, 1), try carried(std.testing.allocator, &chain, &chain_steps, 0, 1));
    try validateCarried(&chain, &chain_steps, 0, 1);
    for (chain_steps[1..]) |step| try std.testing.expectEqual(in_registers, step.read_slot);
    steps[3].read_slot = in_registers;
    try std.testing.expectError(error.BadSlotPlan, validateCarried(&parents, &steps, 23, 16));
}

test "carried plans replay every row from its parent's state and need one slot fewer than build" {
    var random: u64 = 4242;
    for ([_]usize{ 17, 24, 31, 32, 64 }) |width| {
        var parents: [64]i32 = undefined;
        var steps: [64]Step = undefined;
        var plain: [64]Step = undefined;
        var expected: [64]u64 = undefined;
        var actual: [64]u64 = undefined;
        for (0..64) |_| {
            for (parents[0..width], 0..) |*parent, row| {
                random = random *% 6364136223846793005 +% 1442695040888963407;
                parent.* = if (row == 0) -1 else @intCast((random >> 33) % row);
                const value = if (parent.* < 0) @as(u64, 23) else expected[@intCast(parent.*)];
                expected[row] = value *% 65537 +% row +% 1;
            }
            const peak = try carried(std.testing.allocator, parents[0..width], steps[0..width], 7, 16);
            const before = try build(std.testing.allocator, parents[0..width], plain[0..width], 7, 16);
            try std.testing.expect(peak == @max(1, before - 1));
            try validateCarried(parents[0..width], steps[0..width], 7, 16);
            var registers: [max_slots]u64 = @splat(0);
            var held: u64 = 0;
            for (steps[0..width]) |step| {
                const row = step.row - 7;
                const value = if (step.read_slot == in_registers) held else if (step.read_slot < 0) @as(u64, 23) else registers[@intCast(step.read_slot)];
                actual[row] = value *% 65537 +% row +% 1;
                held = actual[row];
                if (step.write_slot >= 0) registers[@intCast(step.write_slot)] = actual[row];
            }
            try std.testing.expectEqualSlices(u64, expected[0..width], actual[0..width]);
        }
    }
}
