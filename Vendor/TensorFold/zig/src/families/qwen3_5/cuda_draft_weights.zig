//! The DFlash2 drafter's weights onto the device: projections quantized to affine-4 on the host, codebooks kept there.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");

const Affine4 = cuda.qlinear.Affine4;
const DeviceBuffer = cuda.DeviceBuffer;
const Engine = engine.Engine;
const aligned = engine.aligned;

pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    q: Affine4,
    k: Affine4,
    v: Affine4,
    o: Affine4,
    q_norm: u64,
    k_norm: u64,
    gate: Affine4,
    up: Affine4,
    down: Affine4,
    attn_conv: Affine4, // kernel_projection: hidden -> 4 * groups dynamic coefficients
    attn_base: u64, // base_kernel (2 branches, 2 lags, hidden) bf16
    mlp_conv: Affine4,
    mlp_base: u64,
};

pub const Weights = struct {
    fc: Affine4, // the taps' concatenation -> hidden
    hidden_norm: u64,
    final_norm: u64,
    selector: Affine4, // hidden_projection: hidden -> rank
    predecessor: []u16, // codebooks (vocab, rank) bf16, on the host
    successor: []u16,
    layers: []Layer,
    arena: DeviceBuffer,
    arena_at: usize = 0,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        w.arena.free();
        gpa.free(w.predecessor);
        gpa.free(w.successor);
        gpa.free(w.layers);
    }

    fn alloc(w: *Weights, bytes: usize) !u64 {
        const at = aligned(w.arena_at);
        if (at + bytes > w.arena.len) return error.ArenaExhausted;
        w.arena_at = at + bytes;
        return w.arena.ptr + at;
    }
};

/// The drafter's widths the loader needs (cuda_draft.zig's Config).
pub const Shape = struct { hidden: usize, intermediate: usize, layers: usize, vocab: usize, q_width: usize, kv_width: usize, head_dim: usize, dynamic: usize, rank: usize, taps: usize };

fn packedBytes(n: usize, k: usize) usize {
    const l = Affine4.layout(n, k);
    return aligned(l.words) + 2 * aligned(l.scales);
}

/// One projection's quantized words, scales and biases as `linear` stages them, back to back.
fn rawBytes(n: usize, k: usize) usize {
    return n * k / 2 + 2 * (n * k / 64 * 2);
}

/// The largest projection `load` stages: the staging buffer it allocates for the load and frees after.
pub fn stageBytes(s: Shape) usize {
    var most: usize = 0;
    const shapes = [_][2]usize{ .{ s.hidden, s.taps * s.hidden }, .{ s.rank, s.hidden }, .{ s.q_width, s.hidden }, .{ s.kv_width, s.hidden }, .{ s.hidden, s.q_width }, .{ s.intermediate, s.hidden }, .{ s.hidden, s.intermediate }, .{ s.dynamic, s.hidden } };
    for (shapes) |nk| most = @max(most, rawBytes(nk[0], nk[1]));
    return most;
}

/// Device bytes of everything `load` packs.
pub fn total(s: Shape) usize {
    var t = packedBytes(s.hidden, s.taps * s.hidden) + packedBytes(s.rank, s.hidden) + 2 * aligned(s.hidden * 2);
    t += s.layers * (2 * aligned(s.hidden * 2) + packedBytes(s.q_width, s.hidden) + 2 * packedBytes(s.kv_width, s.hidden) + packedBytes(s.hidden, s.q_width) +
        2 * aligned(s.head_dim * 2) + 2 * packedBytes(s.intermediate, s.hidden) + packedBytes(s.hidden, s.intermediate) + 2 * packedBytes(s.dynamic, s.hidden) + 2 * aligned(4 * s.hidden * 2) +
        2 * aligned(Affine4.layout(s.intermediate, s.hidden).words) + aligned(Affine4.layout(s.hidden, s.intermediate).words) +
        aligned(Affine4.layout(s.hidden, s.q_width).words)); // the o projection's stream view (oStream)
    return t;
}

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn toBf16(x: f32) u16 {
    // round to nearest even, as the device does
    const u: u32 = @bitCast(x);
    const rounding: u32 = 0x7fff + ((u >> 16) & 1);
    return @intCast((u +% rounding) >> 16);
}

/// MLX affine-4, group 64: minimize weight error with a bf16 scale and bias; pack low nibbles first.
fn quantize(gpa: std.mem.Allocator, w: []align(1) const u16, n: usize, k: usize) !struct { words: []u32, scales: []u16, biases: []u16 } {
    const groups = k / 64;
    const words = try gpa.alloc(u32, n * k / 8);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(biases);
    for (0..n) |r| {
        for (0..groups) |g| {
            const row = w[r * k + g * 64 ..][0..64];
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            for (row) |h| {
                const x = bf16(h);
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
            var fit = [2]f32{ (hi - lo) / 15.0, lo };
            fit = leastSquares(row, fit);
            const scale = toBf16(fit[0]);
            const bias = toBf16(fit[1]);
            scales[r * groups + g] = scale;
            biases[r * groups + g] = bias;
            const s = bf16(scale);
            const b = bf16(bias);
            const inv = if (s == 0) 0.0 else 1.0 / s;
            for (0..8) |j| {
                var word: u32 = 0;
                for (0..8) |i| {
                    const q = std.math.clamp(@round((bf16(row[j * 8 + i]) - b) * inv), 0.0, 15.0);
                    word |= @as(u32, @intFromFloat(q)) << @intCast(4 * i);
                }
                words[(r * k + g * 64) / 8 + j] = word;
            }
        }
    }
    return .{ .words = words, .scales = scales, .biases = biases };
}

/// A group's (scale, bias) improved by alternating code assignment and a least-squares line fit; kept only where the
/// squared error with bf16-rounded scale and bias drops.
fn leastSquares(row: []align(1) const u16, start: [2]f32) [2]f32 {
    var best = start;
    var best_err = groupError(row, start);
    var cur = start;
    for (0..8) |_| {
        const s = cur[0];
        if (s <= 0) break;
        var sq: f64 = 0;
        var sqq: f64 = 0;
        var sw: f64 = 0;
        var sqw: f64 = 0;
        for (row) |h| {
            const x: f64 = bf16(h);
            const q: f64 = std.math.clamp(@round((x - cur[1]) / s), 0.0, 15.0);
            sq += q;
            sqq += q * q;
            sw += x;
            sqw += q * x;
        }
        const det = 64 * sqq - sq * sq;
        if (det <= 0) break;
        const ns = (64 * sqw - sq * sw) / det;
        const nb = (sw - ns * sq) / 64;
        if (!(ns > 0)) break;
        cur = .{ @floatCast(ns), @floatCast(nb) };
        const err = groupError(row, cur);
        if (err < best_err) {
            best_err = err;
            best = cur;
        }
    }
    // Search adjacent representable stored values; keep only an improvement under
    // the same BF16-rounded error used by the base fit. Never alter the target.
    const scale_bits = toBf16(best[0]);
    const bias_bits = toBf16(best[1]);
    for ([_]i32{ -1, 0, 1 }) |ds| {
        for ([_]i32{ -1, 0, 1 }) |db| {
            const su = @as(i32, scale_bits) + ds;
            const bu = @as(i32, bias_bits) + db;
            if (su < 0 or su > 65535 or bu < 0 or bu > 65535) continue;
            const candidate = [2]f32{ bf16(@intCast(su)), bf16(@intCast(bu)) };
            if (!(candidate[0] > 0) or !std.math.isFinite(candidate[0]) or !std.math.isFinite(candidate[1])) continue;
            const err = groupError(row, candidate);
            if (err < best_err) {
                best_err = err;
                best = candidate;
            }
        }
    }
    return best;
}

/// The group's squared error under bf16-rounded scale and bias, as quantize then stores and the kernels read it.
fn groupError(row: []align(1) const u16, fit: [2]f32) f64 {
    const s = bf16(toBf16(fit[0]));
    const b = bf16(toBf16(fit[1]));
    const inv = if (s == 0) 0.0 else 1.0 / s;
    var err: f64 = 0;
    for (row) |h| {
        const x = bf16(h);
        const q = std.math.clamp(@round((x - b) * inv), 0.0, 15.0);
        const d: f64 = x - (q * s + b);
        err += d * d;
    }
    return err;
}

/// TF_DRAFT_O_STREAM=0: the o projection (n = hidden, k = 4096) keeps the cluster tile; otherwise it gets a K-major
/// view and its 16-row windows run tf_qmm_output_stream16 (the same four slices summed in order).
fn oStream() bool {
    const v = @import("sys").c.getenv("TF_DRAFT_O_STREAM") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// A bf16 (n, k) tensor quantized and packed through `stage` into the drafter's arena.
fn linear(gpa: std.mem.Allocator, e: *Engine, stage: *DeviceBuffer, w: *Weights, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !Affine4 {
    const t = try ck.expect(name, .bf16, &.{ n, k });
    const q = try quantize(gpa, std.mem.bytesAsSlice(u16, t.bytes), n, k);
    defer {
        gpa.free(q.words);
        gpa.free(q.scales);
        gpa.free(q.biases);
    }
    const parts = [_][]const u8{ std.mem.sliceAsBytes(q.words), std.mem.sliceAsBytes(q.scales), std.mem.sliceAsBytes(q.biases) };
    var raw: [3]u64 = undefined;
    var at: usize = 0;
    for (parts, &raw) |p, *r| {
        if (at + p.len > stage.len) return error.StageTooSmall;
        try stage.upload(at, p);
        r.* = stage.ptr + at;
        at += p.len;
    }
    const l = Affine4.layout(n, k);
    var out: Affine4 = .{ .w = try w.alloc(l.words), .s = try w.alloc(l.scales), .b = try w.alloc(l.scales), .n = n, .k = k, .npad = l.npad };
    try e.affine.pack(e.stream, raw[0], raw[1], raw[2], out);
    if (e.mlp_pair.kmajor_fn != null and n == 17408 and k == 5120) {
        out.verify_w = try w.alloc(l.words);
        const pack = try e.mods[3].function("tf_pack_kmajor");
        const words = l.npad * k / 8;
        try e.run(pack, .{ (words + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ out.w, out.verify_w, @as(c_int, @intCast(k / 64)), @as(c_int, @intCast(l.npad / 64)) });
    } else if (e.affine.stream16_fns[0] != null and n == 5120 and (k == 17408 or (k == 4096 and oStream()))) {
        out.stream_w = try w.alloc(l.words);
        const pack = try e.mods[3].function("tf_pack_kmajor");
        const words = l.npad * k / 8;
        try e.run(pack, .{ (words + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ out.w, out.stream_w, @as(c_int, @intCast(k / 64)), @as(c_int, @intCast(l.npad / 64)) });
    }
    // the drafter's scales and biases dictionary-coded as the target's (TF_DRAFT_DICT=0 keeps them uncoded); its
    // biases are each group's minimum, so the joint pairs outnumber a pair table and the separate coding serves
    if (e.dict) |*ds| if (!std.mem.eql(u8, std.mem.span(@import("sys").c.getenv("TF_DRAFT_DICT") orelse ""), "0")) {
        const kmajor = e.dict_kmajor and n == 17408 and k == 5120;
        try e.affine.encode(e.stream, &out, ds, kmajor, cuda.qlinear.pairTabRoom(n, k, kmajor));
    };
    try e.stream.synchronize();
    return out;
}

fn vector(w: *Weights, ck: *core.Checkpoint, name: []const u8, shape: []const usize) !u64 {
    const t = try ck.expect(name, .bf16, shape);
    const p = try w.alloc(t.bytes.len);
    try w.arena.upload(p - w.arena.ptr, t.bytes);
    return p;
}

fn hostBf16(gpa: std.mem.Allocator, ck: *core.Checkpoint, name: []const u8, shape: []const usize) ![]u16 {
    const t = try ck.expect(name, .bf16, shape);
    const out = try gpa.alloc(u16, t.bytes.len / 2);
    @memcpy(out, std.mem.bytesAsSlice(u16, t.bytes));
    return out;
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8, s: Shape) !Weights {
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var sponge = @import("cuda_sponge.zig").take(e.d, total(s));
    defer sponge.free();
    var w: Weights = .{ .fc = undefined, .hidden_norm = 0, .final_norm = 0, .selector = undefined, .predecessor = &.{}, .successor = &.{}, .layers = &.{}, .arena = try DeviceBuffer.alloc(e.d, total(s)) };
    errdefer w.arena.free();
    try w.arena.fill8(0, null);
    var stage = try DeviceBuffer.alloc(e.d, stageBytes(s));
    defer stage.free();
    w.layers = try gpa.alloc(Layer, s.layers);
    errdefer gpa.free(w.layers);
    w.fc = try linear(gpa, e, &stage, &w, &ck, "fc.weight", s.hidden, s.taps * s.hidden);
    w.hidden_norm = try vector(&w, &ck, "hidden_norm.weight", &.{s.hidden});
    w.final_norm = try vector(&w, &ck, "norm.weight", &.{s.hidden});
    w.selector = try linear(gpa, e, &stage, &w, &ck, "candidate_selector.hidden_projection.weight", s.rank, s.hidden);
    w.predecessor = try hostBf16(gpa, &ck, "candidate_selector.predecessor_codebook", &.{ s.vocab, s.rank });
    errdefer gpa.free(w.predecessor);
    w.successor = try hostBf16(gpa, &ck, "candidate_selector.successor_codebook", &.{ s.vocab, s.rank });
    errdefer gpa.free(w.successor);
    for (w.layers, 0..) |*l, i| {
        var buf: [96]u8 = undefined;
        const Name = struct {
            buf: *[96]u8,
            i: usize,
            fn of(self: @This(), suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(self.buf, "layers.{d}.{s}", .{ self.i, suffix });
            }
        };
        const nm: Name = .{ .buf = &buf, .i = i };
        l.* = .{
            .input_norm = try vector(&w, &ck, try nm.of("input_layernorm.weight"), &.{s.hidden}),
            .post_norm = try vector(&w, &ck, try nm.of("post_attention_layernorm.weight"), &.{s.hidden}),
            .q = try linear(gpa, e, &stage, &w, &ck, try nm.of("self_attn.q_proj.weight"), s.q_width, s.hidden),
            .k = try linear(gpa, e, &stage, &w, &ck, try nm.of("self_attn.k_proj.weight"), s.kv_width, s.hidden),
            .v = try linear(gpa, e, &stage, &w, &ck, try nm.of("self_attn.v_proj.weight"), s.kv_width, s.hidden),
            .o = try linear(gpa, e, &stage, &w, &ck, try nm.of("self_attn.o_proj.weight"), s.hidden, s.q_width),
            .q_norm = try vector(&w, &ck, try nm.of("self_attn.q_norm.weight"), &.{s.head_dim}),
            .k_norm = try vector(&w, &ck, try nm.of("self_attn.k_norm.weight"), &.{s.head_dim}),
            .gate = try linear(gpa, e, &stage, &w, &ck, try nm.of("mlp.gate_proj.weight"), s.intermediate, s.hidden),
            .up = try linear(gpa, e, &stage, &w, &ck, try nm.of("mlp.up_proj.weight"), s.intermediate, s.hidden),
            .down = try linear(gpa, e, &stage, &w, &ck, try nm.of("mlp.down_proj.weight"), s.hidden, s.intermediate),
            .attn_conv = try linear(gpa, e, &stage, &w, &ck, try nm.of("attention_conv.kernel_projection.weight"), s.dynamic, s.hidden),
            .attn_base = try vector(&w, &ck, try nm.of("attention_conv.base_kernel"), &.{ 2, 2, s.hidden }),
            .mlp_conv = try linear(gpa, e, &stage, &w, &ck, try nm.of("mlp_conv.kernel_projection.weight"), s.dynamic, s.hidden),
            .mlp_base = try vector(&w, &ck, try nm.of("mlp_conv.base_kernel"), &.{ 2, 2, s.hidden }),
        };
    }
    return w;
}

test "bf16 rounding of a quantized group's scale and bias" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(f32, 1.0), bf16(0x3f80));
    const w: [64]u16 = @splat(0x3f80); // a flat group: scale 0, bias 1, every nibble 0
    const q = try quantize(std.testing.allocator, &w, 1, 64);
    defer {
        std.testing.allocator.free(q.words);
        std.testing.allocator.free(q.scales);
        std.testing.allocator.free(q.biases);
    }
    try std.testing.expectEqual(@as(u16, 0), q.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3f80), q.biases[0]);
    try std.testing.expectEqual(@as(u32, 0), q.words[0]);
}


test "drafter refit preserves flat groups and improves a linear weight ramp" {
    const flat: [64]u16 = @splat(0x3f80);
    const unchanged = leastSquares(&flat, .{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 0), unchanged[0]);
    try std.testing.expectEqual(@as(f32, 1), unchanged[1]);
    var row: [64]u16 = undefined;
    for (&row, 0..) |*h, i| h.* = toBf16(@as(f32, @floatFromInt(i)) / 63.0 - 0.5);
    const initial = [2]f32{ 1.0 / 15.0, -0.5 };
    const fitted = leastSquares(&row, initial);
    try std.testing.expect(groupError(&row, fitted) < groupError(&row, initial));
}
