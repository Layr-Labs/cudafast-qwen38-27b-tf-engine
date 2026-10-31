//! The DFlash drafter's span head at fewer bits: the target head's span rows requantized at load (drafter numerics only).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const AffineN = cuda.qlinear.AffineN;

/// The span head's rows (cuda_draft_frame's spans): the first 98,304, then the 64-row tile holding the last 288.
pub const first_rows = 98304;
pub const tail_from = 248032 / 64 * 64;

/// The two projections and the buffer that owns them.
pub const Head = struct {
    buffer: cuda.DeviceBuffer,
    first: AffineN,
    tail: AffineN,

    pub fn deinit(h: *Head) void {
        h.buffer.free();
    }
};

/// The width the drafter reads its span head at: 3 bits (it kept the 4-bit head's accepted tokens a round), 2 by
/// TF_DRAFT_HEAD_BITS=2, the target's own 4-bit head by TF_DRAFT_HEAD_BITS=4 (or 0).
pub fn bitsWanted() u8 {
    const v = @import("sys").c.getenv("TF_DRAFT_HEAD_BITS") orelse return 3;
    const b = std.fmt.parseInt(u8, std.mem.span(v), 10) catch return 3;
    return if (b == 2 or b == 3) b else 0;
}

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn f32bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// Rows [from, from + n) of an MLX affine-4 (vocab, k) head as MLX rows of `bits` codes: each group of 64 dequantized,
/// then min/max round-to-nearest at the narrower width with bf16 scale and bias. Codes are a little-endian bit stream.
pub fn requantize(w4: []align(1) const u32, s4: []align(1) const u16, b4: []align(1) const u16, k: usize, from: usize, n: usize, bits: u8, words: []u32, scales: []u16, biases: []u16) void {
    const groups = k / 64;
    const top: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    const row_words = k * bits / 32;
    @memset(words, 0);
    for (0..n) |r| {
        const src = from + r;
        for (0..groups) |g| {
            var vals: [64]f32 = undefined;
            const s = bf16f(s4[src * groups + g]);
            const b = bf16f(b4[src * groups + g]);
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            for (&vals, 0..) |*v, i| {
                const at = g * 64 + i;
                const code: f32 = @floatFromInt((w4[src * (k / 8) + at / 8] >> @intCast(4 * (at % 8))) & 0xf);
                v.* = code * s + b;
                lo = @min(lo, v.*);
                hi = @max(hi, v.*);
            }
            const ns = f32bf16((hi - lo) / top);
            const nb = f32bf16(lo);
            scales[r * groups + g] = ns;
            biases[r * groups + g] = nb;
            const sf = bf16f(ns);
            const bfv = bf16f(nb);
            const inv: f32 = if (sf == 0) 0 else 1.0 / sf;
            for (vals, 0..) |v, i| {
                const q: u32 = @intFromFloat(std.math.clamp(@round((v - bfv) * inv), 0, top));
                const bit = (g * 64 + i) * bits;
                const word = r * row_words + bit / 32;
                const off: u5 = @intCast(bit % 32);
                words[word] |= q << off;
                if (@as(usize, off) + bits > 32) words[word + 1] |= q >> @intCast(32 - @as(usize, off));
            }
        }
    }
}

/// The span copy of the target head `prefix ++ name`, or null when not wanted or the head is not affine-4 of the span size.
pub fn prepare(e: anytype, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, vocab: usize, k: usize) !?Head {
    const bits = bitsWanted();
    if (bits == 0 or vocab < tail_from + 320 or k % 64 != 0) return null;
    var buf: [160]u8 = undefined;
    const w4 = std.mem.bytesAsSlice(u32, (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, name }), .u32, &.{ vocab, k / 8 })).bytes);
    const s4 = std.mem.bytesAsSlice(u16, (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.scales", .{ prefix, name }), .bf16, &.{ vocab, k / 64 })).bytes);
    const b4 = std.mem.bytesAsSlice(u16, (try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.biases", .{ prefix, name }), .bf16, &.{ vocab, k / 64 })).bytes);
    const parts = [2][2]usize{ .{ 0, first_rows }, .{ tail_from, vocab - tail_from } };
    var size: usize = 0;
    for (parts) |p| {
        const l = AffineN.layout(p[1], k, bits);
        size += l.words + 2 * l.scales;
    }
    var buffer = try cuda.DeviceBuffer.alloc(e.d, size);
    errdefer buffer.free();
    var out: [2]AffineN = undefined;
    var at: usize = 0;
    for (parts, &out) |p, *q| {
        const n = p[1];
        const words = try e.gpa.alloc(u32, n * k * bits / 32);
        defer e.gpa.free(words);
        const scales = try e.gpa.alloc(u16, n * (k / 64));
        defer e.gpa.free(scales);
        const biases = try e.gpa.alloc(u16, n * (k / 64));
        defer e.gpa.free(biases);
        requantize(w4, s4, b4, k, p[0], n, bits, words, scales, biases);
        const raw_bytes = words.len * 4 + 2 * scales.len * 2;
        var stage = try cuda.DeviceBuffer.alloc(e.d, raw_bytes);
        defer stage.free();
        try stage.upload(0, std.mem.sliceAsBytes(words));
        try stage.upload(words.len * 4, std.mem.sliceAsBytes(scales));
        try stage.upload(words.len * 4 + scales.len * 2, std.mem.sliceAsBytes(biases));
        const l = AffineN.layout(n, k, bits);
        q.* = .{ .w = buffer.ptr + at, .s = buffer.ptr + at + l.words, .b = buffer.ptr + at + l.words + l.scales, .n = n, .k = k, .npad = l.npad, .bits = bits };
        at += l.words + 2 * l.scales;
        try e.wide.packWords(e.stream, stage.ptr, bits, n, k, q.w);
        try e.wide.packTable(e.stream, stage.ptr + words.len * 4, n, k, q.npad, q.s);
        try e.wide.packTable(e.stream, stage.ptr + words.len * 4 + scales.len * 2, n, k, q.npad, q.b);
        try e.stream.synchronize(); // the stage is freed next
    }
    return .{ .buffer = buffer, .first = out[0], .tail = out[1] };
}

test "draft head requantize: a two-level 4-bit group keeps its two values at 2 and 3 bits" {
    const k = 64;
    var w4: [k / 8]u32 = undefined;
    for (&w4, 0..) |*w, i| w.* = if (i % 2 == 0) 0x00000000 else 0xffffffff; // codes 0 and 15
    const s4 = [_]u16{f32bf16(0.5)};
    const b4 = [_]u16{f32bf16(-1.0)};
    for ([_]u8{ 2, 3 }) |bits| {
        var words: [k * 3 / 32]u32 = undefined;
        var scales: [1]u16 = undefined;
        var biases: [1]u16 = undefined;
        requantize(&w4, &s4, &b4, k, 0, 1, bits, words[0 .. k * bits / 32], &scales, &biases);
        const top: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
        for (0..k) |i| {
            const bit = i * bits;
            var code = words[bit / 32] >> @intCast(bit % 32);
            if (bit % 32 + bits > 32) code |= words[bit / 32 + 1] << @intCast(32 - bit % 32);
            code &= top;
            const v = @as(f32, @floatFromInt(code)) * bf16f(scales[0]) + bf16f(biases[0]);
            const want: f32 = if ((i / 8) % 2 == 0) -1.0 else 6.5;
            try std.testing.expectApproxEqAbs(want, v, 0.05);
        }
    }
}
