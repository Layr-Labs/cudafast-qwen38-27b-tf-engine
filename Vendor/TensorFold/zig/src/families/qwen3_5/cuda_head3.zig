//! The greedy verify head of up to 32 rows through a 2-bit screen and an exact rescore (kernels/cuda/qwen35_head3.cu); a
//! window of 17 to 32 rows takes it too (Engine.headArgmax), on the screen's two-tile form. Each row's token is
//! the highest exactly recomputed logit among the row's selected columns (ties to the lowest id), kept when the row's cut
//! (the largest screened bound left out) is below the bf16 value under that logit; a window with any other row gets
//! 0xFFFFFFFF there, and Engine.forward runs the full head over it (Engine.head3Settle). TF_HEAD3=0 keeps the full
//! head on every window; TF_HEAD3_KAPPA sets the bound's multiple of the dropped bits' spread (default 8).
const std = @import("std");
const cuda = @import("cuda");

/// Each row's selected columns, and the room for the ties at the cut.
pub const k_select: usize = 512;
pub const cap: usize = 640;
const stages: usize = 8;
/// The 2-bit codes in tile ranges of at most one fast pool piece each (qwen35_head3.cu Chunks).
const chunks_max = 4;
const Chunks = extern struct { p: [chunks_max]u64, tpc: c_int };
/// qwen35_head3.cu Stage<1>::BYTES and Stage<2>::BYTES: one and two m16 row tiles a stage.
const stage_bytes = [2]usize{ 3456, 5632 };
pub const rows_max: usize = 32;
/// Compile-time switch: false keeps the greedy verify head on the 2-bit sieve (Engine.sieve) instead of this screen.
pub const head3_on = false;

pub const Head3 = struct {
    mod: cuda.Module,
    sq: cuda.Function,
    screen: cuda.Function,
    screen32: cuda.Function,
    select: cuda.Function,
    rescore: cuda.Function,
    merge: cuda.Function,
    /// Each column's exact pairs and a window's scratch, plus the codes and screen scales the fast pool did not take.
    buf: cuda.DeviceBuffer,
    codes: Chunks,
    scales: u64,
    tiles: usize,
    kg: usize,
    n: usize,
    kappa: f32,

    fn codesBytes(h: *const Head3) usize {
        return h.tiles * h.kg * 256 * 4;
    }
    fn scalesBytes(h: *const Head3) usize {
        return h.tiles * h.kg * 64 * 4;
    }
    fn sbBytes(h: *const Head3) usize {
        return h.n * h.kg * 4;
    }
    fn scratch(h: *const Head3) u64 {
        return h.buf.ptr + h.sbBytes();
    }

    /// The window scratch: sums of squares, bounds, tile maxima, candidates' values and ids, cuts.
    const Scratch = struct { xq: u64, u: u64, tm: u64, pv: u64, pi: u64, cut: u64 };
    fn parts(h: *const Head3) Scratch {
        const at = h.scratch();
        const xq = at;
        const u = xq + rows_max * h.kg * 4;
        const tm = u + rows_max * h.n * 2;
        const pv = tm + (rows_max * h.tiles * 2 + 15) / 16 * 16;
        const pi = pv + rows_max * cap * 4;
        const cut = pi + rows_max * cap * 4;
        return .{ .xq = xq, .u = u, .tm = tm, .pv = pv, .pi = pi, .cut = cut };
    }
    fn scratchBytes(tiles: usize, kg: usize, n: usize) usize {
        return rows_max * kg * 4 + rows_max * n * 2 + (rows_max * tiles * 2 + 15) / 16 * 16 + 2 * rows_max * cap * 4 + rows_max * 4;
    }

    /// The head's tokens of `rows` (at most 32) rows of x (pitch k, with their group sums xs) into toks.
    pub fn run(h: *const Head3, s: cuda.Stream, x: u64, xs: u64, q: cuda.qlinear.Affine4, rows: usize, toks: u64) !void {
        const p = h.parts();
        const kg: c_int = @intCast(h.kg);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(kg);
        a.add(p.xq);
        try cuda.launch.launch(h.sq, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
        var b: cuda.Args = .{};
        for ([_]u64{ x, xs, p.xq }) |v| b.add(v);
        b.add(h.codes);
        b.add(h.scales);
        b.add(@as(c_int, @intCast(rows)));
        b.add(kg);
        b.add(h.kappa);
        b.add(p.u);
        b.add(@as(c_int, @intCast(h.n)));
        b.add(p.tm);
        b.add(@as(c_int, @intCast(h.tiles)));
        const two = rows > 16;
        try cuda.launch.launch(if (two) h.screen32 else h.screen, .{ .grid = .{ .x = @intCast(h.tiles) }, .block = .{ .x = 128 }, .shared = @intCast(stages * stage_bytes[@intFromBool(two)]) }, s, &b);
        var c: cuda.Args = .{};
        c.add(p.u);
        c.add(@as(c_int, @intCast(h.n)));
        c.add(p.tm);
        c.add(@as(c_int, @intCast(h.tiles)));
        c.add(@as(c_int, @intCast(k_select)));
        c.add(@as(c_int, @intCast(cap)));
        for ([_]u64{ p.pv, p.pi, p.cut }) |v| c.add(v);
        try cuda.launch.launch(h.select, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &c);
        var r: cuda.Args = .{};
        for ([_]u64{ x, xs, q.w, h.buf.ptr }) |v| r.add(v);
        r.add(kg);
        r.add(@as(c_int, @intCast(rows)));
        r.add(p.pv);
        r.add(p.pi);
        r.add(@as(c_int, @intCast(cap)));
        r.add(@as(c_int, @intCast(h.n)));
        try cuda.launch.launch(h.rescore, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(cap / 32) }, .block = .{ .x = 128 } }, s, &r);
        var m: cuda.Args = .{};
        m.add(p.pv);
        m.add(p.pi);
        m.add(@as(c_int, @intCast(cap)));
        m.add(p.cut);
        m.add(toks);
        try cuda.launch.launch(h.merge, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, s, &m);
    }

    pub fn deinit(h: *Head3) void {
        h.buf.free();
        h.mod.unload();
    }
};

fn kappaEnv() f32 {
    const v = @import("sys").c.getenv("TF_HEAD3_KAPPA") orelse return 8.0;
    const k = std.fmt.parseFloat(f32, std.mem.span(v)) catch return 8.0;
    return if (k >= 4.0 and k <= 64.0) k else 8.0;
}

fn headOf(e: anytype) ?cuda.qlinear.Affine4 {
    // this tree's verify head is 14546ab5's 2-bit sieve (head_sieve.zig, cuda_forward.zig): no screened head3
    if (!head3_on) return null;
    if (@import("sys").c.getenv("TF_HEAD3")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return null;
    if (!e.head_argmax) return null;
    const q = switch (e.head) {
        .a4 => |q| q,
        else => return null,
    };
    if (q.k != 5120 or q.n % 64 != 0 or q.npad < q.n or q.n > std.math.maxInt(c_int) or !e.affine.headArgmaxFits(q, 16)) return null;
    return q;
}

/// Tiles a code chunk holds: as many 64-column tiles as fill one fast pool piece.
fn tilesPerChunk(kg: usize) usize {
    return @import("cuda_fastpool.zig").piece / (kg * 256 * 4);
}

/// The codes' chunks and the screen scales, in the order init asks the fast pool for them (after the streams and panels):
/// their sizes into out, returning how many (0 when the copy will not be made).
pub fn poolSizes(e: anytype, out: []usize) usize {
    const q = headOf(e) orelse return 0;
    const kg = q.k / 64;
    const tiles = q.n / 64;
    const tpc = tilesPerChunk(kg);
    const nc = (tiles + tpc - 1) / tpc;
    if (nc > chunks_max or out.len < nc + 1) return 0;
    for (0..nc) |i| out[i] = @min(tpc, tiles - i * tpc) * kg * 256 * 4;
    out[nc] = tiles * kg * 64 * 4;
    return nc + 1;
}

/// The copy of a dictionary-coded 4-bit head the full greedy head takes (Affine4Kernels.headArgmaxFits at 16 rows), packed
/// once: the codes' chunks and the screen scales on the fast pool when it screened room for them (poolSizes), else in the
/// copy's own buffer; null when TF_HEAD3=0 or the head is of another shape.
pub fn init(e: anytype) !?Head3 {
    const q = headOf(e) orelse return null;
    var mod = try cuda.Module.load(e.d, cuda.kernels.qwen35_head3);
    errdefer mod.unload();
    var h: Head3 = .{
        .mod = mod,
        .sq = try mod.function("tf_q35h3_sq"),
        .screen = try mod.function("tf_q35h3_screen"),
        .screen32 = try mod.function("tf_q35h3_screen32"),
        .select = try mod.function("tf_q35h3_select"),
        .rescore = try mod.function("tf_q35h3_rescore"),
        .merge = try mod.function("tf_q35h3_merge"),
        .buf = undefined,
        .codes = undefined,
        .scales = 0,
        .tiles = q.n / 64,
        .kg = q.k / 64,
        .n = q.n,
        .kappa = kappaEnv(),
    };
    const pack = try mod.function("tf_q35h3_pack");
    // the chunks and the scales on the pool's pieces (asked for in poolSizes' order), all or none
    const tpc = tilesPerChunk(h.kg);
    const nc = (h.tiles + tpc - 1) / tpc;
    if (nc > chunks_max) return null;
    var places: [chunks_max + 1]u64 = @splat(0);
    var pooled = false;
    if (e.fast) |*pool| {
        pooled = true;
        for (0..nc) |i| places[i] = pool.alloc(@min(tpc, h.tiles - i * tpc) * h.kg * 256 * 4) orelse {
            pooled = false;
            break;
        };
        if (pooled) places[nc] = pool.alloc(h.scalesBytes()) orelse blk: {
            pooled = false;
            break :blk 0;
        };
    }
    const own = if (pooled) 0 else h.codesBytes() + h.scalesBytes();
    h.buf = try cuda.DeviceBuffer.alloc(e.d, h.sbBytes() + Head3.scratchBytes(h.tiles, h.kg, h.n) + own);
    errdefer h.buf.free();
    if (!pooled) {
        const at = h.buf.ptr + h.sbBytes() + Head3.scratchBytes(h.tiles, h.kg, h.n);
        for (0..nc) |i| places[i] = at + i * tpc * h.kg * 256 * 4;
        places[nc] = at + h.codesBytes();
    }
    h.codes = .{ .p = @splat(places[0]), .tpc = @intCast(tpc) };
    for (0..nc) |i| h.codes.p[i] = places[i];
    h.scales = places[nc];
    var a: cuda.Args = .{};
    for ([_]u64{ q.w, q.s, q.b }) |v| a.add(v);
    a.add(@as(c_int, @intCast(q.npad)));
    a.add(@as(c_int, @intCast(h.kg)));
    a.add(h.codes);
    a.add(h.scales);
    a.add(h.buf.ptr);
    try cuda.launch.launch(pack, .{ .grid = .{ .x = @intCast(h.tiles), .y = @intCast(h.kg) }, .block = .{ .x = 256 } }, e.stream, &a);
    try e.stream.synchronize();
    return h;
}