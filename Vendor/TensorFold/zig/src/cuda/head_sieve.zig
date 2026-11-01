//! The greedy verify head's sieve (qmm_head_sieve.cuh, in the qmm_group module): a 2-bit copy of the vocabulary
//! head's codes (code >> 2, the same scales and biases) built at engine open; a greedy verify then streams it in place
//! of the 4-bit words, recomputes each row's first tile (the n8 tile of its best approximate logit) with the head
//! tile's own arithmetic, and then only the 8-id column groups whose approximate logit plus kappa spreads of the
//! dropped bits' term reaches the lower of that tile's exact best and its best approximate logit. Each row's token is
//! the best of those exact values: the full head's token whenever the best id's dropped bits' term exceeds the first
//! tile's best-approximate id's by less than kappa spreads.
//! TF_HEAD_SIEVE=0: the full head; =check: the full head writes the tokens and the sieve runs beside it, compared.
//! TF_HEAD_SIEVE_K: kappa (10); TF_HEAD_SIEVE_CAP: the most n8 tiles rescored before every tile is (4096).
const std = @import("std");
const linear = @import("qlinear.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const DeviceBuffer = memory.DeviceBuffer;
const Affine4 = linear.Affine4;

fn int(v: usize) c_int {
    return @intCast(v);
}

fn envOff(name: [*:0]const u8) bool {
    const v = @import("sys").c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Mode = enum { on, check };

pub const HeadSieve = struct {
    pub const rows_max = 16;
    pub const sieve_smem: u32 = 4 * (2048 + 4 * 1024 + 4 * 256 + 128); // tf_head_sieve::SMEM
    pub const rescore_smem: u32 = 8 * (2048 + 4 * 288 + 64); // tf_head_sieve::RS_SMEM
    /// The pair table in shared memory when three scan CTAs an SM still fit (GB10: 100 KiB an SM, 1 KiB a CTA).
    const tab_room: u32 = 102400 / 3 - 1024 - sieve_smem;
    const stats_words = 64;
    const rescore_ctas = 144; // three an SM (GB10: 48 SMs)
    const select_ctas = 48;

    pack_fn: Function,
    prep_fn: Function,
    sieve_fn: Function,
    top_fn: Function,
    select_fn: Function,
    rescore_fn: Function,
    finish_fn: Function,
    check_fn: Function,
    rescore_dense_fn: Function,
    fill_fn: Function,
    top4_fn: Function,
    select4_fn: Function,
    buf: DeviceBuffer,
    mode: Mode,
    kappa: f32,
    /// spanTop4's kappa (TF_DRAFT_SIEVE_K, 3): the drafter's proposals only, never a committed token.
    draft_kappa: f32,
    cap: usize,
    tiles: usize,
    kg: usize,
    n8s: usize,
    ctas: usize,
    code: u32, // the head's pair code, or 0: the sieve reads the (kg, npad) arrays
    tab_smem: u32,
    pdl: bool,
    w2: u64,
    afr: u64,
    xv: u64,
    himax: u64,
    himid: u64, // each (n8 tile, row)'s best approximate logit
    list: u64,
    count: u64,
    best: u64,
    pre: u64, // each row's first tile: the n8 tile of its highest bound
    toks: u64,
    stats: u64,
    mrow: u64,
    approx: u64, // check mode: every (id, row)'s approximate logit and spread
    list4: u64, // spanTop4: each row's four first tiles
    count4: u64,
    fourth: u64, // spanTop4: each row's fourth first tile's best a
    rounds: usize = 0,
    next_log: usize = 1,

    /// Whether the sieve takes q: the vocabulary head's tile-major words in 256-id CTAs.
    pub fn fits(q: Affine4) bool {
        return q.n % 256 == 0 and q.k % 64 == 0 and q.n >= 65536 and q.npad >= q.n and q.code & linear.dict_kmajor == 0;
    }

    /// From the loaded qmm_group module and the loaded head; null when TF_HEAD_SIEVE=0 or q does not fit.
    pub fn init(d: *const Driver, group_mod: Module, q: Affine4, s: Stream, pdl: bool) !?HeadSieve {
        const env = @import("sys").c.getenv("TF_HEAD_SIEVE");
        var mode: Mode = .on;
        if (env) |v| {
            const m = std.mem.span(v);
            if (std.mem.eql(u8, m, "0")) return null;
            if (std.mem.eql(u8, m, "check")) mode = .check;
        }
        if (!fits(q)) {
            std.debug.print("info: head sieve off: head n {d} k {d} npad {d} code {x}\n", .{ q.n, q.k, q.npad, q.code });
            return null;
        }
        var kappa: f32 = 10.0;
        if (@import("sys").c.getenv("TF_HEAD_SIEVE_K")) |v| kappa = std.fmt.parseFloat(f32, std.mem.span(v)) catch 10.0;
        var cap: usize = 4096;
        if (@import("sys").c.getenv("TF_HEAD_SIEVE_CAP")) |v| cap = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4096;
        var h: HeadSieve = undefined;
        h.pack_fn = try group_mod.function("tf_head_sieve_pack");
        h.prep_fn = try group_mod.function("tf_head_sieve_prep");
        h.sieve_fn = try group_mod.function("tf_head_sieve_scan");
        h.top_fn = try group_mod.function("tf_head_sieve_top");
        h.select_fn = try group_mod.function("tf_head_sieve_select");
        h.rescore_fn = try group_mod.function("tf_head_sieve_rescore");
        h.finish_fn = try group_mod.function("tf_head_sieve_finish");
        h.check_fn = try group_mod.function("tf_head_sieve_check");
        h.rescore_dense_fn = try group_mod.function("tf_head_sieve_rescore_dense");
        h.fill_fn = try group_mod.function("tf_head_sieve_fill");
        h.top4_fn = try group_mod.function("tf_head_sieve_top4");
        h.select4_fn = try group_mod.function("tf_head_sieve_select4");
        h.mode = mode;
        h.kappa = kappa;
        h.draft_kappa = 3.0;
        if (@import("sys").c.getenv("TF_DRAFT_SIEVE_K")) |v| h.draft_kappa = std.fmt.parseFloat(f32, std.mem.span(v)) catch 3.0;
        h.cap = cap;
        h.tiles = q.n / 64;
        h.kg = q.k / 64;
        h.n8s = q.n / 8;
        h.ctas = h.tiles / 4;
        h.pdl = pdl;
        h.rounds = 0;
        h.next_log = 1;
        h.code = if (linear.pairCode(q.code)) q.code else 0;
        const tab_bytes: u32 = if (h.code != 0) ((h.code >> 5 & 0x1FFF) * 4 + 15) / 16 * 16 else 0;
        h.tab_smem = if (tab_bytes <= tab_room) tab_bytes else 0;
        try h.sieve_fn.allowDynamicShared(sieve_smem + h.tab_smem);
        try h.rescore_fn.allowDynamicShared(rescore_smem);
        try h.rescore_dense_fn.allowDynamicShared(rescore_smem);
        // the buffer: 2-bit words, A fragments, group sums, the sieve's bests, list, count, bests, tokens, counters
        var at: usize = 0;
        const take = struct {
            fn f(p: *usize, bytes: usize) usize {
                const o = p.*;
                p.* += (bytes + 255) / 256 * 256;
                return o;
            }
        }.f;
        const o_w2 = take(&at, h.tiles * h.kg * 1024);
        const o_afr = take(&at, h.kg * 2048);
        const o_xv = take(&at, h.kg * 128);
        const o_himax = take(&at, h.n8s * rows_max * 4);
        const o_himid = take(&at, h.n8s * rows_max * 4);
        const o_list = take(&at, cap * 4);
        const o_count = take(&at, 4);
        const o_best = take(&at, rows_max * 8);
        const o_pre = take(&at, rows_max * 8);
        const o_toks = take(&at, rows_max * 4);
        const o_stats = take(&at, stats_words * 8);
        const o_mrow = take(&at, rows_max * 4);
        const o_approx = if (mode == .check) take(&at, q.n * 32 * 4) else 0;
        const o_list4 = take(&at, rows_max * 4 * 4);
        const o_count4 = take(&at, 4);
        const o_fourth = take(&at, rows_max * 4);
        h.buf = try DeviceBuffer.alloc(d, at);
        errdefer h.buf.free();
        try h.buf.fill8(0, null);
        const b = h.buf.ptr;
        h.w2 = b + o_w2;
        h.afr = b + o_afr;
        h.xv = b + o_xv;
        h.himax = b + o_himax;
        h.himid = b + o_himid;
        h.list = b + o_list;
        h.count = b + o_count;
        h.best = b + o_best;
        h.pre = b + o_pre;
        h.toks = b + o_toks;
        h.stats = b + o_stats;
        h.mrow = b + o_mrow;
        h.approx = if (mode == .check) b + o_approx else 0;
        h.list4 = b + o_list4;
        h.count4 = b + o_count4;
        h.fourth = b + o_fourth;
        const least: u64 = std.math.maxInt(u64);
        try h.buf.upload(o_stats + 50 * 8, std.mem.asBytes(&least));
        try h.buf.upload(o_stats + 52 * 8, std.mem.asBytes(&least));
        var a: launch_.Args = .{};
        a.add(q.w);
        a.add(h.w2);
        try launch_.launch(h.pack_fn, .{ .grid = .{ .x = @intCast(h.tiles * h.kg) }, .block = .{ .x = 256 } }, s, &a);
        try s.synchronize();
        std.debug.print("info: head sieve ({s}): kappa {d:.2}, cap {d} n8 tiles, 2-bit copy {d} MiB, scales {s} (np {d}, table {s}), {d} sieve CTAs\n", .{
            @tagName(mode),                            kappa,                                  cap, h.tiles * h.kg * 1024 >> 20,
            if (h.code != 0) "pair dictionary" else "arrays", if (h.code != 0) h.code >> 5 & 0x1FFF else 0,
            if (h.tab_smem != 0) "shared" else "L2", h.ctas,
        });
        return h;
    }

    pub fn deinit(h: *HeadSieve) void {
        h.buf.free();
    }

    /// A greedy verify's head over `rows` rows of x (bf16, k wide) with group sums xs: each row's token into toks
    /// (check mode: into the sieve's own, compared with toks, which the full head has written).
    pub fn run(h: *const HeadSieve, s: Stream, x: u64, xs: u64, q: Affine4, rows: usize, toks: u64) !void {
        if (rows == 0 or rows > rows_max) return error.WindowTooWide;
        const out = if (h.mode == .check) h.toks else toks;
        {
            var a: launch_.Args = .{};
            a.add(x);
            a.add(xs);
            a.add(int(rows));
            a.add(int(q.k));
            a.add(h.afr);
            a.add(h.xv);
            a.add(h.best);
            a.add(h.pre);
            a.add(h.count);
            try launch_.launch(h.prep_fn, .{ .grid = .{ .x = @intCast(h.kg) }, .block = .{ .x = 128 }, .pdl = h.pdl }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(h.w2);
            a.add(h.afr);
            a.add(h.xv);
            a.add(q.sb);
            a.add(q.tab);
            a.add(@as(c_int, @bitCast(h.code)));
            a.add(@as(c_int, @intCast(h.tab_smem)));
            a.add(q.s);
            a.add(q.b);
            a.add(int(q.npad));
            a.add(int(h.kg));
            a.add(int(rows));
            a.add(h.kappa);
            a.add(h.himax);
            a.add(h.himid);
            a.add(h.approx);
            try launch_.launch(h.sieve_fn, .{ .grid = .{ .x = @intCast(h.ctas) }, .block = .{ .x = 128 }, .shared = sieve_smem + h.tab_smem, .pdl = h.pdl }, s, &a);
        }
        // each row's first tile (its best a) exactly; the row's threshold the lower of its exact best and its best a
        {
            var a: launch_.Args = .{};
            a.add(h.himid);
            a.add(int(rows));
            a.add(int(h.n8s));
            a.add(h.pre);
            try launch_.launch(h.top_fn, .{ .grid = .{ .x = select_ctas }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        try h.rescore(s, q, rows, h.pre);
        {
            var a: launch_.Args = .{};
            a.add(h.himax);
            a.add(h.himid);
            a.add(h.pre);
            a.add(h.best);
            a.add(int(rows));
            a.add(int(h.n8s));
            a.add(h.list);
            a.add(h.count);
            a.add(int(h.cap));
            a.add(h.mrow);
            try launch_.launch(h.select_fn, .{ .grid = .{ .x = select_ctas }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        try h.rescore(s, q, rows, 0);
        {
            var a: launch_.Args = .{};
            a.add(h.best);
            a.add(int(rows));
            a.add(out);
            a.add(h.count);
            a.add(int(h.cap));
            a.add(int(h.n8s));
            a.add(h.stats);
            try launch_.launch(h.finish_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .pdl = h.pdl }, s, &a);
        }
        if (h.mode == .check) {
            var a: launch_.Args = .{};
            a.add(h.approx);
            a.add(int(q.n));
            a.add(h.kappa);
            a.add(h.mrow);
            a.add(toks);
            a.add(h.toks);
            a.add(h.best);
            a.add(h.stats);
            try launch_.launch(h.check_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 }, .pdl = h.pdl }, s, &a);
        }
    }

    /// Whether spanTop4 takes the drafter frame's span: the first `span_tiles` 64-id tiles of the head, `rows` rows.
    pub fn spans(h: *const HeadSieve, rows: usize, span_tiles: usize) bool {
        return h.mode == .on and rows > 0 and rows <= rows_max and span_tiles > 0 and span_tiles % 4 == 0 and span_tiles <= h.tiles and
            !envOff("TF_DRAFT_SIEVE");
    }

    /// The drafter frame's four best ids a row over the head's first `span_tiles` 64-id tiles, for `rows` rows of x
    /// (bf16, k wide) with group sums xs: the sieve's 2-bit scan over those tiles, each row's four tiles of highest
    /// best a exactly, then every 8-id group whose a + kappa spreads reaches the lower of the row's fourth best exact
    /// value and its fourth tile's best a, exactly. Every rescored (row, id) value goes into dense ([row][ld] bf16,
    /// span_tiles * 64 ids a row) and every other id of the span is -inf, so a top-4 over dense takes the four ids the
    /// span's full logits give whenever each of them is within kappa spreads (the verify sieve's bound).
    pub fn spanTop4(h: *const HeadSieve, s: Stream, x: u64, xs: u64, q: Affine4, rows: usize, span_tiles: usize, dense: u64, ld: usize) !void {
        const n8s = span_tiles * 8;
        const cols = span_tiles * 64;
        {
            var a: launch_.Args = .{};
            a.add(dense);
            a.add(int(ld));
            a.add(int(cols));
            try launch_.launch(h.fill_fn, .{ .grid = .{ .x = @intCast((cols + 2047) / 2048), .y = @intCast(rows) }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(x);
            a.add(xs);
            a.add(int(rows));
            a.add(int(q.k));
            a.add(h.afr);
            a.add(h.xv);
            a.add(h.best);
            a.add(h.pre);
            a.add(h.count);
            try launch_.launch(h.prep_fn, .{ .grid = .{ .x = @intCast(h.kg) }, .block = .{ .x = 128 }, .pdl = h.pdl }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(h.w2);
            a.add(h.afr);
            a.add(h.xv);
            a.add(q.sb);
            a.add(q.tab);
            a.add(@as(c_int, @bitCast(h.code)));
            a.add(@as(c_int, @intCast(h.tab_smem)));
            a.add(q.s);
            a.add(q.b);
            a.add(int(q.npad));
            a.add(int(h.kg));
            a.add(int(rows));
            a.add(h.draft_kappa);
            a.add(h.himax);
            a.add(h.himid);
            a.add(@as(u64, 0));
            try launch_.launch(h.sieve_fn, .{ .grid = .{ .x = @intCast(span_tiles / 4) }, .block = .{ .x = 128 }, .shared = sieve_smem + h.tab_smem, .pdl = h.pdl }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(h.himid);
            a.add(int(rows));
            a.add(int(n8s));
            a.add(h.list4);
            a.add(h.count4);
            a.add(h.fourth);
            try launch_.launch(h.top4_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        try h.rescoreDense(s, q, rows, h.list4, h.count4, n8s, dense, ld, (rows * 4 + 3) / 4);
        {
            var a: launch_.Args = .{};
            a.add(h.himax);
            a.add(dense);
            a.add(int(ld));
            a.add(h.list4);
            a.add(h.fourth);
            a.add(int(rows));
            a.add(int(n8s));
            a.add(h.list);
            a.add(h.count);
            a.add(int(h.cap));
            try launch_.launch(h.select4_fn, .{ .grid = .{ .x = select_ctas }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        try h.rescoreDense(s, q, rows, h.list, h.count, n8s, dense, ld, rescore_ctas);
    }

    fn rescoreDense(h: *const HeadSieve, s: Stream, q: Affine4, rows: usize, list: u64, count: u64, n8s: usize, dense: u64, ld: usize, ctas: usize) !void {
        var a: launch_.Args = .{};
        a.add(q.w);
        a.add(q.s);
        a.add(q.b);
        a.add(int(q.npad));
        a.add(h.afr);
        a.add(h.xv);
        a.add(int(h.kg));
        a.add(int(rows));
        a.add(list);
        a.add(count);
        a.add(int(h.cap));
        a.add(int(n8s));
        a.add(h.best);
        a.add(dense);
        a.add(int(ld));
        try launch_.launch(h.rescore_dense_fn, .{ .grid = .{ .x = @intCast(ctas) }, .block = .{ .x = 128 }, .shared = rescore_smem, .pdl = h.pdl }, s, &a);
    }

    /// The listed n8 tiles exactly (keys: each row's first tile instead), each row's best into best.
    fn rescore(h: *const HeadSieve, s: Stream, q: Affine4, rows: usize, keys: u64) !void {
        var a: launch_.Args = .{};
        a.add(q.w);
        a.add(q.s);
        a.add(q.b);
        a.add(int(q.npad));
        a.add(h.afr);
        a.add(h.xv);
        a.add(int(h.kg));
        a.add(int(rows));
        a.add(h.list);
        a.add(h.count);
        a.add(int(h.cap));
        a.add(int(h.n8s));
        a.add(keys);
        a.add(h.best);
        const ctas: usize = if (keys != 0) (rows + 3) / 4 else rescore_ctas;
        try launch_.launch(h.rescore_fn, .{ .grid = .{ .x = @intCast(ctas) }, .block = .{ .x = 128 }, .shared = rescore_smem, .pdl = h.pdl }, s, &a);
    }

    fn unorder(v: u64) f32 {
        const u: u32 = @truncate(v);
        return @bitCast(if (u & 0x80000000 != 0) u & 0x7FFFFFFF else ~u);
    }

    /// After a verify that ran the sieve, its stream drained: the counters at rounds 1, 2, 4, .. (check: every 8th).
    pub fn after(h: *HeadSieve) void {
        h.rounds += 1;
        if (h.rounds != h.next_log) return;
        h.next_log = if (h.mode == .check) h.rounds + 8 else h.rounds * 2;
        var st: [stats_words]u64 = undefined;
        h.buf.download(@intCast(h.stats - h.buf.ptr), std.mem.sliceAsBytes(&st)) catch return;
        const r = @max(st[0], 1);
        std.debug.print("[sieve] rounds={d} rows={d} rescored_n8 avg={d} max={d} overflow={d}\n", .{ st[0], st[4], st[1] / r, st[2], st[3] });
        if (h.mode != .check) return;
        const rows = @max(st[4], 1);
        const margin = unorder(st[50]);
        std.debug.print("[sieve] check true-token error (exact - a)/s in [{d:.3}, {d:.3}]\n", .{ unorder(st[52]), unorder(st[51]) });
        std.debug.print("[sieve] check mismatches={d} least_margin_sigma={d:.3} kappa {d:.1}..{d:.1}: misses {d} {d} {d} {d} | cand avg {d} {d} {d} {d} | max {d} {d} {d} {d}\n", .{
            st[5], margin, h.kappa - 3, h.kappa, st[6], st[7], st[8], st[9], st[10] / rows, st[11] / rows, st[12] / rows, st[13] / rows, st[14], st[15], st[16], st[17],
        });
        for (0..4) |i| std.debug.print("[sieve] check kappa {d:.1} hist <16 <64 <256 <1k <4k <16k <64k more: {any}\n", .{ h.kappa - 3 + @as(f32, @floatFromInt(i)), st[18 + 8 * i .. 26 + 8 * i] });
    }
};
