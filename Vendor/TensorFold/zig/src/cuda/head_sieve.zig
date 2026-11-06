//! The greedy verify head's sieve (qmm_head_sieve.cuh, in the qmm_group module): a 2-bit copy of the vocabulary
//! head's codes (code >> 2, the same scales and biases) built at engine open; a greedy verify then streams it in place
//! of the 4-bit words, recomputes each row's first tile (the n8 tile of its best approximate logit) with the head
//! tile's own arithmetic, and then only the 8-id column groups whose approximate logit plus kappa spreads of the
//! dropped bits' term reaches the lower of that tile's exact best and its best approximate logit. Each row's token is
//! the best of those exact values: the full head's token whenever the best id's dropped bits' term exceeds the first
//! tile's best-approximate id's by less than kappa spreads.
//! TF_HEAD_SIEVE=0: the full head; =check: the full head writes the tokens and the sieve runs beside it, compared.
//! TF_HEAD_SIEVE_K: kappa (10); TF_HEAD_SIEVE_CAP: the most n8 tiles rescored before every tile is (every tile, n / 8).
//! The 16-row windows rescore from an n8-major copy of the head (each n8 tile's words, scales and biases contiguous,
//! the same values and per-item arithmetic): the head's own layout serves 256-byte pieces 2 KiB apart, which stream
//! at ~190 GB/s, the copy contiguous pieces at ~230. TF_HEAD_SIEVE_RN=0: rescore from the head itself, no copy.
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

pub const Mode = enum { on, check };

pub const HeadSieve = struct {
    pub const rows_max = 32; // the sieve's row tiles: 16-row windows on the 16-row kernels, 17-32 on their twins
    pub const sieve_smem: u32 = 4 * (2048 + 4 * 1024 + 4 * 256 + 128); // tf_head_sieve::SMEM
    pub const sieve32_smem: u32 = 4 * (2 * 2048 + 4 * 1024 + 4 * 256 + 2 * 128); // the 32-row scan's stages
    pub const rescore_smem: u32 = 8 * (2048 + 4 * 288 + 64); // tf_head_sieve::RS_SMEM
    pub const rescore32_smem: u32 = 8 * (2048 + 4 * 288 + 128); // the 32-row rescore's stages
    pub const rn_group: usize = 256 + 32; // tf_head_sieve::RN_GROUP
    pub const rn_smem: u32 = 3 * (4 * 2048 + 4 * 4 * 288 + 4 * 64); // rn_smem(RN_CH, RN_STAGES)
    pub const rn_keys_smem: u32 = 3 * (8 * 2048 + 4 * 8 * 288 + 8 * 64); // rn_smem(RN_KCH, RN_STAGES)
    const rn_ctas = 96; // two an SM
    /// The pair table in shared memory when three scan CTAs an SM still fit (GB10: 100 KiB an SM, 1 KiB a CTA).
    const tab_room: u32 = 102400 / 3 - 1024 - sieve_smem;
    /// The 32-row scan's room at two CTAs an SM (its stages double the A fragments and the group sums).
    const tab32_room: u32 = 102400 / 2 - 1024 - sieve32_smem;
    /// PEER-23 (leader 0ede): the prompt tail's own scan — warp-private stages, old scan ABI.
    const prompt_sieve_smem: u32 = 4 * 4 * (1024 + 256);
    const prompt_tab_room: u32 = 102400 / 3 - 1024 - prompt_sieve_smem;
    const stats_words = 64;
    const rescore_ctas = 144; // three an SM (GB10: 48 SMs)
    const select_ctas: usize = 48;
    const select96_ctas: usize = 96; // TF_SEL96 (PEER-05 selector96, jungjipdo 9f882645): two selector CTAs an SM; default on, TF_SEL96=0 rolls back

    pack_fn: Function,
    prep_fn: Function,
    sieve_fn: Function,
    top_fn: Function,
    select_fn: Function,
    rescore_fn: Function,
    finish_fn: Function,
    check_fn: Function,
    prep32_fn: Function, // the 17-32-row windows' twins (the same per-row arithmetic, two m16 row tiles)
    sieve32_fn: Function,
    top32_fn: Function,
    select32_fn: Function,
    rescore32_fn: Function,
    finish32_fn: Function,
    check32_fn: Function,
    rn_fn: ?Function = null, // the n8-major rescore (16-row windows), when the copy is held
    rn_keys_fn: ?Function = null,
    sieve5_fn: ?Function = null, // TF_HS_STAGES=5 probe of the scan
    sieve5_32_fn: ?Function = null,
    top_atomic: bool, // the scan maintains the rows' first tile (HS2); TF_HS_TOPATOMIC=0: the top kernel
    stages5: bool,
    sel96: bool, // TF_SEL96 (PEER-05 selector96): the select launch runs 96 CTAs instead of 48
    rn_buf: ?DeviceBuffer = null,
    /// PEER-23 lane (TF_RNPREFETCH=1): the rn rescore twins with a one-item L2 prefetch lead.
    rn_p_fn: ?Function = null,
    rn_keys_p_fn: ?Function = null,
    /// PEER-23: the prompt-tail scan kernel and its pair-table smem (0 = table via __ldg).
    prompt_sieve_fn: Function,
    prompt_tab_smem: u32,
    buf: DeviceBuffer,
    mode: Mode,
    kappa: f32,
    cap: usize,
    tiles: usize,
    kg: usize,
    n8s: usize,
    ctas: usize,
    code: u32, // the head's pair code, or 0: the sieve reads the (kg, npad) arrays
    gmajor: c_int, // the 2-bit copy's 1 KiB blocks down groups first (the scan's stage reads a contiguous slab); TF_HS_GMAJOR=0: tile-major
    tab_smem: u32,
    tab32_smem: u32 = 0, // the 32-row scan's shared pair table (its two-CTAs-an-SM room)
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
        // the scan maintains the rows' first tile itself (the top kernel's own atomicMax/key rule, HS2):
        // TF_HS_TOPATOMIC=0 puts the separate top kernel back into the chain
        const top_atomic = if (@import("sys").c.getenv("TF_HS_TOPATOMIC")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        // TF_HS_STAGES=5: the five-stage scan probe (two CTAs an SM, the same per-row arithmetic)
        const stages5 = if (@import("sys").c.getenv("TF_HS_STAGES")) |v| std.mem.eql(u8, std.mem.span(v), "5") else false;
        if (!fits(q)) {
            std.debug.print("info: head sieve off: head n {d} k {d} npad {d} code {x}\n", .{ q.n, q.k, q.npad, q.code });
            return null;
        }
        var kappa: f32 = 10.0;
        if (@import("sys").c.getenv("TF_HEAD_SIEVE_K")) |v| kappa = std.fmt.parseFloat(f32, std.mem.span(v)) catch 10.0;
        var cap: usize = q.n / 8; // every n8 tile: a window lists only what it rescores, so no overflow reads the whole head
        if (@import("sys").c.getenv("TF_HEAD_SIEVE_CAP")) |v| cap = std.fmt.parseInt(usize, std.mem.span(v), 10) catch q.n / 8;
        var h: HeadSieve = undefined;
        h.pack_fn = try group_mod.function("tf_head_sieve_pack");
        h.prep_fn = try group_mod.function("tf_head_sieve_prep");
        h.sieve_fn = try group_mod.function("tf_head_sieve_scan");
        h.top_fn = try group_mod.function("tf_head_sieve_top");
        h.select_fn = try group_mod.function("tf_head_sieve_select");
        h.rescore_fn = try group_mod.function("tf_head_sieve_rescore");
        h.finish_fn = try group_mod.function("tf_head_sieve_finish");
        h.check_fn = try group_mod.function("tf_head_sieve_check");
        h.prep32_fn = try group_mod.function("tf_head_sieve_prep32");
        h.sieve32_fn = try group_mod.function("tf_head_sieve_scan32");
        h.top32_fn = try group_mod.function("tf_head_sieve_top32");
        h.select32_fn = try group_mod.function("tf_head_sieve_select32");
        h.rescore32_fn = try group_mod.function("tf_head_sieve_rescore32");
        h.finish32_fn = try group_mod.function("tf_head_sieve_finish32");
        h.check32_fn = try group_mod.function("tf_head_sieve_check32");
        h.prompt_sieve_fn = try group_mod.function("tf_head_sieve_scan_prompt");
        h.mode = mode;
        h.kappa = kappa;
        h.cap = cap;
        h.tiles = q.n / 64;
        h.kg = q.k / 64;
        h.n8s = q.n / 8;
        h.ctas = h.tiles / 4;
        h.pdl = pdl;
        h.rounds = 0;
        h.next_log = 1;
        h.code = if (linear.pairCode(q.code)) q.code else 0;
        h.gmajor = if (@import("sys").c.getenv("TF_HS_GMAJOR")) |v| (if (std.mem.eql(u8, std.mem.span(v), "0")) @as(c_int, 0) else 1) else 1;
        h.top_atomic = top_atomic;
        h.stages5 = stages5;
        h.sel96 = if (@import("sys").c.getenv("TF_SEL96")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        h.sieve5_fn = if (stages5) try group_mod.function("tf_head_sieve_scan5") else null;
        h.sieve5_32_fn = if (stages5) try group_mod.function("tf_head_sieve_scan5_32") else null;
        const scan_smem: u32 = if (stages5) 5 * (2048 + 4 * 1024 + 4 * 256 + 128) else sieve_smem;
        const scan32_smem: u32 = if (stages5) 5 * (2 * 2048 + 4 * 1024 + 4 * 256 + 2 * 128) else sieve32_smem;
        const tab_bytes: u32 = if (h.code != 0) ((h.code >> 5 & 0x1FFF) * 4 + 15) / 16 * 16 else 0;
        h.tab_smem = if (tab_bytes <= tab_room) tab_bytes else 0;
        // PEER-23: the prompt tail's pair table shares the 3-CTA scan budget (its stages are smaller).
        h.prompt_tab_smem = if (tab_bytes <= prompt_tab_room) tab_bytes else 0;
        h.tab32_smem = if (tab_bytes <= tab32_room) tab_bytes else 0;
        try h.sieve_fn.allowDynamicShared(scan_smem + h.tab_smem);
        try h.sieve32_fn.allowDynamicShared(scan32_smem + h.tab32_smem);
        if (h.sieve5_fn) |f| try f.allowDynamicShared(scan_smem + h.tab_smem);
        try h.prompt_sieve_fn.allowDynamicShared(prompt_sieve_smem + h.prompt_tab_smem);
        if (h.sieve5_32_fn) |f| try f.allowDynamicShared(scan32_smem + h.tab32_smem);
        try h.rescore_fn.allowDynamicShared(rescore_smem);
        try h.rescore32_fn.allowDynamicShared(rescore32_smem);
        h.rn_fn = null;
        h.rn_keys_fn = null;
        h.rn_buf = null;
        const rn_env = @import("sys").c.getenv("TF_HEAD_SIEVE_RN");
        const use_rn = if (rn_env) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        if (use_rn and h.kg % 8 == 0) {
            h.rn_fn = try group_mod.function("tf_head_sieve_rescore_rn");
            h.rn_keys_fn = try group_mod.function("tf_head_sieve_rescore_rn_keys");
            try h.rn_fn.?.allowDynamicShared(rn_smem);
            try h.rn_keys_fn.?.allowDynamicShared(rn_keys_smem);
            const rnp = if (@import("sys").c.getenv("TF_RNPREFETCH")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
            if (rnp) {
                h.rn_p_fn = try group_mod.function("tf_head_sieve_rescore_rn_p");
                try h.rn_p_fn.?.allowDynamicShared(rn_smem);
                h.rn_keys_p_fn = try group_mod.function("tf_head_sieve_rescore_rn_keys_p");
                try h.rn_keys_p_fn.?.allowDynamicShared(rn_keys_smem);
            }
        }
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
        const o_count = take(&at, 8); // packed selector counts: high word every qualifying tile, low the compact list
        const o_best = take(&at, rows_max * 8);
        const o_pre = take(&at, rows_max * 8);
        const o_toks = take(&at, rows_max * 4);
        const o_stats = take(&at, stats_words * 8);
        const o_mrow = take(&at, rows_max * 4);
        const o_approx = if (mode == .check) take(&at, q.n * 2 * rows_max * 4) else 0;
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
        const least: u64 = std.math.maxInt(u64);
        try h.buf.upload(o_stats + 50 * 8, std.mem.asBytes(&least));
        try h.buf.upload(o_stats + 52 * 8, std.mem.asBytes(&least));
        var a: launch_.Args = .{};
        a.add(q.w);
        a.add(h.w2);
        a.add(@as(c_int, @intCast(h.kg)));
        a.add(h.gmajor);
        try launch_.launch(h.pack_fn, .{ .grid = .{ .x = @intCast(h.tiles * h.kg) }, .block = .{ .x = 256 } }, s, &a);
        try s.synchronize();
        if (h.rn_fn != null) {
            var rb = try DeviceBuffer.alloc(d, h.n8s * h.kg * rn_group);
            errdefer rb.free();
            var p: launch_.Args = .{};
            p.add(q.w);
            p.add(q.s);
            p.add(q.b);
            p.add(int(q.npad));
            p.add(int(h.kg));
            p.add(rb.ptr);
            const pack_rn = try group_mod.function("tf_head_sieve_pack_rn");
            try launch_.launch(pack_rn, .{ .grid = .{ .x = @intCast(h.n8s) }, .block = .{ .x = 128 } }, s, &p);
            try s.synchronize();
            h.rn_buf = rb;
            std.debug.print("info: head sieve: n8-major rescore copy {d} MiB\n", .{h.n8s * h.kg * rn_group >> 20});
        }
        std.debug.print("info: head sieve ({s}): kappa {d:.2}, cap {d} n8 tiles, 2-bit copy {d} MiB, scales {s} (np {d}, table {s}), {d} sieve CTAs, select {d} CTAs\n", .{
            @tagName(mode),                                   kappa,                                        cap,                                     h.tiles * h.kg * 1024 >> 20,
            if (h.code != 0) "pair dictionary" else "arrays", if (h.code != 0) h.code >> 5 & 0x1FFF else 0, if (h.tab_smem != 0) "shared" else "L2", h.ctas,
            if (h.sel96) select96_ctas else select_ctas,
        });
        return h;
    }

    pub fn deinit(h: *HeadSieve) void {
        if (h.rn_buf) |*rb| rb.free();
        h.buf.free();
    }

    /// A greedy verify's head over `rows` rows of x (bf16, k wide) with group sums xs: each row's token into toks
    /// (check mode: into the sieve's own, compared with toks, which the full head has written).
    pub fn run(h: *const HeadSieve, s: Stream, x: u64, xs: u64, q: Affine4, rows: usize, toks: u64, host_keys: bool) !void {
        return h.runWith(false, s, x, xs, q, rows, toks, host_keys);
    }

    /// PEER-23 (leader 0ede): the prompt tail's verify — its own scan kernel, old scan ABI, rows = 1.
    pub fn runPromptTail(h: *const HeadSieve, s: Stream, x: u64, xs: u64, q: Affine4, toks: u64) !void {
        return h.runWith(true, s, x, xs, q, 1, toks, false);
    }

    fn runWith(h: *const HeadSieve, comptime prompt_tail: bool, s: Stream, x: u64, xs: u64, q: Affine4, rows: usize, toks: u64, host_keys: bool) !void {
        if (rows == 0 or rows > rows_max) return error.WindowTooWide;
        // PEER-18 (265c97a6/c80d6ce): host_keys lets the engine read the winning keys directly and drop
        // the finish launch; check mode and every other path keep the token flow untouched.
        if (host_keys and prompt_tail) return error.InvalidHeadSieveHostKeys;
        if (host_keys and h.mode != .on) return error.InvalidHeadSieveHostKeys;
        // 17-32 rows run the twins: two m16 row tiles through the 16-row kernel's own per-row arithmetic
        // (the same kappa, spreads, thresholds, key(a, id) rule and rescore list semantics a row)
        const wide = rows > 16;
        const prep_fn = if (wide) h.prep32_fn else h.prep_fn;
        const sieve_fn = if (wide) (if (h.stages5) h.sieve5_32_fn.? else h.sieve32_fn) else (if (h.stages5) h.sieve5_fn.? else h.sieve_fn);
        const scan_shared: u32 = if (h.stages5) (if (wide) 5 * (2 * 2048 + 4 * 1024 + 4 * 256 + 2 * 128) else 5 * (2048 + 4 * 1024 + 4 * 256 + 128)) else (if (wide) sieve32_smem else sieve_smem);
        const tab = if (wide) h.tab32_smem else h.tab_smem;
        const top_fn = if (wide) h.top32_fn else h.top_fn;
        const select_fn = if (wide) h.select32_fn else h.select_fn;
        const finish_fn = if (wide) h.finish32_fn else h.finish_fn;
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
            try launch_.launch(prep_fn, .{ .grid = .{ .x = @intCast(h.kg) }, .block = .{ .x = 128 }, .pdl = h.pdl }, s, &a);
        }
        if (prompt_tail) {
            // PEER-23: the prompt tail's own scan (old arguments); the separate top kernel below publishes its key.
            var a: launch_.Args = .{};
            a.add(h.w2);
            a.add(h.afr);
            a.add(h.xv);
            a.add(q.sb);
            a.add(q.tab);
            a.add(@as(c_int, @bitCast(h.code)));
            a.add(@as(c_int, @intCast(h.prompt_tab_smem)));
            a.add(q.s);
            a.add(q.b);
            a.add(int(q.npad));
            a.add(int(h.kg));
            a.add(int(rows));
            a.add(h.kappa);
            a.add(h.himax);
            a.add(h.himid);
            a.add(h.approx);
            try launch_.launch(h.prompt_sieve_fn, .{ .grid = .{ .x = @intCast(h.ctas) }, .block = .{ .x = 128 }, .shared = prompt_sieve_smem + h.prompt_tab_smem, .pdl = h.pdl }, s, &a);
        } else {
            var a: launch_.Args = .{};
            a.add(h.w2);
            a.add(h.afr);
            a.add(h.xv);
            a.add(q.sb);
            a.add(q.tab);
            a.add(@as(c_int, @bitCast(h.code)));
            a.add(@as(c_int, @intCast(tab)));
            a.add(q.s);
            a.add(q.b);
            a.add(int(q.npad));
            a.add(int(h.kg));
            a.add(int(rows));
            a.add(h.kappa);
            a.add(h.himax);
            a.add(h.himid);
            a.add(h.approx);
            a.add(h.pre);
            a.add(if (h.top_atomic) @as(c_int, 1) else 0);
            a.add(h.gmajor);
            try launch_.launch(sieve_fn, .{ .grid = .{ .x = @intCast(h.ctas) }, .block = .{ .x = 128 }, .shared = scan_shared + tab, .pdl = h.pdl }, s, &a);
        }
        // each row's first tile (its best a) exactly; the row's threshold the lower of its exact best and its best a
        // (HS2: with top_atomic the scan maintained pre itself and the top kernel is out of the chain;
        // PEER-23: the prompt scan never maintains pre, so the top kernel always runs on the tail)
        if (prompt_tail or !h.top_atomic) {
            var a: launch_.Args = .{};
            a.add(h.himid);
            a.add(int(rows));
            a.add(int(h.n8s));
            a.add(h.pre);
            try launch_.launch(top_fn, .{ .grid = .{ .x = select_ctas }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        try h.rescore(s, q, rows, h.pre, wide, 0);
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
            try launch_.launch(select_fn, .{ .grid = .{ .x = if (h.sel96) select96_ctas else select_ctas }, .block = .{ .x = 256 }, .pdl = h.pdl }, s, &a);
        }
        try h.rescore(s, q, rows, 0, wide, if (host_keys) h.stats else 0);
        // PEER-18: the host reads the winning keys itself on the host_keys path, so the finish launch
        // (tokens + stats) is skipped; check mode and every other path keep it.
        if (!host_keys) {
            var a: launch_.Args = .{};
            a.add(h.best);
            a.add(int(rows));
            a.add(out);
            a.add(h.count);
            a.add(int(h.cap));
            a.add(int(h.n8s));
            a.add(h.stats);
            try launch_.launch(finish_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 }, .pdl = h.pdl }, s, &a);
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
            const check_fn = if (wide) h.check32_fn else h.check_fn;
            try launch_.launch(check_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 }, .pdl = h.pdl }, s, &a);
        }
    }

    /// The listed n8 tiles exactly (keys: each row's first tile instead), each row's best into best.
    fn rescore(h: *const HeadSieve, s: Stream, q: Affine4, rows: usize, keys: u64, wide: bool, stats: u64) !void {
        if (!wide and h.rn_buf != null) {
            const rb = h.rn_buf.?;
            var a: launch_.Args = .{};
            a.add(rb.ptr);
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
            const ctas: usize = if (keys != 0) (rows + 3) / 4 else rn_ctas;
            const f = if (keys != 0) (if (h.rn_keys_p_fn != null) h.rn_keys_p_fn.? else h.rn_keys_fn.?) else (if (h.rn_p_fn != null) h.rn_p_fn.? else h.rn_fn.?); // TF_RNPREFETCH=1: the _p twins add an L2 prefetch lead (PEER-23 lane)
            const smem: u32 = if (keys != 0) rn_keys_smem else rn_smem;
            try launch_.launch(f, .{ .grid = .{ .x = @intCast(ctas) }, .block = .{ .x = 128 }, .shared = smem, .pdl = h.pdl }, s, &a);
            return;
        }
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
        a.add(stats);
        const ctas: usize = if (keys != 0) (rows + 3) / 4 else rescore_ctas;
        const f = if (wide) h.rescore32_fn else h.rescore_fn;
        const smem: u32 = if (wide) rescore32_smem else rescore_smem;
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(ctas) }, .block = .{ .x = 128 }, .shared = smem, .pdl = h.pdl }, s, &a);
    }

    fn unorder(v: u64) f32 {
        const u: u32 = @truncate(v);
        return @bitCast(if (u & 0x80000000 != 0) u & 0x7FFFFFFF else ~u);
    }

    /// After a verify that ran the sieve, its stream drained: the counters at rounds 1, 2, 4, .. (check: every 8th).
    pub fn after(h: *HeadSieve) void {
        // The counters and the page below are a check-mode instrument: the scored decode runs .on and reads neither,
        // while the download crosses the bus synchronously and drains the verify round it is called from. .on keeps
        // only the call, so the round this returns to carries no page read and no stderr format.
        if (h.mode != .check) return;
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
