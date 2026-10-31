//! Affine-4 decode, verify and prompt launches over the shared tiled weight layout.
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
const Split = linear.Split;
const GemvPart = linear.GemvPart;
const Segs = linear.Segs;
const Parts = linear.Parts;
const symbols = linear.symbols;
const splitK = linear.splitK;
const group_smem = linear.group_smem;
const split_smem = linear.split_smem;
const prefill_smem = linear.prefill_smem;
fn int(v: usize) c_int {
    return @intCast(v);
}
fn u(v: usize) u32 {
    return @intCast(v);
}

/// q coded tile-major: what every twin but tf_qmm_mlp_single16's reads.
fn tileCoded(q: Affine4) bool {
    return q.code != 0 and q.code & linear.dict_kmajor == 0;
}

/// qmm_group.cu's Part for q (its dictionary twin's view when `dict`: indices, table and code in place of the arrays).
fn groupPart(q: Affine4, dict: bool, out: u64, sk: usize, tiles: usize, first: usize) linear.Part {
    return .{ .w = q.w, .scales = if (dict) q.sb else q.s, .biases = if (dict) q.tab else q.b, .out = out, .n = int(q.n), .npad = if (dict) @bitCast(q.code) else int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = int(first) };
}

/// lane_gemv.cu's Part for q, as groupPart.
fn gemvPart(q: Affine4, dict: bool, out: u64, sk: usize, tiles: usize) GemvPart {
    return .{ .w = q.w, .scales = if (dict) q.sb else q.s, .biases = if (dict) q.tab else q.b, .out = out, .n = int(q.n), .npad = if (dict) @bitCast(q.code) else int(q.npad), .sk = int(sk), .tiles = int(tiles) };
}

/// Device memory for dictionary-coded scales and biases (taken in 256 MiB chunks as projections load), and the
/// encoder's scratch: presence bitmaps (2 x 8 KiB), ranks (2 x 8 KiB) and the two value counts, then the pair
/// dictionary's: the separate table (8192 bf16), the 2^24-bit pair bitmap and its word ranks (2 MiB each), the
/// blocks' offsets and the pair count (257 words), and the round-trip check's mismatch count.
pub const DictStore = struct {
    d: *const Driver,
    chunks: [64]DeviceBuffer = undefined,
    n: usize = 0,
    at: usize = 0,
    scratch: DeviceBuffer,
    bytes: usize = 0, // taken so far (reporting)
    coded: usize = 0, // projections coded
    kept: usize = 0, // projections left on their arrays (too many values)
    /// The joint (scale, bias) pair dictionary (TF_DICT_PAIR): off; `check` encodes pairs and checks the round trip but
    /// keeps the separate coding; `on` installs the pair codes (pbits | np << 5, nb = 0) wherever they are smaller.
    pair: PairMode = .off,
    pair_coded: usize = 0, // projections whose pairs were encoded (and installed when `on`)
    pair_declined: usize = 0, // np past dict_pair_max, or no smaller than the separate coding
    pair_bad: u64 = 0, // cells whose pair read back differs from their arrays (must stay 0)
    pair_bytes: usize = 0, // pair index streams' bytes (reporting: the separate streams they replace)
    pair_separate_bytes: usize = 0,
    pair_np_max: u32 = 0, // the largest table (entries) encoded
    pair_pbits: [14]u32 = @splat(0), // projections encoded at each index width

    pub const PairMode = enum { off, check, on };
    const chunk = 256 << 20;
    const sep_scratch = 4 * 2048 * 4 + 16;
    const stab_at = sep_scratch; // 8192 bf16
    const pairs_at = stab_at + 8192 * 2; // 2^19 words
    const prank_at = pairs_at + (1 << 21); // 2^19 words
    const offsets_at = prank_at + (1 << 21); // 257 words (padded)
    const bad_at = offsets_at + 260 * 4;
    pub const scratch_bytes = bad_at + 16;

    pub fn init(d: *const Driver) !DictStore {
        return .{ .d = d, .scratch = try DeviceBuffer.alloc(d, scratch_bytes) };
    }

    pub fn deinit(ds: *DictStore) void {
        for (ds.chunks[0..ds.n]) |*c| c.free();
        ds.n = 0;
        ds.scratch.free();
    }

    fn alloc(ds: *DictStore, bytes: usize) !u64 {
        const need = (bytes + 255) / 256 * 256;
        if (ds.n == 0 or ds.at + need > ds.chunks[ds.n - 1].len) {
            if (ds.n == ds.chunks.len) return error.DictStoreFull;
            ds.chunks[ds.n] = try DeviceBuffer.alloc(ds.d, @max(chunk, need));
            ds.n += 1;
            ds.at = 0;
        }
        const p = ds.chunks[ds.n - 1].ptr + ds.at;
        ds.at += need;
        ds.bytes += need;
        return p;
    }
};

/// The dictionary twins (qmm_dict.cuh) of the decode kernels, and the load-time encoder (affine4_pack.cu).
pub const DictKernels = struct {
    group: Function, // group_kernel_dict<64, 16, 64, 1, 4, 8>: decode()'s cluster tile
    gemv: Function, // gemv_kernel<64, 64, 8>'s twin
    many: Function, // tf_lane_gemv_many's
    mark: Function,
    rank: Function,
    table: Function,
    pack: Function,
    pair_mark: Function,
    pair_rank: Function,
    pair_offsets: Function,
    pair_table: Function,
    pair_pack: Function,
    pair_check: Function,
    gemv_blocks: usize, // resident twin lane CTAs at the largest table kept (GB10: gemv_blocks)
};

/// The affine-4 kernels: qmm_group's cluster tile, lane_gemv and its split-K form, qmm_prefill's prompt GEMM.
pub const Affine4Kernels = struct {
    pub const max_rows = 16; // rows decode() takes: its kernels tile 16 rows
    pub const gdn_stream32_smem: u32 = 17664;
    pub const segs_max = 4; // projections one segmented launch takes
    pub const narrow_tiles = 24; // GB10: column tiles at most that decode() runs in the split form (half its 48 SMs)
    pub const lane_min = 32; // GB10 with short_lane: K slices of fewer groups go to lane_gemv rather than the cluster tile
    pub const lane_groups = 96; // ... when a lane_gemv CTA then streams at least this many groups (K >= 6144)

    /// One projection of a shared input and its output.
    pub const Out = struct { q: Affine4, out: u64 };

    group_fn: Function,
    stream16_fns: [2]?Function = @splat(null),
    stream16_dict_fn: ?Function = null, // tf_qmm_output_stream16 on a dictionary-coded down projection
    gdn_stream16_dict_fn: ?Function = null, // tf_qmm_gdn_stream16 with its dictionary-coded parts
    gdn_stream16_fn: ?Function = null,
    gdn_stream32_fn: ?Function = null,
    gdn_stream32_dict_fn: ?Function = null,
    gemv_fn: Function,
    verify_fn: Function, // two m16 row tiles share each weight panel on the GB10 verify path
    verify64_fn: Function, // four, for wider shared frames
    verify128_fn: Function, // eight
    split_fn: Function,
    seg_fn: Function, // the split form over up to four projections of one input
    many_fn: Function, // lane_gemv over up to four projections of one input, each part's CTAs as its own launch's
    prefill_fn: Function,
    sms: usize = 0,
    prefill22_fn: Function, // the prompt tile on 2x2 warps and two stages (`half_prompt` picks it)
    half_prompt: bool = false, // prompts on prefill22_fn: its four warps a block and two stages are quicker on an RTX 5090
    words_fn: Function, // affine4_pack.cu: MLX words into the tiles
    scales_fn: Function, // affine4_pack.cu: (n, kg) scales or biases into (kg, npad)
    dict: ?DictKernels = null, // GB10: the dictionary twins and their encoder (TF_DICT=0: none, every launch on its arrays)
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    split: ?Split, // every GPU but GB10: lane_gemv's slices on CTAs of their own (one stream's decode() at a time)
    narrow: ?Split, // GB10: the split form for projections too narrow to fill the GPU as lane_gemv column CTAs
    pdl: bool, // GB10: the decode kernels launch with programmatic dependent launch
    wide_verify: bool, // off GB10: verify windows of 17 to 32 rows share one weight pass (sm_120, when its engine says)
    wide_fns: [3]?Function = @splat(null),
    group_wide: bool = false, // group 17 to 32-row projections without changing their tiles or K slices
    cluster_decode: bool = false, // GB10: split-K projections of <= 16 rows on qmm_group's cluster tile, not lane_gemv (same slice order)
    short_lane: bool = false, // GB10: ... except K slices shorter than lane_min groups (out/o: 24), on lane_gemv's column CTAs

    /// From loaded qmm_group, qmm_prefill, lane_gemv and affine4_pack modules; GB10 decodes without the split form.
    pub fn resolve(d: *const Driver, group_mod: Module, prefill_mod: Module, gemv_mod: Module, pack_mod: Module, sms: usize, gb10: bool) !Affine4Kernels {
        var a: Affine4Kernels = undefined;
        a.group_fn = try group_mod.function(symbols.group);
        a.verify_fn = try group_mod.function(symbols.verify32);
        a.verify64_fn = try group_mod.function(symbols.verify64);
        a.verify128_fn = try group_mod.function(symbols.verify128);
        a.prefill_fn = try prefill_mod.function(symbols.prefill);
        a.prefill22_fn = try prefill_mod.function(symbols.prefill22);
        a.sms = sms;
        a.half_prompt = false;
        a.gemv_fn = try gemv_mod.function(symbols.gemv);
        a.split_fn = try gemv_mod.function(symbols.gemv_split);
        a.seg_fn = try gemv_mod.function(symbols.gemv_seg);
        a.many_fn = try gemv_mod.function(symbols.gemv_many);
        a.words_fn = try pack_mod.function(symbols.pack_words);
        a.scales_fn = try pack_mod.function(symbols.pack_scales);
        try a.group_fn.allowDynamicShared(group_smem);
        try a.verify_fn.allowDynamicShared(linear.verify_smem);
        try a.verify64_fn.allowDynamicShared(linear.verify64_smem);
        try a.verify128_fn.allowDynamicShared(linear.verify128_smem);
        try a.gemv_fn.allowDynamicShared(group_smem);
        try a.split_fn.allowDynamicShared(split_smem);
        try a.seg_fn.allowDynamicShared(split_smem);
        try a.many_fn.allowDynamicShared(group_smem);
        try a.prefill_fn.allowDynamicShared(prefill_smem);
        try a.prefill22_fn.allowDynamicShared(linear.prefill22_smem);
        a.gemv_blocks = @max(1, try a.gemv_fn.occupancy(128, group_smem)) * sms;
        a.pdl = gb10;
        a.stream16_fns = @splat(null);
        a.stream16_dict_fn = null;
        a.gdn_stream16_dict_fn = null;
        a.gdn_stream16_fn = null;
        a.gdn_stream32_fn = null;
        a.gdn_stream32_dict_fn = null;
        a.wide_fns = @splat(null);
        a.group_wide = false;
        a.wide_verify = false;
        a.cluster_decode = false;
        a.short_lane = false;
        a.split = if (gb10) null else try Split.initFor(d, segs_max);
        a.narrow = if (gb10) try Split.initFor(d, 1) else null;
        a.dict = null;
        if (gb10) {
            var t: DictKernels = undefined;
            t.group = try group_mod.function(symbols.group_dict);
            t.gemv = try gemv_mod.function(symbols.gemv_dict);
            t.many = try gemv_mod.function(symbols.gemv_many_dict);
            t.mark = try pack_mod.function(symbols.dict_mark);
            t.rank = try pack_mod.function(symbols.dict_rank);
            t.table = try pack_mod.function(symbols.dict_table);
            t.pack = try pack_mod.function(symbols.dict_pack);
            t.pair_mark = try pack_mod.function(symbols.dict_pair_mark);
            t.pair_rank = try pack_mod.function(symbols.dict_pair_rank);
            t.pair_offsets = try pack_mod.function(symbols.dict_pair_offsets);
            t.pair_table = try pack_mod.function(symbols.dict_pair_table);
            t.pair_pack = try pack_mod.function(symbols.dict_pair_pack);
            t.pair_check = try pack_mod.function(symbols.dict_pair_check);
            // the lane twins keep pair tables of up to lane_pair_tab_max in shared memory (lane_gemv.cu LANE_PAIR_SMEM)
            const lane_room = linear.dict_dec + @max(linear.dict_tab_max, linear.lane_pair_tab_max);
            try t.group.allowDynamicShared(group_smem + linear.dict_dec + @max(linear.dict_tab_max, linear.group_pair_tab_one));
            try t.gemv.allowDynamicShared(group_smem + lane_room);
            try t.many.allowDynamicShared(group_smem + lane_room);
            // resident lane CTAs at the largest table a projection keeps coded (GB10: the originals' count)
            t.gemv_blocks = @max(1, try t.gemv.occupancy(128, group_smem + lane_room)) * sms;
            a.dict = t;
        }
        return a;
    }

    /// The dictionary twin's view of q when q is coded (tile-major) and a twin is loaded.
    fn coded(a: *const Affine4Kernels, q: Affine4) bool {
        return a.dict != null and tileCoded(q);
    }

    /// The output stream is selectable only for exactly the geometry cluster() handles.
    fn streamed(a: *const Affine4Kernels, q: Affine4, rows: usize, sk: usize) bool {
        if (q.stream_w == 0) return false;
        const f = a.stream16_fns[@as(usize, if (q.k == 6144) 1 else 0)];
        return f != null and rows == 16 and sk == 4 and q.n == 5120 and (q.k == 6144 or q.k == 17408);
    }

    /// The existing coded four-slice stream can also consume a 16-row out/o projection.
    /// Keep the ordinary short-lane route whenever its view or kernel is unavailable.
    fn dictOutStream(a: *const Affine4Kernels, q: Affine4, rows: usize, sk: usize) bool {
        return a.short_lane and a.cluster_decode and rows == 16 and sk == 4 and
            q.n == 5120 and q.npad == 5120 and q.k == 6144 and q.stream_w != 0 and
            a.stream16_fns[1] != null and a.stream16_dict_fn != null and a.coded(q);
    }

    /// q's scales and biases as dictionary indices and a table (qmm_dict.cuh) out of `store`, from the tiled arrays
    /// pack() wrote: lossless (the table holds every distinct value), so each twin's fmas read the same bf16 values.
    /// A projection of more than 2^12 distinct scales or biases, or a table past dict_tab_max, stays on its arrays.
    /// `room`: the largest pair table q's decode twin keeps in shared memory or reads through L2 at no loss (pairTabRoom);
    /// a larger one keeps the separate coding.
    pub fn encode(a: *const Affine4Kernels, s: Stream, q: *Affine4, store: *DictStore, kmajor: bool, room: u32) !void {
        const t = a.dict orelse return;
        if (q.k % 256 != 0 or q.npad % 64 != 0) return; // kg a multiple of 4: every tile's indices start on a 64-byte line
        const kg = q.k / 64;
        const cells: c_longlong = @intCast(kg * q.npad);
        const bitmap = store.scratch.ptr;
        const rank = bitmap + 2 * 2048 * 4;
        const count = rank + 2 * 2048 * 4;
        try store.scratch.fill8(0, s.handle);
        for ([_]u64{ q.s, q.b }, [_]u64{ bitmap, bitmap + 2048 * 4 }) |src, bm| {
            var g: launch_.Args = .{};
            g.add(src);
            g.add(cells);
            g.add(bm);
            try launch_.launch(t.mark, .{ .grid = .{ .x = 512 }, .block = .{ .x = 256 } }, s, &g);
        }
        {
            var g: launch_.Args = .{};
            g.add(bitmap);
            g.add(rank);
            g.add(count);
            try launch_.launch(t.rank, .{ .grid = .{ .x = 2 }, .block = .{ .x = 1024 } }, s, &g);
        }
        try s.synchronize();
        var counts: [2]c_int = undefined;
        try store.scratch.download(count - store.scratch.ptr, std.mem.asBytes(&counts));
        const ns: u32 = @intCast(counts[0]);
        const nb: u32 = @intCast(counts[1]);
        var bits: u32 = 1;
        while ((@as(u32, 1) << @intCast(bits)) < @max(ns, nb)) bits += 1;
        const tab_bytes = (ns + nb + 7) / 8 * 16;
        // the pair dictionary first (its table's size is the kernels' concern): both indices fit 12 bits
        if (store.pair != .off and bits <= linear.dict_bits_max and try a.encodePair(s, q, store, kmajor, ns, bits, room)) return;
        if (bits > linear.dict_bits_max or tab_bytes > linear.dict_tab_max) {
            store.kept += 1;
            return;
        }
        const tab = try store.alloc(tab_bytes);
        const total: u64 = @as(u64, q.npad / 64) * kg * 4 * bits;
        const sb = try store.alloc(total * 4 + 256); // + the last window's reach past the last group
        {
            var g: launch_.Args = .{};
            g.add(bitmap);
            g.add(rank);
            g.add(@as(c_int, @intCast(ns)));
            g.add(tab);
            try launch_.launch(t.table, .{ .grid = .{ .x = 16 }, .block = .{ .x = 256 } }, s, &g);
        }
        {
            var g: launch_.Args = .{};
            g.add(q.s);
            g.add(q.b);
            for ([_]usize{ kg, q.npad }) |v| g.add(int(v));
            g.add(bitmap);
            g.add(rank);
            g.add(@as(c_int, @intCast(bits)));
            g.add(sb);
            g.add(@as(c_longlong, @intCast(total)));
            g.add(@as(c_int, @intFromBool(kmajor)));
            try launch_.launch(t.pack, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
        }
        try s.synchronize(); // the scratch's next projection
        q.sb = sb;
        q.tab = tab;
        q.code = bits | ns << 5 | nb << 18 | (if (kmajor) linear.dict_kmajor else 0);
        store.coded += 1;
    }

    /// q's (scale, bias) pairs as one dictionary (/root/cf/pairdict/FORMAT.md), from the separate dictionaries' bitmaps
    /// and ranks encode() left in the scratch: key si << 12 | bi marked in a 2^24-bit bitmap and ranked, the table of
    /// np entries (scale | bias << 16, key order), and each group-tile's 64 pair indices of pbits bits, tile-major or
    /// group-major (`kmajor`), 256 bytes of slack. Every cell is read back through the table against its arrays; the
    /// codes are installed (pbits | np << 5, nb = 0) only in `on` mode, with no mismatch, np <= dict_pair_max and a
    /// stream smaller than the separate coding's. Returns whether q now holds pair codes.
    fn encodePair(a: *const Affine4Kernels, s: Stream, q: *Affine4, store: *DictStore, kmajor: bool, ns: u32, bits: u32, room: u32) !bool {
        const t = a.dict.?;
        const d = store.d;
        const kg = q.k / 64;
        const tiles = q.npad / 64;
        const cells: c_longlong = @intCast(kg * q.npad);
        const base = store.scratch.ptr;
        const bitmap = base;
        const rank = bitmap + 2 * 2048 * 4;
        const stab = base + DictStore.stab_at;
        const pairs = base + DictStore.pairs_at;
        const prank = base + DictStore.prank_at;
        const offsets = base + DictStore.offsets_at;
        const bad = base + DictStore.bad_at;
        {
            var g: launch_.Args = .{};
            g.add(bitmap);
            g.add(rank);
            g.add(@as(c_int, @intCast(ns)));
            g.add(stab);
            try launch_.launch(t.table, .{ .grid = .{ .x = 16 }, .block = .{ .x = 256 } }, s, &g);
        }
        try d.check(d.api.cuMemsetD8Async(pairs, 0, 1 << 21, s.handle), "cuMemsetD8Async");
        {
            var g: launch_.Args = .{};
            g.add(q.s);
            g.add(q.b);
            g.add(cells);
            g.add(bitmap);
            g.add(rank);
            g.add(pairs);
            try launch_.launch(t.pair_mark, .{ .grid = .{ .x = 512 }, .block = .{ .x = 256 } }, s, &g);
        }
        {
            var g: launch_.Args = .{};
            g.add(pairs);
            g.add(prank);
            g.add(offsets);
            try launch_.launch(t.pair_rank, .{ .grid = .{ .x = 256 }, .block = .{ .x = 1024 } }, s, &g);
        }
        {
            var g: launch_.Args = .{};
            g.add(offsets);
            try launch_.launch(t.pair_offsets, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &g);
        }
        try s.synchronize();
        var np: u32 = undefined;
        try store.scratch.download(offsets + 256 * 4 - base, std.mem.asBytes(&np));
        var pbits: u32 = 1;
        while ((@as(u32, 1) << @intCast(pbits)) < np) pbits += 1;
        while ((kg * pbits) % 8 != 0) pbits += 1;
        if (np == 0 or np > linear.dict_pair_max or 8 * pbits >= 16 * bits or (np * 4 + 15) / 16 * 16 > room) {
            store.pair_declined += 1;
            return false;
        }
        const tab_bytes = (np * 4 + 15) / 16 * 16;
        const tab = try store.alloc(tab_bytes);
        try d.check(d.api.cuMemsetD8Async(tab, 0, tab_bytes, s.handle), "cuMemsetD8Async");
        const total: u64 = @as(u64, tiles) * kg * 2 * pbits; // index words
        const stream = try store.alloc(total * 4 + 256);
        try d.check(d.api.cuMemsetD8Async(stream + total * 4, 0, 256, s.handle), "cuMemsetD8Async");
        {
            var g: launch_.Args = .{};
            g.add(pairs);
            g.add(prank);
            g.add(offsets);
            g.add(stab);
            g.add(@as(c_int, @intCast(ns)));
            g.add(tab);
            try launch_.launch(t.pair_table, .{ .grid = .{ .x = (1 << 19) / 256 }, .block = .{ .x = 256 } }, s, &g);
        }
        {
            var g: launch_.Args = .{};
            g.add(q.s);
            g.add(q.b);
            for ([_]usize{ kg, q.npad }) |v| g.add(int(v));
            g.add(bitmap);
            g.add(rank);
            g.add(pairs);
            g.add(prank);
            g.add(offsets);
            g.add(@as(c_int, @intCast(pbits)));
            g.add(stream);
            g.add(@as(c_longlong, @intCast(total)));
            g.add(@as(c_int, @intFromBool(kmajor)));
            try launch_.launch(t.pair_pack, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
        }
        // the round trip: every cell's index read back as a kernel reads it, through the table, against its arrays
        try d.check(d.api.cuMemsetD8Async(bad, 0, 8, s.handle), "cuMemsetD8Async");
        {
            var g: launch_.Args = .{};
            g.add(q.s);
            g.add(q.b);
            for ([_]usize{ kg, q.npad }) |v| g.add(int(v));
            g.add(stream);
            g.add(tab);
            g.add(@as(c_int, @intCast(pbits)));
            g.add(@as(c_int, @intFromBool(kmajor)));
            g.add(bad);
            try launch_.launch(t.pair_check, .{ .grid = .{ .x = 512 }, .block = .{ .x = 256 } }, s, &g);
        }
        try s.synchronize();
        var mismatches: u64 = undefined;
        try store.scratch.download(bad - base, std.mem.asBytes(&mismatches));
        store.pair_bad += mismatches;
        store.pair_coded += 1;
        if (@import("sys").c.getenv("TF_DICT_PAIR_LIST") != null) std.debug.print("pair: n {d} k {d} kmajor {} ns {d} bits {d} np {d} pbits {d} table {d} B\n", .{ q.n, q.k, kmajor, ns, bits, np, pbits, np * 4 });
        store.pair_np_max = @max(store.pair_np_max, np);
        store.pair_pbits[pbits] += 1;
        store.pair_bytes += total * 4;
        store.pair_separate_bytes += @as(usize, tiles) * kg * 16 * bits;
        if (store.pair != .on or mismatches != 0) return false;
        q.sb = stream;
        q.tab = tab;
        q.code = pbits | np << 5 | (if (kmajor) linear.dict_kmajor else 0);
        store.coded += 1;
        return true;
    }

    pub fn deinit(a: *Affine4Kernels) void {
        if (a.split) |*sp| sp.deinit();
        if (a.narrow) |*sp| sp.deinit();
    }

    /// qmm_fast.tile: MLX words (n, k/8), scales and biases (n, k/64) on the device into q's tiles (Affine4.layout).
    pub fn pack(a: *const Affine4Kernels, s: Stream, words: u64, scales: u64, biases: u64, q: Affine4) !void {
        const kg = q.k / 64;
        const total: u64 = @as(u64, q.npad / 64) * kg * 512;
        var g: launch_.Args = .{};
        g.add(words);
        for ([_]usize{ q.n, q.k / 8, kg }) |v| g.add(int(v));
        g.add(q.w);
        g.add(@as(c_longlong, @intCast(total)));
        try launch_.launch(a.words_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
        for ([_]u64{ scales, biases }, [_]u64{ q.s, q.b }) |src, out| {
            var t: launch_.Args = .{};
            t.add(src);
            for ([_]usize{ q.n, kg, q.npad }) |v| t.add(int(v));
            t.add(out);
            const cells = @as(u64, kg) * q.npad;
            try launch_.launch(a.scales_fn, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, s, &t);
        }
    }

    /// qmm.matmul: x (rows, k) bf16 with group sums xs -> out (rows, n) bf16, qmm_group's tile-2 bits on every path.
    pub fn decode(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize) !void {
        if (rows > max_rows) return error.WindowTooWide;
        const sk = splitK(q.n, q.k);
        if (sk > 1) if (a.split) |sp| return a.gemvSplit(s, x, xs, q, out, rows, sk, sp);
        // GB10: a projection of few column tiles (GDN a/b, attention k/v) leaves most SMs idle as lane_gemv's
        // column CTAs; the split form runs each K slice on a CTA of its own and sums them in slice order (same bytes)
        if (sk > 1) if (a.narrow) |sp| if ((q.n + 63) / 64 <= narrow_tiles) return a.gemvSplit(s, x, xs, q, out, rows, sk, sp);
        // lane_gemv runs min(tiles, resident) CTAs: an N = 5120 projection is 80 CTAs on GB10's 48 SMs; the cluster
        // tile runs tiles * sk CTAs (320), each K slice summed in slice order over distributed shared memory
        // ... unless its K slices are short (GDN out_proj, attention o_proj: 24 groups a slice): after the GDN and
        // attention kernels, whose CTAs still hold the SMs, 80 lane_gemv CTAs land and prefetch sooner than 80
        // clusters of four, each CTA chaining the tile's slices in order (the same bits); the down projection's long
        // slices (68 groups) stay on the cluster tile, and so does a short K (the drafter's K = 4096, 64 groups a
        // lane CTA: its 80 CTAs keep too few bytes in flight)
        const short = sk > 1 and a.cluster_decode and a.short_lane and q.k / 64 / sk < lane_min and q.k / 64 >= lane_groups and !a.streamed(q, rows, sk);
        if (sk > 1 and a.cluster_decode and !short) return a.cluster(s, x, xs, q, out, rows, sk);
        return if (sk > 1) a.gemv(s, x, xs, q, out, rows, sk) else a.cluster(s, x, xs, q, out, rows, sk);
    }

    /// decode() of each projection of one input; without GB10 one segmented launch, each keeping its split's bits.
    /// On GB10 the K-sliced projections share one lane_gemv launch and the unsliced ones one cluster launch.
    pub fn streamable(a: *const Affine4Kernels, outs: []const Out, rows: usize) bool {
        const gdn_streamed = rows == 16 or (rows > 16 and rows <= 32 and a.gdn_stream32_fn != null);
        if (!gdn_streamed or !a.pdl or a.gdn_stream16_fn == null) return false;
        for (outs) |o| if (o.q.k != 5120 or o.q.stream_w == 0) return false;
        return outs.len == 4 and outs[0].q.n == 10240 and outs[1].q.n == 6144 and outs[2].q.n == 48 and outs[3].q.n == 48;
    }

    pub fn decodeMany(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (a.streamable(outs, rows)) {
            const rows32 = rows > 16;
            const bn: usize = if (rows32) 128 else 256;
            const dict_fn = if (rows32) a.gdn_stream32_dict_fn else a.gdn_stream16_dict_fn;
            // coded parts stream their index windows (the dictionary twin): npad holds the code, tiles the real npad
            var dict = false;
            var tab: u32 = 0;
            if (dict_fn != null) for (outs) |o| if (a.coded(o.q)) {
                dict = true;
                tab = @max(tab, o.q.tabBytesUpTo(linear.gdn_pair_tab_max));
            };
            var parts: Parts = std.mem.zeroes(Parts);
            parts.count = 4;
            var first: usize = 0;
            for (outs, 0..) |o, index| {
                const tiles = if (index < 2) o.q.n / bn else 1;
                const twin = dict and a.coded(o.q);
                parts.p[index] = .{ .w = o.q.stream_w, .scales = if (twin) o.q.sb else o.q.s, .biases = if (twin) o.q.tab else o.q.b, .out = o.out, .n = int(o.q.n), .npad = if (twin) @bitCast(o.q.code) else int(o.q.npad), .sk = int(splitK(o.q.n,o.q.k)), .tiles = if (dict) (if (twin) int(o.q.npad) else 0) else int(tiles), .first = int(first) };
                first += tiles;
            }
            var g: launch_.Args = .{};
            g.add(x); g.add(xs); g.add(parts);
            for ([_]usize{ rows, 5120, 5120 }) |v| g.add(int(v));
            const base: u32 = if (rows32) gdn_stream32_smem else 22656;
            const shared: u32 = if (dict) base + 4 * @as(u32, @intCast(bn)) + tab else base;
            const f = if (dict) dict_fn.? else if (rows32) a.gdn_stream32_fn.? else a.gdn_stream16_fn.?;
            return launch_.launch(f, .{ .grid = .{ .x = u(first) }, .block = .{ .x = 128 }, .shared = shared, .pdl = true }, s, &g);
        }
        if (rows > max_rows) return error.WindowTooWide;
        const sp = a.split orelse return a.decodeGrouped(s, x, xs, outs, rows);
        return a.gemvSeg(s, x, xs, outs, rows, sp);
    }

    /// Attention's full-K Q and ordered sliced K/V tiles in one cluster launch.
    pub fn decodeAttentionMixed(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (rows != 16 or outs.len != 3) return error.InputMismatch;
        var cs: usize = 1;
        for (outs) |o| {
            if (o.q.k != outs[0].q.k) return error.InputMismatch;
            cs = @max(cs, splitK(o.q.n, o.q.k));
        }
        if (cs == 1) return error.InputMismatch;
        var dict = a.dict != null;
        var tab: u32 = 0;
        for (outs) |o| {
            dict = dict and tileCoded(o.q);
            if (tileCoded(o.q)) tab = @max(tab, o.q.tabBytesUpTo(linear.group_pair_tab_max)); // a cluster launch's room
        }
        var parts: Parts = std.mem.zeroes(Parts);
        var clusters: usize = 0;
        for (outs) |o| {
            const tiles = (o.q.n + 63) / 64;
            const sk = splitK(o.q.n, o.q.k);
            parts.p[@intCast(parts.count)] = groupPart(o.q, dict, o.out, sk, tiles, clusters);
            parts.count += 1;
            clusters += (tiles * sk + cs - 1) / cs;
        }
        var args: launch_.Args = .{};
        args.add(x); args.add(xs); args.add(parts);
        for ([_]usize{ rows, outs[0].q.k, outs[0].q.k, 1, cs }) |v| args.add(int(v));
        try launch_.launch(if (dict) a.dict.?.group else a.group_fn, .{ .grid = .{ .x = u(clusters * cs) }, .block = .{ .x = 128 },
            .shared = if (dict) group_smem + linear.dict_dec + tab else group_smem, .cluster = .{ .x = u(cs) }, .pdl = a.pdl }, s, &args);
    }

    /// decode()'s kernels a projection, grouped by kernel: each part's blocks compute exactly what its own launch's would.
    /// With cluster_decode the K-sliced parts share one cluster-tile launch of the widest part's slice count: a cluster
    /// holds one part, C / sk of its tiles, each tile's slices summed by its slice 0 in slice order as cluster() does.
    fn decodeGrouped(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (outs.len == 0 or outs.len > segs_max) return error.TooManyParts;
        const rows_t = (rows + 15) / 16;
        var lanes: linear.Many = std.mem.zeroes(linear.Many);
        var parts: Parts = std.mem.zeroes(Parts);
        var lane_one: usize = 0;
        var part_one: usize = 0;
        var blocks: usize = 0;
        var clusters: usize = 0;
        // dictionary twins: a launch takes its twin when every one of its parts is coded and, for lane CTAs, when
        // they fill the GPU (a few column CTAs, as attention k and v's 32, wait on their latency, not their bytes)
        var lane_tab: u32 = 0;
        var part_tab: u32 = 0;
        var lane_dict = a.dict != null;
        var part_dict = a.dict != null;
        var lane_blocks: usize = 0;
        for (outs) |o| {
            const sk = splitK(o.q.n, o.q.k);
            if (sk > 1) {
                lane_dict = lane_dict and tileCoded(o.q);
                if (tileCoded(o.q)) lane_tab = @max(lane_tab, o.q.tabBytesUpTo(linear.lane_pair_tab_max));
                lane_blocks += @min((o.q.n + 63) / 64, a.gemv_blocks);
            } else {
                part_dict = part_dict and tileCoded(o.q);
                if (tileCoded(o.q)) part_tab = @max(part_tab, o.q.tabBytesUpTo(linear.group_pair_tab_one)); // single CTAs
            }
        }
        if (lane_blocks < a.sms) lane_dict = false;
        const lane_cap = if (lane_dict) a.dict.?.gemv_blocks else a.gemv_blocks;
        for (outs, 0..) |o, i| {
            if (o.q.k != outs[0].q.k) return error.InputMismatch;
            const tiles = (o.q.n + 63) / 64;
            const sk = splitK(o.q.n, o.q.k);
            if (sk > 1) {
                const at: usize = @intCast(lanes.count);
                lanes.p[at] = gemvPart(o.q, lane_dict, o.out, sk, tiles);
                lanes.first[at] = int(blocks);
                lanes.count += 1;
                blocks += @min(tiles, lane_cap);
                lane_one = i;
            } else {
                const at: usize = @intCast(parts.count);
                parts.p[at] = groupPart(o.q, part_dict, o.out, 1, tiles, clusters);
                parts.count += 1;
                clusters += rows_t * tiles;
                part_one = i;
            }
        }
        if (parts.count == 1) {
            const o = outs[part_one];
            try a.cluster(s, x, xs, o.q, o.out, rows, 1);
        } else if (parts.count > 1) {
            var g: launch_.Args = .{};
            g.add(x);
            g.add(xs);
            g.add(parts);
            for ([_]usize{ rows, outs[0].q.k, outs[0].q.k, rows_t, 1 }) |v| g.add(int(v));
            const f = if (part_dict) a.dict.?.group else a.group_fn;
            const shared = if (part_dict) group_smem + linear.dict_dec + part_tab else group_smem;
            try launch_.launch(f, .{ .grid = .{ .x = u(clusters) }, .block = .{ .x = 128 }, .shared = shared, .pdl = a.pdl }, s, &g);
        }
        if (lanes.count == 1) {
            const o = outs[lane_one];
            try a.gemv(s, x, xs, o.q, o.out, rows, splitK(o.q.n, o.q.k));
        } else if (lanes.count > 1) {
            var g: launch_.Args = .{};
            g.add(x);
            g.add(xs);
            g.add(lanes);
            for ([_]usize{ rows, outs[0].q.k, outs[0].q.k }) |v| g.add(int(v));
            const f = if (lane_dict) a.dict.?.many else a.many_fn;
            const shared = if (lane_dict) group_smem + linear.dict_dec + lane_tab else group_smem;
            try launch_.launch(f, .{ .grid = .{ .x = u(blocks) }, .block = .{ .x = 128 }, .shared = shared, .pdl = a.pdl }, s, &g);
        }
    }

    /// One CTA a tile's K slice over every projection of `outs`, the split form's partials and tickets in `sp`.
    pub fn gemvSeg(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize, sp: Split) !void {
        if (outs.len == 0 or outs.len > segs_max) return error.TooManyParts;
        var segs: Segs = std.mem.zeroes(Segs);
        var blocks: usize = 0;
        var work: usize = 0;
        var ticks: usize = 0;
        for (outs, 0..) |o, i| {
            if (o.q.k != outs[0].q.k) return error.InputMismatch;
            const tiles = (o.q.n + 63) / 64;
            const sk = splitK(o.q.n, o.q.k);
            segs.s[i] = .{ .p = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = int(o.q.n), .npad = int(o.q.npad), .sk = int(sk), .tiles = int(tiles) }, .first = int(blocks), .work_at = int(work), .tick_at = int(ticks) };
            blocks += tiles * sk;
            if (sk > 1) {
                work += tiles * sk;
                ticks += tiles;
            }
        }
        segs.count = int(outs.len);
        if (ticks * 4 > sp.tickets.len or work * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(segs);
        for ([_]usize{ rows, outs[0].q.k, outs[0].q.k }) |v| g.add(int(v));
        g.add(sp.work.ptr);
        g.add(sp.tickets.ptr);
        try launch_.launch(a.seg_fn, .{ .grid = .{ .x = u(blocks) }, .block = .{ .x = 128 }, .shared = split_smem }, s, &g);
    }

    /// 16 rows elsewhere; GB10 verify shares a weight panel across up to eight m16 row tiles, sm_120 across two.
    pub fn verifyRows(a: *const Affine4Kernels) usize {
        return if (a.pdl) 128 else if (a.wide_verify) 32 else max_rows;
    }

    /// decode()'s bits a row, the fewest m16 row tiles that hold `rows` sharing each weight panel.
    pub fn verify(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        if (rows == 0 or rows > a.verifyRows()) return error.WindowTooWide;
        if (rows <= max_rows) return a.decode(s, x, xs, weight, out, rows);
        if (rows <= 32) {
            if (weight.wide_w != 0) return a.packed32(s, x, xs, weight, out, rows);
            return a.cluster32(s, x, xs, weight, out, rows);
        }
        if (rows <= 64) return a.wide(a.verify64_fn, linear.verify64_smem, 64, s, x, xs, weight, out, rows);
        return a.wide(a.verify128_fn, linear.verify128_smem, 128, s, x, xs, weight, out, rows);
    }

    /// The same ordered group and K-slice chains as decode(), with both row tiles reusing the loaded weights.
    pub fn cluster32(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        return a.wide(a.verify_fn, linear.verify_smem, 32, s, x, xs, weight, out, rows);
    }

    /// Projections of one 17 to 32-row input, grouped only with projections of the same K-slice count.
    pub fn verifyMany32(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (rows <= max_rows or rows > 32 or rows > a.verifyRows()) return error.WindowTooWide;
        if (outs.len == 0 or outs.len > segs_max) return error.TooManyParts;
        const k = outs[0].q.k;
        for (outs) |o| if (o.q.k != k) return error.InputMismatch;
        const alternate = for (outs) |o| {
            if (o.q.wide_w == 0 or o.q.wide_kind != 1) break false;
        } else true;
        const bn: usize = if (alternate) 128 else 64;
        const function = if (alternate) a.wide_fns[0] orelse return error.FormatNotLoaded else a.verify_fn;
        const shared: u32 = if (alternate) 26496 else linear.verify_smem;
        for ([_]usize{ 1, 2, 4, 8 }) |sk| {
            var parts: Parts = std.mem.zeroes(Parts);
            var clusters: usize = 0;
            for (outs) |o| {
                if (splitK(o.q.n, k) != sk) continue;
                const tiles = (o.q.n + bn - 1) / bn;
                const at: usize = @intCast(parts.count);
                parts.p[at] = .{ .w = if (alternate) o.q.wide_w else o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = int(o.q.n), .npad = int(o.q.npad), .sk = int(sk), .tiles = int(tiles), .first = int(clusters) };
                parts.count += 1;
                clusters += tiles;
            }
            if (parts.count == 0) continue;
            var args: launch_.Args = .{};
            args.add(x);
            args.add(xs);
            args.add(parts);
            for ([_]usize{ rows, k, k, 1, sk }) |v| args.add(int(v));
            try launch_.launch(function, .{ .grid = .{ .x = u(clusters * sk) }, .block = .{ .x = 128 }, .shared = shared, .cluster = if (sk > 1) .{ .x = u(sk) } else null, .pdl = a.pdl }, s, &args);
        }
    }

    /// The same K slices through an input-independent alternate word-panel view.
    fn packed32(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize) !void {
        if (q.wide_kind == 0 or q.wide_kind > 3) return error.FormatNotLoaded;
        const index = q.wide_kind - 1;
        const f = a.wide_fns[index] orelse return error.FormatNotLoaded;
        const bn: usize = if (q.wide_kind == 2) 256 else 128;
        const shared: u32 = switch (q.wide_kind) {
            1 => 26496,
            2 => 32768,
            3 => 17664,
            else => unreachable,
        };
        const sk = splitK(q.n, q.k);
        const tiles = (q.n + bn - 1) / bn;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.wide_w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        var args: launch_.Args = .{};
        args.add(x);
        args.add(xs);
        args.add(parts);
        for ([_]usize{ rows, q.k, q.k, 1, sk }) |v| args.add(int(v));
        try launch_.launch(f, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = shared, .cluster = if (sk > 1) .{ .x = u(sk) } else null, .pdl = a.pdl }, s, &args);
    }

    /// cluster32 with four (64) or eight (128) m16 row tiles a weight panel.
    pub fn clusterRows(a: *const Affine4Kernels, tile: usize, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        return switch (tile) {
            64 => a.wide(a.verify64_fn, linear.verify64_smem, 64, s, x, xs, weight, out, rows),
            128 => a.wide(a.verify128_fn, linear.verify128_smem, 128, s, x, xs, weight, out, rows),
            else => error.WindowTooWide,
        };
    }

    /// One CTA a column tile (a cluster of its K slices), all of `rows` in its m16 row tiles over each loaded panel.
    fn wide(a: *const Affine4Kernels, f: Function, shared: u32, most: usize, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        if (rows == 0 or rows > most) return error.WindowTooWide;
        const tiles = (weight.n + 63) / 64;
        const sk = splitK(weight.n, weight.k);
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = weight.w, .scales = weight.s, .biases = weight.b, .out = out, .n = int(weight.n), .npad = int(weight.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        for ([_]usize{ rows, weight.k, weight.k, 1, sk }) |v| g.add(int(v));
        try launch_.launch(f, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = shared, .cluster = if (sk > 1) .{ .x = u(sk) } else null, .pdl = a.pdl }, s, &g);
    }

    /// Each CTA writes one K slice's partial from zero; the tile's last CTA sums them in slice order.
    pub fn gemvSplit(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize, sp: Split) !void {
        const tiles = (q.n + 63) / 64;
        if (tiles > Split.tiles_max or tiles * sk * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        g.add(sp.work.ptr);
        g.add(sp.tickets.ptr);
        try launch_.launch(a.split_fn, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = split_smem, .pdl = a.pdl }, s, &g);
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in slice order, CTAs looping over the tiles.
    pub fn gemv(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        const dict = a.coded(q) and @min(tiles, a.dict.?.gemv_blocks) >= a.sms;
        const grid = @min(tiles, if (dict) a.dict.?.gemv_blocks else a.gemv_blocks);
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(gemvPart(q, dict, out, sk, tiles));
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        const cfg: launch_.Config = .{ .grid = .{ .x = u(grid) }, .block = .{ .x = 128 }, .shared = if (dict) group_smem + linear.dict_dec + q.tabBytesUpTo(linear.lane_pair_tab_max) else group_smem, .pdl = a.pdl };
        try launch_.launch(if (dict) a.dict.?.gemv else a.gemv_fn, cfg, s, &g);
    }

    /// qmm_group's tile 2: a cluster of `sk` CTAs a column tile, its K slices summed over distributed shared memory.
    pub fn cluster(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const stream_fn = if (q.stream_w == 0) null else a.stream16_fns[@as(usize, if (q.k == 6144) 1 else 0)];
        if (rows == 16 and sk == 4 and q.n == 5120 and (q.k == 6144 or q.k == 17408) and q.stream_w != 0 and stream_fn != null) {
            // A coded down projection, or an eligible out/o projection, streams its index windows.
            const dict = q.npad == 5120 and a.coded(q) and a.stream16_dict_fn != null and
                (q.k == 17408 or a.dictOutStream(q, rows, sk));
            var parts: Parts = std.mem.zeroes(Parts);
            parts.count = 1;
            parts.p[0] = .{ .w = q.stream_w, .scales = if (dict) q.sb else q.s, .biases = if (dict) q.tab else q.b, .out = out, .n = int(q.n), .npad = if (dict) @bitCast(q.code) else int(q.npad), .sk = 4, .tiles = 80, .first = 0 };
            var g: launch_.Args = .{};
            g.add(x); g.add(xs); g.add(parts);
            for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
            const shared = if (dict) group_smem + q.tabBytesUpTo(linear.out_pair_tab_max) else group_smem;
            return launch_.launch(if (dict) a.stream16_dict_fn.? else stream_fn.?, .{ .grid = .{ .x = 80 }, .block = .{ .x = 512 }, .shared = shared, .pdl = a.pdl }, s, &g);
        }
        const tiles = (q.n + 63) / 64;
        const rows_t = (rows + 15) / 16;
        const clusters = rows_t * tiles; // one part: the cluster is its sk K slices of one tile
        const dict = a.coded(q) and clusters * sk >= a.sms;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = groupPart(q, dict, out, sk, tiles, 0);
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| g.add(int(v));
        const cfg: launch_.Config = .{
            .grid = .{ .x = u(clusters * sk) },
            .block = .{ .x = 128 },
            .shared = if (dict) group_smem + linear.dict_dec + q.tabBytesUpTo(if (sk > 1) linear.group_pair_tab_max else linear.group_pair_tab_one) else group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = a.pdl,
        };
        try launch_.launch(if (dict) a.dict.?.group else a.group_fn, cfg, s, &g);
    }

    /// qmm.prefill_matmul (tile 0): weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prompt(a: *const Affine4Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var g: launch_.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| g.add(int(v));
        const grid = rows_t * ((q.n + 127) / 128);
        const f = if (a.half_prompt) a.prefill22_fn else a.prefill_fn;
        const threads: u32 = if (a.half_prompt) 128 else 256;
        const smem: u32 = if (a.half_prompt) linear.prefill22_smem else prefill_smem;
        try launch_.launch(f, .{ .grid = .{ .x = u(grid), .y = 1, .z = 1 }, .block = .{ .x = threads }, .shared = smem }, s, &g);
    }
};
