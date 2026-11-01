// Modified by Layr Labs, Inc. in 2026; see THIRD_PARTY_NOTICES.md.
//! Qwen3.5's hybrids on CUDA: qlinear projections, qwen35.cu glue, one captured forward over every stream's window.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Round = @import("cuda_round.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
pub const verify_weight_layout = @import("cuda_verify_weights.zig");
const caches = @import("cuda_cache.zig");
const routed = @import("cuda_routed.zig");
const hooks = @import("cuda_learn_hooks.zig");
const pocket = @import("cuda_pocket.zig");
const profile = @import("cuda_startup_profile.zig");
const modelopt = @import("cuda_modelopt.zig");
const x3w = @import("cuda_exl3_weights.zig");
pub const recurrent = @import("cuda_recurrent.zig");
pub const projection_stream_layout = @import("cuda_projection_streams.zig");
pub const wide_weight_layout = @import("cuda_wide_weights.zig");

pub const Geometry = geometry.Geometry;
pub const Cache = caches.Cache;
pub const tap_rows = caches.tap_rows;
pub const tap_held = caches.tap_held;
const Seg = caches.Seg;
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;

/// Rows of one forward at most: a prompt chunk, or a shared round's windows.
pub const max_rows = Round.max_rows;
/// Streams' windows in one forward at most.
pub const max_segs = Round.max_segs;
/// Rows of a verify forward at most: each row's logits are kept and drawn.
pub const most_verify = Round.most_verify;
/// L2 set aside for a prompt chunk's residual rows (a 496-row chunk's are 5 MB; GB10 allows 18 MB).
const l2_persist_bytes: usize = 8 << 20;
/// tf_q35_argmax_slices' blocks a row (ARGMAX_SLICE entries each).
fn argmaxSlices(vocab: usize) usize {
    return (vocab + 4095) / 4096;
}
/// A window's rows at most (exact: one-row steps' bits; qwen35.cuh's MAXW, the rows a stream's log holds).
pub const most_window = Round.most_window;
/// Partial attention chunks a row at most (qwen35.cuh's MAXCH): each row's chunk width grows with its own keys.
const max_chunks = 64;

/// Device allocations are 256-byte aligned, as the driver's are.
pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// qwen35.cuh's Geo.
pub const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };

const layers = @import("cuda_layers.zig");
pub const Proj = layers.Proj;
pub const Delta = layers.Delta;
pub const Attention = layers.Attention;
pub const Layer = layers.Layer;
pub const Mlp = layers.Mlp;
const Fns = layers.Fns;
pub const Keeps = layers.Keeps;
const Kv8 = layers.Kv8;

/// A window of one stream in a forward: `parents`, for a tree, names each row's parent row (-1: the first row).
pub const Part = struct { cache: *Cache, rows: usize, parents: ?[]const i32 = null, images: ?*const Images = null };

/// An image prompt's rows for its prompt chunks: the staged tables' rows.
pub const Images = Round.Images;

/// What a forward keeps: a prompt chunk its last row's logits, a verify window every row's draw or logits and log.
pub const Keep = union(enum) { last, draws: []const Draw, logits, hidden };

/// A kept row's draw on the device: sample.cu's keyed rule at `position`, or greedy without one.
pub const Draw = struct { rule: ?cuda.sample.Rule, position: u64 };

/// The one full-window lane beside a page pool, recreated lazily after a phase drain.
pub const FirstWindow = struct {
    active: bool = false,
    pub fn available(w: FirstWindow, enabled: bool) bool {
        return enabled and !w.active;
    }
    pub fn claim(w: *FirstWindow, enabled: bool) bool {
        if (!w.available(enabled)) return false;
        w.active = true;
        return true;
    }
    pub fn release(w: *FirstWindow) void {
        w.active = false;
    }
};

pub const Engine = struct {
    rec_scratch: recurrent.Pool,
    startup: profile.Times = .{},
    gpa: std.mem.Allocator,
    dir: []const u8,
    pockets: pocket.Pockets,
    learning: ?*const hooks.Keep = null,
    mlp: ?hooks.MlpAdd = null,
    taught: u64 = 0,
    d: *const cuda.Driver,
    g: Geometry,
    geo: Geo,
    stream: cuda.Stream,
    blocks: std.ArrayListUnmanaged(caches.Recycled) = .empty, // closed streams' device blocks (cuda_cache.zig take/give)
    mods: [19]cuda.Module,
    kv8: bool, // attention caches as e4m3 rows with a scale each
    fp8: Kv8,
    prompt_attn: @import("cuda_prompt.zig").Fns, // a prompt chunk's attention tiles and delta-rule chain
    int8: @import("cuda_prompt_int8.zig").State = .{}, // prompt projections with int8 activations, when on
    draws: cuda.sample.Draws,
    affine: cuda.qlinear.Affine4Kernels,
    dict: ?cuda.qlinear.DictStore, // the decode kernels' dictionary-coded scales and biases (GB10; TF_DICT=0: none)
    dict_kmajor: bool, // gate and up coded group-major, for tf_qmm_mlp_single16 on the K-major panels (TF_MLP_KMAJOR=2)
    prompt_many: cuda.prompt_many.Kernels, // prompt GEMMs, several projections of one input a launch
    mlp_pair: cuda.mlp_pair.Kernels, // decode gate, up and activation in one launch (qmm_group.cu tf_qmm_mlp_pair)
    /// L2 set aside for persisting lines (0: none) and, during a prompt chunk's forward, the residual rows' window:
    /// each norm writes the rows it reads a sublayer later, past a projection's weights streaming through L2.
    l2_persist: usize = 0,
    h_window: ?cuda.abi.AccessPolicyWindow = null,
    wide_words: ?DeviceBuffer = null,
    projection_streams: ?DeviceBuffer = null,
    wide: cuda.qlinear.AffineNKernels, // the tiles of every width but 4, and their packs
    experts: cuda.experts.Affine4Experts,
    experts3: cuda.experts3.Affine3Experts,
    mo: ?modelopt.Kernels = null, // a ModelOpt checkpoint's lane, prompt GEMM, packers and bf16 rows
    plan: cuda.grouped.Router,
    moe: cuda.moe.Fns,
    fns: Fns,
    keeps: Keeps,
    /// Every weight and scratch buffer in one allocation: WDDM's launch cost grows with the allocation count.
    arena: DeviceBuffer,
    arena_at: usize = 0,
    /// Rows a verify forward takes at most, and a window's rows at most.
    verify_rows: usize,
    window_rows: usize,
    verify_weights: ?DeviceBuffer = null,
    // weights
    embed_w: u64,
    embed_s: u64,
    embed_b: u64,
    embed_bits: u8, // the embedding's rows' width: 4 or 8 (16: bf16 rows, an EXL3 checkpoint's)
    kit: ?*x3w.Kit = null, // an EXL3 checkpoint's kernels (g.exl3)
    head: Proj,
    final_norm: u64,
    layers: []Layer,
    // scratch, max_rows rows each
    h: u64,
    x: u64,
    r: u64,
    xs: u64,
    qkv: u64,
    z: u64,
    a: u64,
    b: u64,
    q: u64,
    k: u64,
    v: u64,
    g_: u64,
    beta: u64,
    y: u64,
    mix: u64,
    qraw: u64,
    kraw: u64,
    vraw: u64,
    qn: u64,
    kn: u64,
    gate: u64,
    up: u64,
    act: u64,
    pm: u64,
    pl: u64,
    po: u64,
    logits: u64,
    routed: routed.Scratch = .{}, // qwen3_5_moe's routed rows
    amax: u64, // the argmax slices' bests: most_verify rows of argmaxSlices(vocab) (f32, i32)
    picks: u64, // drawRows' slice scratch: most_verify rows of cuda.sample.Draws.row_scratch
    // per-forward tables (ids, segments, rows' segment, parent and ancestors, the draws' rules) staged in pinned memory
    tables: DeviceBuffer,
    staged: cuda.HostBuffer,
    host_logits: cuda.HostBuffer,
    /// The kept rows' device draws, when forward was given them.
    tokens: [most_verify]u32 = undefined,
    /// The captured forwards by shape (`Shape.key`): the launches from the embedding to the head.
    graphs: std.AutoHashMapUnmanaged(u32, cuda.graph.Exec) = .empty,
    /// A captured forward's first layer as a graph of its own (keyed as `graphs`): launched first, it starts the GPU
    /// while the host still launches the larger rest, whose launch would otherwise leave the GPU idle until it returns.
    heads: std.AutoHashMapUnmanaged(u32, cuda.graph.Exec) = .empty,
    cut_armed: bool = false,
    cut_head: ?cuda.graph.Exec = null,
    /// A drafter's tapped layers (`setTaps`): each row's residual after them lands in its stream's tap ring.
    taps: [8]u32 = undefined,
    n_taps: usize = 0,
    /// During an image prompt chunk's forward: its feature rows (0: none); the chunk's src and positions are staged.
    features: u64 = 0,

    pub const ids_at = Round.ids_at;
    pub const segs_at = Round.segs_at;
    pub const rowseg_at = Round.rowseg_at;
    pub const rowpar_at = Round.rowpar_at;
    pub const rowanc_at = Round.rowanc_at;
    const rules_at = Round.rules_at;
    const meta_at = Round.meta_at;
    const tokens_at = Round.tokens_at;
    pub const params_at = Round.params_at;
    pub const src_at = Round.src_at;
    pub const mpos_at = Round.mpos_at;
    const tables_len = Round.bytes;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kv8: bool) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const started = profile.now(io);
        const g = try Geometry.read(gpa, io, dir);
        const d = ctx.d;
        const e = try @import("cuda_prompt_int8.zig").Allocation(Engine).create(gpa);
        e.h_window = null; // Allocation.create leaves fields undefined before the first captured forward.
        errdefer gpa.destroy(e);
        e.projection_streams = null;
        errdefer if (e.projection_streams) |*b| b.free();
        e.wide_words = null;
        errdefer if (e.wide_words) |*b| b.free();
        errdefer e.int8.deinit();
        e.gpa = gpa;
        e.dir = try gpa.dupe(u8, dir);
        errdefer gpa.free(e.dir);
        e.pockets = try pocket.Pockets.init(gpa, d, g.hidden, g.layers);
        errdefer e.pockets.deinit();
        e.learning = null;
        e.mlp = null;
        e.taught = 0;
        e.kit = null;
        e.dict = null;
        e.dict_kmajor = false;
        e.d = d;
        e.g = g;
        e.geo = .{ .hid = @intCast(g.hidden), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .qh = @intCast(g.q_heads), .kvh = @intCast(g.kv_heads), .nl = @intCast(g.deltas()) };
        e.verify_rows = most_verify;
        e.kv8 = kv8;
        e.window_rows = most_window;
        e.arena_at = 0;
        e.graphs = .empty;
        e.heads = .empty;
        e.cut_armed = false;
        e.cut_head = null;
        e.rec_scratch = recurrent.Pool.init(gpa, @intFromPtr(e));
        e.n_taps = 0;
        e.features = 0;
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.blocks = .empty; // the field's default does not apply: Allocation(Engine).create leaves the engine undefined
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        // an EXL3 checkpoint's weights are sized by a dry run of its loader (trellis widths come from the tensors)
        const wbytes = if (g.exl3) try x3w.measure(e, &ck) else weights.total(g);
        const arena_bytes = wbytes + try e.scratch(.measure);
        var sponge = @import("cuda_sponge.zig").take(d, arena_bytes);
        defer sponge.free();
        e.arena = try DeviceBuffer.alloc(d, arena_bytes);
        errdefer e.arena.free();
        try e.arena.fill8(0, null); // touched once now: the driver commits memory on first touch, slowly
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qwen35, kk.sample, kk.qwen35_keep, kk.affine8, kk.affine_prefill, kk.qwen35_kv8, kk.experts, kk.experts_prefill, kk.experts_pack, kk.moe, kk.experts3, kk.experts3_prefill, kk.prefill_attention, kk.gdn_prefill, kk.w4a8_prefill };
        var loaded: usize = 0;
        errdefer for (e.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            e.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        const major = try ctx.attribute(.compute_capability_major);
        const minor = try ctx.attribute(.compute_capability_minor);
        const gb10 = major == 12 and minor == 1;
        e.affine = try cuda.qlinear.Affine4Kernels.resolve(d, e.mods[0], e.mods[1], e.mods[2], e.mods[3], sms, gb10);
        e.affine.half_prompt = !gb10; // the 27B's prompt tile (Tile<64, 128, 128, 2, 2, 2>), quicker on an RTX 5090
        // GB10: split-K decode/verify projections on the cluster tile (bits unchanged); TF_CLUSTER_DECODE=0 restores lane_gemv
        e.affine.cluster_decode = gb10 and !envOff("TF_CLUSTER_DECODE");
        // prompt chunks keep the residual rows in L2 across their projections (cache residency only, never a value)
        e.l2_persist = 0;
        if (gb10 and !envOff("TF_L2_PERSIST") and d.api.cuCtxSetLimit(cuda.abi.limit_persisting_l2, l2_persist_bytes) == cuda.abi.success) e.l2_persist = l2_persist_bytes;
        e.affine.group_wide = gb10 and !envOff("TF_WIDE_GROUP");
        // GB10: out/o (short K slices) on lane_gemv's column CTAs instead (bits unchanged); TF_SHORT_LANE=0 restores the cluster tile
        e.affine.short_lane = gb10 and !envOff("TF_SHORT_LANE");
        errdefer e.affine.deinit();
        e.prompt_many = try cuda.prompt_many.Kernels.resolve(e.mods[1], sms);
        e.mlp_pair = try cuda.mlp_pair.Kernels.resolve(e.mods[0], gb10);
        // GB10: the decode projections read their scales and biases as dictionary indices built at load (exact: the
        // tables hold every value); TF_DICT=0 keeps every launch on the (kg, npad) arrays
        if (envOff("TF_DICT")) {
            e.affine.dict = null;
            e.mlp_pair.dict_fn = null;
        }
        if (e.affine.dict != null) e.dict = try cuda.qlinear.DictStore.init(d);
        // the joint (scale, bias) pair dictionary, installed wherever it is smaller; TF_DICT_PAIR=0 keeps the separate
        // coding, =check encodes and checks the pairs but keeps the separate coding
        if (e.dict) |*ds| {
            ds.pair = .on;
            if (@import("sys").c.getenv("TF_DICT_PAIR")) |v| {
                const mode = std.mem.span(v);
                if (std.mem.eql(u8, mode, "0")) ds.pair = .off else if (std.mem.eql(u8, mode, "check")) ds.pair = .check;
            }
        }
        // the same choice as verify_weight_layout.prepare's below: tf_qmm_mlp_single16 unless TF_MLP_KMAJOR says otherwise
        e.dict_kmajor = gb10 and std.mem.eql(u8, std.mem.span(@import("sys").c.getenv("TF_MLP_KMAJOR") orelse "2"), "2");
        errdefer if (e.dict) |*ds| ds.deinit();
        // sm_120 too: a 32-row window of the 27B verifies in 17.8 ms on one weight pass, 22.3 ms on two 16-row ones
        if (major == 12 and minor == 0) e.affine.wide_verify = true;
        e.wide = try cuda.qlinear.AffineNKernels.resolve(d, e.mods[7], e.mods[8], e.mods[3], sms, gb10);
        errdefer e.wide.deinit();
        e.draws = try cuda.sample.Draws.resolve(e.mods[5]);
        e.experts = try cuda.experts.Affine4Experts.resolve(e.mods[10], e.mods[11], e.mods[12], sms);
        e.experts3 = try cuda.experts3.Affine3Experts.resolve(e.mods[14], e.mods[15], sms);
        e.plan = try cuda.grouped.Router.resolve(e.mods[10]);
        e.moe = try cuda.moe.Fns.resolve(e.mods[13]);
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(e.fns, name) = try e.mods[4].function("tf_q35_" ++ name);
        try e.fns.attn_partial.allowDynamicShared(@import("cuda_forward.zig").attn_tiles); // bf16 tiles' two buffers
        try e.fns.attn_rows.allowDynamicShared(@import("cuda_forward.zig").attn_tiles);
        inline for (@typeInfo(Keeps).@"struct".field_names) |name| @field(e.keeps, name) = try e.mods[6].function("tf_q35k_" ++ name);
        e.prompt_attn = try @import("cuda_prompt.zig").resolve(e.mods[16], e.mods[17]);
        try e.int8.resolve(e.mods[18]);
        e.fp8 = .{ .rope = try e.mods[9].function("tf_q35_rope8"), .attn_partial = try e.mods[9].function("tf_q35_attn_partial8"), .attn_rows = try e.mods[9].function("tf_q35_attn_rows8"), .attn_partial_paged = try e.mods[9].function("tf_q35_attn_partial8_paged"), .attn_partial_contiguous = try e.mods[9].function("tf_q35_attn_partial8_contiguous"), .compact = try e.mods[9].function("tf_q35k_compact8") };
        e.mo = if (g.kinds.modelopt) try modelopt.Kernels.load(d, @intCast(try ctx.attribute(.compute_capability_major))) else null;
        errdefer if (e.mo) |*m| m.unload();
        e.layers = try gpa.alloc(Layer, g.layers);
        errdefer gpa.free(e.layers);
        if (g.exl3) {
            e.kit = try x3w.Kit.init(gpa, d, g);
            errdefer e.kit.?.deinit(gpa);
            try x3w.load(e, &ck);
        } else {
            // each projection as stored, packed from here; freed before the server counts what streams can use
            var stage = try DeviceBuffer.alloc(d, weights.stageBytes(g));
            defer stage.free();
            try weights.load(e, &ck, &stage);
        }
        if (e.dict) |ds| if (ds.pair != .off) std.debug.print("info: pair dictionary ({s}): {d} projections encoded, {d} declined, {d} mismatching cells; index streams {d} B (separate coding {d} B); largest table {d} entries; projections by index bits {any}\n", .{ @tagName(ds.pair), ds.pair_coded, ds.pair_declined, ds.pair_bad, ds.pair_bytes, ds.pair_separate_bytes, ds.pair_np_max, ds.pair_pbits });
        if (gb10 and g.hidden == 5120 and g.layers == 64 and g.experts == 0 and
            @import("sys").c.getenv("TF_WIDE_KMAJOR") != null and !envOff("TF_WIDE_KMAJOR"))
            e.wide_words = try wide_weight_layout.prepare(e);
        e.verify_weights = null;
        if (gb10) {
            const layout = @import("sys").c.getenv("TF_MLP_KMAJOR") orelse "2";
            const single = std.mem.eql(u8, std.mem.span(layout), "2");
            if (single or std.mem.eql(u8, std.mem.span(layout), "1"))
                e.verify_weights = try verify_weight_layout.prepare(e, single);
        }
        errdefer if (e.verify_weights) |*b| b.free();
        const output_stream = !envOff("TF_OUTPUT_STREAMS");
        const gdn_stream = !envOff("TF_GDN_STREAM");
        if (gb10 and g.hidden == 5120 and g.layers == 64 and g.experts == 0 and g.pocket == 0 and (output_stream or gdn_stream))
            e.projection_streams = try projection_stream_layout.prepare(e, gdn_stream, output_stream);
        if (e.mo) |*m| try m.reserveParts(e);
        try @import("cuda_engine_tail.zig").finish(e, io, started);
        return e;
    }

    pub fn finishWarm(e: *Engine, io: std.Io, started: i96) !void {
        const warm_start = profile.now(io);
        e.startup = .{ .load = warm_start - started };
        errdefer recurrent.shutdown(e);
        e.startup.add(try profile.withCache(e.gpa, io, e, Cache));
    }

    /// Every path a request takes, run once now: WDDM stalls a process's first forwards for seconds (a 20 s launch).
    pub fn warmCache(e: *Engine, warm_cache: *Cache) !void {
        const ids: [32]u32 = @splat(0);
        const greedy: Draw = .{ .rule = null, .position = 33 };
        try e.forward(&.{.{ .cache = warm_cache, .rows = ids.len }}, &ids, .last); // a prompt chunk: the prompt GEMMs
        try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{greedy} });
        try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = .{ .seed = 1, .inv_t = 1.0 / 0.7, .top_p = 0.95, .near = 20.0, .min_log = -std.math.inf(f32), .top_k = 20 }, .position = 34 }} });
        // a tree window and a branch of it kept: the keep kernels, and the forward's branch path
        try e.forward(&.{.{ .cache = warm_cache, .rows = 3, .parents = &.{ -1, 0, 0 } }}, ids[0..3], .{ .draws = &.{ greedy, greedy, greedy } });
        try warm_cache.keep(e, &.{ 0, 2 });
        try e.stream.synchronize();
        recurrent.completed(e);
    }

    pub fn deinit(e: *Engine) void {
        recurrent.shutdown(e);
        if (e.kit) |k| k.deinit(e.gpa);
        e.pockets.deinit();
        var it = e.graphs.valueIterator();
        while (it.next()) |x| x.deinit();
        e.graphs.deinit(e.gpa);
        var hit = e.heads.valueIterator();
        while (hit.next()) |x| x.deinit();
        e.heads.deinit(e.gpa);
        e.host_logits.free();
        e.staged.free();
        e.tables.free();
        if (e.wide_words) |*b| b.free();
        if (e.projection_streams) |*b| b.free();
        if (e.verify_weights) |*b| b.free();
        e.arena.free();
        e.gpa.free(e.layers);
        e.gpa.free(e.dir);
        for (e.blocks.items) |*b| b.buffer.free();
        e.blocks.deinit(e.gpa);
        @import("cuda_cache.zig").forget(e);
        if (e.dict) |*ds| ds.deinit();
        e.affine.deinit();
        e.wide.deinit();
        e.int8.deinit();
        if (e.mo) |*m| m.unload();
        for (&e.mods) |*m| m.unload();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    /// A stream's device bytes at a `cap`-token window: recurrences, convolution tails, keys and values, taps, log.
    pub fn streamBytes(e: *const Engine, cap: usize) usize {
        return 2 * e.g.stateBytes() + e.g.kvBytes(cap, e.kv8) + e.tapBytes() + 2 * e.g.logBytes(most_window);
    }

    /// A stream's tap ring: tap_rows rows of the tapped layers' residuals.
    pub fn tapBytes(e: *const Engine) usize {
        return tap_rows * e.n_taps * e.g.hidden * 2;
    }

    /// The layers a drafter taps, told before the streams' caches exist; each forward then fills their rings.
    pub fn setTaps(e: *Engine, taps: []const u32) !void {
        if (taps.len > e.taps.len) return error.TooManyTaps;
        for (taps) |t| if (t >= e.g.layers) return error.BadTap;
        @memcpy(e.taps[0..taps.len], taps);
        e.n_taps = taps.len;
    }

    /// The next `bytes` of the arena, 256-byte aligned.
    pub fn alloc(e: *Engine, bytes: usize) !u64 {
        const at = aligned(e.arena_at);
        if (at + bytes > e.arena.len) return error.ArenaExhausted;
        e.arena_at = at + bytes;
        return e.arena.ptr + at;
    }

    /// The scratch rows' bytes (`measure`: summed, nothing taken), or the rows themselves from the arena.
    pub fn scratch(e: *Engine, mode: enum { measure, allocate }) !usize {
        var total: usize = 0;
        const Take = struct {
            e: *Engine,
            mode: @TypeOf(mode),
            total: *usize,
            pub fn alloc(t: @This(), bytes: usize) !u64 {
                t.total.* += aligned(bytes);
                return if (t.mode == .measure) 0 else try t.e.alloc(bytes);
            }
        };
        const t = Take{ .e = e, .mode = mode, .total = &total };
        const g = e.g;
        const R = max_rows;
        const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden); // y and mix: either mixer's rows
        inline for (.{ "h", "x", "r" }) |name| @field(e, name) = try t.alloc(R * g.hidden * 2);
        inline for (.{ "y", "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(R * wide * 2);
        inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(R * g.k_heads * geometry.dk * 2);
        e.qkv = try t.alloc(R * g.convWidth() * 2);
        inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(R * g.intermediate * 2);
        e.xs = try t.alloc(R * (@max(g.intermediate, wide) / 64) * 4);
        inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(R * g.v_heads * 2);
        e.g_ = try t.alloc(R * g.v_heads * 4);
        e.qraw = try t.alloc(R * 2 * g.attnWidth() * 2);
        e.qn = try t.alloc(R * g.attnWidth() * 2);
        inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(R * g.kv_heads * head_dim * 2);
        // the decode partials: verify rows only (a prompt chunk's attention runs cuda_prompt_attention's tiles)
        e.pm = try t.alloc(most_verify * g.q_heads * max_chunks * 4);
        e.pl = try t.alloc(most_verify * g.q_heads * max_chunks * 4);
        e.po = try t.alloc(most_verify * g.q_heads * max_chunks * head_dim * 4);
        e.logits = try t.alloc(most_verify * g.vocab * 2);
        if (g.moe()) e.routed = try routed.Scratch.carve(t, g, R);
        e.amax = try t.alloc(most_verify * argmaxSlices(g.vocab) * 8);
        e.picks = try t.alloc(most_verify * cuda.sample.Draws.row_scratch);
        return total;
    }

    /// While a forward is captured: the work so far becomes the head graph and the capture goes on for the rest.
    pub fn cutCapture(e: *Engine) !void {
        if (!e.cut_armed or e.cut_head != null) return;
        var head = try cuda.graph.endCapture(e.stream);
        defer head.deinit();
        e.cut_head = try head.instantiate();
        try cuda.graph.beginCapture(e.stream, .thread_local);
    }

    pub fn run(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) } };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// `run` with programmatic dependent launch on GB10, for glue kernels that open with griddepcontrol.wait
    /// (qwen35.cu's await_prior): the launch overlaps the preceding kernel's tail, which triggers early.
    pub fn runAfter(e: *Engine, f: cuda.Function, grid: [3]usize, block: [3]usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block[0]), .y = @intCast(block[1]), .z = @intCast(block[2]) }, .pdl = glue_pdl and e.affine.pdl };
        try cuda.launch.launch(f, cfg, e.stream, &a);
    }

    /// The projections: the row tiles, the prompt GEMM and the grouped form, in their own file.
    pub const proj = @import("cuda_projection.zig").proj;
    pub const projSums = @import("cuda_projection.zig").projSums;
    pub const projAgain = @import("cuda_projection.zig").projAgain;
    pub const projMany = @import("cuda_projection.zig").projMany;
    pub const projManySums = @import("cuda_projection.zig").projManySums;
    pub const projectMany = @import("cuda_projection.zig").projectMany;
    pub const project = @import("cuda_projection.zig").project;

    /// Token rows of the embedding into `out` (rows, hidden) bf16, at the width its words are served.
    pub fn embed(e: *Engine, ids: u64, out: u64, rows: usize, params: u64) !void {
        if (e.kit) |kit| return kit.b16.embedRows(e.stream, ids, e.embed_w, out, rows, e.g.hidden, params);
        if (e.embed_bits == 16) return modelopt.embed(e, ids, out, rows); // bf16 rows; padding rows' ids are zeroed
        const f = if (e.embed_bits == 8) e.fns.embed8 else e.fns.embed;
        try e.run(f, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ ids, e.embed_w, e.embed_s, e.embed_b, out, @as(c_int, @intCast(e.g.hidden)), params });
    }

    /// Logits rows first..first + n's greedy tokens into toks: the highest logit, ties to the lower id.
    fn argmax(e: *Engine, first: usize, n: usize, toks: u64) !void {
        if (n == 0) return;
        const slices = argmaxSlices(e.g.vocab);
        const pv = e.amax;
        const pi = e.amax + n * slices * 4;
        try e.run(e.fns.argmax_slices, .{ slices, n, 1 }, .{ 256, 1, 1 }, .{ e.logits + first * e.g.vocab * 2, @as(c_int, @intCast(e.g.vocab)), pv, pi });
        try e.run(e.fns.argmax_merge, .{ n, 1, 1 }, .{ 32, 1, 1 }, .{ pv, pi, @as(c_int, @intCast(slices)), toks + first * 4 });
    }

    pub const norm = @import("cuda_norm.zig").norm;
    pub const normInput = @import("cuda_norm.zig").normInput;

    /// The residual rows of a prompt chunk of `rows` as a persisting L2 window (null: none set aside, or a window).
    pub fn hWindow(e: *const Engine, rows: usize) ?cuda.abi.AccessPolicyWindow {
        if (e.l2_persist == 0 or rows <= 16) return null;
        const bytes = rows * e.g.hidden * 2;
        const ratio: f32 = @min(1.0, @as(f32, @floatFromInt(e.l2_persist)) / @as(f32, @floatFromInt(bytes)));
        return .{ .base = e.h, .bytes = bytes, .hit_ratio = ratio, .hit_prop = cuda.abi.access_persisting, .miss_prop = cuda.abi.access_streaming };
    }
    pub const normTail = @import("cuda_norm.zig").normTail;

    /// Refuse the whole frame before settling any prior window or launching work.
    pub fn admit(e: *const Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        const g = e.g;
        if (!geometry.supported(g, ids.len, keep == .last, most_verify)) return error.UnsupportedQwenGeometry;
        var total: usize = 0;
        if (parts.len == 0 or parts.len > max_segs) return error.InvalidQwenSegments;
        for (parts, 0..) |p, index| {
            for (parts[0..index]) |prior| if (prior.cache == p.cache) return error.InvalidQwenSegments;
            if (p.rows == 0 or p.cache.len > p.cache.cap or p.rows > p.cache.cap - p.cache.len) return error.QwenContextFull;
            if (p.rows > max_rows - total) return error.InvalidQwenSegments;
            total += p.rows;
        }
        if (parts.len == 0 or parts.len > max_segs or total != ids.len or total > max_rows) return error.InvalidQwenSegments;
        if (e.learning != null and total != 1) return error.BadLearningRows;
        if (keep == .hidden and e.learning == null) return error.BadLearningRows;
        const record = keep != .last;
        if (record and total > e.verify_rows) return error.InvalidQwenSegments;
        if (!record and parts.len != 1) return error.InvalidQwenSegments;
        if (keep == .draws and keep.draws.len != total) return error.InvalidQwenSegments;
        for (ids) |id| if (id >= g.vocab) return error.InvalidQwenToken;
        for (parts) |p| if (p.parents != null and !record) return error.InvalidQwenParents;
        for (parts) |p| if (p.images != null and record) return error.InvalidQwenImages;
    }

    /// One forward over `parts`' windows, with every input admitted before an unkept verify is settled.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, keep: Keep) !void {
        try e.admit(parts, ids, keep);
        const g = e.g;
        const total = ids.len;
        const record = keep != .last;
        const heads: usize = if (record) total else 1;
        // the prompt GEMM only for prompt chunks: a window's rows keep the bits of one-row steps
        const prompt = !record;
        // the tables
        const st = e.staged.bytes;
        @memcpy(std.mem.bytesAsSlice(u32, st[ids_at..][0 .. total * 4]), ids);
        @memset(st[ids_at + total * 4 .. ids_at + max_rows * 4], 0); // a bucket's padding rows read token 0
        const segs = std.mem.bytesAsSlice(Seg, st[segs_at..][0 .. parts.len * @sizeOf(Seg)]);
        // past 1 row a window's scan carries each parent's state in the registers to its first child
        // (tree_slots.carried: one slot fewer, -2 steps); TF_CARRIED=0 restores build's plans
        const carry = record and total > 1 and !envOff("TF_CARRIED");
        const slot_peak = try Round.buildCarried(st, parts, record, carry);
        for (parts) |p| {
            try p.cache.settle(e);
            // a pending kept prefix rides in this forward's slot scan only when its log half survives `prepare`
            if (!(record and keep != .hidden and recurrent.logStays(e, p.cache, p.rows))) try p.cache.flush(e);
            try recurrent.prepare(e, p.cache, p.rows, record);
        }
        var row0: usize = 0;
        for (parts, 0..) |p, i| {
            segs[i] = p.cache.fold(p.cache.seg(e, p.cache.len, row0, p.rows));
            row0 += p.rows;
        }
        if (keep == .draws) {
            const rules = std.mem.bytesAsSlice(cuda.sample.Rule, st[rules_at..][0 .. total * @sizeOf(cuda.sample.Rule)]);
            const meta = std.mem.bytesAsSlice(c_int, st[meta_at..][0 .. total * 4]);
            for (keep.draws, rules, meta) |p, *r, *m| {
                if (p.rule) |x| r.* = x;
                m.* = @intCast(p.position -% 1); // sample.cu keys row r at meta + r + 1
            }
        }
        // words 0 and 1 are reserved (each row's attention chunks follow its own keys: qwen35.cu's chunk_keys)
        const params = std.mem.bytesAsSlice(c_int, st[params_at..][0..32]);
        params[0] = 0;
        params[1] = 0;
        params[2] = @intCast(total);
        params[3] = @intCast(parts.len);
        params[4] = @intFromBool(record);
        defer e.taught = 0;
        try @import("cuda_forward_extra.zig").stageScales(e, parts, total);
        // an image prompt's chunk (one part, its rows at prompt positions len..): feature rows and t, h, w positions
        const images: ?*const Images = if (record) null else parts[0].images;
        if (images) |im| {
            const p0 = parts[0].cache.len;
            if (p0 + total > im.src.len or im.pos.len != 3 * im.src.len) return error.InvalidQwenImages;
            @memcpy(std.mem.bytesAsSlice(i32, st[src_at..][0 .. total * 4]), im.src[p0..][0..total]);
            Round.stageImageTables(st, im, p0, total);
        }
        try e.tables.uploadAsync(0, st[0..if (images != null) tables_len else src_at], e.stream.handle); // text: as before
        // carried plans of 2..32 rows (at most 3 slots) scan four value rows a warp on qwen35_slots_lean.cuh (slots_3's
        // arithmetic, its addressing hoisted; 2 slots where they suffice); TF_SLOTS_LEAN=0 keeps prior dispatch
        const lean = record and total > 1 and total <= 32 and e.affine.pdl and !envOff("TF_SLOTS_LEAN");
        const wide = total > 16 and total <= 32 and e.affine.pdl and !envOff("TF_WIDE_SLOTS");
        const slot_variant: usize = if (lean and carry and slot_peak <= 3)
            (if (slot_peak <= 2) 37 else 38)
        else if (wide and slot_peak == 3) 3 else Round.narrowVariant(total, slot_peak, record and e.affine.pdl and !envOff("TF_GDN16"));
        var shape = Shape.init(total, parts.len, record, e.n_taps > 0, @intCast(slot_variant));
        if (record and @import("cuda_round_timing.zig").Trace.enabled())
            std.debug.print("[timing] verify_slots peak={d} variant={d}\n", .{ slot_peak, shape.slots });
        if (record) for (parts) |p| {
            try recurrent.mark(p.cache, false);
            if (p.rows == 1) try recurrent.mark(p.cache, true);
        };
        shape.hidden = keep == .hidden;
        shape.paged = @import("cuda_attention.zig").hasPages(segs);
        // a cache of at most GROUP_LEN positions holds fewer keys than that before any row it verifies
        shape.short = for (parts) |p| {
            if (p.cache.cap > @import("cuda_forward.zig").group_len) break false;
        } else true;
        if (e.learning != null or e.pockets.any()) {
            var exact = shape;
            exact.total = @intCast(total);
            exact.nseg = @intCast(parts.len);
            e.features = if (images) |im| im.features else 0;
            defer e.features = 0;
            try e.body(exact);
        } else if (prompt) {
            // uncaptured: a prompt chunk's GEMMs dwarf its launches, and each length's graph would pay the stall
            e.features = if (images) |im| im.features else 0;
            defer e.features = 0;
            e.h_window = e.hWindow(total);
            defer e.h_window = null;
            try e.body(shape);
        } else {
            const gop = try e.graphs.getOrPut(e.gpa, shape.key());
            if (!gop.found_existing) {
                errdefer _ = e.graphs.remove(shape.key());
                // captured, nothing runs: the replay below is this shape's first run
                e.cut_armed = true;
                defer {
                    e.cut_armed = false;
                    if (e.cut_head) |*h| h.deinit();
                    e.cut_head = null;
                }
                try cuda.graph.beginCapture(e.stream, .thread_local);
                e.body(shape) catch |err| {
                    var broken = cuda.graph.endCapture(e.stream) catch return err;
                    broken.deinit();
                    return err;
                };
                var captured = try cuda.graph.endCapture(e.stream);
                defer captured.deinit();
                gop.value_ptr.* = try captured.instantiate();
                if (e.cut_head) |h| {
                    e.heads.put(e.gpa, shape.key(), h) catch |err| {
                        gop.value_ptr.deinit();
                        return err;
                    };
                    e.cut_head = null;
                }
            }
            if (e.heads.get(shape.key())) |h| try h.launchOn(e.stream);
            try gop.value_ptr.launchOn(e.stream);
        }
        for (parts) |p| p.cache.advanced(p.rows, record);
        switch (keep) {
            .hidden => try e.stream.synchronize(), // mapped layer inputs are ready before the learner reads them
            .draws => |pk| {
                const toks = e.tables.ptr + tokens_at;
                var i: usize = 0; // runs of greedy rows argmaxed together, runs of sampled rows drawn together
                while (i < pk.len) {
                    const sampled = pk[i].rule != null;
                    var n: usize = 1;
                    while (i + n < pk.len and (pk[i + n].rule != null) == sampled) n += 1;
                    if (sampled) {
                        try e.draws.drawRows(e.stream, e.logits + i * g.vocab * 2, g.vocab, e.tables.ptr + rules_at + i * @sizeOf(cuda.sample.Rule), e.tables.ptr + meta_at + i * 4, toks + i * 4, n, e.picks);
                    } else try e.argmax(i, n, toks);
                    i += n;
                }
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(e.host_logits.bytes.ptr, toks, total * 4, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                @memcpy(e.tokens[0..total], e.host_logits.slice(u32)[0..total]);
            },
            .logits, .last => {
                const out = e.host_logits.bytes[0 .. heads * g.vocab * 2];
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, e.logits, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
                try e.stream.synchronize();
                // the chunk is done: its persisting lines turn normal, so decode has all of L2
                if (prompt and e.l2_persist > 0) _ = e.d.api.cuCtxResetPersistingL2Cache();
                if (keep == .last) @memcpy(parts[0].cache.logits, e.rowLogits(0));
            },
        }
        recurrent.completed(e);
    }

    pub const Shape = @import("cuda_shapes.zig").Shape;

    /// The same scan selected by body, exposed for host launch-contract fixtures.
    pub fn deltaScan(e: *Engine, record: bool, slot_count: u8, segments: usize, layer: c_int) !void {
        const scan = if (!record) e.fns.gdn_chain else switch (slot_count) {
            1 => e.fns.gdn_slots_1,
            2 => e.fns.gdn_slots_2,
            3 => e.fns.gdn_slots_3,
            4 => e.fns.gdn_slots_4,
            8 => e.fns.gdn_slots_8,
            16 => e.fns.gdn_slots_16,
            35 => e.fns.gdn_slots16_3,
            37 => e.fns.gdn_lean_2,
            38 => e.fns.gdn_lean_3,
            else => return error.InvalidGdnSlots,
        };
        const input = e.tables.ptr + @as(usize, if (record) Round.steps_at else rowpar_at);
        // qwen35_slots.cuh: up to four slots scan two value rows a warp, so a block covers eight
        const rows_a_warp: usize = if (record and (slot_count == 3 or slot_count == 35 or slot_count == 37 or slot_count == 38)) 4 else if (record and slot_count <= 4) 2 else 1;
        try e.runAfter(scan, .{ 1, geometry.dk / (4 * rows_a_warp), e.g.v_heads * segments }, .{ 32, 4, 1 }, .{ e.geo, e.q, e.k, e.v, e.g_, e.beta, e.tables.ptr + segs_at, input, layer, e.y, e.tables.ptr + params_at });
    }

    /// The launches of a forward of `shape`, from the embedding to the kept rows' logits (captured once a shape).
    pub const body = @import("cuda_forward.zig").body;

    pub const routedRows = routed.rows;

    /// Row `i`'s logits from the newest forward that kept logits.
    pub fn rowLogits(e: *const Engine, i: usize) []const u16 {
        return e.host_logits.slice(u16)[i * e.g.vocab ..][0..e.g.vocab];
    }
};

test "dense read cost: hidden and slot variants have distinct graph keys" {
    var seen: [32]u32 = undefined;
    var n: usize = 0;
    for ([_]u8{ 1, 2, 3, 4, 8, 16, 35, 37, 38 }) |slots| {
        for ([_]bool{ false, true }) |hidden| {
            const shape: Engine.Shape = .{ .total = 1, .nseg = 1, .record = true, .prompt = false, .taps = true, .slots = slots, .hidden = hidden };
            for (seen[0..n]) |key| try std.testing.expect(key != shape.key());
            seen[n] = shape.key();
            n += 1;
        }
    }
}

/// An engine A/B switch: true when `name` is set to "0".
fn envOff(name: [*:0]const u8) bool {
    const v = @import("sys").c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

/// Compile-time switch: decode glue kernels (norm, GDN pre/conv/scan/post, head norms, rope, attention, gate, mlp_act)
/// launched with PDL on GB10, each opening with griddepcontrol.wait; false restores plain stream-ordered launches.
pub const glue_pdl = true;
