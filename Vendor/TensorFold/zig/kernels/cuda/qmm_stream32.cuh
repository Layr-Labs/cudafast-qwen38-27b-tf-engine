// 32-row twins of the 16-row stream kernels, for verify windows of 17 to 32 rows: tf_qmm_gdn_stream16 (and its
// dictionary twin), tf_qmm_output_stream16 / tf_qmm_output_prefetch16 (and the dictionary twin), tf_qmm_mlp_single16 (and
// its dictionary twin). A CTA streams the same weight bytes once and runs both m16 row tiles off each dequantized B
// fragment. Every output element keeps its 16-row kernel's chain: the group order, a group's MMA K order (mma0 at kt 0,
// then kt 1 to 3), its two fmas with the same scale, bias and row sum, the slice-ordered sums, the bf16 roundings and the
// activation, so each row's values are the ones a 16-row launch on its m16 tile computes.
// The output and MLP twins keep one input buffer (both m16 tiles) a slice, refilled once the slice's warps have read it,
// so they fit their 16-row kernels' shared memory and CTAs an SM.
#pragma once
#include "qmm_dict.cuh"

namespace tf_qmm_stream32 {
using namespace tf_qmm_group;

// Every CTA reads all the window's input rows, which the weight stream would push out of L2 between CTAs' reads (a
// 32-row window's rows are twice a 16-row one's); after the grid wait the grid marks their lines evict_last, a line a
// thread, so each read finds them in L2.
__device__ __forceinline__ void pin_rows(const __nv_bfloat16* x, int rows, int K, int ldx) {
#if __CUDA_ARCH__ >= 900
    const int lpr = K / 64, total = rows * lpr;
    for (int l = int(blockIdx.x) + int(gridDim.x) * int(threadIdx.x); l < total; l += int(gridDim.x) * int(blockDim.x))
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(x + static_cast<size_t>(l / lpr) * ldx + (l % lpr) * 64) : "memory");
#endif
}

// ------------------------------------------------------------------------------------------------ GDN in_proj
// tf_qmm_gdn_stream::tile + wide_items (FutureGroups 0, as both 16-row kernels launch) with two m16 row tiles a warp.
template <int BN, int STAGES_ = 2>
struct G32 {
    static constexpr int GS = 64, THREADS = 128, NT = BN / 32, ROW = 128, CHUNKS = 8;
    static constexpr int X = 32 * ROW, W = BN * GS / 2, S = BN * 2, XS = 32 * 4;
    static constexpr int STAGE = X + W + 2 * S + XS, STAGES = STAGES_, SMEM = STAGES * STAGE;
};

// STAGES 2: the GDN stream's CTAs; 6: the lane form (one 64-column CTA a tile chaining its K slices, five groups in
// flight) that GDN out / attention o take at 16 rows (lane_gemv), here on the projection stream's words
template <int BN, bool DICT, int STAGES_ = 2>
__device__ __forceinline__ void gdn_tile32(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P,
                                           int t, int M, int K, int ldx) {
    using T = G32<BN, STAGES_>;
    constexpr int GS = 64, NT = T::NT, STAGES = T::STAGES;
    static_assert(!DICT || 2 * T::S == BN / 64 * 256, "coded stages");
    const int npad = DICT ? P.tiles : P.npad;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS, items = KG, per = KG / P.sk, n0 = t * BN;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(32, M);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = 32 * T::CHUNKS / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    const __nv_bfloat16* xsrc[XC];
    size_t woff[WC];
    int soff[SC], xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC], sbias[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = r < rows;
        xsrc[j] = x + static_cast<size_t>(min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, tt = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        woff[j] = static_cast<size_t>(tt) * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        sbias[j] = which;
        soff[j] = off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64) * TILE_BYTES;
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * (npad / 64) * TILE_BYTES);
        if constexpr (DICT) {
            const int step = qmm_dict::dict_step(P.npad);
            const size_t window = static_cast<size_t>(g) * step & ~static_cast<size_t>(63);
            for (int c = tid; c < BN / 4; c += T::THREADS)
                cp16(p + T::X + T::W + c * 16, reinterpret_cast<const unsigned char*>(P.scales) +
                     static_cast<size_t>(n0 / 64 + c / 16) * KG * step + window + (c % 16) * 16);
        } else {
#pragma unroll
            for (int j = 0; j < SC; ++j)
                if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
        }
    };
    unsigned char* dec = buf + T::SMEM;  // DICT: BN decoded scales, BN biases, then the part's table
    unsigned char* table = dec + 4 * BN;
    // a pair table (aa7c3ca7's coding) in shared memory up to the room two CTAs an SM leave (the 16-row twin's rule with
    // this kernel's room), else read in place through L2; a value table as dict_table loads it
    // (one room a launch: the GDN launch's 64-column a / b CTAs keep its 256-column CTAs' room; qlinear.zig
    // gdn32_pair_tab_max / lane32_pair_tab_max)
    constexpr int ROOM = STAGES_ == 2 ? 22272 : 10752;
    static_assert(STAGES_ == 2 ? 50176 - G32<256, 2>::SMEM - 4 * 256 >= ROOM : (BN == 64 && 50176 - T::SMEM - 4 * BN >= ROOM), "room");
    [[maybe_unused]] const bool pair_smem = DICT && qmm_dict::dict_pair(P.npad) && (qmm_dict::dict_np(P.npad) * 4 + 15) / 16 * 16 <= ROOM;
    if constexpr (DICT) {
        if (qmm_dict::dict_pair(P.npad)) {
            if (pair_smem)
                for (int c = tid; c < (qmm_dict::dict_np(P.npad) * 4 + 15) / 16; c += T::THREADS)
                    cp16(table + c * 16, reinterpret_cast<const unsigned char*>(P.biases) + c * 16);
        } else {
            qmm_dict::dict_table<T::THREADS>(P, table, tid);
        }
    }
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));
    for (int c = tid; c < 32 * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_w(s, s);
        commit();
    }
    grid_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_x(s, s);
        commit();
    }
    pin_rows(x, rows, K, ldx);
    grid_launch();
    float acc[2][NT][4], tot[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] = 0.0f;
    for (int it = 0; it < items; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < items) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        [[maybe_unused]] uint32_t pf[(BN / 4 + 31) / 32] = {};
        if constexpr (DICT) {
            const int code = P.npad, bits = qmm_dict::dict_bits(code), step = qmm_dict::dict_step(code), g = it;
            const uint32_t mask_bits = (1u << bits) - 1u;
            if (qmm_dict::dict_pair(code)) {
                // a pair-coded part: the lane's columns of its warp's BN / 4, one pair index each, the pair table
                // in place; the reads start here and are written after the first tile's MMAs
                const uint32_t* gtab = pair_smem ? nullptr : reinterpret_cast<const uint32_t*>(P.biases);
#pragma unroll
                for (int v = 0; v < (BN / 4 + 31) / 32; ++v) {
                    const int q = lane + 32 * v, col = wn * (BN / 4) + q;
                    if (q < BN / 4) {
                        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + (col / 64) * 256);
                        const int off = (g * step & 63) * 8 + (col % 64) * bits;
                        pf[v] = qmm_dict::dict_pair_entry(table, gtab, __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mask_bits);
                    }
                }
            } else {
#pragma unroll
            for (int v = 0; v < BN / 64; ++v) {
                const int q = lane + 32 * v, half = q / (BN / 4), col = wn * (BN / 4) + q % (BN / 4);
                const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + (col / 64) * 256);
                const int off = (g * step & 63) * 8 + (half * 64 + col % 64) * bits;
                const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mask_bits;
                reinterpret_cast<unsigned short*>(dec)[half * BN + col] =
                    reinterpret_cast<const unsigned short*>(table)[(half ? qmm_dict::dict_ns(code) : 0) + idx];
            }
            }
            __syncwarp();
        }
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(DICT ? dec : p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[NT][GS / 32];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * NT + j) * 32 + lane) * (GS / 32) + v];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float d[NT][4];
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[4];
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
                ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                    const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
                    if (kt == 0) mma0(d[j], a, b0, b1);
                    else mma(d[j], a, b0, b1);
                }
            }
            if constexpr (DICT) {
                if (i == 0 && qmm_dict::dict_pair(P.npad)) {  // the pairs read before the MMAs
#pragma unroll
                    for (int v = 0; v < (BN / 4 + 31) / 32; ++v) {
                        const int q = lane + 32 * v, col = wn * (BN / 4) + q;
                        if (q < BN / 4) {
                            reinterpret_cast<unsigned short*>(dec)[col] = static_cast<unsigned short>(pf[v]);
                            reinterpret_cast<unsigned short*>(dec)[BN + col] = static_cast<unsigned short>(pf[v] >> 16);
                        }
                    }
                    __syncwarp();
                }
            }
            const int row = i * 16 + (lane >> 2);
            const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int col = wn * (BN / 4) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                const float sv[2] = {__low2float(s2), __high2float(s2)};
                const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[i][j][e]));
            }
        }
        if ((it + 1) % per == 0) {  // a K slice's end: its sum joins the slices before it, in slice order
            const bool first = it + 1 == per;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[i][j][e] = first ? acc[i][j][e] : tot[i][j][e] + acc[i][j][e];
                        acc[i][j][e] = 0.0f;
                    }
        }
    }
    const int N = P.n;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = n0 + wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = i * 16 + (lane >> 2) + h * 8;
                if (r >= M) continue;
                const float2 v = make_float2(tot[i][j][2 * h], tot[i][j][2 * h + 1]);
                auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(r) * N + col;
                if (col + 1 < N && (N & 1) == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v.x, v.y);
                } else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v.x);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v.y);
                }
            }
}

// tf_qmm_gdn_stream::body / body_dict: parts 0 and 1 in 256-column CTAs, 2 and 3 in 64-column ones; on a dictionary
// launch a coded part carries its real npad in `tiles` (uncoded parts: tiles 0, npad as stored).
template <bool DICT_LAUNCH>
__device__ __forceinline__ void gdn_body32(const __nv_bfloat16* x, const float* xs, const Parts& parts, int M, int K, int ldx) {
    int which = 0;
#pragma unroll
    for (int i = 1; i < 4; ++i)
        if (int(blockIdx.x) >= parts.p[i].first) which = i;
    const auto p = parts.p[which];
    const int t = int(blockIdx.x) - p.first;
    const bool coded = DICT_LAUNCH && p.tiles > 0;
    if (which < 2) {
        if (coded) gdn_tile32<256, true>(x, xs, p, t, M, K, ldx);
        else gdn_tile32<256, false>(x, xs, p, t, M, K, ldx);
    } else {
        if (coded) gdn_tile32<64, true>(x, xs, p, t, M, K, ldx);
        else gdn_tile32<64, false>(x, xs, p, t, M, K, ldx);
    }
}

// ------------------------------------------------------------------------------------------------ out / o / down
// tf_qmm_output_stream::body (Rolling 0) on 32 rows: the same four K slices a CTA and slice-ordered sum, a slice on two
// warps (each the 16-row kernel's two warps' 32 columns, both m16 row tiles). A slice's stage holds a group's words,
// its scales and biases (or index window) and its 32 row sums; its inputs sit in one buffer after its two stages: each
// warp takes the group's A fragments of both tiles, then the next group's inputs land while the MMAs run.
constexpr int OW = 2048, OS = 256, OXS = 128;
constexpr int OStage = OW + OS + OXS;
constexpr int OX = 32 * 128;
constexpr int OSlice = 2 * OStage + OX;
constexpr int OBytes = 4 * OSlice;  // 35840; the slices' partials (32 KiB) reuse it
constexpr int OTPS = 64;            // threads a slice

template <int FutureGroups, bool DICT>
__device__ __forceinline__ void out_body32(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                           const Parts& parts, int M, int K, int ldx) {
    constexpr int Slices = 4;
    static_assert(!DICT || FutureGroups == 0, "a coded part's L2 prefetch would target its index windows");
    extern __shared__ __align__(128) unsigned char buf[];
    const auto p = parts.p[0];
    const int slice = threadIdx.x / OTPS, tid = threadIdx.x % OTPS, lane = tid & 31, wn = tid >> 5;
    const int KG = K / 64, per = KG / Slices, g0 = slice * per, tile = blockIdx.x, n0 = tile * 64, rows = min(M, 32);
    const int tiles = DICT ? int(gridDim.x) : p.npad / 64;
    unsigned char* const local = buf + slice * OSlice;
    unsigned char* const xb = local + 2 * OStage;
    auto stage = [&](int it) { return local + (it & 1) * OStage; };
    auto weight = [&](int g) { return reinterpret_cast<const unsigned char*>(p.w) + (size_t(g) * tiles + tile) * 2048; };
    // dictionary: the slice's first 16 threads load the group's 256-byte index window (qmm_dict::dict_lanes' chunks);
    // lane l of warp wn decodes column wn * 32 + v * 16 + l % 16's scale (l < 16) or bias, v = 0, 1
    const int code = p.npad, bits = DICT ? qmm_dict::dict_bits(code) : 0, step = DICT ? qmm_dict::dict_step(code) : 0;
    const bool paired = DICT && qmm_dict::dict_pair(code);  // a pair code: 64 pair indices a group
    const unsigned char* const dsrc = reinterpret_cast<const unsigned char*>(p.scales) + static_cast<size_t>(n0 / 64) * KG * step + tid * 16;
    unsigned char* dec = buf + OBytes + slice * qmm_dict::DICT_DEC;
    unsigned char* table = buf + OBytes + Slices * qmm_dict::DICT_DEC;
    auto load_w = [&](int it) {
        auto* dst = stage(it);
        const int g = g0 + it;
#pragma unroll
        for (int h = 0; h < 2; ++h) cp16(dst + (tid + h * OTPS) * 16, weight(g) + (tid + h * OTPS) * 16);
        if constexpr (DICT) {
            if (tid < 16) cp16(dst + OW + tid * 16, dsrc + (static_cast<size_t>(g) * step & ~static_cast<size_t>(63)));
        } else if (tid < 16) {
            const auto* t = tid < 8 ? p.scales : p.biases;
            cp16(dst + OW + tid * 16, t + size_t(g) * p.npad + n0 + (tid % 8) * 8);
        }
    };
    auto load_sums = [&](int it) {
        auto* dst = stage(it);
        const int g = g0 + it;
        if (tid < 32) {
            if (tid < rows) cp4(dst + OW + OS + tid * 4, xs + size_t(tid) * KG + g);
            else *reinterpret_cast<float*>(dst + OW + OS + tid * 4) = 0;
        }
    };
    auto load_x = [&](int it) {
        const int g = g0 + it;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int c = tid + h * OTPS, row = c / 8, ch = c % 8;
            cp16z(xb + row * 128 + swz<8>(row, ch) * 16, x + size_t(min(row, rows - 1)) * ldx + g * 64 + ch * 8, row < rows);
        }
    };
    if constexpr (DICT) qmm_dict::dict_table<Slices * OTPS>(p, table, threadIdx.x);
    load_w(0);
    commit();
#if __CUDA_ARCH__ >= 900
    if (tid == 0) {
#pragma unroll
        for (int g = 1; g <= FutureGroups; ++g)
            if (g < per) {
                const auto* w = weight(g0 + g);
                const auto* s = p.scales + size_t(g0 + g) * p.npad + n0;
                const auto* b = p.biases + size_t(g0 + g) * p.npad + n0;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 2048;" ::"l"(w) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" ::"l"(s) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" ::"l"(b) : "memory");
            }
    }
#endif
    grid_wait();
    load_x(0);
    load_sums(0);
    commit();
    pin_rows(x, rows, K, ldx);
    grid_launch();
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    for (int it = 0; it < per; ++it) {
        wait<0>();
        // a slice's stages are its own: after the first group (whose wait also covers the table) its warps sync alone
        if (it == 0) __syncthreads();
        else asm volatile("bar.sync %0, %1;\n" ::"r"(1 + slice), "n"(OTPS) : "memory");
        if (it + 1 < per) {
            load_w(it + 1);
            load_sums(it + 1);
        }
        commit();
        uint32_t a[2][4][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i][kt], xb + r * 128 + swz<8>(r, ch) * 16);
            }
        asm volatile("bar.sync %0, %1;\n" ::"r"(1 + slice), "n"(OTPS) : "memory");  // the slice's inputs are read
        if (it + 1 < per) load_x(it + 1);
        commit();
        const auto* at = stage(it);
        const auto* wp = reinterpret_cast<const uint32_t*>(at);
        [[maybe_unused]] uint32_t pf = 0;
        if constexpr (DICT) {
            const uint32_t* pk = reinterpret_cast<const uint32_t*>(at + OW);
            const uint32_t dmask = (1u << bits) - 1u;
            if (paired) {  // the lane's column wn * 32 + lane: its pair, read now, written after the MMAs
                const int off = ((g0 + it) * step & 63) * 8 + (wn * 32 + lane) * bits;
                const uint32_t* gtab = qmm_dict::dict_tab_bytes(code) == 0 ? reinterpret_cast<const uint32_t*>(p.biases) : nullptr;
                pf = qmm_dict::dict_pair_entry(table, gtab, __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & dmask);
            } else {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int pos = (lane >> 4) * 64 + wn * 32 + v * 16 + (lane & 15);
                const int off = ((g0 + it) * step & 63) * 8 + pos * bits;
                const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & dmask;
                reinterpret_cast<unsigned short*>(dec)[pos] =
                    reinterpret_cast<const unsigned short*>(table)[((lane >> 4) ? qmm_dict::dict_ns(code) : 0) + idx];
            }
            __syncwarp();
            }
        }
        const auto* sp = reinterpret_cast<const __nv_bfloat16*>(DICT ? dec : at + OW);
        const auto* sums = reinterpret_cast<const float*>(at + OW + OS);
        uint32_t words[4][2];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int v = 0; v < 2; ++v) words[j][v] = wp[((wn * 4 + j) * 32 + lane) * 2 + v];
        float d[2][4][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask), b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if (kt == 0) mma0(d[i][j], a[i][kt], b0, b1);
                    else mma(d[i][j], a[i][kt], b0, b1);
                }
            }
        if constexpr (DICT) {
            if (paired) {
                reinterpret_cast<unsigned short*>(dec)[wn * 32 + lane] = static_cast<unsigned short>(pf);
                reinterpret_cast<unsigned short*>(dec)[64 + wn * 32 + lane] = static_cast<unsigned short>(pf >> 16);
                __syncwarp();
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = (wn * 4 + j) * 8 + (lane & 3) * 2;
            const auto s2 = *reinterpret_cast<const __nv_bfloat162*>(sp + col), b2 = *reinterpret_cast<const __nv_bfloat162*>(sp + 64 + col);
            const float sv[2] = {__low2float(s2), __high2float(s2)}, bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = i * 16 + (lane >> 2);
                const float xv[2] = {sums[r], sums[r + 8]};
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
            }
        }
    }
    wait<0>();
    __syncthreads();
    auto* partial = reinterpret_cast<float*>(buf);
    if (slice != 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) partial[(((slice * 2 + i) * 4 + j) * 4 + e) * OTPS + tid] = acc[i][j][e];
    }
    __syncthreads();
    if (slice == 0) {
        for (int peer = 1; peer < Slices; ++peer)
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = acc[i][j][e] + partial[(((peer * 2 + i) * 4 + j) * 4 + e) * OTPS + tid];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = i * 16 + (lane >> 2) + h * 8, c = n0 + (wn * 4 + j) * 8 + (lane & 3) * 2;
                    if (r < M && c + 1 < p.n)
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.out) + size_t(r) * p.n + c) =
                            __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
    }
}

// ------------------------------------------------------------------------------------------------ MLP gate, up, act
// tf_qmm_mlp_single::mlp_pair_body<2, 512, true> (and mlp_single_dict_body) on 32 rows, on the K-major 2 x 256-column
// panel: eight warps, warp w the 16-row kernel's warp w % 4 on m16 row tile w / 4 (its 128 columns, gate's for w % 4 < 2,
// up's after). A stage holds a group's gate and up words, their scales and biases (or index runs) and the 32 row sums;
// both tiles' inputs sit in one buffer after the two stages, refilled once the group's last pass has read them. A warp
// takes its 16 n8 tiles in four passes of four (a pass's MMAs, then its fmas), the order of each element's chain kept.
constexpr int MW = 512 * 32, MS = 512 * 2, MXS = 32 * 4;
constexpr int MStage = MW + 2 * MS + MXS;
constexpr int MX = 32 * 128;
constexpr int MBytes = 2 * MStage + MX;  // 41216; the activation panel (32 KiB) reuses it
constexpr int MDec = 8 * 128;            // eight warps' 32 scales and 32 biases of a pass (bf16)

// PAIR: 0 both parts value-coded, 1 both pair-coded, 2 each part as its code says (registers: the first two keep one
// decode path; a mixed launch is rare)
template <bool DICT, int PAIR>
__device__ __forceinline__ void mlp_body32(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                           const Parts& parts, __nv_bfloat16* __restrict__ act, float* __restrict__ as,
                                           int M, int K, int ldx) {
    constexpr int GS = 64, BN = 512, WN = 4, THREADS = 256, TILE_BYTES = 64 * GS / 2;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wi = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const Part P = parts.p[0];  // gate (rank 0) and up (rank 1): same n, npad, tiles
    const int n0 = blockIdx.x * (BN / 2), per = KG, rows = min(32, M);
    auto stage = [&](int s) { return buf + s * MStage; };
    unsigned char* const xb = buf + 2 * MStage;
    // a group's panel: chunk tid + 256 j is 16 bytes of 64-column tile tid / 128 + 2 j (gate's four, then up's four)
    static_assert(TILE_BYTES / 16 == 128 && MW / 16 == 4 * THREADS, "four chunks a thread");
    const int half = tid / 128;
    const unsigned char* const wgate = reinterpret_cast<const unsigned char*>(parts.p[0].w) + static_cast<size_t>(n0 / 64) * TILE_BYTES + (tid % 128) * 16;
    const unsigned char* const wup = reinterpret_cast<const unsigned char*>(parts.p[1].w) + static_cast<size_t>(n0 / 64) * TILE_BYTES + (tid % 128) * 16;
    // plain: scales then biases of the 512 columns (gate's 256, up's 256), one 16-byte chunk a thread of the first 128
    const int sw = (tid % 128) / (MS / 16), so = tid % (MS / 16);
    const auto SQ = parts.p[(so * 8) / (BN / 2)];
    const __nv_bfloat16* ssrc = (sw ? SQ.biases : SQ.scales) + n0 + (so * 8) % (BN / 2);
    const int sdst = MW + sw * MS + so * 16;
    // dictionary: each part's `first` holds its code, its scales the group-major indices [kg][npad / 64][16 * bits],
    // its biases its table (ns scale values, then nb bias values)
    const int code0 = parts.p[0].first, code1 = parts.p[1].first;
    const int bits0 = DICT ? qmm_dict::dict_bits(code0) : 0, bits1 = DICT ? qmm_dict::dict_bits(code1) : 0;
    // a pair-coded part: 64 pair indices a tile's group, its table read in place (global, prefetched into L1) as tf_qmm_mlp_single16 does
    const bool pair0 = DICT && (PAIR == 1 || (PAIR == 2 && qmm_dict::dict_pair(code0)));
    const bool pair1 = DICT && (PAIR == 1 || (PAIR == 2 && qmm_dict::dict_pair(code1)));
    const int tab0 = DICT && !pair0 ? (qmm_dict::dict_ns(code0) + qmm_dict::dict_nb(code0) + 7) / 8 * 16 : 0;
    unsigned char* const dec = buf + MBytes;
    const unsigned short* const tabs = reinterpret_cast<const unsigned short*>(buf + MBytes + MDec);
    const int run0 = (pair0 ? 2 : 4) * bits0, run1 = (pair1 ? 2 : 4) * bits1;
    const bool dok = tid < run0 + run1;
    const int dpart = tid < run0 ? 0 : 1, dchunk = tid < run0 ? tid : tid - run0;
    const int dstep = DICT ? qmm_dict::dict_step(dpart ? code1 : code0) : 0;
    const unsigned char* dsrc = reinterpret_cast<const unsigned char*>(parts.p[dpart].scales) +
                                static_cast<size_t>(n0 / 64) * dstep + dchunk * 16;
    const int dpitch = P.npad / 64 * dstep;
    const int ddst = MW + (dpart ? 16 * run0 : 0) + dchunk * 16;
    const int mpart = wn / 2, mbits = mpart ? bits1 : bits0, mns = DICT ? qmm_dict::dict_ns(mpart ? code1 : code0) : 0;
    const uint32_t mmask = (1u << mbits) - 1u;
    const int mrun = mpart ? 16 * run0 : 0;
    const bool mpair = mpart ? pair1 : pair0;
    const uint32_t* const mpairs = reinterpret_cast<const uint32_t*>(parts.p[mpart].biases);
    const unsigned short* const mtab = tabs + (mpart ? tab0 / 2 : 0);
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(min(tid, rows - 1)) * KG;
    const int xr = tid / 8, xch = tid % 8;  // one 16-byte input chunk a thread
    auto load_x = [&](int g) {
        if (xr < rows) cp16(xb + xr * 128 + swz<8>(xr, xch) * 16, x + static_cast<size_t>(xr) * ldx + g * GS + xch * 8);
    };
    auto load_sums = [&](int s, int g) {
        if (xsok) cp4(stage(s) + MW + 2 * MS + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
        const size_t gofs = static_cast<size_t>(g) * (P.npad / 64) * TILE_BYTES;
#pragma unroll
        for (int j = 0; j < 4; ++j) cp16(p + (tid + j * THREADS) * 16, (j < 2 ? wgate : wup) + ((half + 2 * j) % 4) * TILE_BYTES + gofs);
        if constexpr (DICT) {
            if (dok) cp16(p + ddst, dsrc + static_cast<size_t>(g) * dpitch);
        } else {
            if (tid < 128) cp16(p + sdst, ssrc + static_cast<size_t>(g) * P.npad);
        }
    };
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    if (xr >= rows) *reinterpret_cast<uint4*>(xb + xr * 128 + xch * 16) = uint4{};
    if constexpr (DICT) {  // a pair-coded part's table, static data: its lines into L1 before the wait
#pragma unroll
        for (int part = 0; part < 2; ++part)
            if (part ? pair1 : pair0) {
                const uint32_t* pt = reinterpret_cast<const uint32_t*>(parts.p[part].biases);
                for (int i = tid * 32; i < qmm_dict::dict_ns(part ? code1 : code0); i += THREADS * 32)
                    asm volatile("prefetch.global.L1 [%0];" ::"l"(pt + i));
            }
    }
    if constexpr (DICT) {  // both tables, static data: loaded with the first stage, before the wait
        const int c0 = tab0 / 16, c1 = pair1 ? 0 : (qmm_dict::dict_ns(code1) + qmm_dict::dict_nb(code1) + 7) / 8;
        for (int c = tid; c < c0 + c1; c += THREADS)
            cp16(buf + MBytes + MDec + c * 16, c < c0 ? reinterpret_cast<const unsigned char*>(parts.p[0].biases) + c * 16
                                                      : reinterpret_cast<const unsigned char*>(parts.p[1].biases) + (c - c0) * 16);
    }
    if (per > 0) load_w(0, 0);
    commit();
    grid_wait();
    if (per > 0) {
        load_x(0);
        load_sums(0, 0);
    }
    commit();
    pin_rows(x, rows, K, ldx);
    grid_launch();
    // this warp's 32 + 32 values of a pass; PAIR 1: of all four passes (written once a group, after pass 0's MMAs)
    __nv_bfloat16* const dw0 = reinterpret_cast<__nv_bfloat16*>(dec) + warp * (PAIR == 1 ? 256 : 64);
    for (int it = 0; it < per; ++it) {
        wait<0>();
        __syncthreads();
        if (it + 1 < per) {
            load_w((it + 1) & 1, it + 1);
            load_sums((it + 1) & 1, it + 1);
        }
        commit();
        const unsigned char* p = stage(it & 1);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p);
        const float* px = reinterpret_cast<const float*>(p + MW + 2 * MS);
        const int row = wi * 16 + (lane >> 2);
        [[maybe_unused]] uint32_t pf[4] = {0u, 0u, 0u, 0u};
        if constexpr (DICT && PAIR == 1) {  // the lane's column's pair for each pass, read before pass 0's MMAs
            const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + MW + mrun);
#pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int off = ((wn % 2) * 128 + pass * 32 + lane) * mbits;
                pf[pass] = __ldg(mpairs + (__funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mmask));
            }
        }
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
            __nv_bfloat16* const dw = dw0 + (PAIR == 1 ? pass * 64 : 0);
            if constexpr (DICT && PAIR == 1) {
            } else if constexpr (DICT) {
                const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + MW + mrun);
                const int c = (wn % 2) * 128 + pass * 32 + lane, tt = c / 64, ct = c % 64;
                if (mpair) {  // the lane's column's pair (one index a column, the table in place)
                    const int off = c * mbits;
                    const uint32_t f = __ldg(mpairs + (__funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mmask));
                    reinterpret_cast<unsigned short*>(dw)[lane] = static_cast<unsigned short>(f);
                    reinterpret_cast<unsigned short*>(dw)[32 + lane] = static_cast<unsigned short>(f >> 16);
                } else {
#pragma unroll
                    for (int which = 0; which < 2; ++which) {
                        const int off = (tt * 128 + which * 64 + ct) * mbits;
                        const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mmask;
                        reinterpret_cast<unsigned short*>(dw)[which * 32 + lane] = mtab[(which ? mns : 0) + idx];
                    }
                }
                __syncwarp();
            } else {  // the pass's 32 columns' scales and biases from the stage, a lane a column
                const unsigned short* ss = reinterpret_cast<const unsigned short*>(p + MW) + wn * (BN / WN) + pass * 32 + lane;
                reinterpret_cast<unsigned short*>(dw)[lane] = ss[0];
                reinterpret_cast<unsigned short*>(dw)[32 + lane] = ss[BN];
                __syncwarp();
            }
            uint32_t words[4][2];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
#pragma unroll
                for (int v = 0; v < 2; ++v) words[jj][v] = pw[((wn * 16 + pass * 4 + jj) * 32 + lane) * 2 + v];
            float d[4][4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                const int r = wi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
                ldmatrix4(a, xb + r * 128 + swz<8>(r, ch) * 16);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t b0 = pairm(words[jj][kt / 2], (kt & 1) * 8, mask);
                    const uint32_t b1 = pairm(words[jj][kt / 2], (kt & 1) * 8 + 4, mask);
                    if (kt == 0) mma0(d[jj], a, b0, b1);
                    else mma(d[jj], a, b0, b1);
                }
            }
            if (pass == 3) {  // every warp has read the group's inputs: the next group's may land in them
                __syncthreads();
                if (it + 1 < per) load_x(it + 1);
                commit();
            }
            if constexpr (DICT && PAIR == 1) {
                if (pass == 0) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        reinterpret_cast<unsigned short*>(dw0)[q * 64 + lane] = static_cast<unsigned short>(pf[q]);
                        reinterpret_cast<unsigned short*>(dw0)[q * 64 + 32 + lane] = static_cast<unsigned short>(pf[q] >> 16);
                    }
                    __syncwarp();
                }
            }
            const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int j = pass * 4 + jj;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(dw + jj * 8 + (lane & 3) * 2);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(dw + 32 + jj * 8 + (lane & 3) * 2);
                const float sv[2] = {__low2float(s2), __high2float(s2)};
                const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[jj][e], sv[e & 1], acc[j][e]));
            }
            if constexpr (PAIR != 1) __syncwarp();  // the next pass rewrites the warp's values
        }
    }
    wait<0>();
    // both projections use the same activation panel, gate then up columns
    __nv_bfloat16* own = reinterpret_cast<__nv_bfloat16*>(buf);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = wi * 16 + (lane >> 2) + h * 8;
            const int c = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            *reinterpret_cast<__nv_bfloat162*>(own + r * BN + c) = __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
        }
    __syncthreads();
    if (wn >= WN / 2) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wi * 16 + (lane >> 2) + h * 8;
                const int c = wn * (BN / WN) + j * 8 + (lane & 3) * 2 - BN / 2;
                const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(own + r * BN + c);
                const __nv_bfloat162 u2 = *reinterpret_cast<const __nv_bfloat162*>(own + r * BN + c + BN / 2);
                const float gf[2] = {__low2float(g2), __high2float(g2)};
                const float uf[2] = {__low2float(u2), __high2float(u2)};
                __nv_bfloat16 o[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) o[e] = __float2bfloat16_rn(gf[e] / (1.0f + expf(-gf[e])) * uf[e]);
                const __nv_bfloat162 o2 = __halves2bfloat162(o[0], o[1]);
                *reinterpret_cast<__nv_bfloat162*>(own + r * BN + c) = o2;
                if (r < M) *reinterpret_cast<__nv_bfloat162*>(act + static_cast<size_t>(r) * P.n + n0 + c) = o2;
            }
    }
    __syncthreads();
    if (tid < rows * (BN / 128)) {
        const int r = tid / (BN / 128), group = tid % (BN / 128);
        as[static_cast<size_t>(r) * (P.n / 64) + n0 / 64 + group] = act_group_sum(own + r * BN + group * 64);
    }
}

}  // namespace tf_qmm_stream32

extern "C" __global__ void __launch_bounds__(128) tf_qmm_gdn_stream32(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::gdn_body32<false>(x, xs, parts, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_gdn_stream32_dict(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::gdn_body32<true>(x, xs, parts, M, K, ldx);
}
// GDN out / attention o (and down, TF_W32_DOWN_LANE=1) on 17 to 32 rows in the lane form: one CTA a 64-column tile of
// the projection stream's words, its K slices chained and summed in slice order (lane_gemv's bits), six stages
extern "C" __global__ void __launch_bounds__(128) tf_qmm_lane32(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::gdn_tile32<64, false, 6>(x, xs, parts.p[0], blockIdx.x, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_lane32_dict(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::gdn_tile32<64, true, 6>(x, xs, parts.p[0], blockIdx.x, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_output_stream32(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::out_body32<0, false>(x, xs, parts, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_output_stream32_dict(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::out_body32<0, true>(x, xs, parts, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_output_prefetch32(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx) {
    tf_qmm_stream32::out_body32<4, false>(x, xs, parts, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_mlp_single32(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_stream32::mlp_body32<false, 0>(x, xs, parts, act, sums, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_mlp_single32_dict(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_stream32::mlp_body32<true, 0>(x, xs, parts, act, sums, M, K, ldx);
}
// pair-coded gate and up (both), and a launch whose parts differ
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_mlp_single32_pair(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_stream32::mlp_body32<true, 1>(x, xs, parts, act, sums, M, K, ldx);
}
extern "C" __global__ void __launch_bounds__(256, 2) tf_qmm_mlp_single32_mixed(const __nv_bfloat16* x, const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts, __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_stream32::mlp_body32<true, 2>(x, xs, parts, act, sums, M, K, ldx);
}

// group_kernel<64, 32, 64, 1, 4, 5> (verify32: GB10's 17 to 32-row cluster tile)'s dictionary twin, the same tiles, K
// slices, MMA chains, fmas and slice-ordered sums as the 16-row group_kernel_dict decode() runs on each m16 row tile.
template __global__ void tf_qmm_group::group_kernel_dict<64, 32, 64, 1, 4, 5, false>(
    const __nv_bfloat16* __restrict__, const float* __restrict__, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);
