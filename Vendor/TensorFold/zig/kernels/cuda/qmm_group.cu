// Device code of src/tensorfold/cuda/kernels/qmm_group.cu (lines 1-348, comments and ATen includes dropped), checked by zig/tests/cuda/copies.py.

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

#include "qmm_frag.cuh"

namespace tf_qmm_group {

using namespace qmm_frag;

constexpr int PARTS = 4;

struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, sk, tiles, first;
};

struct Parts {
    Part p[PARTS];
    int count;
};

__device__ __forceinline__ uint32_t pairm(uint32_t w, int s, uint32_t mask) {
    const uint32_t t = ((w >> s) & mask) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
#endif
    return r;
}

template <int N>
using ic = std::integral_constant<int, N>;

template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool SWAP = false, bool SKIP = false,
          bool SPREAD = false>
__global__ void __launch_bounds__(WM * WN * 32) group_kernel(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const __grid_constant__ Parts parts,
        int M, int K, int ldx, int rows_t, int C) {
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    static_assert(!SWAP || (BM == 8 && WM == 1 && T::NT % 2 == 0), "swapped tiles: 8 rows, column pairs a warp");
    constexpr int I = SWAP ? T::NT / 2 : T::MT;     // MMA m16 tiles a warp: column pairs (swapped) or row tiles
    constexpr int J = SWAP ? 1 : T::NT;             // MMA n8 tiles a warp: the 8 rows (swapped) or column tiles
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x / C, rank = blockIdx.x % C;
    int q = 0;
#pragma unroll
    for (int i = 1; i < PARTS; ++i)
        if (i < parts.count && cid >= parts.p[i].first) q = i;
    const Part P = parts.p[q];                   // in registers: the loads below index it every group
    const int sk = P.sk, lc = cid - P.first;
    const int tile = lc / rows_t * (C / sk) + rank / sk, slice = rank % sk;
    const int m0 = lc % rows_t * BM, n0 = tile * BN, per = KG / sk, g0 = slice * per;
    const bool live = tile < P.tiles;            // a cluster's spare blocks past the last tile only join its syncs

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);           // rows past M stay zero in every stage (set once below)
    constexpr int TILE_BYTES = 64 * GS / 2;     // one stored 64-column tile's group block
    constexpr int XC = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    static_assert(BM <= T::THREADS, "one input sum a thread");
    const __nv_bfloat16* xsrc[XC];
    const unsigned char* wsrc[WC];
    const __nv_bfloat16* ssrc[SC];
    int xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < BM * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(m0 + min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES +
                  off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        ssrc[j] = (which ? P.biases : P.scales) + n0 + off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], ssrc[j] + static_cast<size_t>(g) * P.npad);
    };
    uint32_t mask;                              // 0x000F000F, opaque to the compiler (kept in a register)
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    float acc[I][J][4];
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    if (live) {
#pragma unroll
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS)
            if (c / T::CHUNKS >= rows)
#pragma unroll
                for (int s = 0; s < STAGES; ++s)
                    *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load_w(s, g0 + s);
            commit();
        }
        grid_wait();
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load_x(s, g0 + s);
            commit();
        }
        grid_launch();
        auto steps = [&](auto live_c) {                   // LIVE: the warp's m16 tiles holding rows below M
            [[maybe_unused]] constexpr int LIVE = decltype(live_c)::value;
            for (int it = 0; it < per; ++it) {
                wait<STAGES - 2>();
                __syncthreads();
                const int next = it + STAGES - 1;
                if (next < per) {
                    load_x(next % STAGES, g0 + next);
                    load_w(next % STAGES, g0 + next);
                }
                commit();
                const unsigned char* p = stage(it % STAGES);
                const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
                const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
                uint32_t words[T::NT][GS / 32];
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int v = 0; v < GS / 32; ++v)
                        words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
                float d[I][J][4];
                if constexpr (SWAP) {
#pragma unroll
                    for (int kt = 0; kt < GS / 16; ++kt) {
                        uint32_t bx[2];                           // rows 0-7 at k lo, k hi: the B fragment
                        const int ch = kt * 2 + ((lane >> 3) & 1);
                        ldmatrix2(bx, p + (lane & 7) * T::ROW + swz<T::CHUNKS>(lane & 7, ch) * 16);
#pragma unroll
                        for (int i = 0; i < I; ++i) {             // A rows g, g + 8 are columns of n8 tiles 2i, 2i + 1
                            const int s0 = (kt & 1) * 8;
                            const uint32_t a[4] = {pairm(words[2 * i][kt / 2], s0, mask),
                                                   pairm(words[2 * i + 1][kt / 2], s0, mask),
                                                   pairm(words[2 * i][kt / 2], s0 + 4, mask),
                                                   pairm(words[2 * i + 1][kt / 2], s0 + 4, mask)};
                            if (kt == 0) mma0(d[i][0], a, bx[0], bx[1]);
                            else mma(d[i][0], a, bx[0], bx[1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < I; ++i)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {             // d: column g (+ 8 for e >= 2), rows 2t, 2t + 1
                            const int col = wn * (BN / WN) + i * 16 + (lane >> 2) + (e >> 1) * 8;
                            const float sv = __bfloat162float(ps[col]), bv = __bfloat162float(ps[BN + col]);
                            const float xv = px[(lane & 3) * 2 + (e & 1)];
                            acc[i][0][e] = __fmaf_rn(xv, bv, __fmaf_rn(d[i][0][e], sv, acc[i][0][e]));
                        }
                } else {
#pragma unroll
                    for (int kt = 0; kt < GS / 16; ++kt) {
                        uint32_t a[LIVE > 0 ? LIVE : 1][4];
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                            const int ch = kt * 2 + (lane >> 4);
                            ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                        }
#pragma unroll
                        for (int j = 0; j < T::NT; ++j) {
                            const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                            const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                            for (int i = 0; i < LIVE; ++i) {
                                if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                                else mma(d[i][j], a[i], b0, b1);
                            }
                        }
                    }
#pragma unroll
                    for (int j = 0; j < T::NT; ++j) {
                        const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                        const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                        const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                        const float sv[2] = {__low2float(s2), __high2float(s2)};
                        const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                            const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
                            for (int e = 0; e < 4; ++e)           // acc = fma(xs, b, fma(p, s, acc)): qmm.cu's order
                                acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1],
                                                         __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
                        }
                    }
                }
            }
        };
        if constexpr (!SKIP || SWAP || T::MT == 1) {
            steps(ic<T::MT>());
        } else {
            const int live = min(T::MT, max(0, (M - m0 - wm * (BM / WM) + 15) / 16));
            if (live == 0) steps(ic<0>());
            else if (live == 1) steps(ic<1>());
            else if (T::MT > 2 && live == 2) steps(ic<(T::MT > 2 ? 2 : T::MT)>());
            else if (T::MT > 3 && live == 3) steps(ic<(T::MT > 3 ? 3 : T::MT)>());
            else if (T::MT > 4 && live <= 4) steps(ic<(T::MT > 4 ? 4 : T::MT)>());
            else if (T::MT > 6 && live <= 6) steps(ic<(T::MT > 6 ? 6 : T::MT)>());
            else steps(ic<T::MT>());
        }
        wait<0>();
        __syncthreads();
    } else {
        grid_launch();
    }
    const int N = P.n;
    auto put = [&](int row, int col, float v) {
        if (row >= M || col >= N) return;
        if (F32) reinterpret_cast<float*>(P.out)[static_cast<size_t>(row) * N + col] = v;
        else reinterpret_cast<__nv_bfloat16*>(P.out)[static_cast<size_t>(row) * N + col] = __float2bfloat16_rn(v);
    };
    auto emit = [&](int i, int j, int h, float v0, float v1) {      // entries 2h, 2h + 1 of fragment (i, j)
        if constexpr (SWAP) {                    // column g (+ 8 for h 1), rows 2t, 2t + 1
            const int col = n0 + wn * (BN / WN) + i * 16 + (lane >> 2) + h * 8;
            put(m0 + (lane & 3) * 2, col, v0);
            put(m0 + (lane & 3) * 2 + 1, col, v1);
        } else {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
            if (row >= M) return;
            if (!F32 && col + 1 < N && (N & 1) == 0) {
                auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(row) * N + col;
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
            } else {
                put(row, col, v0);
                put(row, col + 1, v1);
            }
        }
    };
    if (sk > 1) {                                // uniform in a cluster: it holds one part
#if __CUDA_ARCH__ >= 900
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (SPREAD || slice != 0) {
#pragma unroll
            for (int i = 0; i < I; ++i)
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * J + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if constexpr (SPREAD) {                  // slice s adds pairs s, s + sk, .. over the tile's slices in order
            if (live) {
                for (int pr = slice; pr < I * J * 2; pr += sk) {
                    const float* first = cluster.map_shared_rank(mine, rank - slice);
                    float v0 = first[2 * pr * T::THREADS + tid], v1 = first[(2 * pr + 1) * T::THREADS + tid];
#pragma unroll 7
                    for (int peer = 1; peer < sk; ++peer) {
                        const float* theirs = cluster.map_shared_rank(mine, rank - slice + peer);
                        v0 = v0 + theirs[2 * pr * T::THREADS + tid];
                        v1 = v1 + theirs[(2 * pr + 1) * T::THREADS + tid];
                    }
                    emit(pr / (J * 2), pr / 2 % J, pr % 2, v0, v1);
                }
            }
            cluster.sync();                      // peers keep their memory until every slice has read it
            return;
        } else {
            if (slice == 0 && live) {            // slice 0 adds its tile's peers in slice order, as qmm.cu does
                for (int peer = 1; peer < sk; ++peer) {
                    const float* theirs = cluster.map_shared_rank(mine, rank + peer);
#pragma unroll
                    for (int i = 0; i < I; ++i)
#pragma unroll
                        for (int j = 0; j < J; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = acc[i][j][e] + theirs[((i * J + j) * 4 + e) * T::THREADS + tid];
                }
            }
            cluster.sync();                      // peers keep their memory until slice 0 has read it
            if (slice != 0) return;
        }
#else
        __trap();
        return;
#endif
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) emit(i, j, h, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
}

} // namespace tf_qmm_group

// The instantiations Nemotron's windows launch on sm_121 (tile 2: rows <= 16, bf16 out).
template __global__ void tf_qmm_group::group_kernel<64, 16, 64, 1, 4, 8, false, false, false, false>(
    const __nv_bfloat16* __restrict__, const float* __restrict__, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);

// Two independent m16 accumulators reuse each weight panel; K groups, fmas and slice sums keep the narrow order.
template __global__ void tf_qmm_group::group_kernel<64, 32, 64, 1, 4, 5, false, false, false, false>(
    const __nv_bfloat16* __restrict__, const float* __restrict__, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);

// Four and eight m16 accumulators a weight panel for wide shared frames, in the same order a row.
template __global__ void tf_qmm_group::group_kernel<64, 64, 64, 1, 4, 4, false, false, false, false>(
    const __nv_bfloat16* __restrict__, const float* __restrict__, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);
template __global__ void tf_qmm_group::group_kernel<64, 128, 64, 1, 4, 3, false, false, false, false>(
    const __nv_bfloat16* __restrict__, const float* __restrict__, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);

#include "qmm_dict.cuh"

namespace tf_qmm_group {

using namespace qmm_dict;

// group_kernel_dict's shared pair-table room (qlinear.zig group_pair_tab_max, group_pair_tab_one): two CTAs an SM past the
// stages and dec[] in a cluster launch; any table (np <= 8191) in a launch of single CTAs
constexpr int GROUP_PAIR_SMEM = 14592;
constexpr int GROUP_PAIR_SMEM_ONE = 32768;

// group_kernel with dictionary-coded scales and biases (dict_lanes): no swapped or spread tiles.
template <int GS, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool SKIP = false>
__global__ void __launch_bounds__(WM * WN * 32) group_kernel_dict(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const __grid_constant__ Parts parts,
        int M, int K, int ldx, int rows_t, int C) {
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    constexpr bool SWAP = false, SPREAD = false;
    constexpr int I = SWAP ? T::NT / 2 : T::MT;     // MMA m16 tiles a warp: column pairs (swapped) or row tiles
    constexpr int J = SWAP ? 1 : T::NT;             // MMA n8 tiles a warp: the 8 rows (swapped) or column tiles
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x / C, rank = blockIdx.x % C;
    int q = 0;
#pragma unroll
    for (int i = 1; i < PARTS; ++i)
        if (i < parts.count && cid >= parts.p[i].first) q = i;
    const Part P = parts.p[q];                   // in registers: the loads below index it every group
    const int sk = P.sk, lc = cid - P.first;
    const int tile = lc / rows_t * (C / sk) + rank / sk, slice = rank % sk;
    const int m0 = lc % rows_t * BM, n0 = tile * BN, per = KG / sk, g0 = slice * per;
    const bool live = tile < P.tiles;            // a cluster's spare blocks past the last tile only join its syncs

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);           // rows past M stay zero in every stage (set once below)
    constexpr int TILE_BYTES = 64 * GS / 2;     // one stored 64-column tile's group block
    constexpr int XC = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    static_assert(BM <= T::THREADS, "one input sum a thread");
    const __nv_bfloat16* xsrc[XC];
    const unsigned char* wsrc[WC];
    const __nv_bfloat16* ssrc[SC];
    int xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < BM * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(m0 + min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES +
                  off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        ssrc[j] = (which ? P.biases : P.scales) + n0 + off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    // the pair-table room: two CTAs an SM for a cluster launch (its CTAs co-scheduled a GPC at a time), a whole
    // table for a launch of single CTAs (one an SM beats reading a large table through L2)
    const int room = C > 1 ? GROUP_PAIR_SMEM : GROUP_PAIR_SMEM_ONE;
    DictLanes D = dict_lanes<BN, WN>(P, n0, KG, tid);
    dict_room(D, P, room);
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * TILE_BYTES);
        if (D.ok) cp16(p + T::X + T::W + tid * 16, D.src + dict_window(D, g));
    };
    uint32_t mask;                              // 0x000F000F, opaque to the compiler (kept in a register)
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    float acc[I][J][4];
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    if (live) {
#pragma unroll
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS)
            if (c / T::CHUNKS >= rows)
#pragma unroll
                for (int s = 0; s < STAGES; ++s)
                    *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
        dict_table<T::THREADS>(P, buf + T::SMEM + DICT_DEC, tid, room);
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load_w(s, g0 + s);
            commit();
        }
        grid_wait();
        qmm_dict::x_keep_l2(x + static_cast<size_t>(m0) * ldx, rows, K, ldx);
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load_x(s, g0 + s);
            commit();
        }
        grid_launch();
        const int b_off = D.pair ? 0 : 16;
        auto steps = [&](auto live_c) {                   // LIVE: the warp's m16 tiles holding rows below M
            [[maybe_unused]] constexpr int LIVE = decltype(live_c)::value;
            for (int it = 0; it < per; ++it) {
                wait<STAGES - 2>();
                __syncthreads();
                const int next = it + STAGES - 1;
                if (next < per) {
                    load_x(next % STAGES, g0 + next);
                    load_w(next % STAGES, g0 + next);
                }
                commit();
                const unsigned char* p = stage(it % STAGES);
                const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
                // the lane's scale or bias, fetched now and stored after the MMAs (a global pair table's latency)
                const uint32_t dv = SWAP ? dict_fetch(D, p + T::X + T::W, buf + T::SMEM + DICT_DEC, g0 + it)
                                         : qmm_dict::dict_fetch_raw(D, p + T::X + T::W, buf + T::SMEM + DICT_DEC, g0 + it);
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(buf + T::SMEM);
                const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
                uint32_t words[T::NT][GS / 32];
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int v = 0; v < GS / 32; ++v)
                        words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
                float d[I][J][4];
                if constexpr (SWAP) {
                    dict_put(D, buf + T::SMEM, dv);
                    __syncwarp();
#pragma unroll
                    for (int kt = 0; kt < GS / 16; ++kt) {
                        uint32_t bx[2];                           // rows 0-7 at k lo, k hi: the B fragment
                        const int ch = kt * 2 + ((lane >> 3) & 1);
                        ldmatrix2(bx, p + (lane & 7) * T::ROW + swz<T::CHUNKS>(lane & 7, ch) * 16);
#pragma unroll
                        for (int i = 0; i < I; ++i) {             // A rows g, g + 8 are columns of n8 tiles 2i, 2i + 1
                            const int s0 = (kt & 1) * 8;
                            const uint32_t a[4] = {pairm(words[2 * i][kt / 2], s0, mask),
                                                   pairm(words[2 * i + 1][kt / 2], s0, mask),
                                                   pairm(words[2 * i][kt / 2], s0 + 4, mask),
                                                   pairm(words[2 * i + 1][kt / 2], s0 + 4, mask)};
                            if (kt == 0) mma0(d[i][0], a, bx[0], bx[1]);
                            else mma(d[i][0], a, bx[0], bx[1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < I; ++i)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {             // d: column g (+ 8 for e >= 2), rows 2t, 2t + 1
                            const int col = wn * (BN / WN) + i * 16 + (lane >> 2) + (e >> 1) * 8;
                            const float sv = __bfloat162float(ps[col]), bv = __bfloat162float(ps[BN + col]);
                            const float xv = px[(lane & 3) * 2 + (e & 1)];
                            acc[i][0][e] = __fmaf_rn(xv, bv, __fmaf_rn(d[i][0][e], sv, acc[i][0][e]));
                        }
                } else {
#pragma unroll
                    for (int kt = 0; kt < GS / 16; ++kt) {
                        uint32_t a[LIVE > 0 ? LIVE : 1][4];
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                            const int ch = kt * 2 + (lane >> 4);
                            ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                        }
#pragma unroll
                        for (int j = 0; j < T::NT; ++j) {
                            const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                            const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                            for (int i = 0; i < LIVE; ++i) {
                                if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                                else mma(d[i][j], a[i], b0, b1);
                            }
                        }
                    }
                    float sf = 0.0f, bf = 0.0f;
                    qmm_dict::dict_raw_vals(D, dv, sf, bf);
#pragma unroll
                    for (int j = 0; j < T::NT; ++j) {
                        const int c0 = j * 8 + (lane & 3) * 2;
                        const float sv[2] = {__shfl_sync(0xffffffffu, sf, c0), __shfl_sync(0xffffffffu, sf, c0 + 1)};
                        const float bv[2] = {__shfl_sync(0xffffffffu, bf, b_off + c0), __shfl_sync(0xffffffffu, bf, b_off + c0 + 1)};
#pragma unroll
                        for (int i = 0; i < LIVE; ++i) {
                            const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                            const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
                            for (int e = 0; e < 4; ++e)           // acc = fma(xs, b, fma(p, s, acc)): qmm.cu's order
                                acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1],
                                                         __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
                        }
                    }
                }
            }
        };
        if constexpr (!SKIP || SWAP || T::MT == 1) {
            steps(ic<T::MT>());
        } else {
            const int live = min(T::MT, max(0, (M - m0 - wm * (BM / WM) + 15) / 16));
            if (live == 0) steps(ic<0>());
            else if (live == 1) steps(ic<1>());
            else if (T::MT > 2 && live == 2) steps(ic<(T::MT > 2 ? 2 : T::MT)>());
            else if (T::MT > 3 && live == 3) steps(ic<(T::MT > 3 ? 3 : T::MT)>());
            else if (T::MT > 4 && live <= 4) steps(ic<(T::MT > 4 ? 4 : T::MT)>());
            else if (T::MT > 6 && live <= 6) steps(ic<(T::MT > 6 ? 6 : T::MT)>());
            else steps(ic<T::MT>());
        }
        wait<0>();
        __syncthreads();
    } else {
        grid_launch();
    }
    const int N = P.n;
    auto put = [&](int row, int col, float v) {
        if (row >= M || col >= N) return;
        if (F32) reinterpret_cast<float*>(P.out)[static_cast<size_t>(row) * N + col] = v;
        else reinterpret_cast<__nv_bfloat16*>(P.out)[static_cast<size_t>(row) * N + col] = __float2bfloat16_rn(v);
    };
    auto emit = [&](int i, int j, int h, float v0, float v1) {      // entries 2h, 2h + 1 of fragment (i, j)
        if constexpr (SWAP) {                    // column g (+ 8 for h 1), rows 2t, 2t + 1
            const int col = n0 + wn * (BN / WN) + i * 16 + (lane >> 2) + h * 8;
            put(m0 + (lane & 3) * 2, col, v0);
            put(m0 + (lane & 3) * 2 + 1, col, v1);
        } else {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
            if (row >= M) return;
            if (!F32 && col + 1 < N && (N & 1) == 0) {
                auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(row) * N + col;
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
            } else {
                put(row, col, v0);
                put(row, col + 1, v1);
            }
        }
    };
    if (sk > 1) {                                // uniform in a cluster: it holds one part
#if __CUDA_ARCH__ >= 900
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (SPREAD || slice != 0) {
#pragma unroll
            for (int i = 0; i < I; ++i)
#pragma unroll
                for (int j = 0; j < J; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * J + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if constexpr (SPREAD) {                  // slice s adds pairs s, s + sk, .. over the tile's slices in order
            if (live) {
                for (int pr = slice; pr < I * J * 2; pr += sk) {
                    const float* first = cluster.map_shared_rank(mine, rank - slice);
                    float v0 = first[2 * pr * T::THREADS + tid], v1 = first[(2 * pr + 1) * T::THREADS + tid];
#pragma unroll 7
                    for (int peer = 1; peer < sk; ++peer) {
                        const float* theirs = cluster.map_shared_rank(mine, rank - slice + peer);
                        v0 = v0 + theirs[2 * pr * T::THREADS + tid];
                        v1 = v1 + theirs[(2 * pr + 1) * T::THREADS + tid];
                    }
                    emit(pr / (J * 2), pr / 2 % J, pr % 2, v0, v1);
                }
            }
            cluster.sync();                      // peers keep their memory until every slice has read it
            return;
        } else {
            if (slice == 0 && live) {            // slice 0 adds its tile's peers in slice order, as qmm.cu does
                for (int peer = 1; peer < sk; ++peer) {
                    const float* theirs = cluster.map_shared_rank(mine, rank + peer);
#pragma unroll
                    for (int i = 0; i < I; ++i)
#pragma unroll
                        for (int j = 0; j < J; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = acc[i][j][e] + theirs[((i * J + j) * 4 + e) * T::THREADS + tid];
                }
            }
            cluster.sync();                      // peers keep their memory until slice 0 has read it
            if (slice != 0) return;
        }
#else
        __trap();
        return;
#endif
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) emit(i, j, h, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
}

} // namespace tf_qmm_group

// A dense MLP's decode gate and up projections and its activation in one launch: clusters of two CTAs over the same
// 64 columns of one 16-row tile, rank 0 gate's and rank 1 up's, each the group_kernel<64, 16, 64, 1, 4, 8> tile's
// arithmetic (its loads, MMA chain, per-group fmas and bf16 rounding). Rank 0 stores its bf16 gate values into rank
// 1's shared memory (distributed shared memory) and is done; rank 1 writes act = bf16(silu(gate) * up) as
// tf_q35_mlp_act does, and each row's group sum of those 64 act values in tf_q35_mlp_act's order (group_sum), into AS
// (the down projection's input sums).
namespace tf_qmm_group {

__device__ __forceinline__ float act_group_sum(const __nv_bfloat16* x) {   // qwen35.cu's group_sum
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(x + j);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int k = 0; k < 8; k++) s += __bfloat162float(b[k]);
    }
    return s;
}

template <int STAGES, bool DICT = false>
__device__ __forceinline__ void mlp_pair_body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                              const Parts& parts, __nv_bfloat16* __restrict__ act,
                                              float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    constexpr int GS = 64, BM = 16, BN = 64, WM = 1, WN = 4;
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    constexpr int I = T::MT, J = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x / 2, rank = blockIdx.x % 2;
    const Part P = parts.p[rank];                // rank 0: gate, 1: up (same n, npad, tiles)
    const int tile = cid / rows_t;
    const int m0 = cid % rows_t * BM, n0 = tile * BN, per = KG, g0 = 0;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    const __nv_bfloat16* xsrc[XC];
    const unsigned char* wsrc[WC];
    const __nv_bfloat16* ssrc[SC];
    int xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < BM * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(m0 + min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES +
                  off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        ssrc[j] = (which ? P.biases : P.scales) + n0 + off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    [[maybe_unused]] const DictLanes D = [&] {
        if constexpr (DICT) return dict_lanes<BN, WN>(P, n0, KG, tid);
        else return DictLanes{};
    }();
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * TILE_BYTES);
        if constexpr (DICT) {
            if (D.ok) cp16(p + T::X + T::W + tid * 16, D.src + dict_window(D, g));
        } else {
#pragma unroll
            for (int j = 0; j < SC; ++j)
                if (sok[j]) cp16(p + sdst[j], ssrc[j] + static_cast<size_t>(g) * P.npad);
        }
    };
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    float acc[I][J][4];
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
    if constexpr (DICT) dict_table<T::THREADS>(P, buf + T::SMEM + BM * BN * 2 + DICT_DEC, tid);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_w(s, g0 + s);
        commit();
    }
    grid_wait();
    qmm_dict::x_keep_l2(x + static_cast<size_t>(m0) * ldx, rows, K, ldx);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_x(s, g0 + s);
        commit();
    }
    grid_launch();
    const int b_off = DICT ? (D.pair ? 0 : 16) : 0;
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) {
            load_x(next % STAGES, g0 + next);
            load_w(next % STAGES, g0 + next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        // the lane's scale or bias, fetched now and stored after the MMAs (a global pair table's latency)
        uint32_t dv = 0;
        if constexpr (DICT) dv = qmm_dict::dict_fetch_raw(D, p + T::X + T::W, buf + T::SMEM + BM * BN * 2 + DICT_DEC, g0 + it);
        const __nv_bfloat16* ps = DICT ? reinterpret_cast<const __nv_bfloat16*>(buf + T::SMEM + BM * BN * 2)
                                       : reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[T::NT][GS / 32];
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
        float d[I][J][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[I][4];
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                for (int i = 0; i < I; ++i) {
                    if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
        }
        [[maybe_unused]] float sf = 0.0f, bf = 0.0f;
        if constexpr (DICT) qmm_dict::dict_raw_vals(D, dv, sf, bf);
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            float sv[2], bv[2];
            if constexpr (DICT) {
                const int c0 = j * 8 + (lane & 3) * 2;
                sv[0] = __shfl_sync(0xffffffffu, sf, c0);
                sv[1] = __shfl_sync(0xffffffffu, sf, c0 + 1);
                bv[0] = __shfl_sync(0xffffffffu, bf, b_off + c0);
                bv[1] = __shfl_sync(0xffffffffu, bf, b_off + c0 + 1);
            } else {
                const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                sv[0] = __low2float(s2);
                sv[1] = __high2float(s2);
                bv[0] = __low2float(b2);
                bv[1] = __high2float(b2);
            }
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
            }
        }
    }
    wait<0>();
#if __CUDA_ARCH__ >= 900
    // rank 0 stores gate's tile, bf16 (BM, BN), into rank 1's memory past the stages (no load of rank 1's lands
    // there), the cluster barrier orders those stores, and rank 0 is done; rank 1 then works from its own memory
    auto cluster = cooperative_groups::this_cluster();
    __nv_bfloat16* gate_tile = reinterpret_cast<__nv_bfloat16*>(buf + T::SMEM);
    if (rank == 0) {
        __nv_bfloat16* to = cluster.map_shared_rank(gate_tile, 1);
#pragma unroll
        for (int i = 0; i < I; ++i)
#pragma unroll
            for (int j = 0; j < J; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8, c = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                    *reinterpret_cast<__nv_bfloat162*>(to + r * BN + c) = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
    }
    cluster.sync();                              // gate's tile in rank 1's memory; every stage of both CTAs drained
    if (rank == 1) {
        __nv_bfloat16* own = reinterpret_cast<__nv_bfloat16*>(buf);
        const int N = P.n;
#pragma unroll
        for (int i = 0; i < I; ++i)
#pragma unroll
            for (int j = 0; j < J; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8, c = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                    const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(gate_tile + r * BN + c);
                    const __nv_bfloat162 u2 = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                    const float gf[2] = {__low2float(g2), __high2float(g2)};
                    const float uf[2] = {__low2float(u2), __high2float(u2)};
                    __nv_bfloat16 o[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) o[e] = __float2bfloat16_rn(gf[e] / (1.0f + expf(-gf[e])) * uf[e]);
                    const __nv_bfloat162 o2 = __halves2bfloat162(o[0], o[1]);
                    *reinterpret_cast<__nv_bfloat162*>(own + r * BN + c) = o2;
                    if (m0 + r < M) *reinterpret_cast<__nv_bfloat162*>(act + static_cast<size_t>(m0 + r) * N + n0 + c) = o2;
                }
        __syncthreads();                         // the CTA's act tile complete
        if (tid < rows) as[static_cast<size_t>(m0 + tid) * (N / 64) + n0 / 64] = act_group_sum(own + tid * BN);
    }
#else
    __trap();
#endif
}

} // namespace tf_qmm_group

extern "C" __global__ void __launch_bounds__(128) tf_qmm_mlp_pair(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const __grid_constant__ tf_qmm_group::Parts parts, __nv_bfloat16* __restrict__ act,
        float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    tf_qmm_group::mlp_pair_body<8>(x, xs, parts, act, as, M, K, ldx, rows_t);
}

#include "qmm_wide_kmajor.cuh"
#include "qmm_mlp_layout.cuh"

#include "qmm_mlp_single.cuh"

#include "qmm_output_stream.cuh"

#include "qmm_gdn_stream.cuh"

// Dictionary-coded twins of decode()'s cluster tile and tf_qmm_mlp_pair: the same loads,
// MMA chains, fmas and sums, each scale and bias decoded from its dictionary index (dict_lanes).
template __global__ void tf_qmm_group::group_kernel_dict<64, 16, 64, 1, 4, 8, false>(
    const __nv_bfloat16* __restrict__, const float* __restrict__, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);

extern "C" __global__ void __launch_bounds__(128) tf_qmm_mlp_pair_dict(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const __grid_constant__ tf_qmm_group::Parts parts, __nv_bfloat16* __restrict__ act,
        float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    tf_qmm_group::mlp_pair_body<8, true>(x, xs, parts, act, as, M, K, ldx, rows_t);
}

#include "qmm_mlp_single_dict.cuh"
#include "qmm_mlp_single32.cuh"

// ercumentyildirim's 32-row output-stream, lane and verify32 twins (366d8a17), past f3445bf7's GDN-in and MLP twins
#include "qmm_stream32.cuh"
