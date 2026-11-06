// The qmm_prompt_pipe.cuh bodies with the stage count as a template parameter (3 and 4 stage variants for the
// PTILE retune). Bit-safety by construction: the accumulation loops, the MMA operands and their order, the
// swizzle and every address are the crown pipe's; only how many cp.async groups stay in flight (the depth of
// the weight/activation prefetch) and which shared buffer holds which group change. The STAGES=2 schedule of
// this file is exactly the crown pipe's, so a variant differs from the crown kernel only in memory schedule.
//
// Schedule invariant (same argument as the crown pipe's comment): at the top of group `it` the copies fill
// buffer (it + STAGES - 1) % STAGES — the buffer group it-1 just finished reading (its last reader passed the
// barrier inside its last k16 step). The last k16 step waits for all but the newest STAGES-2 commit groups, so
// the next group's stage (buffer (it+1) % STAGES, filled STAGES-1 groups ago) has landed, and the newest copy
// (for group it+STAGES-1) may still be in flight under this group's MMAs.
#pragma once

#include "qmm_frag.cuh"

namespace tf_qmm_pipe3 {
using namespace tf_qmm_prefill;

using tf_qmm_pipe::cp16p;

// mlp_tile with a deeper pipeline: gate and up through silu(gate) * up, the crown pipe's bits.
template <int GS, int BM, int WM, int WN, int STAGES>
__device__ __forceinline__ void mlp_tile3(const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int m0, int n0) {
    constexpr int BN = 128;
    static_assert(STAGES >= 2, "at least the current and the next stage");
    static_assert(WN % 2 == 0, "gate and up use equal column-warp groups");
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    static_assert(T::NT % 2 == 0, "paired gate and up fragments");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    auto b_column = [&](int j) { return (j / (T::NT / 2)) * 8 + wn * (T::NT / 2) + j % (T::NT / 2); };
    const ManyPart& G = parts.p[0];
    const ManyPart& U = parts.p[1];
    const int N = G.n, npad = G.npad;
    const int KG = K / GS, per = KG, g0 = 0;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    constexpr int XC = BM * T::CHUNKS / T::THREADS, RSTEP = T::THREADS / T::CHUNKS;
    static_assert(T::THREADS % T::CHUNKS == 0 && XC * T::THREADS == BM * T::CHUNKS && RSTEP % T::CHUNKS == 0, "whole chunks");
    constexpr int TILE_BYTES = 64 * GS / 2, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const int ch = tid % T::CHUNKS, r0 = tid / T::CHUNKS;
    const int xdst = r0 * T::ROW + swz<T::CHUNKS>(r0, ch) * 16;
    const __nv_bfloat16* xsrc = x + static_cast<size_t>(m0 + r0) * ldx + ch * 8;
    const unsigned char* wsrc[WC];
    bool wok[WC];
#pragma unroll
    for (int k = 0; k < WC; ++k) {
        const int c = tid + k * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[k] = c < T::W / 16;
        wsrc[k] = reinterpret_cast<const unsigned char*>(t ? U.w : G.w) + static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + off * 16;
    }
    const int s_which = tid / (T::S / 16), s_off = tid % (T::S / 16), s_t = s_off / 8;
    const bool sok = tid < 2 * (T::S / 16);
    const __nv_bfloat16* ssrc = (s_which ? (s_t ? U.biases : G.biases) : (s_t ? U.scales : G.scales)) + n0 + (s_off % 8) * 8;
    const int sdst = s_which * T::S + s_off * 16;
    unsigned xok = 0;
#pragma unroll
    for (int i = 0; i < XC; ++i) xok |= (m0 + r0 + i * RSTEP < M ? 1u : 0u) << i;
    auto load = [&](int s, int g, bool go) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XC; ++i)
            cp16p(p + xdst + i * RSTEP * T::ROW, xsrc + static_cast<size_t>(i * RSTEP) * ldx + g * GS, go && (xok >> i & 1u));
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k) cp16p(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES, go && wok[k]);
        cp16p(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad, go && sok);
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
    auto read_w = [&](const unsigned char* p) {
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[(b_column(j) * 32 + lane) * (GS / 32) + v];
            const int col = b_column(j) * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
    };
    uint32_t a[T::MT][4];
    auto load_a_row = [&](const unsigned char* p, int kt, int i) {
        const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kt * 2 + (lane >> 4);
        ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
    };
    auto weight_pair = [&](int j, int kt) {
        return make_uint2(fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]),
                          fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]));
    };
    // prime stages 0..STAGES-2; the loop's first top-of-group copy fills buffer STAGES-1
    load(0, g0, true);
    commit();
#pragma unroll
    for (int s = 1; s < STAGES - 1; ++s) {
        load(s, g0 + s, per > s);
        commit();
    }
    wait<STAGES - 2>();
    __syncthreads();
    read_w(stage(0));
#pragma unroll
    for (int i = 0; i < T::MT; ++i) load_a_row(stage(0), 0, i);
    uint2 b[2];
    b[0] = weight_pair(0, 0);
    for (int it = 0; it < per; ++it) {
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pn = stage((it + 1) % STAGES);
        const int fill = it + STAGES - 1;
        const bool more = fill < per;
        load(fill % STAGES, g0 + (more ? fill : it), more);
        commit();
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const bool last = kt + 1 == GS / 16 && j + 1 == T::NT;
                const int next_j = (j + 1) % T::NT, next_kt = kt + (j + 1 == T::NT);
                const int slot = (kt * T::NT + j) & 1;
                if (last) {
                    wait<STAGES - 2>();
                    __syncthreads();
                    read_w(pn);
                    b[slot ^ 1] = weight_pair(0, 0);
                } else {
                    b[slot ^ 1] = weight_pair(next_j, next_kt);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    mma(acc[i][j], a[i], b[slot].x, b[slot].y);
                    if (j + 1 == T::NT) load_a_row(kt + 1 < GS / 16 ? p : pn, (kt + 1) % (GS / 16), i);
                }
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT / 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + wn * (64 / WN) + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                const __nv_bfloat162 g2 = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                const __nv_bfloat162 u2 = __floats2bfloat162_rn(acc[i][j + T::NT / 2][2 * h], acc[i][j + T::NT / 2][2 * h + 1]);
                const float gf[2] = {__low2float(g2), __high2float(g2)};
                const float uf[2] = {__low2float(u2), __high2float(u2)};
                __nv_bfloat16 o[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) o[e] = __float2bfloat16_rn(gf[e] / (1.0f + expf(-gf[e])) * uf[e]);
                __nv_bfloat16* dst = act + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(o[0], o[1]);
                else {
                    if (col < N) dst[0] = o[0];
                    if (col + 1 < N) dst[1] = o[1];
                }
            }
}

// tile with a deeper pipeline: one projection's BN columns, the crown pipe's bits.
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void tile3(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int N, int npad, int M, int K, int ldx, int m0,
        int n0) {
    static_assert(STAGES >= 2, "at least the current and the next stage");
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG, g0 = 0;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    static_assert(T::THREADS % T::CHUNKS == 0 && (BM * T::CHUNKS) % T::THREADS == 0, "whole input chunks a thread");
    constexpr int XC = BM * T::CHUNKS / T::THREADS, RSTEP = T::THREADS / T::CHUNKS;
    constexpr int TILE_BYTES = 64 * GS / 2, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const int ch = tid % T::CHUNKS, r0 = tid / T::CHUNKS;
    const int xdst = r0 * T::ROW + swz<T::CHUNKS>(r0, ch) * 16;
    static_assert(RSTEP % T::CHUNKS == 0, "the swizzle's row phase is the thread's");
    const __nv_bfloat16* xsrc = x + static_cast<size_t>(m0 + r0) * ldx + ch * 8;
    const unsigned char* wsrc[WC];
    bool wok[WC];
#pragma unroll
    for (int k = 0; k < WC; ++k) {
        const int c = tid + k * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[k] = c < T::W / 16 && n0 + t * 64 < npad;
        wsrc[k] = reinterpret_cast<const unsigned char*>(w) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES + off * 16;
    }
    const bool sok = tid < 2 * (T::S / 16) && n0 + (tid % (T::S / 16)) * 8 < npad;
    const __nv_bfloat16* ssrc = ((tid / (T::S / 16)) ? biases : scales) + n0 + (tid % (T::S / 16)) * 8;
    const int sdst = (tid / (T::S / 16)) * T::S + (tid % (T::S / 16)) * 16;
    unsigned xok = 0;
#pragma unroll
    for (int i = 0; i < XC; ++i) xok |= (m0 + r0 + i * RSTEP < M ? 1u : 0u) << i;
    auto load = [&](int s, int g, bool go) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XC; ++i)
            cp16p(p + xdst + i * RSTEP * T::ROW, xsrc + static_cast<size_t>(i * RSTEP) * ldx + g * GS, go && (xok >> i & 1u));
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k) cp16p(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES, go && wok[k]);
        cp16p(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad, go && sok);
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
    auto read_w = [&](const unsigned char* p) {
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
    };
    uint32_t a[T::MT][4];
    auto load_a_row = [&](const unsigned char* p, int kt, int i) {
        const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kt * 2 + (lane >> 4);
        ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
    };
    auto weight_pair = [&](int j, int kt) {
        return make_uint2(fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]),
                          fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]));
    };
    load(0, g0, true);
    commit();
#pragma unroll
    for (int s = 1; s < STAGES - 1; ++s) {
        load(s, g0 + s, per > s);
        commit();
    }
    wait<STAGES - 2>();
    __syncthreads();
    read_w(stage(0));
#pragma unroll
    for (int i = 0; i < T::MT; ++i) load_a_row(stage(0), 0, i);
    uint2 b[2];
    b[0] = weight_pair(0, 0);
    static_assert((GS / 16 * T::NT) % 2 == 0, "B slots alternate evenly through a group");
    for (int it = 0; it < per; ++it) {
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pn = stage((it + 1) % STAGES);
        const int fill = it + STAGES - 1;
        const bool more = fill < per;
        load(fill % STAGES, g0 + (more ? fill : it), more);
        commit();
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const bool last = kt + 1 == GS / 16 && j + 1 == T::NT;
                const int next_j = (j + 1) % T::NT, next_kt = kt + (j + 1 == T::NT);
                const int slot = (kt * T::NT + j) & 1;
                if (last) {
                    wait<STAGES - 2>();
                    __syncthreads();
                    read_w(pn);
                    b[slot ^ 1] = weight_pair(0, 0);
                } else {
                    b[slot ^ 1] = weight_pair(next_j, next_kt);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    mma(acc[i][j], a[i], b[slot].x, b[slot].y);
                    if (j + 1 == T::NT) load_a_row(kt + 1 < GS / 16 ? p : pn, (kt + 1) % (GS / 16), i);
                }
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }
}

} // namespace tf_qmm_pipe3
