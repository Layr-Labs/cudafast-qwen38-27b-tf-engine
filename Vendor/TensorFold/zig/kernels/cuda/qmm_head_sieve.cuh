// The greedy verify head's sieve (head_sieve.zig): a 2-bit copy of the head's codes (code >> 2, the same scales and
// biases) gives each (row, id) an approximate logit a and the spread s of the dropped bits' term (sum over groups of
// scale^2 times the group's sum of squares, times 1.25, the variance of a uniform 2-bit residual about its middle 1.5).
// Each row's first tile (the n8 tile of its best a) is recomputed exactly; the row's threshold is the lower of that
// tile's exact best and its best a, and the 8-id column groups whose best a + k s reaches it are recomputed with the
// head tile's own loads, MMA chain, fmas and bf16 rounding (group_kernel_dict's AMAX tile). Each row's token is the
// best of those exact values: the highest bf16 logit, ties to the lowest id, the head merge's rule. A row's true best
// u escapes only if its dropped bits' term exceeds that of the first tile's best-a id g by more than k s_u (the
// 2-bit terms of high logits move together, so the margin is on their difference).
// The kernels are templated on the window's row count R (16 or 32): every m16 row tile runs the R = 16 kernel's own
// loads, MMA chain, fmas and reductions — MT = R / 16 independent row tiles through the same group loop, the same
// key(a, id) comparisons and the same rescore list semantics — so a row's arithmetic is bit-identical at either width
// (the R = 16 instantiations are the original kernels' code).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_head_sieve {

constexpr int ROWS = 16;
constexpr int TILES = 4;                  // 64-id tiles a sieve CTA, a warp each
constexpr int STAGES = 4;               // the scan's stages: three CTAs an SM (GB10, measured quickest)
constexpr int AFR = 4 * 32 * 16;          // a group's A fragments: four k16 steps, a lane's four registers
constexpr int W2 = TILES * 1024;          // the tiles' 2-bit codes of a group
constexpr int DW = TILES * 256;           // the tiles' dictionary index windows of a group
constexpr int XV = 2 * ROWS * 4;          // a group's input sums, then sums of squares
constexpr int STAGE = AFR + W2 + DW + XV;
constexpr int SMEM = STAGES * STAGE;
constexpr int RS_STAGES = 8;
constexpr int RS_WARP = 256 + 32;         // a rescoring warp's stage: its n8 tile's words, then 8 scales and 8 biases
constexpr int RS_STAGE = AFR + 4 * RS_WARP + 64;
constexpr int RS_SMEM = RS_STAGES * RS_STAGE;
constexpr float SLACK_REL = 1.0f / 128.0f; // a bf16 logit's spacing at most, relative to its size
constexpr float SLACK_ABS = 1e-2f;         // the f32 sums' rounding, generously

// R-row shapes: MT m16 row tiles; a group's A fragments (MT planes), input sums and sums of squares.
constexpr int afr_of(int R) { return R / 16 * AFR; }
constexpr int xv_of(int R) { return 2 * R * 4; }

__device__ __forceinline__ uint32_t pair2(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x00030003u) | 0x43004300u;
    uint32_t r;
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
    return r;
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// Order of (value, id) as one integer: the higher value, then the lower id (-0 counts as +0).
__device__ __forceinline__ unsigned long long key(float v, int id) {
    uint32_t u = __float_as_uint(v == 0.0f ? 0.0f : v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | (0xFFFFFFFFu - static_cast<uint32_t>(id));
}

__device__ __forceinline__ uint32_t ordered(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

} // namespace tf_head_sieve

// One (tile, group) block of the head's tile-major words (a lane's two words an n8 tile, eight codes each) into its
// 2-bit block of 256 words, [n8 tile / 4][lane][n8 tile % 4]: a lane's sixteen codes >> 2 of the tile, placed so that
// pair2(w, 4 kt) and pair2(w, 4 kt + 2) are its B fragment registers of k step kt. Grid tiles x groups, 256 threads.
// gmajor: the 1 KiB blocks go down groups first (g * tiles + tile) instead of tiles first (tile * KG + g): the scan's
// stage g then reads the whole grid one contiguous slab (the tile-major layout gives a CTA its four tiles' pieces
// KG KiB apart, the chunked pattern that streams ~216-221 GB/s against ~240-248 adjacent). The same bytes, only the
// address mapping: the scan's loads and smem placement are unchanged, so a row's arithmetic is bit-identical.
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_pack(const uint32_t* __restrict__ w, uint32_t* __restrict__ w2, int KG, int gmajor) {
    const size_t blk = blockIdx.x;
    const int tile = static_cast<int>(blk / KG), g = static_cast<int>(blk % KG);
    const size_t out_blk = gmajor ? static_cast<size_t>(g) * (gridDim.x / KG) + tile : blk;
    const int nt = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t* src = w + blk * 512 + nt * 64 + lane * 2;
    uint32_t out = 0;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
        const uint32_t v = src[kt >> 1];
        const int sh = (kt & 1) * 8;
        const uint32_t c0 = (v >> sh) & 15u, c1 = (v >> (sh + 16)) & 15u, c8 = (v >> (sh + 4)) & 15u, c9 = (v >> (sh + 20)) & 15u;
        out |= (c0 >> 2) << (kt * 4) | (c8 >> 2) << (kt * 4 + 2) | (c1 >> 2) << (16 + kt * 4) | (c9 >> 2) << (16 + kt * 4 + 2);
    }
    w2[out_blk * 256 + (nt >> 2) * 128 + lane * 4 + (nt & 3)] = out;
}

namespace tf_head_sieve {

// The rows' A fragments by (group, row tile, k step, lane), zero past `rows`; each group's input sums (xs, zero past
// rows) and sums of squares; the rows' bests, their first tiles and the candidate count cleared. Grid K / 64, 128 threads.
template <int R>
__device__ __forceinline__ void prep(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, int rows, int K, uint4* __restrict__ afr, float* __restrict__ xv,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ pre, unsigned long long* __restrict__ count) {
    constexpr int MT = R / 16;
    qmm_frag::grid_wait();
    const int g = blockIdx.x, KG = gridDim.x, tid = threadIdx.x, lane = tid & 31, kt = tid >> 5;
    const int r = lane >> 2, h = K / 2, k2 = (g * 64 + kt * 16 + (lane & 3) * 2) / 2;
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(x);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        uint4 a;
        a.x = r + mt * 16 < rows ? x32[static_cast<size_t>(r + mt * 16) * h + k2] : 0u;
        a.y = r + mt * 16 + 8 < rows ? x32[static_cast<size_t>(r + mt * 16 + 8) * h + k2] : 0u;
        a.z = r + mt * 16 < rows ? x32[static_cast<size_t>(r + mt * 16) * h + k2 + 4] : 0u;
        a.w = r + mt * 16 + 8 < rows ? x32[static_cast<size_t>(r + mt * 16 + 8) * h + k2 + 4] : 0u;
        afr[((g * MT + mt) * 4 + kt) * 32 + lane] = a;
    }
    const int row = tid >> 3, part = tid & 7;   // eight threads a row, eight inputs each
    float s2[MT]{};
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        if (row + mt * 16 < rows) {
            const uint4 v = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(row + mt * 16) * K + g * 64 + part * 8);
            const uint32_t q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float lo = __uint_as_float(q[i] << 16), hi = __uint_as_float(q[i] & 0xFFFF0000u);
                s2[mt] = fmaf(lo, lo, s2[mt]);
                s2[mt] = fmaf(hi, hi, s2[mt]);
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        s2[mt] += __shfl_xor_sync(0xffffffffu, s2[mt], 1);
        s2[mt] += __shfl_xor_sync(0xffffffffu, s2[mt], 2);
        s2[mt] += __shfl_xor_sync(0xffffffffu, s2[mt], 4);
    }
    if (part == 0) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int rr = row + mt * 16;
            xv[g * 2 * R + rr] = rr < rows ? xs[static_cast<size_t>(rr) * KG + g] : 0.0f;
            xv[g * 2 * R + R + rr] = s2[mt];
        }
    }
    if (g == 0 && tid < R) {
        best[tid] = 0ull;
        pre[tid] = 0ull;
    }
    if (g == 0 && tid == 0) *count = 0ull;
    qmm_frag::grid_launch();
}

// The sieve: a CTA's four 64-id tiles, a warp each, through every group of the 2-bit copy (bf16 MMA on the rows' A
// fragments, the pair dictionary's scales and biases). Each (row, n8 tile)'s best a + k s + slack into himax
// [n8][R], its best a into himid [n8][R]; approx (when set) gets every (id, row)'s a and s
// ([id][2 * R]: a for R rows, then s). Pair-coded heads (code) read the dictionary, others (code 0) the (kg, npad)
// scale and bias arrays. Grid tiles / 4, 128 threads; shared NS * stage_of(R) + tab_smem (the pair table, or 0: L2).
template <int NS, int R>
__device__ __forceinline__ void scan(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx, unsigned long long* __restrict__ pre, int pre_atomic, int gmajor) {
    constexpr int MT = R / 16;
    constexpr int AFR_R = afr_of(R), XV_R = xv_of(R);
    constexpr int STAGE_R = AFR_R + W2 + DW + XV_R;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int bits = code & 31, step = 8 * bits;
    const uint32_t imask = (1u << bits) - 1u;
    const size_t tt = static_cast<size_t>(gridDim.x) * TILES; // the copy's 1 KiB blocks: g * tt + tile when gmajor
    unsigned char* tabs = buf + NS * STAGE_R;
    for (int c = tid; c < tab_smem / 16; c += 128) qmm_frag::cp16(tabs + c * 16, reinterpret_cast<const unsigned char*>(tab) + c * 16);
    const size_t t0 = static_cast<size_t>(blockIdx.x) * TILES;
    auto stage = [&](int s) { return buf + s * STAGE_R; };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = tid + j * 128, t = c >> 6, off = c & 63;
            const size_t blk2 = gmajor ? static_cast<size_t>(g) * tt + t0 + t : (t0 + t) * KG + g;
            qmm_frag::cp16(p + AFR_R + t * 1024 + off * 16, reinterpret_cast<const unsigned char*>(w2) + blk2 * 1024 + off * 16);
        }
        if (tid < 64) {
            const int t = tid >> 4, part = tid & 15;
            if (code != 0)
                qmm_frag::cp16(p + AFR_R + W2 + t * 256 + part * 16,
                               sb + (t0 + t) * KG * step + (static_cast<size_t>(g) * step & ~static_cast<size_t>(63)) + part * 16);
            else   // no pair code: the tile's 64 scales, then its 64 biases, from the (kg, npad) arrays
                qmm_frag::cp16(p + AFR_R + W2 + t * 256 + part * 16,
                               (part < 8 ? sc : bi) + static_cast<size_t>(g) * npad + (t0 + t) * 64 + (part & 7) * 8);
        }
    };
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            qmm_frag::cp16(p + mt * AFR + tid * 16, afr + (static_cast<size_t>(g) * MT + mt) * 128 + tid);
        if (tid < R / 2) qmm_frag::cp16(p + AFR_R + W2 + DW + tid * 16, xv + static_cast<size_t>(g) * 2 * R + tid * 4);
    };
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) {
        if (s < KG) load_w(s, s);
        qmm_frag::commit();
    }
    qmm_frag::grid_wait();
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) {
        if (s < KG) load_x(s, s);
        qmm_frag::commit();
    }
    qmm_frag::grid_launch();
    float acc[MT][8][4], sg[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc[mt][i][e] = 0.0f;
            sg[mt][i][e] = 0.0f;
        }
    const int r0 = lane >> 2;
    const bool stab = tab_smem > 0;
    for (int g = 0; g < KG; ++g) {
        qmm_frag::wait<NS - 2>();
        __syncthreads();
        const int next = g + NS - 1;
        if (next < KG) {
            load_w(next % NS, next);
            load_x(next % NS, next);
        }
        qmm_frag::commit();
        const unsigned char* p = stage(g % NS);
        uint32_t a[MT][4][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const uint4 v = reinterpret_cast<const uint4*>(p)[(mt * 4 + kt) * 32 + lane];
            a[mt][kt][0] = v.x;
            a[mt][kt][1] = v.y;
            a[mt][kt][2] = v.z;
            a[mt][kt][3] = v.w;
        }
        const uint4 wlo = reinterpret_cast<const uint4*>(p + AFR_R + warp * 1024)[lane];
        const uint4 whi = reinterpret_cast<const uint4*>(p + AFR_R + warp * 1024 + 512)[lane];
        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + AFR_R + W2 + warp * 256);
        const int base = (g * step & 63) * 8;
        uint32_t ent[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            if (code != 0) {
                const int off = base + (lane + 32 * hh) * bits;
                const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & imask;
                ent[hh] = stab ? reinterpret_cast<const uint32_t*>(tabs)[idx] : __ldg(tab + idx);
            } else {
                const unsigned short* ph = reinterpret_cast<const unsigned short*>(pk);
                ent[hh] = static_cast<uint32_t>(ph[lane + 32 * hh]) | static_cast<uint32_t>(ph[64 + lane + 32 * hh]) << 16;
            }
        }
        const float* pv = reinterpret_cast<const float*>(p + AFR_R + W2 + DW);
        float xsv[MT][2], qv[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            xsv[mt][0] = pv[mt * 16 + r0];
            xsv[mt][1] = pv[mt * 16 + r0 + 8];
            qv[mt][0] = pv[R + mt * 16 + r0];
            qv[mt][1] = pv[R + mt * 16 + r0 + 8];
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const uint4 wq = nt < 4 ? wlo : whi;
            const uint32_t wv = (nt & 3) == 0 ? wq.x : (nt & 3) == 1 ? wq.y : (nt & 3) == 2 ? wq.z : wq.w;
            float d[MT][4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const uint32_t b0 = pair2(wv, kt * 4), b1 = pair2(wv, kt * 4 + 2);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (kt == 0) qmm_frag::mma0(d[mt], a[mt][kt], b0, b1);
                    else qmm_frag::mma(d[mt], a[mt][kt], b0, b1);
                }
            }
            const int src = (nt & 3) * 8 + (lane & 3) * 2;
            const uint32_t eh = nt < 4 ? ent[0] : ent[1];
            const uint32_t ea = __shfl_sync(0xffffffffu, eh, src), eb = __shfl_sync(0xffffffffu, eh, src + 1);
            const float s0 = __uint_as_float(ea << 16), c0 = __uint_as_float(ea & 0xFFFF0000u);
            const float s1 = __uint_as_float(eb << 16), c1 = __uint_as_float(eb & 0xFFFF0000u);
            const float m0 = fmaf(1.5f, s0, c0), m1 = fmaf(1.5f, s1, c1);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const float s[2] = {s0, s1}, m[2] = {m0, m1};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[mt][nt][e] = fmaf(4.0f * s[e & 1], d[mt][e], fmaf(m[e & 1], xsv[mt][e >> 1], acc[mt][nt][e]));
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    sg[mt][nt][e] = fmaf(s[e & 1] * s[e & 1], qv[mt][e >> 1], sg[mt][nt][e]);
            }
        }
    }
    qmm_frag::wait<0>();
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        float hi[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float sd = sqrtf(1.25f * sg[mt][nt][e]), ks = kappa * sd, v = acc[mt][nt][e];
            hi[mt][e] = v + ks + fmaf(fabsf(v) + ks, SLACK_REL, SLACK_ABS);
            if (approx != nullptr) {
                const size_t id = (t0 + warp) * 64 + nt * 8 + (lane & 3) * 2 + (e & 1);
                const int row = mt * 16 + r0 + (e >> 1) * 8;
                approx[id * 2 * R + row] = v;
                approx[id * 2 * R + R + row] = sd;
            }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float h0 = fmaxf(hi[mt][0], hi[mt][1]), h1 = fmaxf(hi[mt][2], hi[mt][3]);
            h0 = fmaxf(h0, __shfl_xor_sync(0xffffffffu, h0, 1));
            h0 = fmaxf(h0, __shfl_xor_sync(0xffffffffu, h0, 2));
            h1 = fmaxf(h1, __shfl_xor_sync(0xffffffffu, h1, 1));
            h1 = fmaxf(h1, __shfl_xor_sync(0xffffffffu, h1, 2));
            float mm0 = fmaxf(acc[mt][nt][0], acc[mt][nt][1]), mm1 = fmaxf(acc[mt][nt][2], acc[mt][nt][3]);
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, 1));
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, 2));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, 1));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, 2));
            if ((lane & 3) == 0) {
                const size_t n8 = (t0 + warp) * 8 + nt;
                if (r0 + mt * 16 < rows) {
                    himax[n8 * R + r0 + mt * 16] = h0;
                    himid[n8 * R + r0 + mt * 16] = mm0;
                    // the rows' first tile, sieve_top's own rule (key = the higher value, then the lower tile),
                    // maintained here by atomicMax: prep left 0 = unset and keys of real values are never 0, and
                    // the key order is the same amax_take order, so pre ends exactly tf_head_sieve_top's write
                    if (pre_atomic) atomicMax(pre + r0 + mt * 16, key(h0, static_cast<int>(n8)));
                }
                if (r0 + 8 + mt * 16 < rows) {
                    himax[n8 * R + r0 + 8 + mt * 16] = h1;
                    himid[n8 * R + r0 + 8 + mt * 16] = mm1;
                    // the second band's rows need their pre too: a window of 9-16 rows leaves pre[row] at
                    // prep's 0 otherwise, and the keys rescore turns 0 into tile (int)(0xFFFFFFFF - 0) = -1
                    if (pre_atomic) atomicMax(pre + r0 + 8 + mt * 16, key(h1, static_cast<int>(n8)));
                }
            }
        }
    }
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(128, 3) tf_head_sieve_scan(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx, unsigned long long* __restrict__ pre, int pre_atomic, int gmajor) {
    tf_head_sieve::scan<tf_head_sieve::STAGES, 16>(w2, afr, xv, sb, tab, code, tab_smem, sc, bi, npad, KG, rows, kappa, himax, himid, approx, pre, pre_atomic, gmajor);
}

// The 32-row sieve: two m16 row tiles a warp, the same stages, group loop and per-row arithmetic.
extern "C" __global__ void __launch_bounds__(128, 2) tf_head_sieve_scan32(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx, unsigned long long* __restrict__ pre, int pre_atomic, int gmajor) {
    tf_head_sieve::scan<tf_head_sieve::STAGES, 32>(w2, afr, xv, sb, tab, code, tab_smem, sc, bi, npad, KG, rows, kappa, himax, himid, approx, pre, pre_atomic, gmajor);
}

// Five-stage probes of the same scan (TF_HS_STAGES=5): the deeper pipeline runs at two CTAs an SM
// (5 x STAGE an CTA no longer fits three), the per-row arithmetic and the loads per group unchanged.
namespace tf_head_sieve {
constexpr int STAGES5 = 5;
}
extern "C" __global__ void __launch_bounds__(128, 2) tf_head_sieve_scan5(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx, unsigned long long* __restrict__ pre, int pre_atomic, int gmajor) {
    tf_head_sieve::scan<tf_head_sieve::STAGES5, 16>(w2, afr, xv, sb, tab, code, tab_smem, sc, bi, npad, KG, rows, kappa, himax, himid, approx, pre, pre_atomic, gmajor);
}

extern "C" __global__ void __launch_bounds__(128, 2) tf_head_sieve_scan5_32(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx, unsigned long long* __restrict__ pre, int pre_atomic, int gmajor) {
    tf_head_sieve::scan<tf_head_sieve::STAGES5, 32>(w2, afr, xv, sb, tab, code, tab_smem, sc, bi, npad, KG, rows, kappa, himax, himid, approx, pre, pre_atomic, gmajor);
}

namespace tf_head_sieve {

// Each row's first tile: the n8 tile of its highest value in h (himid: the best a; key: then the lower tile) into
// pre[row], for an exact value to bound the row's best from below. Grid any, 256 threads.
template <int R>
__device__ __forceinline__ void sieve_top(const float* __restrict__ himax, int rows, int n8s,
        unsigned long long* __restrict__ pre) {
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    const int tid = threadIdx.x, lane = tid & 31;
    float bv[16];
    int bt[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        bv[r] = tf_head_sieve::neg_inf();
        bt[r] = 0x7fffffff;
    }
    for (int t = blockIdx.x * 256 + tid; t < n8s; t += gridDim.x * 256) {
        const float4* h = reinterpret_cast<const float4*>(himax + static_cast<size_t>(t) * R);
        // two 16-row bands share the registers: a later band replaces a value only when strictly higher, so the
        // kept tile stays the lowest t of the best value (the R = 16 kernel's rule)
#pragma unroll
        for (int band = 0; band < R / 16; ++band) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = h[band * 4 + i];
                const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (vv[j] > bv[i * 4 + j]) {
                        bv[i * 4 + j] = vv[j];
                        bt[i * 4 + j] = t;
                    }
            }
        }
    }
    __shared__ unsigned long long sk[8][R];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        float b = bv[r];
        int id = bt[r];
#pragma unroll
        for (int m = 16; m; m >>= 1)
            tf_qmm_group::amax_take(__shfl_xor_sync(0xffffffffu, b, m), __shfl_xor_sync(0xffffffffu, id, m), b, id);
        if (lane == 0) sk[tid >> 5][r] = id != 0x7fffffff ? tf_head_sieve::key(b, id) : 0ull;
    }
    __syncthreads();
    if (tid < rows) {
        unsigned long long k = sk[0][tid];
        for (int w = 1; w < 8; ++w) k = sk[w][tid] > k ? sk[w][tid] : k;
        if (k != 0ull) atomicMax(pre + tid, k);
    }
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_top(const float* __restrict__ himax, int rows, int n8s,
        unsigned long long* __restrict__ pre) {
    tf_head_sieve::sieve_top<16>(himax, rows, n8s, pre);
}

extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_top32(const float* __restrict__ himax, int rows, int n8s,
        unsigned long long* __restrict__ pre) {
    tf_head_sieve::sieve_top<32>(himax, rows, n8s, pre);
}

namespace tf_head_sieve {

// Each row's threshold: the lower of the exact best value its first tile gave (best[row], a lower bound of the row's
// top logit) and that tile's best a (himid at pre[row]'s tile); then every n8 tile some row's himax reaches into list
// (count past cap: an overflow, and the rescoring takes every tile). mrow (when set) gets the thresholds. A row whose
// best is unset takes every tile.
template <int R>
__device__ __forceinline__ void sieve_select(const float* __restrict__ himax,
        const float* __restrict__ himid, const unsigned long long* __restrict__ pre,
        const unsigned long long* __restrict__ best, int rows, int n8s, int* __restrict__ list, unsigned long long* __restrict__ count,
        int cap, float* __restrict__ mrow) {
    qmm_frag::grid_wait();
    __shared__ float M[R];
    __shared__ int first_tile[R];
    const int tid = threadIdx.x, lane = tid & 31;
    if (tid < R) {
        const unsigned long long k = best[tid], f = pre[tid];
        // The preceding keys-mode rescore evaluates every selected first tile for every live row.
        first_tile[tid] = f == 0ull ? -1 : static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(f));
        const uint32_t u = static_cast<uint32_t>(k >> 32);
        float v = k == 0ull || f == 0ull ? tf_head_sieve::neg_inf() : __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
        if (k != 0ull && f != 0ull) {
            const size_t t = 0xFFFFFFFFu - static_cast<uint32_t>(f);
            v = fminf(v, himid[t * R + tid]);
        }
        M[tid] = v;
        if (blockIdx.x == 0 && mrow != nullptr) mrow[tid] = v;
    }
    __syncthreads();
    qmm_frag::grid_launch();
    for (int base = blockIdx.x * 256; base < n8s; base += gridDim.x * 256) {
        const int t = base + tid;
        bool hit = false;
        if (t < n8s) {
            const float4* h = reinterpret_cast<const float4*>(himax + static_cast<size_t>(t) * R);
#pragma unroll
            for (int i = 0; i < R / 4; ++i) {
                const float4 v = h[i];
                const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) hit |= (i * 4 + j < rows) && vv[j] >= M[i * 4 + j];
            }
        }
        // PEER-10 (jungjipdo, submission 3e139917 / source 835f2e61): a qualifying tile that is some live
        // row's first tile was already rescored exactly by the keys pass, so it leaves the compact list;
        // the overflow rule still counts every qualifying tile (the count's high word).
        bool already_scored = false;
        if (hit) {
#pragma unroll
            for (int r = 0; r < R; ++r) already_scored |= r < rows && first_tile[r] == t;
        }
        const unsigned hits = __ballot_sync(0xffffffffu, hit);
        const unsigned emit = __ballot_sync(0xffffffffu, hit && !already_scored);
        if (hits != 0u) {
            unsigned long long at = 0ull;
            // High word counts all hits for the original overflow rule; low word reserves compact entries.
            if (lane == 0) at = atomicAdd(count, (static_cast<unsigned long long>(__popc(hits)) << 32) |
                                                  static_cast<unsigned long long>(__popc(emit)));
            const unsigned base_at = __shfl_sync(0xffffffffu, static_cast<unsigned>(at), 0);
            if (hit && !already_scored) {
                const unsigned pos = base_at + __popc(emit & ((1u << lane) - 1u));
                if (pos < static_cast<unsigned>(cap)) list[pos] = t;
            }
        }
    }
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_select(const float* __restrict__ himax,
        const float* __restrict__ himid, const unsigned long long* __restrict__ pre,
        const unsigned long long* __restrict__ best, int rows, int n8s, int* __restrict__ list, unsigned long long* __restrict__ count,
        int cap, float* __restrict__ mrow) {
    tf_head_sieve::sieve_select<16>(himax, himid, pre, best, rows, n8s, list, count, cap, mrow);
}

extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_select32(const float* __restrict__ himax,
        const float* __restrict__ himid, const unsigned long long* __restrict__ pre,
        const unsigned long long* __restrict__ best, int rows, int n8s, int* __restrict__ list, unsigned long long* __restrict__ count,
        int cap, float* __restrict__ mrow) {
    tf_head_sieve::sieve_select<32>(himax, himid, pre, best, rows, n8s, list, count, cap, mrow);
}

namespace tf_head_sieve {

// The listed n8 tiles (every tile on an overflow; with keys, the rows' first tiles) exactly: a warp a tile, the head tile's words, A fragments, MMA chain
// (mma0 then three mma a group), acc = fma(xs, b, fma(d, s, acc)) a group in order, bf16 rounding, then each row's best
// of the tile into best[row] (key: the higher value, then the lower id). Grid any, 128 threads, RS_SMEM shared.
template <int R>
__device__ __forceinline__ void sieve_rescore(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const unsigned long long* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ stats) {
    using namespace tf_head_sieve;
    constexpr int MT = R / 16;
    constexpr int RS_STAGE_R = AFR + 4 * RS_WARP + R * 4;
    extern __shared__ __align__(128) unsigned char buf[];
    qmm_frag::grid_wait();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // PEER-10 (jungjipdo 3e139917 / 835f2e61): the select's packed count - high word every qualifying
    // tile (the original overflow rule), low word the compact list's entries.
    const unsigned long long counts = keys != nullptr ? 0ull : *count;
    const int total = static_cast<int>(counts >> 32);
    const int n = keys != nullptr ? rows : static_cast<int>(static_cast<unsigned>(counts));
    const bool all = keys == nullptr && total > cap;
    const int items = all ? n8s : n;
    const int r0 = lane >> 2;
    // PEER-18 (265c97a6/c80d6ce): the second rescore records the same completed selector counts the
    // finish kernel wrote, so the host-keys path can drop that launch. Single block/tid, no atomics.
    if (stats != nullptr && blockIdx.x == 0 && tid == 0) {
        const unsigned long long selected = all ? static_cast<unsigned long long>(n8s) : static_cast<unsigned long long>(total);
        stats[0] += 1;
        stats[1] += selected;
        if (selected > stats[2]) stats[2] = selected;
        if (all) stats[3] += 1;
        stats[4] += static_cast<unsigned long long>(rows);
    }
    qmm_frag::grid_launch();
    for (int base = blockIdx.x * 4; base < items; base += gridDim.x * 4) {
        const int it = base + warp;
        const bool in_range = it < items;
        const int item = !in_range ? 0 : keys != nullptr ? static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(keys[it])) : all ? it : list[it];
        // A first tile is rescored for every live row, so later references to the same valid tile do not
        // need another identical exact projection. Unset/invalid keys keep the original path.
        bool duplicate_first = false;
        if (keys != nullptr && in_range && keys[it] != 0ull && item >= 0 && item < n8s) {
            const uint32_t encoded_tile = static_cast<uint32_t>(keys[it]);
            for (int prev = 0; prev < it; ++prev) {
                const unsigned long long earlier = keys[prev];
                if (earlier != 0ull && static_cast<uint32_t>(earlier) == encoded_tile) {
                    duplicate_first = true;
                    break;
                }
            }
        }
        // Compact-list entries are nonnegative; all threads take these uniform branches and barriers.
        const bool live = in_range && (keys != nullptr || all || item >= 0) && !duplicate_first;
        if (keys != nullptr && !__syncthreads_or(live)) continue;
        const int t = live ? item : 0;
        const size_t tile = static_cast<size_t>(t >> 3);
        const int nt = t & 7;
        const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(w) + tile * KG * 2048 + nt * 256;
        const size_t col0 = tile * 64 + nt * 8;
        auto load = [&](int s, int g) {
            unsigned char* p = buf + s * RS_STAGE_R;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                qmm_frag::cp16(p + mt * AFR + tid * 16, afr + (static_cast<size_t>(g) * MT + mt) * 128 + tid);
            if (live) {
                if (lane < 16) qmm_frag::cp16(p + AFR + warp * RS_WARP + lane * 16, wsrc + static_cast<size_t>(g) * 2048 + lane * 16);
                else if (lane == 16) qmm_frag::cp16(p + AFR + warp * RS_WARP + 256, scales + static_cast<size_t>(g) * npad + col0);
                else if (lane == 17) qmm_frag::cp16(p + AFR + warp * RS_WARP + 272, biases + static_cast<size_t>(g) * npad + col0);
            }
            if (tid >= 64 && tid < 64 + R / 4) qmm_frag::cp16(p + AFR + 4 * RS_WARP + (tid - 64) * 16, xv + static_cast<size_t>(g) * 2 * R + (tid - 64) * 4);
        };
#pragma unroll
        for (int s = 0; s < RS_STAGES - 1; ++s) {
            if (s < KG) load(s, s);
            qmm_frag::commit();
        }
        float acc[MT][4]{};
        for (int g = 0; g < KG; ++g) {
            qmm_frag::wait<RS_STAGES - 2>();
            __syncthreads();
            const int next = g + RS_STAGES - 1;
            if (next < KG) load(next % RS_STAGES, next);
            qmm_frag::commit();
            if (live) {
                const unsigned char* p = buf + (g % RS_STAGES) * RS_STAGE_R;
                const uint2 wv = reinterpret_cast<const uint2*>(p + AFR + warp * RS_WARP)[lane];
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + AFR + warp * RS_WARP + 256);
                const float* px = reinterpret_cast<const float*>(p + AFR + 4 * RS_WARP);
                float d[MT][4];
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const uint4 vm = reinterpret_cast<const uint4*>(p)[(mt * 4 + kt) * 32 + lane];
                        const uint32_t a[4] = {vm.x, vm.y, vm.z, vm.w};
                        const uint32_t word = kt < 2 ? wv.x : wv.y;
                        const uint32_t b0 = qmm_frag::pair(word, (kt & 1) * 8), b1 = qmm_frag::pair(word, (kt & 1) * 8 + 4);
                        if (kt == 0) qmm_frag::mma0(d[mt], a, b0, b1);
                        else qmm_frag::mma(d[mt], a, b0, b1);
                    }
                }
                const int c = (lane & 3) * 2;
                const float sv[2] = {__bfloat162float(ps[c]), __bfloat162float(ps[c + 1])};
                const float bv[2] = {__bfloat162float(ps[8 + c]), __bfloat162float(ps[8 + c + 1])};
                float xr[MT][2];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    xr[mt][0] = px[mt * 16 + r0];
                    xr[mt][1] = px[mt * 16 + r0 + 8];
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[mt][e] = __fmaf_rn(xr[mt][e >> 1], bv[e & 1], __fmaf_rn(d[mt][e], sv[e & 1], acc[mt][e]));
            }
        }
        qmm_frag::wait<0>();
        __syncthreads();
        if (live) {
            const int col = static_cast<int>(col0) + (lane & 3) * 2;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float b = neg_inf();
                int id = 0x7fffffff;
#pragma unroll
                for (int e2 = 0; e2 < 2; ++e2) {
                    const float v = __bfloat162float(__float2bfloat16_rn(acc[mt][2 * hh + e2]));
                    if (v > b) {
                        b = v;
                        id = col + e2;
                    }
                }
#pragma unroll
                for (int m = 1; m <= 2; m <<= 1)
                    tf_qmm_group::amax_take(__shfl_xor_sync(0xffffffffu, b, m), __shfl_xor_sync(0xffffffffu, id, m), b, id);
                const int row = mt * 16 + r0 + 8 * hh;
                if ((lane & 3) == 0 && row < rows && id != 0x7fffffff) atomicMax(best + row, key(b, id));
            }
        }
    }
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_rescore(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const unsigned long long* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ stats) {
    tf_head_sieve::sieve_rescore<16>(w, scales, biases, npad, afr, xv, KG, rows, list, count, cap, n8s, keys, best, stats);
}

extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_rescore32(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const unsigned long long* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ stats) {
    tf_head_sieve::sieve_rescore<32>(w, scales, biases, npad, afr, xv, KG, rows, list, count, cap, n8s, keys, best, stats);
}

namespace tf_head_sieve {
constexpr int RN_GROUP = 256 + 32;   // an n8 tile's group in the n8-major copy: its head words, 8 scales, 8 biases
constexpr int RN_WARPS = 4;          // the n8-major rescore: four warps a CTA, an item a warp
constexpr int RN_CH = 4;             // groups a stage (1152 contiguous bytes an item)
constexpr int RN_STAGES = 3;
constexpr int RN_KCH = 8;            // the first tiles' pass: eight groups a stage
constexpr int rn_smem(int ch, int stages) { return stages * (ch * AFR + RN_WARPS * ch * RN_GROUP + ch * ROWS * 4); }
}

// The n8-major copy of the head for the rescore: n8 tile t's KG groups back to back, each the 256 bytes of head words
// the rescore reads at tile * KG * 2048 + g * 2048 + nt * 256, then its 8 bf16 scales and its 8 bf16 biases.
// Grid n8 tiles, 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_pack_rn(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad, int KG,
        unsigned char* __restrict__ wr) {
    using namespace tf_head_sieve;
    const size_t t = blockIdx.x;
    const size_t tile = t >> 3;
    const int nt = static_cast<int>(t & 7);
    const size_t col0 = tile * 64 + nt * 8;
    const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(w) + tile * KG * 2048 + nt * 256;
    unsigned char* dst = wr + t * KG * RN_GROUP;
    for (int c = threadIdx.x; c < KG * 18; c += blockDim.x) {
        const int g = c / 18, part = c % 18;
        uint4 v;
        if (part < 16) v = *reinterpret_cast<const uint4*>(wsrc + static_cast<size_t>(g) * 2048 + part * 16);
        else v = *reinterpret_cast<const uint4*>((part == 16 ? scales : biases) + static_cast<size_t>(g) * npad + col0);
        *reinterpret_cast<uint4*>(dst + static_cast<size_t>(g) * RN_GROUP + part * 16) = v;
    }
}

namespace tf_head_sieve {

// sieve_rescore<16> from the n8-major copy: the same items (the list, every tile on an overflow, or with keys the rows'
// first tiles), the same per-item loads, MMA chain (mma0 then three mma a group), acc = fma(xs, b, fma(d, s, acc)) in
// group order, bf16 rounding and atomicMax(key) into best[row]. A warp an item; the CTA's batches of RN_WARPS items run
// through one pipeline of CH-group steps, each step an item's CH groups in one contiguous piece (the head's own layout
// gives 256-byte pieces 2 KiB apart, which the memory serves at ~190 GB/s; contiguous 1152-byte pieces at ~230).
// KG % CH == 0. Grid any, RN_WARPS * 32 threads, rn_smem(CH, STAGES) shared.
template <int CH, int STAGES>
__device__ __forceinline__ void sieve_rescore_rn(const unsigned char* __restrict__ wr, const uint4* __restrict__ afr,
        const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list, const unsigned long long* __restrict__ count,
        int cap, int n8s, const unsigned long long* __restrict__ keys, unsigned long long* __restrict__ best) {
    constexpr int R = ROWS, NW = RN_WARPS, NT = NW * 32;
    constexpr int O_W = CH * AFR, O_X = O_W + NW * CH * RN_GROUP;
    constexpr int STG = O_X + CH * R * 4;
    extern __shared__ __align__(128) unsigned char buf[];
    qmm_frag::grid_wait();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // PEER-10 (jungjipdo 3e139917 / 835f2e61): the select's packed count, and the keys pass skips a
    // first tile an earlier row already rescored exactly (the atomicMax keys are order-independent).
    const unsigned long long counts = keys != nullptr ? 0ull : *count;
    const int total = static_cast<int>(counts >> 32);
    const int n = keys != nullptr ? rows : static_cast<int>(static_cast<unsigned>(counts));
    const bool all = keys == nullptr && total > cap;
    const int items = all ? n8s : n;
    const int r0 = lane >> 2;
    qmm_frag::grid_launch();
    const int per = gridDim.x * NW;
    const int first = blockIdx.x * NW;
    if (first >= items) return;
    const int nb = (items - first + per - 1) / per;
    const int cpi = KG / CH;
    const int steps = nb * cpi;
    auto item_of = [&](int j) { return first + j * per + warp; };
    auto dup_of = [&](int it) {
        if (keys == nullptr || it < 0 || it >= items) return false;
        const unsigned long long k = keys[it];
        if (k == 0ull) return false;
        const int tile = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(k));
        if (tile < 0 || tile >= n8s) return false;
        for (int prev = 0; prev < it; ++prev) {
            const unsigned long long earlier = keys[prev];
            if (earlier != 0ull && static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(earlier)) == tile) return true;
        }
        return false;
    };
    auto tile_of = [&](int it) {
        return keys != nullptr ? static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(keys[it])) : all ? it : list[it];
    };
    auto load = [&](int s, int f) {
        const int j = f / cpi, g0 = (f - j * cpi) * CH;
        unsigned char* p = buf + s * STG;
        for (int c = tid; c < CH * 128; c += NT) qmm_frag::cp16(p + c * 16, afr + static_cast<size_t>(g0) * 128 + c);
        const int it = item_of(j);
        if (it < items && !dup_of(it)) {
            const unsigned char* src = wr + (static_cast<size_t>(tile_of(it)) * KG + g0) * RN_GROUP;
            for (int c = lane; c < CH * 18; c += 32) qmm_frag::cp16(p + O_W + warp * CH * RN_GROUP + c * 16, src + c * 16);
        }
        for (int c = tid; c < CH * R / 4; c += NT)
            qmm_frag::cp16(p + O_X + c * 16, xv + static_cast<size_t>(g0 + c / (R / 4)) * 2 * R + (c % (R / 4)) * 4);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < steps) load(s, s);
        qmm_frag::commit();
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int f = 0; f < steps; ++f) {
        qmm_frag::wait<STAGES - 2>();
        __syncthreads();
        const int nx = f + STAGES - 1;
        if (nx < steps) load(nx % STAGES, nx);
        qmm_frag::commit();
        const int j = f / cpi, ch = f - j * cpi;
        const int it = item_of(j);
        if (it < items && !dup_of(it)) {
            const unsigned char* p = buf + (f % STAGES) * STG;
#pragma unroll
            for (int q = 0; q < CH; ++q) {
                const unsigned char* pw = p + O_W + (warp * CH + q) * RN_GROUP;
                const uint2 wv = reinterpret_cast<const uint2*>(pw)[lane];
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(pw + 256);
                const float* px = reinterpret_cast<const float*>(p + O_X + q * R * 4);
                float d[4];
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const uint4 vm = reinterpret_cast<const uint4*>(p + q * AFR)[kt * 32 + lane];
                    const uint32_t a[4] = {vm.x, vm.y, vm.z, vm.w};
                    const uint32_t word = kt < 2 ? wv.x : wv.y;
                    const uint32_t b0 = qmm_frag::pair(word, (kt & 1) * 8), b1 = qmm_frag::pair(word, (kt & 1) * 8 + 4);
                    if (kt == 0) qmm_frag::mma0(d, a, b0, b1);
                    else qmm_frag::mma(d, a, b0, b1);
                }
                const int c = (lane & 3) * 2;
                const float sv[2] = {__bfloat162float(ps[c]), __bfloat162float(ps[c + 1])};
                const float bv[2] = {__bfloat162float(ps[8 + c]), __bfloat162float(ps[8 + c + 1])};
                const float xr[2] = {px[r0], px[r0 + 8]};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[e] = __fmaf_rn(xr[e >> 1], bv[e & 1], __fmaf_rn(d[e], sv[e & 1], acc[e]));
            }
            if (ch == cpi - 1) {
                const int t = tile_of(it);
                const int col = (t >> 3) * 64 + (t & 7) * 8 + (lane & 3) * 2;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    float b = neg_inf();
                    int id = 0x7fffffff;
#pragma unroll
                    for (int e2 = 0; e2 < 2; ++e2) {
                        const float v = __bfloat162float(__float2bfloat16_rn(acc[2 * hh + e2]));
                        if (v > b) {
                            b = v;
                            id = col + e2;
                        }
                    }
#pragma unroll
                    for (int m = 1; m <= 2; m <<= 1)
                        tf_qmm_group::amax_take(__shfl_xor_sync(0xffffffffu, b, m), __shfl_xor_sync(0xffffffffu, id, m), b, id);
                    const int row = r0 + 8 * hh;
                    if ((lane & 3) == 0 && row < rows && id != 0x7fffffff) atomicMax(best + row, key(b, id));
                }
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[e] = 0.0f;
            }
        }
    }
    qmm_frag::wait<0>();
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_rescore_rn(const unsigned char* __restrict__ wr,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const unsigned long long* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best) {
    tf_head_sieve::sieve_rescore_rn<tf_head_sieve::RN_CH, tf_head_sieve::RN_STAGES>(wr, afr, xv, KG, rows, list, count, cap, n8s, keys, best);
}

// The rows' first tiles (keys): at most 16 items, so fewer, longer steps.
extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_rescore_rn_keys(const unsigned char* __restrict__ wr,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const unsigned long long* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best) {
    tf_head_sieve::sieve_rescore_rn<tf_head_sieve::RN_KCH, tf_head_sieve::RN_STAGES>(wr, afr, xv, KG, rows, list, count, cap, n8s, keys, best);
}

// Each row's token from its best; the round's counters: stats[0] rounds, [1] rescored n8 tiles, [2] their most,
// [3] overflows, [4] rows. One warp.
extern "C" __global__ void __launch_bounds__(32) tf_head_sieve_finish(const unsigned long long* __restrict__ best,
        int rows, unsigned* __restrict__ toks, const unsigned long long* __restrict__ count, int cap, int n8s,
        unsigned long long* __restrict__ stats) {
    qmm_frag::grid_wait();
    const int tid = threadIdx.x;
    if (tid < rows) toks[tid] = 0xFFFFFFFFu - static_cast<unsigned>(best[tid] & 0xFFFFFFFFull);
    if (tid == 0) {
        // PEER-10 (jungjipdo 3e139917 / 835f2e61): the high word counts every qualifying tile, the
        // original overflow rule's denominator; the compact list only held the low word's entries.
        const int n = static_cast<int>(*count >> 32);
        const unsigned long long items = n > cap ? static_cast<unsigned long long>(n8s) : static_cast<unsigned long long>(n);
        stats[0] += 1;
        stats[1] += items;
        if (items > stats[2]) stats[2] = items;
        if (n > cap) stats[3] += 1;
        stats[4] += static_cast<unsigned long long>(rows);
    }
}

// The 17-32-row twin: one warp covers every row, so the body is the 16-row kernel's.
extern "C" __global__ void __launch_bounds__(32) tf_head_sieve_finish32(const unsigned long long* __restrict__ best,
        int rows, unsigned* __restrict__ toks, const unsigned long long* __restrict__ count, int cap, int n8s,
        unsigned long long* __restrict__ stats) {
    qmm_frag::grid_wait();
    const int tid = threadIdx.x;
    if (tid < rows) toks[tid] = 0xFFFFFFFFu - static_cast<unsigned>(best[tid] & 0xFFFFFFFFull);
    if (tid == 0) {
        const int n = static_cast<int>(*count >> 32);
        const unsigned long long items = n > cap ? static_cast<unsigned long long>(n8s) : static_cast<unsigned long long>(n);
        stats[0] += 1;
        stats[1] += items;
        if (items > stats[2]) stats[2] = items;
        if (n > cap) stats[3] += 1;
        stats[4] += static_cast<unsigned long long>(rows);
    }
}

namespace tf_head_sieve {

// TF_HEAD_SIEVE=check, a block a row, against the full head's token `full` and the row's threshold mrow: stats[5] tokens
// differing from the sieve's (`mine`); for k = kappa - 3 .. kappa (index i): [6 + i] rows whose true token falls outside
// the column candidates (a + k s + slack >= threshold),
// [10 + i] the candidates' sum, [14 + i] their most, [18 + 8 i + b] a histogram (b: < 16, 64, 256, 1024, 4096, 16384,
// 65536, more); [50] the least margin at kappa of the true token, (a + k s + slack - threshold) / s, as ordered bits;
// [51] and [52] the most and least of the true token's (exact - a) / s, as ordered bits (rows whose tokens agree).
template <int R>
__device__ __forceinline__ void sieve_check(const float* __restrict__ approx, int V,
        float kappa, const float* __restrict__ mrow, const unsigned* __restrict__ full, const unsigned* __restrict__ mine,
        const unsigned long long* __restrict__ best, unsigned long long* __restrict__ stats) {
    using namespace tf_head_sieve;
    qmm_frag::grid_wait();
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ unsigned redu[4][32];
    __shared__ float M[4];
    const float k[4] = {kappa - 3.0f, kappa - 2.0f, kappa - 1.0f, kappa};
    if (tid < 4) M[tid] = mrow[r];
    __syncthreads();
    unsigned c[4] = {0u, 0u, 0u, 0u};
    for (int v = tid; v < V; v += 1024) {
        const float a = approx[static_cast<size_t>(v) * 2 * R + r], sd = approx[static_cast<size_t>(v) * 2 * R + R + r];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float ks = k[i] * sd;
            c[i] += (a + ks + fmaf(fabsf(a) + ks, SLACK_REL, SLACK_ABS) >= M[i]) ? 1u : 0u;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int o = 16; o; o >>= 1) c[i] += __shfl_xor_sync(0xffffffffu, c[i], o);
        if (lane == 0) redu[i][warp] = c[i];
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned u = full[r];
        const float a = approx[static_cast<size_t>(u) * 2 * R + r], sd = approx[static_cast<size_t>(u) * 2 * R + R + r];
        if (u != mine[r]) atomicAdd(stats + 5, 1ull);
        else {   // the true token's exact logit is the row's best: its 2-bit error in spreads, both ways
            const uint32_t o = static_cast<uint32_t>(best[r] >> 32);
            const float t = __uint_as_float((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
            const unsigned long long z = ordered((t - a) / sd);
            atomicMax(stats + 51, z);
            atomicMin(stats + 52, z);
        }
        for (int i = 0; i < 4; ++i) {
            unsigned cnt = 0;
            for (int w2 = 0; w2 < 32; ++w2) cnt += redu[i][w2];
            const float ks = k[i] * sd;
            const float hi = a + ks + fmaf(fabsf(a) + ks, SLACK_REL, SLACK_ABS);
            if (hi < M[i]) atomicAdd(stats + 6 + i, 1ull);
            atomicAdd(stats + 10 + i, static_cast<unsigned long long>(cnt));
            atomicMax(stats + 14 + i, static_cast<unsigned long long>(cnt));
            const int b = cnt < 16 ? 0 : cnt < 64 ? 1 : cnt < 256 ? 2 : cnt < 1024 ? 3 : cnt < 4096 ? 4 : cnt < 16384 ? 5 : cnt < 65536 ? 6 : 7;
            atomicAdd(stats + 18 + 8 * i + b, 1ull);
            if (i == 3) atomicMin(stats + 50, static_cast<unsigned long long>(ordered((hi - M[i]) / sd)));
        }
    }
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(1024) tf_head_sieve_check(const float* __restrict__ approx, int V,
        float kappa, const float* __restrict__ mrow, const unsigned* __restrict__ full, const unsigned* __restrict__ mine,
        const unsigned long long* __restrict__ best, unsigned long long* __restrict__ stats) {
    tf_head_sieve::sieve_check<16>(approx, V, kappa, mrow, full, mine, best, stats);
}

extern "C" __global__ void __launch_bounds__(1024) tf_head_sieve_check32(const float* __restrict__ approx, int V,
        float kappa, const float* __restrict__ mrow, const unsigned* __restrict__ full, const unsigned* __restrict__ mine,
        const unsigned long long* __restrict__ best, unsigned long long* __restrict__ stats) {
    tf_head_sieve::sieve_check<32>(approx, V, kappa, mrow, full, mine, best, stats);
}

namespace tf_head_sieve {
// The 32-row prep runs at the 16-row kernel's grid (K / 64, 128 threads); its A fragments and
// per-group sums double, the per-row arithmetic does not change.
} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_prep32(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, int rows, int K, uint4* __restrict__ afr, float* __restrict__ xv,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ pre, unsigned long long* __restrict__ count) {
    tf_head_sieve::prep<32>(x, xs, rows, K, afr, xv, best, pre, count);
}

extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_prep(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, int rows, int K, uint4* __restrict__ afr, float* __restrict__ xv,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ pre, unsigned long long* __restrict__ count) {
    tf_head_sieve::prep<16>(x, xs, rows, K, afr, xv, best, pre, count);
}
