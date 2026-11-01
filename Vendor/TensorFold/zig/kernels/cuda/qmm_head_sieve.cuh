// The greedy verify head's sieve (head_sieve.zig): a 2-bit copy of the head's codes (code >> 2, the same scales and
// biases) gives each (row, id) an approximate logit a and the spread s of the dropped bits' term (sum over groups of
// scale^2 times the group's sum of squares, times 1.25, the variance of a uniform 2-bit residual about its middle 1.5).
// Each row's first tile (the n8 tile of its best a) is recomputed exactly; the row's threshold is the lower of that
// tile's exact best and its best a, and the 8-id column groups whose best a + k s reaches it are recomputed with the
// head tile's own loads, MMA chain, fmas and bf16 rounding (group_kernel_dict's AMAX tile). Each row's token is the
// best of those exact values: the highest bf16 logit, ties to the lowest id, the head merge's rule. A row's true best
// u escapes only if its dropped bits' term exceeds that of the first tile's best-a id g by more than k s_u (the
// 2-bit terms of high logits move together, so the margin is on their difference).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_head_sieve {

using namespace qmm_frag;

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
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_pack(const uint32_t* __restrict__ w, uint32_t* __restrict__ w2) {
    const size_t blk = blockIdx.x;
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
    w2[blk * 256 + (nt >> 2) * 128 + lane * 4 + (nt & 3)] = out;
}

// The rows' A fragments by (group, k step, lane), zero past `rows`; each group's input sums (xs, zero past rows) and
// sums of squares; the rows' bests, their first tiles and the candidate count cleared. Grid K / 64, 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_prep(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, int rows, int K, uint4* __restrict__ afr, float* __restrict__ xv,
        unsigned long long* __restrict__ best, unsigned long long* __restrict__ pre, int* __restrict__ count) {
    qmm_frag::grid_wait();
    const int g = blockIdx.x, KG = gridDim.x, tid = threadIdx.x, lane = tid & 31, kt = tid >> 5;
    const int r = lane >> 2, h = K / 2, k2 = (g * 64 + kt * 16 + (lane & 3) * 2) / 2;
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(x);
    uint4 a;
    a.x = r < rows ? x32[static_cast<size_t>(r) * h + k2] : 0u;
    a.y = r + 8 < rows ? x32[static_cast<size_t>(r + 8) * h + k2] : 0u;
    a.z = r < rows ? x32[static_cast<size_t>(r) * h + k2 + 4] : 0u;
    a.w = r + 8 < rows ? x32[static_cast<size_t>(r + 8) * h + k2 + 4] : 0u;
    afr[(g * 4 + kt) * 32 + lane] = a;
    const int row = tid >> 3, part = tid & 7;   // eight threads a row, eight inputs each
    float s2 = 0.0f;
    if (row < rows) {
        const uint4 v = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * K + g * 64 + part * 8);
        const uint32_t q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = __uint_as_float(q[i] << 16), hi = __uint_as_float(q[i] & 0xFFFF0000u);
            s2 = fmaf(lo, lo, s2);
            s2 = fmaf(hi, hi, s2);
        }
    }
    s2 += __shfl_xor_sync(0xffffffffu, s2, 1);
    s2 += __shfl_xor_sync(0xffffffffu, s2, 2);
    s2 += __shfl_xor_sync(0xffffffffu, s2, 4);
    if (part == 0) {
        xv[g * 32 + row] = row < rows ? xs[static_cast<size_t>(row) * KG + g] : 0.0f;
        xv[g * 32 + 16 + row] = s2;
    }
    if (g == 0 && tid < tf_head_sieve::ROWS) {
        best[tid] = 0ull;
        pre[tid] = 0ull;
    }
    if (g == 0 && tid == 0) *count = 0;
    qmm_frag::grid_launch();
}

// The sieve: a CTA's four 64-id tiles, a warp each, through every group of the 2-bit copy (bf16 MMA on the rows' A
// fragments, the pair dictionary's scales and biases). Each (row, n8 tile)'s best a + k s + slack into himax
// [n8][16], its best a into himid [n8][16]; approx (when set) gets every (id, row)'s a and s
// ([id][32]: a for 16 rows, then s). Pair-coded heads (code) read the dictionary, others (code 0) the (kg, npad)
// scale and bias arrays. Grid tiles / 4, 128 threads; shared SMEM + tab_smem (the pair table, or 0: through L2).
namespace tf_head_sieve {

template <int NS>
__device__ __forceinline__ void scan(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int bits = code & 31, step = 8 * bits;
    const uint32_t imask = (1u << bits) - 1u;
    unsigned char* tabs = buf + NS * STAGE;
    for (int c = tid; c < tab_smem / 16; c += 128) qmm_frag::cp16(tabs + c * 16, reinterpret_cast<const unsigned char*>(tab) + c * 16);
    const size_t t0 = static_cast<size_t>(blockIdx.x) * TILES;
    auto stage = [&](int s) { return buf + s * STAGE; };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = tid + j * 128, t = c >> 6, off = c & 63;
            qmm_frag::cp16(p + AFR + t * 1024 + off * 16, reinterpret_cast<const unsigned char*>(w2) + ((t0 + t) * KG + g) * 1024 + off * 16);
        }
        if (tid < 64) {
            const int t = tid >> 4, part = tid & 15;
            if (code != 0)
                qmm_frag::cp16(p + AFR + W2 + t * 256 + part * 16,
                               sb + (t0 + t) * KG * step + (static_cast<size_t>(g) * step & ~static_cast<size_t>(63)) + part * 16);
            else   // no pair code: the tile's 64 scales, then its 64 biases, from the (kg, npad) arrays
                qmm_frag::cp16(p + AFR + W2 + t * 256 + part * 16,
                               (part < 8 ? sc : bi) + static_cast<size_t>(g) * npad + (t0 + t) * 64 + (part & 7) * 8);
        }
    };
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
        qmm_frag::cp16(p + tid * 16, afr + static_cast<size_t>(g) * 128 + tid);
        if (tid < 8) qmm_frag::cp16(p + AFR + W2 + DW + tid * 16, xv + static_cast<size_t>(g) * 32 + tid * 4);
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
    float acc[8][4], sg[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc[i][e] = 0.0f;
            sg[i][e] = 0.0f;
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
        uint32_t a[4][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const uint4 v = reinterpret_cast<const uint4*>(p)[kt * 32 + lane];
            a[kt][0] = v.x;
            a[kt][1] = v.y;
            a[kt][2] = v.z;
            a[kt][3] = v.w;
        }
        const uint4 wlo = reinterpret_cast<const uint4*>(p + AFR + warp * 1024)[lane];
        const uint4 whi = reinterpret_cast<const uint4*>(p + AFR + warp * 1024 + 512)[lane];
        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + AFR + W2 + warp * 256);
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
        const float* pv = reinterpret_cast<const float*>(p + AFR + W2 + DW);
        const float xs0 = pv[r0], xs1 = pv[r0 + 8], q0 = pv[16 + r0], q1 = pv[16 + r0 + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const uint4 wq = nt < 4 ? wlo : whi;
            const uint32_t wv = (nt & 3) == 0 ? wq.x : (nt & 3) == 1 ? wq.y : (nt & 3) == 2 ? wq.z : wq.w;
            float d[4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const uint32_t b0 = pair2(wv, kt * 4), b1 = pair2(wv, kt * 4 + 2);
                if (kt == 0) qmm_frag::mma0(d, a[kt], b0, b1);
                else qmm_frag::mma(d, a[kt], b0, b1);
            }
            const int src = (nt & 3) * 8 + (lane & 3) * 2;
            const uint32_t eh = nt < 4 ? ent[0] : ent[1];
            const uint32_t ea = __shfl_sync(0xffffffffu, eh, src), eb = __shfl_sync(0xffffffffu, eh, src + 1);
            const float s0 = __uint_as_float(ea << 16), c0 = __uint_as_float(ea & 0xFFFF0000u);
            const float s1 = __uint_as_float(eb << 16), c1 = __uint_as_float(eb & 0xFFFF0000u);
            const float m0 = fmaf(1.5f, s0, c0), m1 = fmaf(1.5f, s1, c1);
            acc[nt][0] = fmaf(4.0f * s0, d[0], fmaf(m0, xs0, acc[nt][0]));
            acc[nt][1] = fmaf(4.0f * s1, d[1], fmaf(m1, xs0, acc[nt][1]));
            acc[nt][2] = fmaf(4.0f * s0, d[2], fmaf(m0, xs1, acc[nt][2]));
            acc[nt][3] = fmaf(4.0f * s1, d[3], fmaf(m1, xs1, acc[nt][3]));
            sg[nt][0] = fmaf(s0 * s0, q0, sg[nt][0]);
            sg[nt][1] = fmaf(s1 * s1, q0, sg[nt][1]);
            sg[nt][2] = fmaf(s0 * s0, q1, sg[nt][2]);
            sg[nt][3] = fmaf(s1 * s1, q1, sg[nt][3]);
        }
    }
    qmm_frag::wait<0>();
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        float hi[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float sd = sqrtf(1.25f * sg[nt][e]), ks = kappa * sd, v = acc[nt][e];
            hi[e] = v + ks + fmaf(fabsf(v) + ks, SLACK_REL, SLACK_ABS);
            if (approx != nullptr) {
                const size_t id = (t0 + warp) * 64 + nt * 8 + (lane & 3) * 2 + (e & 1);
                const int row = r0 + (e >> 1) * 8;
                approx[id * 32 + row] = v;
                approx[id * 32 + 16 + row] = sd;
            }
        }
        float h0 = fmaxf(hi[0], hi[1]), h1 = fmaxf(hi[2], hi[3]);
        h0 = fmaxf(h0, __shfl_xor_sync(0xffffffffu, h0, 1));
        h0 = fmaxf(h0, __shfl_xor_sync(0xffffffffu, h0, 2));
        h1 = fmaxf(h1, __shfl_xor_sync(0xffffffffu, h1, 1));
        h1 = fmaxf(h1, __shfl_xor_sync(0xffffffffu, h1, 2));
        float m0 = fmaxf(acc[nt][0], acc[nt][1]), m1 = fmaxf(acc[nt][2], acc[nt][3]);
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        if ((lane & 3) == 0) {
            const size_t n8 = (t0 + warp) * 8 + nt;
            if (r0 < rows) {
                himax[n8 * 16 + r0] = h0;
                himid[n8 * 16 + r0] = m0;
            }
            if (r0 + 8 < rows) {
                himax[n8 * 16 + r0 + 8] = h1;
                himid[n8 * 16 + r0 + 8] = m1;
            }
        }
    }
}

} // namespace tf_head_sieve

extern "C" __global__ void __launch_bounds__(128, 3) tf_head_sieve_scan(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx) {
    tf_head_sieve::scan<tf_head_sieve::STAGES>(w2, afr, xv, sb, tab, code, tab_smem, sc, bi, npad, KG, rows, kappa, himax, himid, approx);
}

// Each row's first tile: the n8 tile of its highest value in h (himid: the best a; key: then the lower tile) into
// pre[row], for an exact value to bound the row's best from below. Grid any, 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_top(const float* __restrict__ himax, int rows, int n8s,
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
        const float4* h = reinterpret_cast<const float4*>(himax + static_cast<size_t>(t) * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = h[i];
            const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (vv[j] > bv[i * 4 + j]) {
                    bv[i * 4 + j] = vv[j];
                    bt[i * 4 + j] = t;
                }
        }
    }
    __shared__ unsigned long long sk[8][16];
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

// Each row's threshold: the lower of the exact best value its first tile gave (best[row], a lower bound of the row's
// top logit) and that tile's best a (himid at pre[row]'s tile); then every n8 tile some row's himax reaches into list
// (count past cap: an overflow, and the rescoring takes every tile). mrow (when set) gets the thresholds. A row whose
// best is unset takes every tile.
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_select(const float* __restrict__ himax,
        const float* __restrict__ himid, const unsigned long long* __restrict__ pre,
        const unsigned long long* __restrict__ best, int rows, int n8s, int* __restrict__ list, int* __restrict__ count,
        int cap, float* __restrict__ mrow) {
    qmm_frag::grid_wait();
    __shared__ float M[16];
    const int tid = threadIdx.x, lane = tid & 31;
    if (tid < 16) {
        const unsigned long long k = best[tid], f = pre[tid];
        const uint32_t u = static_cast<uint32_t>(k >> 32);
        float v = k == 0ull || f == 0ull ? tf_head_sieve::neg_inf() : __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
        if (k != 0ull && f != 0ull) {
            const size_t t = 0xFFFFFFFFu - static_cast<uint32_t>(f);
            v = fminf(v, himid[t * 16 + tid]);
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
            const float4* h = reinterpret_cast<const float4*>(himax + static_cast<size_t>(t) * 16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = h[i];
                const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) hit |= (i * 4 + j < rows) && vv[j] >= M[i * 4 + j];
            }
        }
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (mask != 0u) {
            int at = 0;
            if (lane == 0) at = atomicAdd(count, __popc(mask));
            at = __shfl_sync(0xffffffffu, at, 0);
            if (hit) {
                const int pos = at + __popc(mask & ((1u << lane) - 1u));
                if (pos < cap) list[pos] = t;
            }
        }
    }
}

// The listed n8 tiles (every tile on an overflow; with keys, the rows' first tiles) exactly: a warp a tile, the head tile's words, A fragments, MMA chain
// (mma0 then three mma a group), acc = fma(xs, b, fma(d, s, acc)) a group in order, bf16 rounding, then each row's best
// of the tile into best[row] (key: the higher value, then the lower id). Grid any, 128 threads, RS_SMEM shared.
namespace tf_head_sieve {
// tf_head_sieve_rescore's body; DENSE: every (row, id) exact value of a listed tile also into dense[row * ld + id].
template <bool DENSE>
__device__ __forceinline__ void rescore_body(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const int* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best, __nv_bfloat16* __restrict__ dense, int ld) {
    extern __shared__ __align__(128) unsigned char buf[];
    qmm_frag::grid_wait();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = keys != nullptr ? rows : *count;
    const bool all = keys == nullptr && n > cap;
    const int items = all ? n8s : n;
    const int r0 = lane >> 2;
    qmm_frag::grid_launch();
    for (int base = blockIdx.x * 4; base < items; base += gridDim.x * 4) {
        const int it = base + warp;
        const bool live = it < items;
        const int t = !live ? 0 : keys != nullptr ? static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(keys[it])) : all ? it : list[it];
        const size_t tile = static_cast<size_t>(t >> 3);
        const int nt = t & 7;
        const unsigned char* wsrc = reinterpret_cast<const unsigned char*>(w) + tile * KG * 2048 + nt * 256;
        const size_t col0 = tile * 64 + nt * 8;
        auto load = [&](int s, int g) {
            unsigned char* p = buf + s * RS_STAGE;
            qmm_frag::cp16(p + tid * 16, afr + static_cast<size_t>(g) * 128 + tid);
            if (live) {
                if (lane < 16) qmm_frag::cp16(p + AFR + warp * RS_WARP + lane * 16, wsrc + static_cast<size_t>(g) * 2048 + lane * 16);
                else if (lane == 16) qmm_frag::cp16(p + AFR + warp * RS_WARP + 256, scales + static_cast<size_t>(g) * npad + col0);
                else if (lane == 17) qmm_frag::cp16(p + AFR + warp * RS_WARP + 272, biases + static_cast<size_t>(g) * npad + col0);
            }
            if (tid >= 64 && tid < 68) qmm_frag::cp16(p + AFR + 4 * RS_WARP + (tid - 64) * 16, xv + static_cast<size_t>(g) * 32 + (tid - 64) * 4);
        };
#pragma unroll
        for (int s = 0; s < RS_STAGES - 1; ++s) {
            if (s < KG) load(s, s);
            qmm_frag::commit();
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int g = 0; g < KG; ++g) {
            qmm_frag::wait<RS_STAGES - 2>();
            __syncthreads();
            const int next = g + RS_STAGES - 1;
            if (next < KG) load(next % RS_STAGES, next);
            qmm_frag::commit();
            if (live) {
                const unsigned char* p = buf + (g % RS_STAGES) * RS_STAGE;
                const uint2 wv = reinterpret_cast<const uint2*>(p + AFR + warp * RS_WARP)[lane];
                const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + AFR + warp * RS_WARP + 256);
                const float* px = reinterpret_cast<const float*>(p + AFR + 4 * RS_WARP);
                float d[4];
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const uint4 v = reinterpret_cast<const uint4*>(p)[kt * 32 + lane];
                    const uint32_t a[4] = {v.x, v.y, v.z, v.w};
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
        }
        qmm_frag::wait<0>();
        __syncthreads();
        if (live) {
            const int col = static_cast<int>(col0) + (lane & 3) * 2;
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
                if constexpr (DENSE) {
                    if (row < rows) {
                        __nv_bfloat162 pv;
                        pv.x = __float2bfloat16_rn(acc[2 * hh]);
                        pv.y = __float2bfloat16_rn(acc[2 * hh + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(dense + static_cast<size_t>(row) * ld + col) = pv;
                    }
                }
            }
        }
    }
}
} // namespace tf_head_sieve

// The listed n8 tiles (every tile on an overflow; with keys, the rows' first tiles) exactly, each row's best into best.
extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_rescore(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const int* __restrict__ count, int cap, int n8s, const unsigned long long* __restrict__ keys,
        unsigned long long* __restrict__ best) {
    tf_head_sieve::rescore_body<false>(w, scales, biases, npad, afr, xv, KG, rows, list, count, cap, n8s, keys, best, nullptr, 0);
}

// The drafter frame's span head (head_sieve.zig spanTop4): tf_head_sieve_rescore's tiles, every (row, id) exact
// value, rounded to bf16 as the head's logits are, also into dense[row * ld + id] for the frame's top-k.
extern "C" __global__ void __launch_bounds__(128) tf_head_sieve_rescore_dense(const uint32_t* __restrict__ w,
        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases, int npad,
        const uint4* __restrict__ afr, const float* __restrict__ xv, int KG, int rows, const int* __restrict__ list,
        const int* __restrict__ count, int cap, int n8s, unsigned long long* __restrict__ best,
        __nv_bfloat16* __restrict__ dense, int ld) {
    tf_head_sieve::rescore_body<true>(w, scales, biases, npad, afr, xv, KG, rows, list, count, cap, n8s, nullptr, best, dense, ld);
}

// The frame's packed span logits set to -inf before the rescoring fills its candidates: `cols` of each of `rows`
// rows `ld` apart, eight a thread. Grid (cols / 2048, rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_fill(__nv_bfloat16* __restrict__ dense, int ld, int cols) {
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    const int c = (blockIdx.x * 256 + threadIdx.x) * 8;
    if (c < cols) *reinterpret_cast<uint4*>(dense + static_cast<size_t>(blockIdx.y) * ld + c) = make_uint4(0xff80ff80u, 0xff80ff80u, 0xff80ff80u, 0xff80ff80u);
}

// Each row's four n8 tiles of highest best a (himid; ties to the lower tile) into list[row * 4 + i], their count
// (rows * 4) into count, and the fourth's best a into fourth[row]. Grid rows, 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_top4(const float* __restrict__ himid, int rows, int n8s,
        int* __restrict__ list, int* __restrict__ count, float* __restrict__ fourth) {
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned long long k[4] = {0ull, 0ull, 0ull, 0ull};  // a thread's four best keys, descending
    for (int t = tid; t < n8s; t += 256) {
        const unsigned long long kk = tf_head_sieve::key(himid[static_cast<size_t>(t) * 16 + row], t);
        if (kk > k[3]) {
            k[3] = kk;
#pragma unroll
            for (int i = 3; i > 0; --i)
                if (k[i] > k[i - 1]) {
                    const unsigned long long x = k[i];
                    k[i] = k[i - 1];
                    k[i - 1] = x;
                }
        }
    }
    __shared__ unsigned long long wk[8];
    int next = 0;
#pragma unroll 1
    for (int round = 0; round < 4; ++round) {
        unsigned long long m = next < 4 ? k[next] : 0ull;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const unsigned long long y = __shfl_xor_sync(0xffffffffu, m, o);
            m = y > m ? y : m;
        }
        if (lane == 0) wk[warp] = m;
        __syncthreads();
        unsigned long long top = wk[0];
        for (int i = 1; i < 8; ++i) top = wk[i] > top ? wk[i] : top;
        __syncthreads();
        if (next < 4 && k[next] == top && top != 0ull) ++next;  // keys are unique: one thread owns it
        if (tid == 0) {
            list[row * 4 + round] = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(top));
            if (round == 3) {
                const uint32_t u = static_cast<uint32_t>(top >> 32);
                fourth[row] = __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
            }
        }
    }
    if (row == 0 && tid == 0) *count = rows * 4;
}

// tf_head_sieve_select for a row's four best: its threshold the lower of the fourth best exact value over its four
// first tiles (dense, which tf_head_sieve_rescore_dense filled from list4) and the fourth tile's best a (fourth);
// every n8 tile some row's himax reaches into list. Grid any, 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_head_sieve_select4(const float* __restrict__ himax,
        const __nv_bfloat16* __restrict__ dense, int ld, const int* __restrict__ list4, const float* __restrict__ fourth,
        int rows, int n8s, int* __restrict__ list, int* __restrict__ count, int cap) {
    qmm_frag::grid_wait();
    __shared__ float M[16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid < 16) M[tid] = tf_head_sieve::neg_inf();
    __syncthreads();
    for (int row = warp; row < rows; row += 8) {
        // lane: id (lane & 7) of the row's (lane >> 3)-th tile; the fourth highest (value, lane) of the 32
        const int t = list4[row * 4 + (lane >> 3)];
        const float v = __bfloat162float(dense[static_cast<size_t>(row) * ld + t * 8 + (lane & 7)]);
        unsigned long long mine = (static_cast<unsigned long long>(tf_head_sieve::ordered(v)) << 32) | static_cast<unsigned>(31 - lane);
        unsigned long long m = 0ull;
#pragma unroll 1
        for (int i = 0; i < 4; ++i) {
            m = mine;
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const unsigned long long y = __shfl_xor_sync(0xffffffffu, m, o);
                m = y > m ? y : m;
            }
            if (i < 3 && mine == m) mine = 0ull;
        }
        if (lane == 0) {
            const uint32_t u = static_cast<uint32_t>(m >> 32);
            const float e4 = __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
            M[row] = fminf(e4, fourth[row]);
        }
    }
    __syncthreads();
    qmm_frag::grid_launch();
    for (int base = blockIdx.x * 256; base < n8s; base += gridDim.x * 256) {
        const int t = base + tid;
        bool hit = false;
        if (t < n8s) {
            const float4* h = reinterpret_cast<const float4*>(himax + static_cast<size_t>(t) * 16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = h[i];
                const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) hit |= (i * 4 + j < rows) && vv[j] >= M[i * 4 + j];
            }
        }
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (mask != 0u) {
            int at = 0;
            if (lane == 0) at = atomicAdd(count, __popc(mask));
            at = __shfl_sync(0xffffffffu, at, 0);
            if (hit) {
                const int pos = at + __popc(mask & ((1u << lane) - 1u));
                if (pos < cap) list[pos] = t;
            }
        }
    }
}

// Each row's token from its best; the round's counters: stats[0] rounds, [1] rescored n8 tiles, [2] their most,
// [3] overflows, [4] rows. One warp.
extern "C" __global__ void __launch_bounds__(32) tf_head_sieve_finish(const unsigned long long* __restrict__ best,
        int rows, unsigned* __restrict__ toks, const int* __restrict__ count, int cap, int n8s,
        unsigned long long* __restrict__ stats) {
    qmm_frag::grid_wait();
    const int tid = threadIdx.x;
    if (tid < rows) toks[tid] = 0xFFFFFFFFu - static_cast<unsigned>(best[tid] & 0xFFFFFFFFull);
    if (tid == 0) {
        const int n = *count;
        const unsigned long long items = n > cap ? static_cast<unsigned long long>(n8s) : static_cast<unsigned long long>(n);
        stats[0] += 1;
        stats[1] += items;
        if (items > stats[2]) stats[2] = items;
        if (n > cap) stats[3] += 1;
        stats[4] += static_cast<unsigned long long>(rows);
    }
}

// TF_HEAD_SIEVE=check, a block a row, against the full head's token `full` and the row's threshold mrow: stats[5] tokens
// differing from the sieve's (`mine`); for k = kappa - 3 .. kappa (index i): [6 + i] rows whose true token falls outside
// the column candidates (a + k s + slack >= threshold),
// [10 + i] the candidates' sum, [14 + i] their most, [18 + 8 i + b] a histogram (b: < 16, 64, 256, 1024, 4096, 16384,
// 65536, more); [50] the least margin at kappa of the true token, (a + k s + slack - threshold) / s, as ordered bits;
// [51] and [52] the most and least of the true token's (exact - a) / s, as ordered bits (rows whose tokens agree).
extern "C" __global__ void __launch_bounds__(1024) tf_head_sieve_check(const float* __restrict__ approx, int V,
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
        const float a = approx[static_cast<size_t>(v) * 32 + r], sd = approx[static_cast<size_t>(v) * 32 + 16 + r];
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
        const float a = approx[static_cast<size_t>(u) * 32 + r], sd = approx[static_cast<size_t>(u) * 32 + 16 + r];
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
