// tf_qmm_mlp_single16's tile for verify windows of 17 to 32 rows: two m16 row tiles share each K-major weight panel
// in one CTA (one weight pass for the window, where the 16-row tiles stream every weight once a tile). 128 gate and
// 128 up columns a CTA; each output keeps tf_qmm_mlp_single16's chain: the same m16n8k16 MMAs over K, the same fmas,
// the same bf16 activation and group sums, so the same bits a row as a 16-row window computes.
#pragma once
#include "qmm_dict.cuh"
namespace tf_qmm_mlp_rows32 {
using namespace tf_qmm_group;
using namespace qmm_dict;
template <int STAGES, int BN, int BM, bool DICT = false>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                     const Parts& parts, __nv_bfloat16* __restrict__ act,
                                     float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    constexpr int GS = 64, WM = 1, WN = 4;
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    constexpr int I = T::MT, J = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x;
    const Part P = parts.p[0];
    const int tile = cid / rows_t;
    const int m0 = cid % rows_t * BM, n0 = tile * (BN / 2), per = KG;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);
    constexpr int TILE_BYTES = 64 * GS / 2;
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
        wsrc[j] = reinterpret_cast<const unsigned char*>(parts.p[t / (BN / 128)].w) +
                  static_cast<size_t>(n0 / 64 + t % (BN / 128)) * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        const auto Q = parts.p[(off * 8) / (BN / 2)];
        ssrc[j] = (which ? Q.biases : Q.scales) + n0 + (off * 8) % (BN / 2);
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const int code0 = DICT ? parts.p[0].first : 0, code1 = DICT ? parts.p[1].first : 0;
    const int bits0 = dict_bits(code0), bits1 = dict_bits(code1);
    const bool pair0 = dict_pair(code0), pair1 = dict_pair(code1);
    const int tab0 = pair0 ? 0 : (dict_ns(code0) + dict_nb(code0) + 7) / 8 * 16;
    const unsigned short* const tabs = reinterpret_cast<const unsigned short*>(buf + STAGES * T::STAGE);
    const int run0 = (pair0 ? 1 : 2) * bits0, run1 = (pair1 ? 1 : 2) * bits1;
    const bool dok = DICT && tid < run0 + run1;
    const int dpart = tid < run0 ? 0 : 1, dchunk = tid < run0 ? tid : tid - run0;
    const int dstep = DICT ? dict_step(dpart ? code1 : code0) : 0;
    const unsigned char* dsrc = DICT ? reinterpret_cast<const unsigned char*>(parts.p[dpart].scales) +
                                       static_cast<size_t>(n0 / 64) * dstep + dchunk * 16
                                     : nullptr;
    const int dpitch = P.npad / 64 * dstep;
    const int ddst = T::X + T::W + (dpart ? 16 * run0 : 0) + dchunk * 16;
    const int mpart = wn / 2, mbits = mpart ? bits1 : bits0, mns = dict_ns(mpart ? code1 : code0);
    const uint32_t mmask = (1u << mbits) - 1u;
    const int mrun = mpart ? 16 * run0 : 0;
    const bool mpair = mpart ? pair1 : pair0;
    const uint32_t* const mgtab = reinterpret_cast<const uint32_t*>(parts.p[mpart].biases);
    const unsigned short* const mtab = tabs + (mpart ? tab0 / 2 : 0);
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    const size_t wstride = static_cast<size_t>(P.npad / 64) * TILE_BYTES;
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
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * wstride);
        if constexpr (DICT) {
            if (dok) cp16(p + ddst, dsrc + static_cast<size_t>(g) * dpitch);
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
    if constexpr (DICT) {
        const int c0 = tab0 / 16, c1 = pair1 ? 0 : (dict_ns(code1) + dict_nb(code1) + 7) / 8;
        for (int c = tid; c < c0 + c1; c += T::THREADS)
            cp16(buf + STAGES * T::STAGE + c * 16, c < c0 ? reinterpret_cast<const unsigned char*>(parts.p[0].biases) + c * 16
                                                          : reinterpret_cast<const unsigned char*>(parts.p[1].biases) + (c - c0) * 16);
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_w(s, s);
        commit();
    }
    grid_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_x(s, s);
        commit();
    }
    grid_launch();
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + mrun);
        uint32_t pe[2] = {0u, 0u};
        if constexpr (DICT) {
            if (mpair) {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const int c = (wn % 2) * 64 + pass * 32 + lane, off = c * mbits;
                    pe[pass] = __ldg(mgtab + (__funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mmask));
                }
            }
        }
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
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
        if constexpr (DICT) {
            float xv[I][2];
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                xv[i][0] = px[row];
                xv[i][1] = px[row + 8];
            }
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                unsigned short s_bits, b_bits;
                if (mpair) {
                    s_bits = static_cast<unsigned short>(pe[pass] & 0xFFFFu);
                    b_bits = static_cast<unsigned short>(pe[pass] >> 16);
                } else {
                    const int c = (wn % 2) * 64 + pass * 32 + lane, tt = c / 64, ct = c % 64;
                    const int offs = (tt * 128 + ct) * mbits, offb = (tt * 128 + 64 + ct) * mbits;
                    const uint32_t idxs = __funnelshift_r(pk[offs >> 5], pk[(offs >> 5) + 1], offs & 31) & mmask;
                    const uint32_t idxb = __funnelshift_r(pk[offb >> 5], pk[(offb >> 5) + 1], offb & 31) & mmask;
                    s_bits = mtab[idxs];
                    b_bits = mtab[mns + idxb];
                }
                const float sf = __bfloat162float(__ushort_as_bfloat16(s_bits));
                const float bf = __bfloat162float(__ushort_as_bfloat16(b_bits));
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = pass * 4 + jj;
                    const int col = jj * 8 + (lane & 3) * 2;
                    const float sv[2] = {__shfl_sync(0xffffffffu, sf, col), __shfl_sync(0xffffffffu, sf, col + 1)};
                    const float bv[2] = {__shfl_sync(0xffffffffu, bf, col), __shfl_sync(0xffffffffu, bf, col + 1)};
#pragma unroll
                    for (int i = 0; i < I; ++i) {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = __fmaf_rn(xv[i][e >> 1], bv[e & 1], __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                const float sv[2] = {__low2float(s2), __high2float(s2)};
                const float bv[2] = {__low2float(b2), __high2float(b2)};
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
    }
    wait<0>();
    // gate's and up's activations of the CTA's columns, gate then up, as tf_qmm_mlp_single16 parks them
    __nv_bfloat16* own = reinterpret_cast<__nv_bfloat16*>(buf);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                const int c = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                *reinterpret_cast<__nv_bfloat162*>(own + r * BN + c) =
                    __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
            }
    __syncthreads();
    if (wn >= WN / 2) {
#pragma unroll
        for (int i = 0; i < I; ++i)
#pragma unroll
            for (int j = 0; j < J; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
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
                    if (m0 + r < M) *reinterpret_cast<__nv_bfloat162*>(act + static_cast<size_t>(m0 + r) * P.n + n0 + c) = o2;
                }
    }
    __syncthreads();
    if (tid < rows * (BN / 128)) {
        const int r = tid / (BN / 128), group = tid % (BN / 128);
        as[static_cast<size_t>(m0 + r) * (P.n / 64) + n0 / 64 + group] = act_group_sum(own + r * BN + group * 64);
    }
}
}

// 32 rows, 128 gate and 128 up columns a CTA (136 CTAs at the 27B's 17408), two stages.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_mlp_single32(
    const __nv_bfloat16* x, const float* xs, const __grid_constant__ tf_qmm_group::Parts parts,
    __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_mlp_rows32::body<2, 256, 32, false>(x, xs, parts, act, sums, M, K, ldx, rows_t);
}

extern "C" __global__ void __launch_bounds__(128) tf_qmm_mlp_single32_dict(
    const __nv_bfloat16* x, const float* xs, const __grid_constant__ tf_qmm_group::Parts parts,
    __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_mlp_rows32::body<2, 256, 32, true>(x, xs, parts, act, sums, M, K, ldx, rows_t);
}
