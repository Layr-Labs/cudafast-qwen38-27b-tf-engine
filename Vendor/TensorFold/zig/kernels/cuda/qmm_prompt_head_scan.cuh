// Prompt-tail-only scan ownership. Decode/verify use tf_head_sieve_scan unchanged.
#pragma once
#include "qmm_head_sieve.cuh"
namespace tf_prompt_head_scan {
using namespace tf_head_sieve;
constexpr int SCAN_WARP_STAGE = 1024 + 256;
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
    unsigned char* tabs = buf + NS * TILES * SCAN_WARP_STAGE;
    for (int c = tid; c < tab_smem / 16; c += 128) qmm_frag::cp16(tabs + c * 16, reinterpret_cast<const unsigned char*>(tab) + c * 16);
    const size_t t0 = static_cast<size_t>(blockIdx.x) * TILES;
    auto stage = [&](int st) { return buf + (warp * NS + st) * SCAN_WARP_STAGE; };
    auto load_w = [&](int st, int g) {
        auto* p=stage(st);
#pragma unroll
        for(int j=0;j<2;++j)
            qmm_frag::cp16(p+(lane+j*32)*16,reinterpret_cast<const unsigned char*>(w2)+((t0+warp)*KG+g)*1024+(lane+j*32)*16);
        if(lane<16) {
            if(code!=0)qmm_frag::cp16(p+1024+lane*16,sb+(t0+warp)*KG*step+(size_t(g)*step&~size_t(63))+lane*16);
            else qmm_frag::cp16(p+1024+lane*16,(lane<8?sc:bi)+size_t(g)*npad+(t0+warp)*64+(lane&7)*8);
        }
    };
    auto load_x = [&](int, int) {}; // immutable prepared fragments/sums read by their consuming warp
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
    qmm_frag::wait<0>();
    __syncthreads(); // shared immutable pair table initialized across all four warps
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
        __syncwarp();
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
            const uint4 v = afr[size_t(g)*128+kt*32+lane];
            a[kt][0] = v.x;
            a[kt][1] = v.y;
            a[kt][2] = v.z;
            a[kt][3] = v.w;
        }
        const uint4 wlo = reinterpret_cast<const uint4*>(p)[lane];
        const uint4 whi = reinterpret_cast<const uint4*>(p+512)[lane];
        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + 1024);
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
        const float* pv = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(xv + size_t(g)*32));
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

} // namespace tf_prompt_head_scan

extern "C" __global__ void __launch_bounds__(128, 3) tf_head_sieve_scan_prompt(const uint32_t* __restrict__ w2,
        const uint4* __restrict__ afr, const float* __restrict__ xv, const unsigned char* __restrict__ sb,
        const uint32_t* __restrict__ tab, int code, int tab_smem, const __nv_bfloat16* __restrict__ sc,
        const __nv_bfloat16* __restrict__ bi, int npad, int KG, int rows, float kappa, float* __restrict__ himax,
        float* __restrict__ himid, float* __restrict__ approx) {
    tf_prompt_head_scan::scan<tf_head_sieve::STAGES>(w2, afr, xv, sb, tab, code, tab_smem, sc, bi, npad, KG, rows, kappa, himax, himid, approx);
}
