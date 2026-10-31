// Exact sixteen-row attention Q/K/V: smaller column panels and a deep weight/index ring.
#pragma once
#include "qmm_dict.cuh"
namespace tf_qmm_attention_stream {
using namespace tf_qmm_group;
__device__ __forceinline__ uint32_t load_a(const __nv_bfloat16* p) {
    uint32_t word;
    asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(word) : "l"(p) : "memory");
    return word;
}
template<int GS, int BN, int STAGES>
struct GemvTile : qmm_frag::LaneTile<GS,16,BN,1,4,STAGES> {
    using Base = qmm_frag::LaneTile<GS,16,BN,1,4,STAGES>;
    static constexpr int X=0, STAGE=Base::W+2*Base::S+Base::XS, SMEM=STAGES*STAGE;
};
template <int GS, int BN, int STAGES, int FutureGroups, bool DICT, typename Map, typename After>
__device__ __forceinline__ void wide_items(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                           int K, int ldx, int items, Map map, float (&acc)[GemvTile<GS, BN, STAGES>::NT][4],
                                           After after) {
    using T = GemvTile<GS, BN, STAGES>;
    static_assert(DICT && GS==64 && BN==128 && STAGES==6 && FutureGroups==0, "fixed attention stream geometry");
    static_assert(!DICT || (FutureGroups == 0 && GS == 64 && BN % 64 == 0 && 2 * T::S == BN / 64 * 256), "coded stages");
    const int npad = DICT ? P.tiles : P.npad;
    static_assert(T::MT == 1 && T::THREADS == 128, "one m16 row tile, four column warps");
    constexpr int NT = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(16, M);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    size_t woff[WC];
    int soff[SC], wdst[WC], sdst[SC];
    bool wok[WC], sok[SC], sbias[SC];
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        woff[j] = static_cast<size_t>(t) * TILE_BYTES + off * 16;
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
    auto load_x = [&](int s, int i) {
        const int g = map(i).x;
        unsigned char* p = stage(s);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int i) {
        const int2 at = map(i);
        const int g = at.x, n0 = at.y;
        unsigned char* p = stage(s);
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(P.w) +
                                  static_cast<size_t>(n0 / 64) * TILE_BYTES + (n0 % 64) * GS / 2;
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * (npad/64) * TILE_BYTES);
        if constexpr (DICT) {
            // the CTA's tiles' index windows: tile n0 / 64 + c / 16, chunk c % 16, from each group's 64-byte line
            const int step = qmm_dict::dict_step(P.npad);
            const size_t window = static_cast<size_t>(g) * step & ~static_cast<size_t>(63);
            if (tid < BN / 4)
                cp16(p + T::X + T::W + tid * 16, reinterpret_cast<const unsigned char*>(P.scales) +
                     static_cast<size_t>(n0 / 64 + tid / 16) * KG * step + window + (tid % 16) * 16);
        } else {
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
        }
    };
    const unsigned char* table = reinterpret_cast<const unsigned char*>(P.biases);
    const unsigned char* dec = nullptr; // BN>=128 keeps decoded scale and bias bits in registers.
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    grid_wait();
    const __nv_bfloat16* x_ready = x;
    asm volatile("" : "+l"(x_ready) :: "memory");
    qmm_dict::x_keep_l2(x_ready, rows, K, ldx);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) { load_w(s, s); load_x(s, s); }
        commit();
    }
    grid_launch();
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
        // the lane's BN / 64 values of its warp's BN / 4 columns: value q is column q % (BN / 4)'s scale, then bias;
        // fetched before the MMAs (the immutable global table lookup can overlap them). A pair
        // code (qmm_dict.cuh, FORMAT.md): the column's one index, the entry's scale or bias half
        [[maybe_unused]] uint32_t dv[BN / 64];
        if constexpr (DICT) {
            const int code = P.npad, bits = qmm_dict::dict_bits(code), step = qmm_dict::dict_step(code), g = map(it).x;
            const bool pair = qmm_dict::dict_pair(code);
            const uint32_t* gtab = reinterpret_cast<const uint32_t*>(P.biases);
            const uint32_t mask_bits = (1u << bits) - 1u;
            if constexpr (BN >= 128) {
                if (pair) {
#pragma unroll
                    for (int pass = 0; pass < BN / 128; ++pass) {
                        const int col = wn * (BN / 4) + pass * 32 + lane;
                        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + (col / 64) * 256);
                        const int off = (g * step & 63) * 8 + (col % 64) * bits;
                        const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mask_bits;
                        const uint32_t entry = qmm_dict::dict_pair_entry(table, gtab, idx);
                        dv[pass] = entry & 0xFFFFu;
                        dv[BN / 128 + pass] = entry >> 16;
                    }
                } else {
#pragma unroll
                    for (int v = 0; v < BN / 64; ++v) {
                        const int q = lane + 32 * v, half = q / (BN / 4), col = wn * (BN / 4) + q % (BN / 4);
                        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + (col / 64) * 256);
                        const int off = (g * step & 63) * 8 + (half * 64 + col % 64) * bits;
                        const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mask_bits;
                        dv[v] = __ldg(reinterpret_cast<const unsigned short*>(table) + (half ? qmm_dict::dict_ns(code) : 0) + idx);
                    }
                }
            } else {
                const int q = lane, half = q / 16, col = wn * 16 + (q & 15);
                const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W);
                const int off = (g * step & 63) * 8 + (pair ? col : half * 64 + col) * bits;
                const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mask_bits;
                dv[0] = pair ? qmm_dict::dict_pair_half(qmm_dict::dict_pair_entry(table, gtab, idx), half * 16)
                             : __ldg(reinterpret_cast<const unsigned short*>(table) + (half ? qmm_dict::dict_ns(code) : 0) + idx);
            }
        }
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(DICT ? dec : p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[NT][GS / 32];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * NT + j) * 32 + lane) * (GS / 32) + v];
        float d[NT][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            const int r = lane >> 2, c = map(it).x * GS + kt * 16 + 2 * (lane & 3);
            const __nv_bfloat16* a0 = x_ready + static_cast<size_t>(r) * ldx + c;
            const __nv_bfloat16* a1 = a0 + 8 * ldx;
            a[0] = load_a(a0); a[1] = load_a(a1);
            a[2] = load_a(a0 + 8); a[3] = load_a(a1 + 8);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
                if (kt == 0) mma0(d[j], a, b0, b1);
                else mma(d[j], a, b0, b1);
            }
        }
        const int row = lane >> 2;
        const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float sv[2], bv[2];
            if constexpr (DICT) {
                if constexpr (BN >= 128) {
                    const int pass = j / 4, jj = j % 4, c0 = jj * 8 + (lane & 3) * 2;
                    const float sf = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(dv[pass])));
                    const float bf = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(dv[BN / 128 + pass])));
                    sv[0] = __shfl_sync(0xffffffffu, sf, c0);
                    sv[1] = __shfl_sync(0xffffffffu, sf, c0 + 1);
                    bv[0] = __shfl_sync(0xffffffffu, bf, c0);
                    bv[1] = __shfl_sync(0xffffffffu, bf, c0 + 1);
                } else {
                    const int c0 = j * 8 + (lane & 3) * 2;
                    const float dval = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(dv[0])));
                    sv[0] = __shfl_sync(0xffffffffu, dval, c0);
                    sv[1] = __shfl_sync(0xffffffffu, dval, c0 + 1);
                    bv[0] = __shfl_sync(0xffffffffu, dval, 16 + c0);
                    bv[1] = __shfl_sync(0xffffffffu, dval, 16 + c0 + 1);
                }
            } else {
                const int col = wn * (BN / 4) + j * 8 + (lane & 3) * 2;
                const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
                const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
                sv[0] = __low2float(s2);
                sv[1] = __high2float(s2);
                bv[0] = __low2float(b2);
                bv[1] = __high2float(b2);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[j][e]));
        }
        after(it);
    }
}

__device__ __forceinline__ void body(const __nv_bfloat16* x, const float* xs, const Parts& parts, int M, int K, int ldx) {
    constexpr int BN=128, STAGES=6, NT=BN/4/8;
    static_assert(GemvTile<64,BN,STAGES>::SMEM == 28032);
    int which=0;
    for(int i=1;i<3;++i) if(int(blockIdx.x)>=parts.p[i].first) which=i;
    const Part p=parts.p[which];
    const int tile=int(blockIdx.x)-p.first, KG=K/64, per=KG/p.sk;
    float acc[NT][4]{}, total[NT][4]{};
    wide_items<64,BN,STAGES,0,true>(x,xs,p,M,K,ldx,KG,[&](int g){return make_int2(g,tile*BN);},acc,[&](int g){
        if((g+1)%per!=0)return;
        const bool first=g+1==per;
#pragma unroll
        for(int j=0;j<NT;++j)
#pragma unroll
            for(int e=0;e<4;++e){total[j][e]=first?acc[j][e]:total[j][e]+acc[j][e];acc[j][e]=0;}
        if(g!=KG-1)return;
        tf_qmm_gdn_stream::gemv_store<BN,NT>(p,M,tile*BN,[&](int j,int h,int,int){return make_float2(total[j][2*h],total[j][2*h+1]);});
    });
}
}
extern "C" __global__ void __launch_bounds__(128,3) tf_qmm_attention_stream16_dict(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx) {
 tf_qmm_attention_stream::body(x,xs,parts,M,K,ldx);
}
// Deterministic finite nonzero BF16 data, used only to validate immutable projection views at initialization.
extern "C" __global__ void tf_attention_validate_fill(unsigned short* x,int n) {
 const int at=blockIdx.x*blockDim.x+threadIdx.x;
 if(at<n){ uint32_t bits=static_cast<uint32_t>(at)*1664525u+1013904223u; bits^=bits>>13; bits*=0x5bd1e995u; bits^=bits>>15; x[at]=static_cast<unsigned short>((0x3d00u+(bits&127u))|((bits&128u)<<8)); }
}
extern "C" __global__ void tf_attention_validate_equal(const unsigned short* a,const unsigned short* b,int n,unsigned int* bad) {
 const int at=blockIdx.x*blockDim.x+threadIdx.x;
 if(at<n && a[at]!=b[at])atomicAdd(bad,1u);
}
