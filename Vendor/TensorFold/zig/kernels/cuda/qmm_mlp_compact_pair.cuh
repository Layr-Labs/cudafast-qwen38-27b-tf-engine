#pragma once
#include "qmm_mlp_single_dict.cuh"
namespace tf_qmm_mlp_single {
__device__ __forceinline__ uint32_t compact_load_a(const __nv_bfloat16* p) {
    uint32_t word;
    asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(word) : "l"(p) : "memory");
    return word;
}

template <int STAGES = 2, int BN = 256, bool KMAJOR = true>
__device__ __forceinline__ void mlp_single_compact_body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                              const Parts& parts, __nv_bfloat16* __restrict__ act,
                                              float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    constexpr int GS = 64, BM = 16, WM = 1, WN = 4;
    using Base = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    // Gate and up keep the original two-warp column ownership, each with its own ring and sums.
    // Two groups advance independently; they meet only for the existing BF16 activation epilogue.
    struct T : Base {
        enum { X=0, W=BN*GS/4, S=128, STAGE=W+2*S+BM*4, SMEM=2*STAGES*STAGE };
    };
    constexpr int PASSES=BN/WN/32;
    static_assert(T::SMEM==17664,"two independent two-warp compact pair pipelines");
    constexpr int I=T::MT,J=T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int wm=warp/WN,wn=warp%WN,mpart=wn/2,local_tid=tid&63;
    const int KG=K/GS,cid=blockIdx.x,rank=0;
    const Part P=parts.p[rank];
    const int tile=cid/rows_t;
    const int m0=cid%rows_t*BM,n0=tile*(BN/2),per=KG,g0=0;
    auto stage=[&](int st){return buf+(mpart*STAGES+st)*T::STAGE;};
    const int rows=min(BM,M-m0);
    constexpr int TILE_BYTES=64*GS/2;
    constexpr int WC=(T::W/16+63)/64;
    const unsigned char* wsrc[WC];int wdst[WC];bool wok[WC];
#pragma unroll
    for(int j=0;j<WC;++j){
        const int c=local_tid+j*64,t=c/(TILE_BYTES/16),off=c%(TILE_BYTES/16);
        wok[j]=c<T::W/16;
        wsrc[j]=reinterpret_cast<const unsigned char*>(parts.p[mpart].w)+size_t(n0/64+t)*(KMAJOR?1:KG)*TILE_BYTES+off*16;
        wdst[j]=c*16;
    }
    const int code=parts.p[mpart].first,mbits=dict_bits(code),dstep=dict_step(code);
    const bool dok=local_tid<mbits;
    const auto* dsrc=reinterpret_cast<const unsigned char*>(parts.p[mpart].scales)+size_t(n0/64)*dstep+local_tid*16;
    const int dpitch=P.npad/64*dstep;
    const uint32_t mmask=(1u<<mbits)-1u;
    const uint32_t* const mgtab=reinterpret_cast<const uint32_t*>(parts.p[mpart].biases);
    const bool xsok=local_tid<rows;
    const float* xssrc=xs+size_t(m0+min(local_tid,rows-1))*KG;
    auto load_x=[&](int st,int g){if(xsok)cp4(stage(st)+T::W+2*T::S+local_tid*4,xssrc+g);};
    auto load_w=[&](int st,int g){
        auto* dst=stage(st);
#pragma unroll
        for(int j=0;j<WC;++j)if(wok[j])cp16(dst+wdst[j],wsrc[j]+size_t(g)*(KMAJOR?P.npad/64:1)*TILE_BYTES);
        if(dok)cp16(dst+T::W+local_tid*16,dsrc+size_t(g)*dpitch);
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
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_w(s, g0 + s);
        commit();
    }
    grid_wait();
    const __nv_bfloat16* x_ready = x;
    asm volatile("" : "+l"(x_ready) :: "memory");
    qmm_dict::x_keep_l2(x_ready + static_cast<size_t>(m0) * ldx, rows, K, ldx);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_x(s, g0 + s);
        commit();
    }
    grid_launch();
    for (int it = 0; it < per; ++it) {
        wait<0>();
        if(mpart==0)asm volatile("bar.sync 1, 64;" ::: "memory");
        else asm volatile("bar.sync 2, 64;" ::: "memory");
        const int next = it + 1;
        if (next < per) { load_x(next % STAGES, next); load_w(next % STAGES, next); }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W);
        uint32_t pe[PASSES] = {};
        {
#pragma unroll
            for (int pass = 0; pass < PASSES; ++pass) {
                const int c = (wn % 2) * (BN / WN) + pass * 32 + lane, off = (c / 64 * 64 + c % 64) * mbits;
                pe[pass] = __ldg(mgtab + (__funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mmask));
            }
        }
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[T::NT][GS / 32];
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[(((wn % 2) * T::NT + j) * 32 + lane) * (GS / 32) + v];
        uint32_t av[GS / 16][I][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt)
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane >> 2);
                const int c = it * GS + kt * 16 + 2 * (lane & 3);
                const __nv_bfloat16* a0 = x_ready + static_cast<size_t>(m0 + r) * ldx + c;
                const __nv_bfloat16* a1 = a0 + 8 * ldx;
                av[kt][i][0] = compact_load_a(a0);
                av[kt][i][1] = compact_load_a(a1);
                av[kt][i][2] = compact_load_a(a0 + 8);
                av[kt][i][3] = compact_load_a(a1 + 8);
            }
        float xv[I][2];
#pragma unroll
        for (int i = 0; i < I; ++i) {
            const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
            xv[i][0] = px[row];
            xv[i][1] = px[row + 8];
        }
        float d[I][J][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                for (int i = 0; i < I; ++i) {
                    if (kt == 0) mma0(d[i][j], av[kt][i], b0, b1);
                    else mma(d[i][j], av[kt][i], b0, b1);
                }
            }
        }
#pragma unroll
        for (int pass = 0; pass < PASSES; ++pass) {
            const unsigned short s_bits = static_cast<unsigned short>(pe[pass]);
            const unsigned short b_bits = static_cast<unsigned short>(pe[pass] >> 16);
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
    }
    wait<0>();
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
                    __floats2bfloat162_rn(acc[i][j][2*h], acc[i][j][2*h+1]);
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
extern "C" __global__ void __launch_bounds__(128, 3) tf_qmm_mlp_single16_compact_pair(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,
 __nv_bfloat16* act,float* sums,int M,int K,int ldx,int rows_t) {
 tf_qmm_mlp_single::mlp_single_compact_body<>(x,xs,parts,act,sums,M,K,ldx,rows_t);
}
