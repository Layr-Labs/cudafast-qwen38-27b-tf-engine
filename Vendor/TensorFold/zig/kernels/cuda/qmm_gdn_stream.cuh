// Native grouped-input MMA and slice order, with one wider word-only K-major stream per CTA.
#pragma once
namespace tf_qmm_gdn_stream {
using namespace tf_qmm_group;
template<int GS,int BN,int STAGES>using GemvTile=qmm_frag::LaneTile<GS,16,BN,1,4,STAGES>;
template <int GS, int BN, int STAGES, int FutureGroups, typename Map, typename After>
__device__ __forceinline__ void wide_items(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                           int K, int ldx, int items, Map map, float (&acc)[GemvTile<GS, BN, STAGES>::NT][4],
                                           After after) {
    using T = GemvTile<GS, BN, STAGES>;
    static_assert(T::MT == 1 && T::THREADS == 128, "one m16 row tile, four column warps");
    constexpr int NT = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(16, M);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = (16 * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    const __nv_bfloat16* xsrc[XC];
    size_t woff[WC];
    int soff[SC], xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC], sbias[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < 16 * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
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
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
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
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * (P.npad/64) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
    };
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    for (int c = tid; c < 16 * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_w(s, s);
        commit();
    }
#if __CUDA_ARCH__ >= 900
    if(tid==0) {
#pragma unroll
        for(int g=1;g<=FutureGroups;++g)if(g<items){
            const auto at=map(g);const auto* w=reinterpret_cast<const unsigned char*>(P.w)+(size_t(at.x)*(P.npad/64)+at.y/64)*TILE_BYTES;
            const auto* a=P.scales+size_t(at.x)*P.npad+at.y;const auto* b=P.biases+size_t(at.x)*P.npad+at.y;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w), "n"(BN*GS/2) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "n"(BN*2) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(b), "n"(BN*2) : "memory");
        }
    }
#endif
    grid_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_x(s, s);
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
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
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
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
            ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
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
            const int col = wn * (BN / 4) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
            const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
            const float sv[2] = {__low2float(s2), __high2float(s2)};
            const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[j][e]));
        }
        after(it);
    }
}


template <int BN, int NT, typename Got>
__device__ __forceinline__ void gemv_store(const Part& P, int M, int n0, Got got) {
    const int lane = threadIdx.x & 31, wn = threadIdx.x >> 5, row = lane >> 2, N = P.n;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = n0 + wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = row + h * 8;
            if (r >= M) continue;
            const float2 v = got(j, h, r, col - n0);
            auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(r) * N + col;
            if (col + 1 < N && (N & 1) == 0) {
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v.x, v.y);
            } else {
                if (col < N) dst[0] = __float2bfloat16_rn(v.x);
                if (col + 1 < N) dst[1] = __float2bfloat16_rn(v.y);
            }
        }
}


template<int BN,int FutureGroups>
__device__ __forceinline__ void tile(const __nv_bfloat16* x,const float* xs,const tf_qmm_group::Part p,int t,int M,int K,int ldx){
    constexpr int NT=GemvTile<64,BN,2>::NT;
    const auto original=p;
    const int KG=K/64,per=KG/p.sk;float acc[NT][4]{},tot[NT][4]{};
    wide_items<64,BN,2,FutureGroups>(x,xs,original,M,K,ldx,KG,[&](int g){return make_int2(g,t*BN);},acc,[&](int g){
        if((g+1)%per!=0)return;bool first=g+1==per;
#pragma unroll
        for(int j=0;j<NT;++j)
#pragma unroll
            for(int e=0;e<4;++e){tot[j][e]=first?acc[j][e]:tot[j][e]+acc[j][e];acc[j][e]=0;}
        if(g!=KG-1)return;
        gemv_store<BN,NT>(original,M,t*BN,[&](int j,int h,int,int){return make_float2(tot[j][2*h],tot[j][2*h+1]);});
    });
}
template<int FutureGroups>
__device__ __forceinline__ void body(const __nv_bfloat16* x,const float* xs,const tf_qmm_group::Parts& parts,int M,int K,int ldx){
    int which=0;
#pragma unroll
    for(int i=1;i<4;++i)if(int(blockIdx.x)>=parts.p[i].first)which=i;
    const auto p=parts.p[which];int t=int(blockIdx.x)-p.first;
    if(which<2)tile<256,FutureGroups>(x,xs,p,t,M,K,ldx);
    else tile<64,FutureGroups>(x,xs,p,t,M,K,ldx);
}
}

extern "C" __global__ void __launch_bounds__(128) tf_qmm_gdn_stream16(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_qmm_gdn_stream::body<0>(x,xs,parts,M,K,ldx);
}
