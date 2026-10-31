// Native grouped-input MMA and slice order, with one wider word-only K-major stream per CTA.
#pragma once
#include "qmm_dict.cuh"
namespace tf_qmm_gdn_stream {
using namespace tf_qmm_group;
template<int GS,int BN,int STAGES>using GemvTile=qmm_frag::LaneTile<GS,16,BN,1,4,STAGES>;
// A dictionary-coded part (qmm_dict.cuh, DICT): its npad holds the code and its tiles field the real npad. A stage's
// 2 * S bytes after the words hold the CTA's BN / 64 index windows (256 bytes each, a 64-column tile's group) in place
// of its scales and biases; each warp decodes its BN / 4 columns' scales and biases into `dec` (BN scales, then BN
// biases, as the arrays' stage holds them) from the part's table after it, so every fma reads the same values.
template <int GS, int BN, int STAGES, int FutureGroups, bool DICT, typename Map, typename After>
__device__ __forceinline__ void wide_items(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                           int K, int ldx, int items, Map map, float (&acc)[GemvTile<GS, BN, STAGES>::NT][4],
                                           After after) {
    using T = GemvTile<GS, BN, STAGES>;
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
    unsigned char* dec = buf + T::SMEM;  // DICT: BN decoded scales, BN biases, then the part's table
    unsigned char* table = dec + 4 * BN;
    // a pair table of up to GDN_PAIR_SMEM bytes in shared memory (the twin's two-CTAs-an-SM room, qlinear.zig
    // gdn_pair_tab_max), else read through L2
    constexpr int GDN_PAIR_SMEM = 26496;
    [[maybe_unused]] const int pair_bytes = (qmm_dict::dict_np(P.npad) * 4 + 15) / 16 * 16;
    if constexpr (DICT) {
        if (qmm_dict::dict_pair(P.npad) && pair_bytes <= GDN_PAIR_SMEM) {
            for (int c = tid; c < pair_bytes / 16; c += T::THREADS) cp16(table + c * 16, reinterpret_cast<const unsigned char*>(P.biases) + c * 16);
        } else {
            qmm_dict::dict_table<T::THREADS>(P, table, tid);
        }
    }
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
    qmm_dict::x_keep_l2(x, rows, K, ldx);
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
        // the lane's BN / 64 values of its warp's BN / 4 columns: value q is column q % (BN / 4)'s scale, then bias;
        // fetched before the MMAs (a pair table read through L2 waits under them), stored into dec after them. A pair
        // code (qmm_dict.cuh, FORMAT.md): the column's one index, the entry's scale or bias half
        [[maybe_unused]] uint32_t dv[BN / 64];
        if constexpr (DICT) {
            const int code = P.npad, bits = qmm_dict::dict_bits(code), step = qmm_dict::dict_step(code), g = map(it).x;
            const bool pair = qmm_dict::dict_pair(code);
            const uint32_t* gtab = pair && pair_bytes > GDN_PAIR_SMEM ? reinterpret_cast<const uint32_t*>(P.biases) : nullptr;
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
                        dv[v] = reinterpret_cast<const unsigned short*>(table)[(half ? qmm_dict::dict_ns(code) : 0) + idx];
                    }
                }
            } else {
                const int q = lane, half = q / 16, col = wn * 16 + (q & 15);
                const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W);
                const int off = (g * step & 63) * 8 + (pair ? col : half * 64 + col) * bits;
                const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mask_bits;
                dv[0] = pair ? qmm_dict::dict_pair_half(qmm_dict::dict_pair_entry(table, gtab, idx), half * 16)
                             : reinterpret_cast<const unsigned short*>(table)[(half ? qmm_dict::dict_ns(code) : 0) + idx];
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


template<int BN,int FutureGroups,bool DICT=false>
__device__ __forceinline__ void tile(const __nv_bfloat16* x,const float* xs,const tf_qmm_group::Part p,int t,int M,int K,int ldx){
    constexpr int NT=GemvTile<64,BN,2>::NT;
    const auto original=p;
    const int KG=K/64,per=KG/p.sk;float acc[NT][4]{},tot[NT][4]{};
    wide_items<64,BN,2,FutureGroups,DICT>(x,xs,original,M,K,ldx,KG,[&](int g){return make_int2(g,t*BN);},acc,[&](int g){
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
// body() on a launch whose coded parts carry their real npad in `tiles` (uncoded parts: tiles 0, npad as stored).
__device__ __forceinline__ void body_dict(const __nv_bfloat16* x,const float* xs,const tf_qmm_group::Parts& parts,int M,int K,int ldx){
    int which=0;
#pragma unroll
    for(int i=1;i<4;++i)if(int(blockIdx.x)>=parts.p[i].first)which=i;
    const auto p=parts.p[which];int t=int(blockIdx.x)-p.first;const bool coded=p.tiles>0;
    if(which<2){if(coded)tile<256,0,true>(x,xs,p,t,M,K,ldx);else tile<256,0>(x,xs,p,t,M,K,ldx);}
    else{if(coded)tile<64,0,true>(x,xs,p,t,M,K,ldx);else tile<64,0>(x,xs,p,t,M,K,ldx);}
}
}

extern "C" __global__ void __launch_bounds__(128) tf_qmm_gdn_stream16(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_qmm_gdn_stream::body<0>(x,xs,parts,M,K,ldx);
}
// tf_qmm_gdn_stream16 with its dictionary-coded parts' index windows: the same loads, MMA chains, fmas and sums.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_gdn_stream16_dict(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_qmm_gdn_stream::body_dict(x,xs,parts,M,K,ldx);
}

// Three native attention matrices retain each matrix's original K partitions.
template<bool DICT>
__device__ __forceinline__ void tf_attention_input_stream(const __nv_bfloat16* x,const float* xs,
        const tf_qmm_group::Parts& parts,int M,int K,int ldx){
    int which=0;
#pragma unroll
    for(int i=1;i<4;++i)if(i<parts.count&&int(blockIdx.x)>=parts.p[i].first)which=i;
    const auto p=parts.p[which];const int tile=int(blockIdx.x)-p.first;
    if constexpr(DICT){
        if(p.tiles>0)tf_qmm_gdn_stream::tile<256,0,true>(x,xs,p,tile,M,K,ldx);
        else tf_qmm_gdn_stream::tile<256,0>(x,xs,p,tile,M,K,ldx);
    }else tf_qmm_gdn_stream::tile<256,0>(x,xs,p,tile,M,K,ldx);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_attention_stream16(const __nv_bfloat16* x,const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_attention_input_stream<false>(x,xs,parts,M,K,ldx);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_attention_stream16_dict(const __nv_bfloat16* x,const float* xs,
        const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_attention_input_stream<true>(x,xs,parts,M,K,ldx);
}
