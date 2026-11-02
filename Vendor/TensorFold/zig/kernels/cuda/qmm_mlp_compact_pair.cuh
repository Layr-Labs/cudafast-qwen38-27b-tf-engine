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
    struct T : Base {
        // XT: the group's x tile (BM rows of 64 bf16), staged with the weights
        enum { X = 0, W = BN * GS / 2, S = 256, XT = W + 2 * S + BM * 4, STAGE = XT + BM * GS * 2,
               SMEM = STAGES * STAGE };
    };
    constexpr int PASSES = BN / WN / 32;
    static_assert(T::SMEM == 21632, "compact pair pipeline footprint");
    constexpr int I = T::MT, J = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x, rank = 0;
    const Part P = parts.p[rank];
    const int tile = cid / rows_t;
    const int m0 = cid % rows_t * BM, n0 = tile * (BN / 2), per = KG, g0 = 0;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const unsigned char* wsrc[WC];
    int wdst[WC];
    bool wok[WC];
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(parts.p[t / (BN / 128)].w) + static_cast<size_t>(n0 / 64 + t % (BN / 128)) * (KMAJOR ? 1 : KG) * TILE_BYTES +
                  off * 16;
        wdst[j] = T::X + c * 16;
    }
    static_assert(KMAJOR && BN == 256 && WN == 4);
    const int code0 = parts.p[0].first, code1 = parts.p[1].first;
    const int bits0 = dict_bits(code0), bits1 = dict_bits(code1);
    const int run0 = bits0, run1 = bits1;
    const bool dok = tid < run0 + run1;
    const int dpart = tid < run0 ? 0 : 1, dchunk = tid < run0 ? tid : tid - run0;
    const int dstep = dict_step(dpart ? code1 : code0);
    const unsigned char* dsrc = reinterpret_cast<const unsigned char*>(parts.p[dpart].scales) +
                                static_cast<size_t>(n0 / 64) * dstep + dchunk * 16;
    const int dpitch = P.npad / 64 * dstep;
    const int ddst = T::X + T::W + (dpart ? 16 * run0 : 0) + dchunk * 16;
    const int mpart = wn / 2, mbits = mpart ? bits1 : bits0;
    const uint32_t mmask = (1u << mbits) - 1u;
    const int mrun = mpart ? 16 * run0 : 0;
    const uint32_t* const mgtab = reinterpret_cast<const uint32_t*>(parts.p[mpart].biases);
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    // x's 16 rows of the group: 16-byte chunk ch of row r at slot ch ^ (r & 7) (rows past M repeat the last one: their
    // products are never stored); the fragments below read the same bf16 words the x rows hold
    const int xr = tid >> 3, xch = tid & 7;
    const __nv_bfloat16* xtsrc = x + static_cast<size_t>(m0 + min(xr, rows - 1)) * ldx + xch * 8;
    const int xtdst = T::XT + xr * 128 + ((xch ^ (xr & 7)) * 16);
    auto load_x = [&](int s, int g) {
        if (xsok) cp4(stage(s) + T::W + 2 * T::S + tid * 4, xssrc + g);
        cp16(stage(s) + xtdst, xtsrc + g * GS);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * (KMAJOR ? P.npad / 64 : 1) * TILE_BYTES);
        if (dok) cp16(p + ddst, dsrc + static_cast<size_t>(g) * dpitch);
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
        __syncthreads();
        uint32_t av[GS / 16][I][4];
        {
            const unsigned char* xt = stage(it % STAGES) + T::XT;
            const int b = (2 * (lane & 3)) * 2;
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt)
#pragma unroll
                for (int i = 0; i < I; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane >> 2);
                    auto at = [&](int row, int ch) {
                        return *reinterpret_cast<const uint32_t*>(xt + row * 128 + ((ch ^ (row & 7)) * 16) + b);
                    };
                    av[kt][i][0] = at(r, 2 * kt);
                    av[kt][i][1] = at(r + 8, 2 * kt);
                    av[kt][i][2] = at(r, 2 * kt + 1);
                    av[kt][i][3] = at(r + 8, 2 * kt + 1);
                }
        }
        const int next = it + 1;
        if (next < per) { load_x(next % STAGES, next); load_w(next % STAGES, next); }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + mrun);
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
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
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

// Warp-specialized twin of mlp_single_compact_body: a fifth warp issues every cp.async of a stage (weights, dictionary
// indices, input sums, the swizzled input tile) into a ring of STAGES stages and marks each one full on an mbarrier
// (cp.async.mbarrier.arrive.noinc); the four compute warps wait for a stage, read it into registers, release it on a
// second mbarrier and run exactly mlp_single_compact_body's arithmetic (the same fragments, MMA chain, FMA order, bf16
// rounding and epilogue), so act and its group sums are the same bits. The loads of later stages no longer wait at a
// CTA-wide barrier for the slowest compute warp, and three stages keep two loads in flight while one is consumed.
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "mlp_ws_wait:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra mlp_ws_wait;\n"
                 "}\n" ::"r"(smem(b)), "r"(phase) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(smem(b)) : "memory");
}
__device__ __forceinline__ void mbar_cp_arrive(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem(b)) : "memory");
}

template <int STAGES = 3>
__device__ __forceinline__ void mlp_single_compact_ws_body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                                 const Parts& parts, __nv_bfloat16* __restrict__ act,
                                                 float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    constexpr int GS = 64, BM = 16, BN = 256, WM = 1, WN = 4;
    using Base = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    struct T : Base {
        enum { X = 0, W = BN * GS / 2, S = 256, XT = W + 2 * S + BM * 4, STAGE = XT + BM * GS * 2,
               SMEM = STAGES * STAGE };
    };
    constexpr int PASSES = BN / WN / 32;
    constexpr int CONSUMERS = T::THREADS;  // the four compute warps; warp WN loads
    static_assert(T::STAGE == 10816 && CONSUMERS == 128, "compact pair stage");
    constexpr int I = T::MT, J = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const bool loader = warp == WN;
    const int wm = loader ? 0 : warp / WN, wn = loader ? 0 : warp % WN;
    const int KG = K / GS, cid = blockIdx.x, rank = 0;
    const Part P = parts.p[rank];
    const int tile = cid / rows_t;
    const int m0 = cid % rows_t * BM, n0 = tile * (BN / 2), per = KG;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);
    const int code0 = parts.p[0].first, code1 = parts.p[1].first;
    const int bits0 = dict_bits(code0), bits1 = dict_bits(code1);
    const int run0 = bits0, run1 = bits1;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 32);
            mbar_init(&empty[s], CONSUMERS);
        }
    }
    __syncthreads();

    if (loader) {
        // the loader lane's share of a stage: 16 weight chunks, one dictionary chunk, one input sum, four input chunks
        constexpr int TILE_BYTES = 64 * GS / 2;
        constexpr int WC = T::W / 16 / 32;
        const unsigned char* wsrc[WC];
#pragma unroll
        for (int j = 0; j < WC; ++j) {
            const int c = lane + j * 32, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            wsrc[j] = reinterpret_cast<const unsigned char*>(parts.p[t / (BN / 128)].w) +
                      static_cast<size_t>(n0 / 64 + t % (BN / 128)) * TILE_BYTES + off * 16;
        }
        const size_t wpitch = static_cast<size_t>(P.npad / 64) * TILE_BYTES;
        const bool dok = lane < run0 + run1;
        const int dpart = lane < run0 ? 0 : 1, dchunk = lane < run0 ? lane : lane - run0;
        const int dstep = dict_step(dpart ? code1 : code0);
        const unsigned char* dsrc = reinterpret_cast<const unsigned char*>(parts.p[dpart].scales) +
                                    static_cast<size_t>(n0 / 64) * dstep + dchunk * 16;
        const int dpitch = P.npad / 64 * dstep;
        const int ddst = T::X + T::W + (dpart ? 16 * run0 : 0) + dchunk * 16;
        const bool xsok = lane < rows;
        const float* xssrc = xs + static_cast<size_t>(m0 + min(lane, rows - 1)) * KG;
        auto load_w = [&](int s, int g) {
            unsigned char* p = stage(s);
#pragma unroll
            for (int j = 0; j < WC; ++j) cp16(p + T::X + (lane + j * 32) * 16, wsrc[j] + static_cast<size_t>(g) * wpitch);
            if (dok) cp16(p + ddst, dsrc + static_cast<size_t>(g) * dpitch);
        };
        auto load_x = [&](int s, int g) {
            unsigned char* p = stage(s);
            if (xsok) cp4(p + T::W + 2 * T::S + lane * 4, xssrc + g);
#pragma unroll
            for (int q = 0; q < BM * 8 / 32; ++q) {
                const int c = lane + q * 32, xr = c >> 3, xch = c & 7;
                cp16(p + T::XT + xr * 128 + ((xch ^ (xr & 7)) * 16),
                     x + static_cast<size_t>(m0 + min(xr, rows - 1)) * ldx + xch * 8 + static_cast<size_t>(g) * GS);
            }
        };
        const int pre = min(STAGES, per);
        // the weights do not depend on the previous kernel: they stream before its results are visible
        for (int s = 0; s < pre; ++s) load_w(s, s);
        grid_wait();
        const __nv_bfloat16* x_ready = x;
        asm volatile("" : "+l"(x_ready) :: "memory");
        qmm_dict::x_keep_l2(x_ready + static_cast<size_t>(m0) * ldx, rows, K, ldx);
        for (int s = 0; s < pre; ++s) {
            load_x(s, s);
            mbar_cp_arrive(&full[s]);
        }
        grid_launch();
        for (int g = pre; g < per; ++g) {
            const int s = g % STAGES;
            mbar_wait(&empty[s], (g / STAGES - 1) & 1);
            load_w(s, g);
            load_x(s, g);
            mbar_cp_arrive(&full[s]);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    }

    const int mpart = wn / 2, mbits = mpart ? bits1 : bits0;
    const uint32_t mmask = (1u << mbits) - 1u;
    const int mrun = mpart ? 16 * run0 : 0;
    const uint32_t* const mgtab = reinterpret_cast<const uint32_t*>(parts.p[mpart].biases);
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));
    float acc[I][J][4];
#pragma unroll
    for (int i = 0; i < I; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    if (!loader) {
        grid_wait();
        const __nv_bfloat16* x_ready = x;
        asm volatile("" : "+l"(x_ready) :: "memory");
        qmm_dict::x_keep_l2(x_ready + static_cast<size_t>(m0) * ldx, rows, K, ldx);
        grid_launch();
        for (int it = 0; it < per; ++it) {
            const int s = it % STAGES;
            mbar_wait(&full[s], (it / STAGES) & 1);
            const unsigned char* p = stage(s);
            uint32_t av[GS / 16][I][4];
            {
                const unsigned char* xt = p + T::XT;
                const int b = (2 * (lane & 3)) * 2;
#pragma unroll
                for (int kt = 0; kt < GS / 16; ++kt)
#pragma unroll
                    for (int i = 0; i < I; ++i) {
                        const int r = wm * (BM / WM) + i * 16 + (lane >> 2);
                        auto at = [&](int row, int ch) {
                            return *reinterpret_cast<const uint32_t*>(xt + row * 128 + ((ch ^ (row & 7)) * 16) + b);
                        };
                        av[kt][i][0] = at(r, 2 * kt);
                        av[kt][i][1] = at(r + 8, 2 * kt);
                        av[kt][i][2] = at(r, 2 * kt + 1);
                        av[kt][i][3] = at(r + 8, 2 * kt + 1);
                    }
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
            const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + T::X + T::W + mrun);
            uint32_t idx[PASSES];
#pragma unroll
            for (int pass = 0; pass < PASSES; ++pass) {
                const int c = (wn % 2) * (BN / WN) + pass * 32 + lane, off = (c / 64 * 64 + c % 64) * mbits;
                idx[pass] = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & mmask;
            }
            const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
            uint32_t words[T::NT][GS / 32];
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            float xv[I][2];
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                xv[i][0] = px[row];
                xv[i][1] = px[row + 8];
            }
            // every read of the stage is in registers: the loader may refill it
            mbar_arrive(&empty[s]);
            uint32_t pe[PASSES];
#pragma unroll
            for (int pass = 0; pass < PASSES; ++pass) pe[pass] = __ldg(mgtab + idx[pass]);
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
    }
    __nv_bfloat16* own = reinterpret_cast<__nv_bfloat16*>(buf);
    __syncthreads();
    if (!loader) {
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
    }
    __syncthreads();
    if (!loader && wn >= WN / 2) {
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
extern "C" __global__ void __maxnreg__(136) tf_qmm_mlp_single16_compact_pair_ws(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,
 __nv_bfloat16* act,float* sums,int M,int K,int ldx,int rows_t) {
 tf_qmm_mlp_single::mlp_single_compact_ws_body<>(x,xs,parts,act,sums,M,K,ldx,rows_t);
}
