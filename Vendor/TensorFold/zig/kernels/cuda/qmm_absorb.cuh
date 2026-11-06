// Absorb-only dictionary tile. Read A fragments and the small scale/bias table through the cache, leaving shared
// memory for the packed weight pipeline and the unchanged ordered DSM reduction. No weight view or scratch copy.
#pragma once
namespace tf_qmm_group {
using namespace qmm_dict;

template<int STAGES, bool SHARED_X = false>
__device__ __forceinline__ void absorb_l2(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
        const Parts& parts, int M, int K, int ldx, int rows_t, int C) {
    constexpr int THREADS = 128, WS = 2048, IS = 256, STAGE = WS + IS;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / 64, cid = blockIdx.x / C, rank = blockIdx.x % C;
    int q = 0;
#pragma unroll
    for (int i = 1; i < PARTS; ++i) if (i < parts.count && cid >= parts.p[i].first) q = i;
    const Part P = parts.p[q];
    const int sk = P.sk, lc = cid - P.first;
    const int tile = lc / rows_t * (C / sk) + rank / sk, slice = rank % sk;
    const int m0 = lc % rows_t * 16, n0 = tile * 64, per = KG / sk, g0 = slice * per;
    const bool live = tile < P.tiles;
    DictLanes D = dict_lanes<64, 4>(P, n0, KG, tid);
    const unsigned short* tab = reinterpret_cast<const unsigned short*>(P.biases);
    const uint32_t* pairtab = reinterpret_cast<const uint32_t*>(P.biases);
    auto stage = [&](int s) { return buf + s * STAGE; };
    auto load = [&](int s, int g) {
        auto* p = stage(s);
        cp16(p + tid * 16, reinterpret_cast<const unsigned char*>(P.w) +
             (static_cast<size_t>(n0 / 64) * KG + g) * WS + tid * 16);
        if (D.ok) cp16(p + WS + tid * 16, D.src + dict_window(D, g));
    };
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));
    float acc[2][4] = {};
    if (live) {
        if constexpr (SHARED_X) {
            const int r = tid / 8, ch = tid % 8;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                auto* xb = buf + STAGES * STAGE + z * 2112;
                if (m0 + r >= M) *reinterpret_cast<uint4*>(xb + r * 128 + swz<8>(r, ch) * 16) = uint4{};
                if (tid < 16 && m0 + tid >= M) reinterpret_cast<float*>(xb + 2048)[tid] = 0.0f;
            }
        }
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) { if (s < per) load(s, g0 + s); commit(); }
        grid_wait();
        qmm_dict::x_keep_l2(x + static_cast<size_t>(m0) * ldx, min(16, M - m0), K, ldx);
        grid_launch();
        for (int it = 0; it < per; ++it) {
            uint32_t av[4][4];
            float xv[2];
            auto* xb = buf + STAGES * STAGE + (it & 1) * 2112;
            if constexpr (SHARED_X) {
                const int r = tid / 8, ch = tid % 8;
                if (m0 + r < M) *reinterpret_cast<uint4*>(xb + r * 128 + swz<8>(r, ch) * 16) =
                    __ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(m0 + r) * ldx + (g0 + it) * 64 + ch * 8));
                if (tid < 16 && m0 + tid < M) reinterpret_cast<float*>(xb + 2048)[tid] =
                    __ldg(xs + static_cast<size_t>(m0 + tid) * KG + g0 + it);
            } else {
                // A fragment mapping is exactly ldmatrix.m8n8.x4 of the original row-major 16x16 tile.
                // Issue the cache reads before waiting for the independent packed-weight stage.
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const int row = m0 + (lane >> 2) + (h & 1) * 8;
                        const int col = (g0 + it) * 64 + kt * 16 + (lane & 3) * 2 + (h >> 1) * 8;
                        av[kt][h] = row < M ? __ldg(reinterpret_cast<const uint32_t*>(x + static_cast<size_t>(row) * ldx + col)) : 0u;
                    }
                }
                const int xr = m0 + (lane >> 2);
                xv[0] = xr < M ? __ldg(xs + static_cast<size_t>(xr) * KG + g0 + it) : 0.0f;
                xv[1] = xr + 8 < M ? __ldg(xs + static_cast<size_t>(xr + 8) * KG + g0 + it) : 0.0f;
            }
            wait<STAGES - 2>();
            __syncthreads();
            if constexpr (SHARED_X) {
                const int r = lane >> 2;
                xv[0] = reinterpret_cast<const float*>(xb + 2048)[r];
                xv[1] = reinterpret_cast<const float*>(xb + 2048)[r + 8];
            }
            const int next = it + STAGES - 1;
            if (next < per) load(next % STAGES, g0 + next);
            commit();
            const auto* p = stage(it % STAGES);
            const uint32_t* pk = reinterpret_cast<const uint32_t*>(p + WS);
            const int off = ((g0 + it) * D.step & 63) * 8 + D.bit0;
            const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & D.mask;
            const uint32_t dv = D.pair ? (D.shift == 0 ? __ldg(pairtab + idx) : 0u) : __ldg(tab + D.base + idx);
            float sf = 0.0f, bf = 0.0f;
            dict_raw_vals(D, dv, sf, bf);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p);
            uint32_t words[2][2];
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int v = 0; v < 2; ++v) words[j][v] = pw[((wn * 2 + j) * 32 + lane) * 2 + v];
            float d[2][4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                if constexpr (SHARED_X) {
                    const int r = (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(av[kt], xb + r * 128 + swz<8>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                    const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
                    if (kt == 0) mma0(d[j], av[kt], b0, b1); else mma(d[j], av[kt], b0, b1);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c0 = j * 8 + (lane & 3) * 2;
                const float sv[2] = {__shfl_sync(0xffffffffu, sf, c0), __shfl_sync(0xffffffffu, sf, c0 + 1)};
                const int bo = D.pair ? 0 : 16;
                const float bv[2] = {__shfl_sync(0xffffffffu, bf, bo + c0), __shfl_sync(0xffffffffu, bf, bo + c0 + 1)};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[j][e]));
            }
        }
        wait<0>();
        __syncthreads();
    } else grid_launch();
    if (sk > 1) {
#if __CUDA_ARCH__ >= 900
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (slice != 0) {
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) mine[(j * 4 + e) * THREADS + tid] = acc[j][e];
        }
        cluster.sync();
        if (slice == 0 && live) for (int peer = 1; peer < sk; ++peer) {
            const float* theirs = cluster.map_shared_rank(mine, rank + peer);
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[j][e] = acc[j][e] + theirs[(j * 4 + e) * THREADS + tid];
        }
        cluster.sync();
        if (slice != 0) return;
#else
        __trap(); return;
#endif
    }
    if (!live) return;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = m0 + (lane >> 2) + h * 8, col = n0 + wn * 16 + j * 8 + (lane & 3) * 2;
            if (row < M && col + 1 < P.n)
                *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(row) * P.n + col) =
                    __floats2bfloat162_rn(acc[j][h * 2], acc[j][h * 2 + 1]);
        }
}
} // namespace tf_qmm_group

#define TF_ABSORB_ENTRY(S) \
extern "C" __global__ void __launch_bounds__(128) tf_qmm_absorb_l2_##S( \
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, \
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx, int rows_t, int C) { \
    tf_qmm_group::absorb_l2<S>(x, xs, parts, M, K, ldx, rows_t, C); \
}
TF_ABSORB_ENTRY(4)
TF_ABSORB_ENTRY(8)
#undef TF_ABSORB_ENTRY

// Two input tiles alternate: iteration i's entry barrier completed every read of i-2 before that tile is overwritten.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_absorb_l2x_4(
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
        const __grid_constant__ tf_qmm_group::Parts parts, int M, int K, int ldx, int rows_t, int C) {
    tf_qmm_group::absorb_l2<4, true>(x, xs, parts, M, K, ldx, rows_t, C);
}
