// Prompt MLP A-transport via a BF16 tensor map; arithmetic is qmm_prompt_pipe.cuh's.
#pragma once
#include <cuda.h>
namespace tf_qmm_tma {
using namespace tf_qmm_prefill;
using tf_qmm_pipe::cp16p;
template <int GS, int BM, int WM, int WN, int STAGES>
__device__ __forceinline__ void mlp_tile(const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int m0, int n0, const CUtensorMap& amap) {
    constexpr int BN = 128;
    static_assert(STAGES == 2, "two stages: the next stage is read after the current one's last read");
    static_assert(WN % 2 == 0, "gate and up use equal column-warp groups");
    // TMA always copies a 128x64 box; a shortened final tile still computes
    // only its original BM rows. OOB rows are zero-filled by the descriptor.
    struct T : Tile<GS, BM, BN, WM, WN, STAGES> {
        enum { X = 128 * GS * 2, STAGE = (X + BN * GS / 2 + 2 * BN * 2 + 1023) / 1024 * 1024 };
    };
    static_assert(T::NT % 2 == 0, "paired gate and up fragments");
    extern __shared__ __align__(1024) unsigned char buf[];
    __shared__ __align__(8) uint64_t abar[STAGES];
    if (threadIdx.x < STAGES) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem(&abar[threadIdx.x])) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    auto b_column = [&](int j) { return (j / (T::NT / 2)) * 8 + wn * (T::NT / 2) + j % (T::NT / 2); };
    const ManyPart& G = parts.p[0];
    const ManyPart& U = parts.p[1];
    const int N = G.n, npad = G.npad;
    const int KG = K / GS, per = KG, g0 = 0;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    constexpr int XC = BM * T::CHUNKS / T::THREADS, RSTEP = T::THREADS / T::CHUNKS;
    static_assert(T::THREADS % T::CHUNKS == 0 && XC * T::THREADS == BM * T::CHUNKS && RSTEP % T::CHUNKS == 0, "whole chunks");
    constexpr int TILE_BYTES = 64 * GS / 2, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const int ch = tid % T::CHUNKS, r0 = tid / T::CHUNKS;
    const unsigned char* wsrc[WC];
    bool wok[WC];
#pragma unroll
    for (int k = 0; k < WC; ++k) {
        const int c = tid + k * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[k] = c < T::W / 16;
        wsrc[k] = reinterpret_cast<const unsigned char*>(t ? U.w : G.w) + static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + off * 16;
    }
    const int s_which = tid / (T::S / 16), s_off = tid % (T::S / 16), s_t = s_off / 8;
    const bool sok = tid < 2 * (T::S / 16);
    const __nv_bfloat16* ssrc = (s_which ? (s_t ? U.biases : G.biases) : (s_t ? U.scales : G.scales)) + n0 + (s_off % 8) * 8;
    const int sdst = s_which * T::S + s_off * 16;
    auto load = [&](int s, int g, bool go) {
        unsigned char* p = stage(s);
        if (tid == 0 && go) {
            const uint32_t bar = smem(&abar[s]);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(128 * GS * 2) : "memory");
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                         :: "r"(smem(p)), "l"(&amap), "r"(g * GS), "r"(m0), "r"(bar) : "memory");
        }
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k) cp16p(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES, go && wok[k]);
        cp16p(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad, go && sok);
    };

    auto wait_a = [&](int group) {
        const uint32_t bar = smem(&abar[group % STAGES]);
        const uint32_t phase = (group / STAGES) & 1;
        asm volatile("{ .reg .pred p;\n"
                     "Lwait: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "@!p bra Lwait;\n}"
                     :: "r"(bar), "r"(phase) : "memory");
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
    auto read_w = [&](const unsigned char* p) {
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[(b_column(j) * 32 + lane) * (GS / 32) + v];
            const int col = b_column(j) * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
    };
    uint32_t a[T::MT][4];
    auto load_a_row = [&](const unsigned char* p, int kt, int i) {
        const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kt * 2 + (lane >> 4);
        ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
    };
    auto weight_pair = [&](int j, int kt) {
        return make_uint2(fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]),
                          fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]));
    };
    // stage 0 in, visible, and its first operands in registers
    load(0, g0, true);
    commit();
    wait<0>();
    wait_a(0);
    __syncthreads();
    read_w(stage(0));
#pragma unroll
    for (int i = 0; i < T::MT; ++i) load_a_row(stage(0), 0, i);
    uint2 b[2];
    b[0] = weight_pair(0, 0);
    for (int it = 0; it < per; ++it) {
        const unsigned char* p = stage(it & 1);
        const unsigned char* pn = stage((it + 1) & 1);
        const bool more = it + 1 < per;
        // the other stage's last reader passed the barrier of the previous group (or the prologue's); a tile's last
        // group copies nothing
        load((it + 1) & 1, g0 + (more ? it + 1 : it), more);
        commit();
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const bool last = kt + 1 == GS / 16 && j + 1 == T::NT;
                const int next_j = (j + 1) % T::NT, next_kt = kt + (j + 1 == T::NT);
                const int slot = (kt * T::NT + j) & 1;
                if (last) {
                    // every read of this stage is issued (A rows for this k16 step came in the previous one): the
                    // next stage may be read once landed, and this one refilled after the barrier. A tile's last
                    // group reads the other stage too; nothing uses what it reads.
                    wait<0>();
                    if (it + 1 < KG) wait_a(it + 1);
                    __syncthreads();
                    read_w(pn);
                    b[slot ^ 1] = weight_pair(0, 0);
                } else {
                    b[slot ^ 1] = weight_pair(next_j, next_kt);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    mma(acc[i][j], a[i], b[slot].x, b[slot].y);
                    if (j + 1 == T::NT) load_a_row(kt + 1 < GS / 16 ? p : pn, (kt + 1) % (GS / 16), i);
                }
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT / 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + wn * (64 / WN) + j * 8 + (lane & 3) * 2;
                if (row >= M) continue;
                const __nv_bfloat162 g2 = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                const __nv_bfloat162 u2 = __floats2bfloat162_rn(acc[i][j + T::NT / 2][2 * h], acc[i][j + T::NT / 2][2 * h + 1]);
                const float gf[2] = {__low2float(g2), __high2float(g2)};
                const float uf[2] = {__low2float(u2), __high2float(u2)};
                __nv_bfloat16 o[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) o[e] = __float2bfloat16_rn(gf[e] / (1.0f + expf(-gf[e])) * uf[e]);
                __nv_bfloat16* dst = act + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(o[0], o[1]);
                else {
                    if (col < N) dst[0] = o[0];
                    if (col + 1 < N) dst[1] = o[1];
                }
            }
    __syncthreads();
    if (tid < STAGES) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(smem(&abar[tid])) : "memory");
    __syncthreads();
}


} // namespace tf_qmm_tma
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_mlp_tma(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ tf_qmm_prefill::ManyParts parts,
        __nv_bfloat16* __restrict__ act, int M, int K, int ldx, int group, int full_cols, int total, int rotate,
        const __grid_constant__ CUtensorMap amap) {
    using namespace tf_qmm_prefill;
    for (int work = blockIdx.x; work < total; work += gridDim.x) {
        int tile = work;
        const int rows = (M + 127) / 128;
        if (rotate && group == rows && total % rows == 0 && (int)gridDim.x % rows == 0) {
            const int base = work / rows * rows;
            tile = base + (work % rows + work / gridDim.x) % rows;
        }
        const int2 at = tile_of(tile, M, parts.p[0].n, 128, 64, group);
        if (tf_qmm_rows0::short_tile<128,1>(M - at.x))
            tf_qmm_tma::mlp_tile<64,112,1,4,2>(x,parts,act,M,K,ldx,at.x,at.y,amap);
        else
            tf_qmm_tma::mlp_tile<64,128,1,4,2>(x,parts,act,M,K,ldx,at.x,at.y,amap);
        __syncthreads();
    }
}
