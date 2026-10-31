// Prompt GEMMs of up to four projections of one input in one launch: prefill_kernel's chain a row and column.
//
// Every output keeps prefill_kernel's arithmetic exactly: the same bf16 weights (fma2 of the nibble pair, scale
// and bias), the same m16n8k16 MMAs over K in group then k16 order into one fp32 accumulator from zero, and the
// same bf16 store. Only the tile shape, the warps' shares and which CTA computes which tile differ, and an
// element's MMA chain does not depend on those: its m16 row block and n8 column block start at multiples of 16
// and 8 in every tile shape here, as in prefill_kernel's.
#pragma once

namespace tf_qmm_prefill {

constexpr int MANY = 4;

struct ManyPart {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, first;  // first: the part's first CTA in the launch
};

struct ManyParts {
    ManyPart p[MANY];
    int count;
};

// One BM x BN output tile at (m0, n0) of x through (w, scales, biases) into out (rows N wide): prefill_kernel's chain.
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_tile(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int N, int npad, int M, int K, int ldx, int m0,
        int n0, int ga = 0, int gb = -1, const float* __restrict__ in = nullptr, float* __restrict__ park = nullptr) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, g0 = ga, per = (gb < 0 ? KG : gb) - ga;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // each thread's copies, addressed once: its input chunk column and rows r0 + i * (THREADS / CHUNKS), its weight
    // tiles' chunk, and (threads below the scales' chunks) its scale or bias chunk; a stage only adds g's offsets
    static_assert(T::THREADS % T::CHUNKS == 0 && (BM * T::CHUNKS) % T::THREADS == 0, "whole input chunks a thread");
    constexpr int XC = BM * T::CHUNKS / T::THREADS, RSTEP = T::THREADS / T::CHUNKS;
    constexpr int TILE_BYTES = 64 * GS / 2, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const int ch = tid % T::CHUNKS, r0 = tid / T::CHUNKS;
    const int xdst = r0 * T::ROW + swz<T::CHUNKS>(r0, ch) * 16;  // r0 + RSTEP * i keeps r % CHUNKS (RSTEP % CHUNKS == 0)
    static_assert(RSTEP % T::CHUNKS == 0, "the swizzle's row phase is the thread's");
    const __nv_bfloat16* xsrc = x + static_cast<size_t>(m0 + r0) * ldx + ch * 8;
    const unsigned char* wsrc[WC];
    bool wok[WC];
#pragma unroll
    for (int k = 0; k < WC; ++k) {
        const int c = tid + k * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[k] = c < T::W / 16 && n0 + t * 64 < npad;
        wsrc[k] = reinterpret_cast<const unsigned char*>(w) + static_cast<size_t>(n0 / 64 + t) * KG * TILE_BYTES + off * 16;
    }
    const bool sok = tid < 2 * (T::S / 16) && n0 + (tid % (T::S / 16)) * 8 < npad;
    const __nv_bfloat16* ssrc = ((tid / (T::S / 16)) ? biases : scales) + n0 + (tid % (T::S / 16)) * 8;
    const int sdst = (tid / (T::S / 16)) * T::S + (tid % (T::S / 16)) * 16;
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XC; ++i) {
            const bool ok = m0 + r0 + i * RSTEP < M;
            cp16z(p + xdst + i * RSTEP * T::ROW, ok ? xsrc + static_cast<size_t>(i * RSTEP) * ldx + g * GS : x, ok);
        }
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k)
            if (wok[k]) cp16(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES);
        if (sok) cp16(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad);
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            if (in) {
                const float4 v = __ldcg(reinterpret_cast<const float4*>(in) + i * T::NT + j);
                acc[i][j][0] = v.x; acc[i][j][1] = v.y; acc[i][j][2] = v.z; acc[i][j][3] = v.w;
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
            }
        }
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
        // Prefetch A along K and B along N while preserving each accumulator's MMA chain.
        uint32_t a[2][T::MT][4];
        auto load_a = [&](int kt, int slot) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[slot][i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
        };
        load_a(0, 0);
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            if (kt + 1 < GS / 16) load_a(kt + 1, (kt + 1) & 1);
            uint32_t b[2][2];
            auto load_b = [&](int j, int slot) {
                b[slot][0] = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                b[slot][1] = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
            };
            load_b(0, 0);
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                if (j + 1 < T::NT) load_b(j + 1, (j + 1) & 1);
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
                    mma(acc[i][j], a[kt & 1][i], b[j & 1][0], b[j & 1][1]);
            }
        }
    }
    wait<0>();
    if (park) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
                reinterpret_cast<float4*>(park)[i * T::NT + j] = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
        return;
    }
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }

}

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_many_body(
        const __nv_bfloat16* __restrict__ x, const ManyParts& parts, int M, int K, int ldx, int group) {
    int q = 0;
#pragma unroll
    for (int i = 1; i < MANY; ++i)
        if (i < parts.count && (int)blockIdx.x >= parts.p[i].first) q = i;
    const ManyPart& P = parts.p[q];
    const int2 at = tile_of((int)blockIdx.x - P.first, M, P.n, BM, BN, group);
    prompt_tile<GS, BM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_sk_body(const __nv_bfloat16* __restrict__ x, const ManyParts& parts, int M,
                                               int K, int ldx, int group, int tiles, float* __restrict__ ws,
                                               unsigned* __restrict__ flags, unsigned epoch) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    constexpr int ACC = T::MT * T::NT * 4;
    const int tid = threadIdx.x, c = blockIdx.x, KG = K / GS;
    const long long I = (long long)tiles * KG;
    const long long s0 = I * c / gridDim.x, e0 = I * (c + 1) / gridDim.x;
    const int t_first = (int)(s0 / KG), off_first = (int)(s0 % KG);
    const int t_last = (int)((e0 - 1) / KG), end_last = (int)((e0 - 1) % KG) + 1;
    auto tile = [&](int t, int ga, int gb, const float* in, float* park) {
        int q = 0;
#pragma unroll
        for (int i = 1; i < MANY; ++i)
            if (i < parts.count && t >= parts.p[i].first) q = i;
        const ManyPart& P = parts.p[q];
        const int2 at = tile_of(t - P.first, M, P.n, BM, BN, group);
        __syncthreads();
        prompt_tile<GS, BM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y,
                                                ga, gb, in, park);
    };
    const bool head = end_last < KG;
    if (head) {
        tile(t_last, 0, end_last, nullptr, ws + (static_cast<size_t>(c) * T::THREADS + tid) * ACC);
        __threadfence();
        __syncthreads();
        if (tid == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(flags + c), "r"(epoch) : "memory");
    }
    for (int t = off_first ? t_first + 1 : t_first; t < (head ? t_last : t_last + 1); ++t) tile(t, 0, -1, nullptr, nullptr);
    if (off_first) {
        if (tid == 0) {
            unsigned f;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(f) : "l"(flags + c - 1) : "memory");
            } while (f != epoch);
        }
        __syncthreads();
        __threadfence();
        tile(t_first, off_first, -1, ws + (static_cast<size_t>(c - 1) * T::THREADS + tid) * ACC, nullptr);
    }
}

// One projection whose last column panels run as half-width tiles: the first CTAs take BN-wide panels over columns
// [0, full_cols), the rest BN/2-wide panels over [full_cols, N). A last wave of whole tiles leaves SMs idle; half
// tiles there even the SMs' shares. Each tile keeps prompt_tile's chain, so the bits are prefill_kernel's.
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_split_body(
        const __nv_bfloat16* __restrict__ x, const ManyPart& P, int M, int K, int ldx, int group, int full_cols) {
    const int rows_t = (M + BM - 1) / BM, full = rows_t * (full_cols / BN);
    if ((int)blockIdx.x < full) {
        const int2 at = tile_of((int)blockIdx.x, M, full_cols, BM, BN, group);
        prompt_tile<GS, BM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
    } else {
        const int2 at = tile_of((int)blockIdx.x - full, M, P.n - full_cols, BM, BN / 2, group);
        prompt_tile<GS, BM, BN / 2, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x,
                                                    full_cols + at.y);
    }
}


// A dense MLP's gate and up prompt GEMMs and its activation in one launch: out = bf16(silu(gate) * up), the bits
// of prefill_kernel's gate and up (each rounded to bf16 as it stores them) through tf_q35_mlp_act's formula.
// Each warp owns corresponding gate and up fragments. Both projections round to bf16 before the original
// SiLU formula, so the activation stays in registers without a shared gate tile or an epilogue barrier.
template <int GS, int BM, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_mlp_body(
        const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int group) {
    constexpr int BN = 128;
    static_assert(WN % 2 == 0, "gate and up use equal column-warp groups");
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    static_assert(T::NT % 2 == 0, "paired gate and up fragments");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    auto b_column = [&](int j) { return (j / (T::NT / 2)) * 8 + wn * (T::NT / 2) + j % (T::NT / 2); };
    const ManyPart& G = parts.p[0];
    const ManyPart& U = parts.p[1];
    const int N = G.n, npad = G.npad;
    const int KG = K / GS, per = KG, g0 = 0;
    const int2 at = tile_of((int)blockIdx.x, M, N, BM, 64, group);
    const int m0 = at.x, n0 = at.y;                 // n0: the 64 output columns, of gate and of up

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // each thread's copies addressed once (prompt_tile's): a stage only adds g's offsets
    constexpr int XC = BM * T::CHUNKS / T::THREADS, RSTEP = T::THREADS / T::CHUNKS;
    static_assert(T::THREADS % T::CHUNKS == 0 && XC * T::THREADS == BM * T::CHUNKS && RSTEP % T::CHUNKS == 0, "whole chunks");
    constexpr int TILE_BYTES = 64 * GS / 2, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const int ch = tid % T::CHUNKS, r0 = tid / T::CHUNKS;
    const int xdst = r0 * T::ROW + swz<T::CHUNKS>(r0, ch) * 16;
    const __nv_bfloat16* xsrc = x + static_cast<size_t>(m0 + r0) * ldx + ch * 8;
    const unsigned char* wsrc[WC];
    bool wok[WC];
#pragma unroll
    for (int k = 0; k < WC; ++k) {
        const int c = tid + k * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);   // t 0: gate's tile, 1: up's
        wok[k] = c < T::W / 16;
        wsrc[k] = reinterpret_cast<const unsigned char*>(t ? U.w : G.w) + static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + off * 16;
    }
    const int s_which = tid / (T::S / 16), s_off = tid % (T::S / 16), s_t = s_off / 8;
    const bool sok = tid < 2 * (T::S / 16);
    const __nv_bfloat16* ssrc = (s_which ? (s_t ? U.biases : G.biases) : (s_t ? U.scales : G.scales)) + n0 + (s_off % 8) * 8;
    const int sdst = s_which * T::S + s_off * 16;
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XC; ++i) {
            const bool ok = m0 + r0 + i * RSTEP < M;
            cp16z(p + xdst + i * RSTEP * T::ROW, ok ? xsrc + static_cast<size_t>(i * RSTEP) * ldx + g * GS : x, ok);
        }
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k)
            if (wok[k]) cp16(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES);
        if (sok) cp16(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad);
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[(b_column(j) * 32 + lane) * (GS / 32) + v];
            const int col = b_column(j) * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
        // Refill each A row after its last column use; carry B prefetch across K steps.
        uint32_t a[T::MT][4];
        auto load_a_row = [&](int kt, int i) {
            const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            const int ch = kt * 2 + (lane >> 4);
            ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
        };
#pragma unroll
        for (int i = 0; i < T::MT; ++i) load_a_row(0, i);
        auto weight_pair = [&](int j, int kt) {
            return make_uint2(fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]),
                              fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]));
        };
        uint2 b[2];
        b[0] = weight_pair(0, 0);
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int next_j = (j + 1) % T::NT, next_kt = kt + (j + 1 == T::NT);
                const int slot = (kt * T::NT + j) & 1;
                if (next_kt < GS / 16) b[slot ^ 1] = weight_pair(next_j, next_kt);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    mma(acc[i][j], a[i], b[slot].x, b[slot].y);
                    if (j + 1 == T::NT && kt + 1 < GS / 16) load_a_row(kt + 1, i);
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
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) prompt_many_kernel(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ ManyParts parts, int M, int K, int ldx, int group) {
    prompt_many_body<GS, BM, BN, WM, WN, STAGES>(x, parts, M, K, ldx, group);
}

} // namespace tf_qmm_prefill

#define TF_PROMPT_MANY_V(NAME, WM, WN, ST)                                                                          \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(const __nv_bfloat16* __restrict__ x,            \
            const __grid_constant__ tf_qmm_prefill::ManyParts parts, int M, int K, int ldx, int group) {            \
        tf_qmm_prefill::prompt_many_body<64, 128, 128, WM, WN, ST>(x, parts, M, K, ldx, group);                     \
    }
#define TF_PROMPT_SK_V(NAME, WM, WN, ST)                                                                            \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(const __nv_bfloat16* __restrict__ x,            \
            const __grid_constant__ tf_qmm_prefill::ManyParts parts, int M, int K, int ldx, int group, int tiles,   \
            float* __restrict__ ws, unsigned* __restrict__ flags, unsigned epoch) {                                 \
        tf_qmm_prefill::prompt_sk_body<64, 128, 128, WM, WN, ST>(x, parts, M, K, ldx, group, tiles, ws, flags,      \
                                                                 epoch);                                            \
    }
#define TF_PROMPT_SPLIT_V(NAME, WM, WN, ST)                                                                         \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(const __nv_bfloat16* __restrict__ x,            \
            const __grid_constant__ tf_qmm_prefill::ManyPart part, int M, int K, int ldx, int group,                \
            int full_cols) {                                                                                        \
        tf_qmm_prefill::prompt_split_body<64, 128, 128, WM, WN, ST>(x, part, M, K, ldx, group, full_cols);          \
    }
#define TF_PROMPT_MLP_V(NAME, WM, WN, ST)                                                                           \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32) NAME(const __nv_bfloat16* __restrict__ x,            \
            const __grid_constant__ tf_qmm_prefill::ManyParts parts, __nv_bfloat16* __restrict__ act, int M, int K, \
            int ldx, int group) {                                                                                   \
        tf_qmm_prefill::prompt_mlp_body<64, 128, WM, WN, ST>(x, parts, act, M, K, ldx, group);                      \
    }

TF_PROMPT_MANY_V(tf_pmv_many_222, 2, 2, 2)
TF_PROMPT_MANY_V(tf_pmv_many_242, 2, 4, 2)
TF_PROMPT_MANY_V(tf_pmv_many_223, 2, 2, 3)
TF_PROMPT_MANY_V(tf_pmv_many_143, 1, 4, 3)
TF_PROMPT_SK_V(tf_pmv_sk_222, 2, 2, 2)
TF_PROMPT_SK_V(tf_pmv_sk_242, 2, 4, 2)
TF_PROMPT_SPLIT_V(tf_pmv_split_242, 2, 4, 2)
TF_PROMPT_SPLIT_V(tf_pmv_split_143, 1, 4, 3)
TF_PROMPT_MLP_V(tf_pmv_mlp_222, 2, 2, 2)
TF_PROMPT_MLP_V(tf_pmv_mlp_242, 2, 4, 2)
TF_PROMPT_MLP_V(tf_pmv_mlp_143, 1, 4, 3)
