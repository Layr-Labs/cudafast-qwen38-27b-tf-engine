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

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_tile(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int N, int npad, int M, int K, int ldx, int m0,
        int n0) {
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG, g0 = 0;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            if (n0 + t * 64 < npad)
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
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
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
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
// A CTA's 128 GEMM columns are 64 gate columns (warps wn 0) and the same 64 up columns (warps wn 1); the gate
// warps park their bf16 values in shared memory and the up warps write the activation.
template <int GS, int BM, int WM, int STAGES>
__device__ __forceinline__ void prompt_mlp_body(
        const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int group) {
    constexpr int BN = 128, WN = 2;
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    static_assert(BM * 64 * 2 <= T::SMEM, "the gate tile fits the stages' memory");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const ManyPart& G = parts.p[0];
    const ManyPart& U = parts.p[1];
    const int N = G.n, npad = G.npad;
    const int KG = K / GS, per = KG, g0 = 0;
    const int2 at = tile_of((int)blockIdx.x, M, N, BM, 64, group);
    const int m0 = at.x, n0 = at.y;                 // n0: the 64 output columns, of gate and of up

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);   // t 0: gate's tile, 1: up's
            const size_t tile = static_cast<size_t>(n0 / 64) * KG + g;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(t ? U.w : G.w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16), t = off / 8;
            const ManyPart& P = t ? U : G;
            const __nv_bfloat16* src = (which ? P.biases : P.scales) + static_cast<size_t>(g) * npad + n0 + (off % 8) * 8;
            cp16(ps + which * T::S + off * 16, src);
        }
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
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    __syncthreads();                                  // every stage read: the gate tile takes their memory
    __nv_bfloat16* gt = reinterpret_cast<__nv_bfloat16*>(buf);   // (BM, 64) bf16 gate values
    if (wn == 0) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8, c = j * 8 + (lane & 3) * 2;
                    *reinterpret_cast<__nv_bfloat162*>(gt + r * 64 + c) =
                        __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
    }
    __syncthreads();
    if (wn == 1) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8, c = j * 8 + (lane & 3) * 2;
                    const int row = m0 + r, col = n0 + c;
                    if (row >= M) continue;
                    const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(gt + r * 64 + c);
                    const __nv_bfloat162 u2 = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                    const float gf[2] = {__low2float(g2), __high2float(g2)};
                    const float uf[2] = {__low2float(u2), __high2float(u2)};
                    __nv_bfloat16 o[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) o[e] = __float2bfloat16_rn(gf[e] / (1.0f + expf(-gf[e])) * uf[e]);
                    __nv_bfloat16* dst = act + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N) {
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(o[0], o[1]);
                    } else if (col < N) {
                        dst[0] = o[0];
                    }
                }
    }
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) prompt_many_kernel(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ ManyParts parts, int M, int K, int ldx, int group) {
    prompt_many_body<GS, BM, BN, WM, WN, STAGES>(x, parts, M, K, ldx, group);
}

} // namespace tf_qmm_prefill
