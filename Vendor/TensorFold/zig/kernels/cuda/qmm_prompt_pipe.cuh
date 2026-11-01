// Prompt tile bodies whose stage barrier sits inside the group, for the persistent prompt schedules.
//
// Each output keeps prefill_kernel's bits: the same bf16 weights (fma2 of the nibble pair, scale and bias), the same
// m16n8k16 MMAs over K in group then k16 order into one fp32 accumulator from zero, the same bf16 stores (and for the
// MLP tile the same silu(gate) * up of the rounded gate and up). Only the order of independent MMAs and where the
// copies, the barrier and the next stage's register loads sit in the instruction stream differ.
//
// prompt_tile waits for a stage and syncs at the top of each group, then issues the next copies, reads the stage's
// words and scales, loads the first A rows and dequantizes the first B pair before its first MMA: about 260 cycles a
// group with no MMA a warp can issue, and on GB10 the SM's other CTA hides only part of that. Here the barrier comes in
// a group's last k16 step, after its last read of the stage: the next stage's words, scales, first A rows and first B
// pair load under that step's MMAs, and the next group opens with its operands in registers and its copies issued
// among its first MMAs. Copies are predicated instructions rather than branches so they schedule among the MMAs.
#pragma once

namespace tf_qmm_pipe {
using namespace tf_qmm_prefill;

// A 16-byte cp.async issued only where p holds: a predicated instruction rather than a branch, so the copies schedule
// among the MMAs. Rows past M are not read; their A rows only reach accumulator rows that are never stored.
__device__ __forceinline__ void cp16p(void* dst, const void* src, bool p) {
    asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %2, 0;\n\t@q cp.async.cg.shared.global [%0], [%1], 16;\n\t}\n" ::"r"(smem(dst)), "l"(src), "r"((int)p));
}

// The final group barrier ends all shared reads; persistent workers can reuse the stages on return.

// prompt_mlp_tile's outputs (gate and up through silu(gate) * up) with the barrier inside the group, two stages.
template <int GS, int BM, int WM, int WN, int STAGES>
__device__ __forceinline__ void mlp_tile(const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int m0, int n0) {
    constexpr int BN = 128;
    static_assert(STAGES == 2, "two stages: the next stage is read after the current one's last read");
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
    auto stage = [&](int s) { return buf + s * T::STAGE; };
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
        const int c = tid + k * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[k] = c < T::W / 16;
        wsrc[k] = reinterpret_cast<const unsigned char*>(t ? U.w : G.w) + static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + off * 16;
    }
    const int s_which = tid / (T::S / 16), s_off = tid % (T::S / 16), s_t = s_off / 8;
    const bool sok = tid < 2 * (T::S / 16);
    const __nv_bfloat16* ssrc = (s_which ? (s_t ? U.biases : G.biases) : (s_t ? U.scales : G.scales)) + n0 + (s_off % 8) * 8;
    const int sdst = s_which * T::S + s_off * 16;
    // row i of this thread's copies exists below M: bit i (fixed for the tile)
    unsigned xok = 0;
#pragma unroll
    for (int i = 0; i < XC; ++i) xok |= (m0 + r0 + i * RSTEP < M ? 1u : 0u) << i;
    auto load = [&](int s, int g, bool go) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XC; ++i)
            cp16p(p + xdst + i * RSTEP * T::ROW, xsrc + static_cast<size_t>(i * RSTEP) * ldx + g * GS, go && (xok >> i & 1u));
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k) cp16p(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES, go && wok[k]);
        cp16p(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad, go && sok);
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
                    wait<0>();
                    __syncthreads();
                    if (more) {
                        read_w(pn);
                        b[slot ^ 1] = weight_pair(0, 0);
                    }
                } else {
                    b[slot ^ 1] = weight_pair(next_j, next_kt);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    mma(acc[i][j], a[i], b[slot].x, b[slot].y);
                    if (j + 1 == T::NT && (kt + 1 < GS / 16 || more))
                        load_a_row(kt + 1 < GS / 16 ? p : pn, (kt + 1) % (GS / 16), i);
                }
            }
        }
    }
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

// prompt_tile's outputs (one projection, BN columns) with the barrier inside the group: mlp_tile's order (each A row
// held for its k16 step and refilled after its last column, B pairs double buffered), two stages.
template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void tile(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int N, int npad, int M, int K, int ldx, int m0,
        int n0) {
    static_assert(STAGES == 2, "two stages: the next stage is read after the current one's last read");
    using T = Tile<GS, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, per = KG, g0 = 0;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    static_assert(T::THREADS % T::CHUNKS == 0 && (BM * T::CHUNKS) % T::THREADS == 0, "whole input chunks a thread");
    constexpr int XC = BM * T::CHUNKS / T::THREADS, RSTEP = T::THREADS / T::CHUNKS;
    constexpr int TILE_BYTES = 64 * GS / 2, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    const int ch = tid % T::CHUNKS, r0 = tid / T::CHUNKS;
    const int xdst = r0 * T::ROW + swz<T::CHUNKS>(r0, ch) * 16;
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
    unsigned xok = 0;
#pragma unroll
    for (int i = 0; i < XC; ++i) xok |= (m0 + r0 + i * RSTEP < M ? 1u : 0u) << i;
    auto load = [&](int s, int g, bool go) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XC; ++i)
            cp16p(p + xdst + i * RSTEP * T::ROW, xsrc + static_cast<size_t>(i * RSTEP) * ldx + g * GS, go && (xok >> i & 1u));
        unsigned char* pw = p + T::X;
#pragma unroll
        for (int k = 0; k < WC; ++k) cp16p(pw + (tid + k * T::THREADS) * 16, wsrc[k] + static_cast<size_t>(g) * TILE_BYTES, go && wok[k]);
        cp16p(pw + T::W + sdst, ssrc + static_cast<size_t>(g) * npad, go && sok);
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
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
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
    load(0, g0, true);
    commit();
    wait<0>();
    __syncthreads();
    read_w(stage(0));
#pragma unroll
    for (int i = 0; i < T::MT; ++i) load_a_row(stage(0), 0, i);
    uint2 b[2];
    b[0] = weight_pair(0, 0);
    static_assert((GS / 16 * T::NT) % 2 == 0, "B slots alternate evenly through a group");
    for (int it = 0; it < per; ++it) {
        const unsigned char* p = stage(it & 1);
        const unsigned char* pn = stage((it + 1) & 1);
        const bool more = it + 1 < per;
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
                    wait<0>();
                    __syncthreads();
                    if (more) {
                        read_w(pn);
                        b[slot ^ 1] = weight_pair(0, 0);
                    }
                } else {
                    b[slot ^ 1] = weight_pair(next_j, next_kt);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    mma(acc[i][j], a[i], b[slot].x, b[slot].y);
                    if (j + 1 == T::NT && (kt + 1 < GS / 16 || more))
                        load_a_row(kt + 1 < GS / 16 ? p : pn, (kt + 1) % (GS / 16), i);
                }
            }
        }
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

} // namespace tf_qmm_pipe
