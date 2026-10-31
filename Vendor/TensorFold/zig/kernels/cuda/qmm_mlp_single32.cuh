// tf_qmm_mlp_single16's tile for verify windows of 17 to 32 rows: two m16 row tiles share each K-major weight panel
// in one CTA (one weight pass for the window, where the 16-row tiles stream every weight once a tile). 128 gate and
// 128 up columns a CTA; each output keeps tf_qmm_mlp_single16's chain: the same m16n8k16 MMAs over K, the same fmas,
// the same bf16 activation and group sums, so the same bits a row as a 16-row window computes.
#pragma once
namespace tf_qmm_mlp_rows32 {
using namespace tf_qmm_group;
template <int STAGES, int BN, int BM>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                     const Parts& parts, __nv_bfloat16* __restrict__ act,
                                     float* __restrict__ as, int M, int K, int ldx, int rows_t) {
    constexpr int GS = 64, WM = 1, WN = 4;
    using T = LaneTile<GS, BM, BN, WM, WN, STAGES>;
    constexpr int I = T::MT, J = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, cid = blockIdx.x;
    const Part P = parts.p[0];
    const int tile = cid / rows_t;
    const int m0 = cid % rows_t * BM, n0 = tile * (BN / 2), per = KG;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(BM, M - m0);
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = (BM * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    static_assert(BM <= T::THREADS, "one input sum a thread");
    const __nv_bfloat16* xsrc[XC];
    const unsigned char* wsrc[WC];
    const __nv_bfloat16* ssrc[SC];
    int xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < BM * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(m0 + min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        wsrc[j] = reinterpret_cast<const unsigned char*>(parts.p[t / (BN / 128)].w) +
                  static_cast<size_t>(n0 / 64 + t % (BN / 128)) * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        const auto Q = parts.p[(off * 8) / (BN / 2)];
        ssrc[j] = (which ? Q.biases : Q.scales) + n0 + (off * 8) % (BN / 2);
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(m0 + min(tid, rows - 1)) * KG;
    const size_t wstride = static_cast<size_t>(P.npad / 64) * TILE_BYTES;
    auto load_x = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wsrc[j] + static_cast<size_t>(g) * wstride);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], ssrc[j] + static_cast<size_t>(g) * P.npad);
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
    for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_w(s, s);
        commit();
    }
    grid_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load_x(s, s);
        commit();
    }
    grid_launch();
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[T::NT][GS / 32];
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
        float d[I][J][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[I][4];
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
#pragma unroll
                for (int i = 0; i < I; ++i) {
                    if (kt == 0) mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
            const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
            const float sv[2] = {__low2float(s2), __high2float(s2)};
            const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
            for (int i = 0; i < I; ++i) {
                const int row = wm * (BM / WM) + i * 16 + (lane >> 2);
                const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[i][j][e], sv[e & 1], acc[i][j][e]));
            }
        }
    }
    wait<0>();
    // gate's and up's activations of the CTA's columns, gate then up, as tf_qmm_mlp_single16 parks them
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
                    __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
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

// 32 rows, 128 gate and 128 up columns a CTA (136 CTAs at the 27B's 17408), two stages.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_mlp_single32(
    const __nv_bfloat16* x, const float* xs, const __grid_constant__ tf_qmm_group::Parts parts,
    __nv_bfloat16* act, float* sums, int M, int K, int ldx, int rows_t) {
    tf_qmm_mlp_rows32::body<2, 256, 32>(x, xs, parts, act, sums, M, K, ldx, rows_t);
}
