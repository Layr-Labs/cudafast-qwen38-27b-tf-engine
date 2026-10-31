// Row-fragment streaming preserves each output's original K64 chain.
#pragma once
namespace tf_qmm_rows0 {
using namespace tf_qmm_prefill;
// One BM x BN output tile at (m0, n0) of x through (w, scales, biases) into out (rows N wide): prefill_kernel's chain.
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
        uint32_t a[2][4];
        auto load_a = [&](int kt,int i,int slot){
            const int r=wm*(BM/WM)+i*16+(lane&7)+((lane>>3)&1)*8;
            const int ch=kt*2+(lane>>4);
            ldmatrix4(a[slot],p+r*T::ROW+swz<T::CHUNKS>(r,ch)*16);
        };
#pragma unroll
        for(int kt=0;kt<GS/16;kt++){
            uint2 b[T::NT];
#pragma unroll
            for(int j=0;j<T::NT;j++){
                const uint32_t word=words[j][kt/2], scale=sv[j], bias=bv[j];
                b[j]=make_uint2(fma2(pair(word,(kt&1)*8),scale,bias),
                                fma2(pair(word,(kt&1)*8+4),scale,bias));
            }
            load_a(kt,0,0);
#pragma unroll
            for(int i=0;i<T::MT;i++){
                if(i+1<T::MT)load_a(kt,i+1,(i+1)&1);
#pragma unroll
                for(int j=0;j<T::NT;j++)mma(acc[i][j],a[i&1],b[j].x,b[j].y);
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

// A row tile whose rows (below M) fill no more than its first BM - 16 * WM rows: every warp's last m16 row tile is
// padding only. Such a tile runs the instantiation one m16 tile a warp shorter (its loader zero-fills the same rows,
// its stored rows keep their MMA chains), so no warp spends ldmatrix or MMA issue on rows it never stores. The tile's
// position (m0, n0) still comes from the BM-row grid.
template <int BM, int WM>
__device__ __forceinline__ bool short_tile(int rows) {
    return BM / WM > 16 && rows > 0 && rows <= BM - 16 * WM && rows > BM - 32 * WM;
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
    if (tf_qmm_rows0::short_tile<BM, WM>(M - at.x))
        tf_qmm_rows0::prompt_tile<GS, BM - 16 * WM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
    else
        tf_qmm_rows0::prompt_tile<GS, BM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
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
        if (tf_qmm_rows0::short_tile<BM, WM>(M - at.x))
            tf_qmm_rows0::prompt_tile<GS, BM - 16 * WM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
        else
            tf_qmm_rows0::prompt_tile<GS, BM, BN, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
    } else {
        const int2 at = tile_of((int)blockIdx.x - full, M, P.n - full_cols, BM, BN / 2, group);
        if (tf_qmm_rows0::short_tile<BM, WM>(M - at.x))
            tf_qmm_rows0::prompt_tile<GS, BM - 16 * WM, BN / 2, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx,
                                                                at.x, full_cols + at.y);
        else
            tf_qmm_rows0::prompt_tile<GS, BM, BN / 2, WM, WN, STAGES>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x,
                                                        full_cols + at.y);
    }
}


// A dense MLP's gate and up prompt GEMMs and its activation in one launch: out = bf16(silu(gate) * up), the bits
// of prefill_kernel's gate and up (each rounded to bf16 as it stores them) through tf_q35_mlp_act's formula.
// Each warp owns corresponding gate and up fragments. Both projections round to bf16 before the original
// SiLU formula, so the activation stays in registers without a shared gate tile or an epilogue barrier.
template <int GS, int BM, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_mlp_tile(
        const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int m0, int n0) {
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
    const int KG = K / GS, per = KG, g0 = 0;      // n0: the 64 output columns, of gate and of up

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
        uint32_t a[2][4];
        auto load_a = [&](int kt,int i,int slot){
            const int r=wm*(BM/WM)+i*16+(lane&7)+((lane>>3)&1)*8;
            const int ch=kt*2+(lane>>4);
            ldmatrix4(a[slot],p+r*T::ROW+swz<T::CHUNKS>(r,ch)*16);
        };
#pragma unroll
        for(int kt=0;kt<GS/16;kt++){
            uint2 b[T::NT];
#pragma unroll
            for(int j=0;j<T::NT;j++){
                const uint32_t word=words[j][kt/2], scale=sv[j], bias=bv[j];
                b[j]=make_uint2(fma2(pair(word,(kt&1)*8),scale,bias),
                                fma2(pair(word,(kt&1)*8+4),scale,bias));
            }
            load_a(kt,0,0);
#pragma unroll
            for(int i=0;i<T::MT;i++){
                if(i+1<T::MT)load_a(kt,i+1,(i+1)&1);
#pragma unroll
                for(int j=0;j<T::NT;j++)mma(acc[i][j],a[i&1],b[j].x,b[j].y);
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

// prompt_mlp_tile at the BM-row grid's (m0, n0); a short last row tile (short_tile) runs one m16 tile a warp less.
template <int GS, int BM, int WM, int WN, int STAGES>
__device__ __forceinline__ void prompt_mlp_body(
        const __nv_bfloat16* __restrict__ x, const ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K,
        int ldx, int group) {
    const int2 at = tile_of((int)blockIdx.x, M, parts.p[0].n, BM, 64, group);
    if (tf_qmm_rows0::short_tile<BM, WM>(M - at.x))
        tf_qmm_rows0::prompt_mlp_tile<GS, BM - 16 * WM, WM, WN, STAGES>(x, parts, act, M, K, ldx, at.x, at.y);
    else
        tf_qmm_rows0::prompt_mlp_tile<GS, BM, WM, WN, STAGES>(x, parts, act, M, K, ldx, at.x, at.y);
}

template <int GS, int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) prompt_many_kernel(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ ManyParts parts, int M, int K, int ldx, int group) {
    prompt_many_body<GS, BM, BN, WM, WN, STAGES>(x, parts, M, K, ldx, group);
}


}
