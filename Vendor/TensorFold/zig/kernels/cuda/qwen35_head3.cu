#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_q35h3 {

constexpr int GS = 64;
constexpr int TILE_WORDS = 512;
constexpr int CODE_WORDS = 256;
constexpr int STAGES = 8;
constexpr int CB = CODE_WORDS * 4, SB = 64 * 4;
template <int RT>
struct Stage {
    static constexpr int XB = RT * 16 * GS * 2, PB = RT * 16 * 4, QB = RT * 16 * 4;
    static constexpr int BYTES = XB + CB + SB + PB + QB;
};
constexpr int TCAP = 1024;
constexpr int RS_KG = 80;

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}
__device__ __forceinline__ uint32_t pairm(uint32_t w, int s, uint32_t mask) {
    const uint32_t t = ((w >> s) & mask) | 0x43004300u;
    uint32_t r;
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
    return r;
}
template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) { return c ^ (r % CHUNKS); }

struct Chunks {
    uint32_t* p[4];
    int tpc;
};
__device__ __forceinline__ uint32_t* chunk_at(const Chunks& c, int tile, int KG) {
    const int k = tile / c.tpc;
    return c.p[k] + static_cast<size_t>(tile - k * c.tpc) * KG * CODE_WORDS;
}

__device__ __forceinline__ unsigned key16(unsigned b) { return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u); }
__device__ __forceinline__ unsigned unkey16(unsigned k) { return (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu); }

__device__ __forceinline__ void pack(const uint32_t* __restrict__ w4, const __nv_bfloat16* __restrict__ s,
                                     const __nv_bfloat16* __restrict__ b, int npad, int KG, const Chunks& codes,
                                     __nv_bfloat162* __restrict__ scales, __nv_bfloat162* __restrict__ sbcol) {
    const int tile = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const uint32_t* src = w4 + (static_cast<size_t>(tile) * KG + g) * TILE_WORDS + t * 2;
    const uint32_t w[2] = {src[0], src[1]};
    uint32_t out = 0;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int sh = (kt & 1) * 8 + h * 4, p = 2 * kt + h;
            const uint32_t lo = (w[kt / 2] >> sh) & 15u, hi = (w[kt / 2] >> (sh + 16)) & 15u;
            out |= (lo >> 2) << (2 * p) | (hi >> 2) << (16 + 2 * p);
        }
    chunk_at(codes, tile, KG)[static_cast<size_t>(g) * CODE_WORDS + t] = out;
    if (t < 64) {
        const int col = tile * 64 + t;
        const float sv = __bfloat162float(s[static_cast<size_t>(g) * npad + col]);
        const float bv = __bfloat162float(b[static_cast<size_t>(g) * npad + col]);
        scales[(static_cast<size_t>(tile) * KG + g) * 64 + t] = __floats2bfloat162_rn(4.0f * sv, bv + 1.5f * sv);
        __nv_bfloat162 sb;
        sb.x = s[static_cast<size_t>(g) * npad + col];
        sb.y = b[static_cast<size_t>(g) * npad + col];
        sbcol[static_cast<size_t>(col) * KG + g] = sb;
    }
}

__device__ __forceinline__ void sq(const __nv_bfloat16* __restrict__ x, int KG, float* __restrict__ xq) {
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const __nv_bfloat162* row = reinterpret_cast<const __nv_bfloat162*>(x + static_cast<size_t>(r) * KG * GS);
    for (int g = warp; g < KG; g += warps) {
        const float2 v = __bfloat1622float2(row[g * (GS / 2) + lane]);
        float s = v.x * v.x + v.y * v.y;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (lane == 0) xq[static_cast<size_t>(r) * KG + g] = s;
    }
}

template <int RT>
__device__ __forceinline__ void screen(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                       const float* __restrict__ xq, const Chunks& codes,
                                       const __nv_bfloat162* __restrict__ scales, int M, int KG, float kappa,
                                       __nv_bfloat16* __restrict__ out, int ldo, __nv_bfloat16* __restrict__ tmax,
                                       int tiles) {
    using St = Stage<RT>;
    constexpr int XB = St::XB, PB = St::PB, STAGE = St::BYTES;
    extern __shared__ __align__(128) unsigned char buf[];
    __shared__ float wmax[4][RT * 16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, tile = blockIdx.x;
    const int rows = min(RT * 16, M);
    const uint32_t* tile_codes = chunk_at(codes, tile, KG);
    auto stage = [&](int s) { return buf + s * STAGE; };
    for (int c = tid; c < RT * 16 * 8; c += 128)
        if (c / 8 >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s) *reinterpret_cast<uint4*>(stage(s) + (c / 8) * 128 + (c % 8) * 16) = uint4{};
    __syncthreads();
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
#pragma unroll
        for (int c = tid; c < RT * 16 * 8; c += 128) {
            const int r = c / 8, ch = c % 8;
            if (r < rows) cp16(p + r * 128 + swz<8>(r, ch) * 16, x + static_cast<size_t>(r) * KG * GS + g * GS + ch * 8);
        }
        const unsigned char* cb = reinterpret_cast<const unsigned char*>(tile_codes + static_cast<size_t>(g) * CODE_WORDS);
        if (tid < CB / 16) cp16(p + XB + tid * 16, cb + tid * 16);
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(scales + (static_cast<size_t>(tile) * KG + g) * 64);
        if (tid >= 64 && tid < 64 + SB / 16) cp16(p + XB + CB + (tid - 64) * 16, sb + (tid - 64) * 16);
        if (tid < rows) cp4(p + XB + CB + SB + tid * 4, xs + static_cast<size_t>(tid) * KG + g);
        if (tid >= 64 && tid < 64 + rows) cp4(p + XB + CB + SB + PB + (tid - 64) * 4, xq + static_cast<size_t>(tid - 64) * KG + g);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    float acc[RT][2][4] = {}, var[RT][2][4] = {};
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < KG) load((it + STAGES - 1) % STAGES, it + STAGES - 1);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* cw = reinterpret_cast<const uint32_t*>(p + XB);
        const __nv_bfloat162* sc = reinterpret_cast<const __nv_bfloat162*>(p + XB + CB);
        const float* px = reinterpret_cast<const float*>(p + XB + CB + SB);
        const float* pq = reinterpret_cast<const float*>(p + XB + CB + SB + PB);
        const uint32_t w[2] = {cw[(warp * 2) * 32 + lane], cw[(warp * 2 + 1) * 32 + lane]};
        float d[RT][2][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t b0[2], b1[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                b0[j] = pairm(w[j], 2 * (2 * kt), 0x00030003u);
                b1[j] = pairm(w[j], 2 * (2 * kt + 1), 0x00030003u);
            }
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                uint32_t a[4];
                const int r = rt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
                ldmatrix4(a, p + r * 128 + swz<8>(r, ch) * 16);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if (kt == 0) mma0(d[rt][j], a, b0[j], b1[j]);
                    else mma(d[rt][j], a, b0[j], b1[j]);
                }
            }
        }
#pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
            const float xv[2] = {px[rt * 16 + (lane >> 2)], px[rt * 16 + (lane >> 2) + 8]};
            const float qv[2] = {pq[rt * 16 + (lane >> 2)], pq[rt * 16 + (lane >> 2) + 8]};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c0 = (warp * 2 + j) * 8 + (lane & 3) * 2;
                const float2 s0 = __bfloat1622float2(sc[c0]), s1 = __bfloat1622float2(sc[c0 + 1]);
                acc[rt][j][0] += d[rt][j][0] * s0.x + xv[0] * s0.y;
                acc[rt][j][1] += d[rt][j][1] * s1.x + xv[0] * s1.y;
                acc[rt][j][2] += d[rt][j][2] * s0.x + xv[1] * s0.y;
                acc[rt][j][3] += d[rt][j][3] * s1.x + xv[1] * s1.y;
                const float q0 = s0.x * s0.x, q1 = s1.x * s1.x;
                var[rt][j][0] += q0 * qv[0];
                var[rt][j][1] += q1 * qv[0];
                var[rt][j][2] += q0 * qv[1];
                var[rt][j][3] += q1 * qv[1];
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        float m[2] = {__int_as_float(0xff800000), __int_as_float(0xff800000)};
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = rt * 16 + (lane >> 2) + h * 8, col = tile * 64 + (warp * 2 + j) * 8 + (lane & 3) * 2;
                const __nv_bfloat16 u0 = __float2bfloat16_ru(acc[rt][j][2 * h] + kappa * sqrtf(0.078125f * var[rt][j][2 * h]));
                const __nv_bfloat16 u1 = __float2bfloat16_ru(acc[rt][j][2 * h + 1] + kappa * sqrtf(0.078125f * var[rt][j][2 * h + 1]));
                m[h] = fmaxf(m[h], fmaxf(__bfloat162float(u0), __bfloat162float(u1)));
                if (row < rows) {
                    __nv_bfloat162 pair;
                    pair.x = u0;
                    pair.y = u1;
                    *reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(row) * ldo + col) = pair;
                }
            }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            m[h] = fmaxf(m[h], __shfl_xor_sync(0xffffffffu, m[h], 1));
            m[h] = fmaxf(m[h], __shfl_xor_sync(0xffffffffu, m[h], 2));
            if ((lane & 3) == 0) wmax[warp][rt * 16 + (lane >> 2) + h * 8] = m[h];
        }
    }
    __syncthreads();
    if (tid < rows) {
        const float v = fmaxf(fmaxf(wmax[0][tid], wmax[1][tid]), fmaxf(wmax[2][tid], wmax[3][tid]));
        tmax[static_cast<size_t>(tid) * tiles + tile] = __float2bfloat16_rn(v);
    }
}

template <typename Get>
__device__ __forceinline__ unsigned kth(Get get, int n, int K, unsigned* hist, unsigned* s_bin, unsigned* s_above) {
    const int t = threadIdx.x, T = blockDim.x;
    unsigned above = 0, prefix = 0;
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = t; i < 256; i += T) hist[i] = 0;
        __syncthreads();
        for (int i = t; i < n; i += T) {
            const unsigned k = get(i);
            const bool in = pass == 0 || (k >> 8) == prefix;
            const unsigned bin = pass == 0 ? k >> 8 : (k & 255u);
            const unsigned mask = __match_any_sync(__activemask(), in ? bin : 0x100u);
            if (in && (t & 31) == __ffs(mask) - 1) atomicAdd(&hist[bin], __popc(mask));
        }
        __syncthreads();
        if (t == 0) {
            unsigned a = above;
            int b = 255;
            for (; b > 0; --b) {
                if (a + hist[b] >= static_cast<unsigned>(K)) break;
                a += hist[b];
            }
            *s_bin = b;
            *s_above = a;
        }
        __syncthreads();
        prefix = pass == 0 ? *s_bin : (prefix << 8 | *s_bin);
        above = *s_above;
        __syncthreads();
    }
    return prefix;
}

__device__ __forceinline__ void select(const __nv_bfloat16* __restrict__ U, int ldo, const __nv_bfloat16* __restrict__ TM,
                                       int tiles, int K, int cap, float* __restrict__ PV, unsigned* __restrict__ PI,
                                       float* __restrict__ CUT) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_bin, s_above, s_count;
    __shared__ int tl[TCAP];
    const int row = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const unsigned short* tm = reinterpret_cast<const unsigned short*>(TM + static_cast<size_t>(row) * tiles);
    const unsigned short* u = reinterpret_cast<const unsigned short*>(U + static_cast<size_t>(row) * ldo);
    const unsigned tk = kth([&](int i) { return key16(tm[i]); }, tiles, K, hist, &s_bin, &s_above);
    if (t == 0) s_count = 0;
    __syncthreads();
#pragma unroll 1
    for (int tie = 0; tie < 2; ++tie) {
        for (int i = t; i < tiles; i += T) {
            const unsigned k = key16(tm[i]);
            if (tie ? k == tk : k > tk) {
                const unsigned at = atomicAdd(&s_count, 1u);
                if (at < static_cast<unsigned>(TCAP)) tl[at] = i;
            }
        }
        __syncthreads();
    }
    const int nt = min(static_cast<int>(s_count), TCAP), n = nt * 64;
    auto col = [&](int i) { return tl[i >> 6] * 64 + (i & 63); };
    const unsigned cut = kth([&](int i) { return key16(u[col(i)]); }, n, K, hist, &s_bin, &s_above);
    if (t == 0) s_count = 0;
    __syncthreads();
    float* pv = PV + static_cast<size_t>(row) * cap;
    unsigned* pi = PI + static_cast<size_t>(row) * cap;
#pragma unroll 1
    for (int tie = 0; tie < 2; ++tie) {
        for (int i = t; i < n; i += T) {
            const unsigned k = key16(u[col(i)]);
            if (tie ? k == cut : k > cut) {
                const unsigned at = atomicAdd(&s_count, 1u);
                if (at < static_cast<unsigned>(cap)) pi[at] = static_cast<unsigned>(col(i));
            }
        }
        __syncthreads();
    }
    for (unsigned at = min(s_count, static_cast<unsigned>(cap)) + t; at < static_cast<unsigned>(cap); at += T) pi[at] = 0xFFFFFFFFu;
    for (int at = t; at < cap; at += T) pv[at] = __int_as_float(0xff800000);
    if (t == 0) CUT[row] = __uint_as_float(unkey16(cut) << 16);
}

__device__ __forceinline__ void rescore(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                        const uint32_t* __restrict__ w4, const __nv_bfloat162* __restrict__ sbcol,
                                        int KG, int M, float* __restrict__ PV, const unsigned* __restrict__ PI, int n,
                                        int V) {
    __shared__ float dsm[RS_KG][32];
    __shared__ int cols[32];
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int at = blockIdx.y * 32;
    if (at >= n || KG > RS_KG) return;
    const int rt = r / 16, rl = r % 16;
    x += static_cast<size_t>(rt) * 16 * KG * GS;
    M -= rt * 16;
    const size_t base = static_cast<size_t>(r) * n + at;
    if (threadIdx.x < 32) {
        int c = -1;
        if (at + lane < n) {
            const unsigned u = PI[base + lane];
            if (u < static_cast<unsigned>(V)) c = static_cast<int>(u);
        }
        cols[lane] = c;
    }
    __syncthreads();
    bool any = false;
#pragma unroll
    for (int i = 0; i < 32; ++i) any = any || cols[i] >= 0;
    if (!any) return;
    const int g4 = lane >> 2, t4 = lane & 3;
    const bool r0 = g4 < M, r1 = g4 + 8 < M;
    const __nv_bfloat16* x0 = x + static_cast<size_t>(r0 ? g4 : 0) * KG * GS + 2 * t4;
    const __nv_bfloat16* x1 = x + static_cast<size_t>(r1 ? g4 + 8 : 0) * KG * GS + 2 * t4;
    const uint32_t* wcol[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ub = max(cols[j * 8 + g4], 0);
        wcol[j] = w4 + static_cast<size_t>(ub / 64) * KG * TILE_WORDS + (((ub % 64) / 8) * 32 + (ub % 8) * 4 + t4) * 2;
    }
    const bool mine = (rl < 8 ? g4 == rl : g4 == rl - 8);
    const int e0 = rl < 8 ? 0 : 2;
    const int per = (KG + 3) / 4, g_lo = warp * per, g_hi = min(KG, g_lo + per);
#pragma unroll 2
    for (int g = g_lo; g < g_hi; ++g) {
        uint32_t a[4][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int k = g * GS + kt * 16;
            a[kt][0] = r0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
            a[kt][1] = r1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
            a[kt][2] = r0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
            a[kt][3] = r1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t w[2] = {wcol[j][static_cast<size_t>(g) * TILE_WORDS], wcol[j][static_cast<size_t>(g) * TILE_WORDS + 1]};
            float d[4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const uint32_t b0 = pairm(w[kt / 2], (kt & 1) * 8, 0x000F000Fu), b1 = pairm(w[kt / 2], (kt & 1) * 8 + 4, 0x000F000Fu);
                if (kt == 0) mma0(d, a[kt], b0, b1);
                else mma(d, a[kt], b0, b1);
            }
            if (mine) {
                dsm[g][j * 8 + t4 * 2] = d[e0];
                dsm[g][j * 8 + t4 * 2 + 1] = d[e0 + 1];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int c = cols[lane];
        if (c >= 0) {
            const __nv_bfloat162* sb = sbcol + static_cast<size_t>(c) * KG;
            const float* px = xs + static_cast<size_t>(r) * KG;
            float acc = 0.0f;
            for (int g = 0; g < KG; ++g) {
                const float2 p = __bfloat1622float2(sb[g]);
                acc = __fmaf_rn(px[g], p.y, __fmaf_rn(dsm[g][lane], p.x, acc));
            }
            PV[base + lane] = __bfloat162float(__float2bfloat16_rn(acc));
        }
    }
}

__device__ __forceinline__ void merge(const float* __restrict__ PV, const unsigned* __restrict__ PI, int n,
                                      const float* __restrict__ CUT, unsigned* __restrict__ toks) {
    const int r = blockIdx.x, lane = threadIdx.x;
    float best = __int_as_float(0xff800000);
    unsigned id = 0xFFFFFFFFu;
    for (int i = lane; i < n; i += 32) {
        const float v = PV[static_cast<size_t>(r) * n + i];
        const unsigned c = PI[static_cast<size_t>(r) * n + i];
        if (c != 0xFFFFFFFFu && (v > best || (v == best && c < id))) {
            best = v;
            id = c;
        }
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, m);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, id, m);
        if (ov > best || (ov == best && oi < id)) {
            best = ov;
            id = oi;
        }
    }
    if (lane == 0) {
        const unsigned b = __float_as_uint(best) >> 16;
        const unsigned under = best > 0.0f ? b - 1u : (best < 0.0f ? b + 1u : 0x8001u);
        const bool ok = id != 0xFFFFFFFFu && best > __int_as_float(0xff800000) && CUT[r] < __uint_as_float(under << 16);
        toks[r] = ok ? id : 0xFFFFFFFFu;
    }
}

}

extern "C" __global__ void __launch_bounds__(256) tf_q35h3_pack(const uint32_t* __restrict__ w4,
        const __nv_bfloat16* __restrict__ s, const __nv_bfloat16* __restrict__ b, int npad, int KG,
        const tf_q35h3::Chunks codes, __nv_bfloat162* __restrict__ scales, __nv_bfloat162* __restrict__ sbcol) {
    tf_q35h3::pack(w4, s, b, npad, KG, codes, scales, sbcol);
}

extern "C" __global__ void __launch_bounds__(256) tf_q35h3_sq(const __nv_bfloat16* __restrict__ x, int KG,
        float* __restrict__ xq) {
    tf_q35h3::sq(x, KG, xq);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35h3_screen(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const float* __restrict__ xq, const tf_q35h3::Chunks codes,
        const __nv_bfloat162* __restrict__ scales, int M, int KG, float kappa, __nv_bfloat16* __restrict__ out, int ldo,
        __nv_bfloat16* __restrict__ tmax, int tiles) {
    tf_q35h3::screen<1>(x, xs, xq, codes, scales, M, KG, kappa, out, ldo, tmax, tiles);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35h3_screen32(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const float* __restrict__ xq, const tf_q35h3::Chunks codes,
        const __nv_bfloat162* __restrict__ scales, int M, int KG, float kappa, __nv_bfloat16* __restrict__ out, int ldo,
        __nv_bfloat16* __restrict__ tmax, int tiles) {
    tf_q35h3::screen<2>(x, xs, xq, codes, scales, M, KG, kappa, out, ldo, tmax, tiles);
}

extern "C" __global__ void __launch_bounds__(1024) tf_q35h3_select(const __nv_bfloat16* __restrict__ U, int ldo,
        const __nv_bfloat16* __restrict__ TM, int tiles, int K, int cap, float* __restrict__ PV,
        unsigned* __restrict__ PI, float* __restrict__ CUT) {
    tf_q35h3::select(U, ldo, TM, tiles, K, cap, PV, PI, CUT);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35h3_rescore(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const uint32_t* __restrict__ w4, const __nv_bfloat162* __restrict__ sbcol, int KG,
        int M, float* __restrict__ PV, const unsigned* __restrict__ PI, int n, int V) {
    tf_q35h3::rescore(x, xs, w4, sbcol, KG, M, PV, PI, n, V);
}

extern "C" __global__ void __launch_bounds__(32) tf_q35h3_merge(const float* __restrict__ PV,
        const unsigned* __restrict__ PI, int n, const float* __restrict__ CUT, unsigned* __restrict__ toks) {
    tf_q35h3::merge(PV, PI, n, CUT, toks);
}
