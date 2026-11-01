#pragma once

namespace tf_q35d_head2 {

constexpr int GS = 64;
constexpr int TILE_WORDS = 512;
constexpr int CODE_WORDS = 256;
constexpr int STAGES = 8;

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(read ? 16 : 0));
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

__device__ __forceinline__ void pack(const uint32_t* __restrict__ w4, const __nv_bfloat16* __restrict__ s,
                                            const __nv_bfloat16* __restrict__ b, int npad, int KG,
                                            uint32_t* __restrict__ codes, __nv_bfloat162* __restrict__ scales,
                                            __nv_bfloat162* __restrict__ sbcol) {
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
    codes[(static_cast<size_t>(tile) * KG + g) * CODE_WORDS + t] = out;
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

__device__ __forceinline__ void screen(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                              const uint32_t* __restrict__ codes,
                                              const __nv_bfloat162* __restrict__ scales, int M, int KG,
                                              __nv_bfloat16* __restrict__ out, int ldo) {
    constexpr int XB = 16 * GS * 2, CB = CODE_WORDS * 4, SB = 64 * 4, PB = 16 * 4;
    constexpr int STAGE = XB + CB + SB + PB;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, tile = blockIdx.x;
    const int rows = min(16, M);
    auto stage = [&](int s) { return buf + s * STAGE; };
    for (int c = tid; c < 16 * 8; c += 128)
        if (c / 8 >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s) *reinterpret_cast<uint4*>(stage(s) + (c / 8) * 128 + (c % 8) * 16) = uint4{};
    __syncthreads();
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        {
            const int r = tid / 8, ch = tid % 8;
            if (r < rows) cp16(p + r * 128 + swz<8>(r, ch) * 16, x + static_cast<size_t>(r) * KG * GS + g * GS + ch * 8);
        }
        const unsigned char* cb = reinterpret_cast<const unsigned char*>(codes + (static_cast<size_t>(tile) * KG + g) * CODE_WORDS);
        if (tid < CB / 16) cp16(p + XB + tid * 16, cb + tid * 16);
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(scales + (static_cast<size_t>(tile) * KG + g) * 64);
        if (tid >= 64 && tid < 64 + SB / 16) cp16(p + XB + CB + (tid - 64) * 16, sb + (tid - 64) * 16);
        if (tid >= 96 && tid < 96 + rows) cp4(p + XB + CB + SB + (tid - 96) * 4, xs + static_cast<size_t>(tid - 96) * KG + g);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    float acc[2][4] = {};
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < KG) load((it + STAGES - 1) % STAGES, it + STAGES - 1);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* cw = reinterpret_cast<const uint32_t*>(p + XB);
        const __nv_bfloat162* sc = reinterpret_cast<const __nv_bfloat162*>(p + XB + CB);
        const float* px = reinterpret_cast<const float*>(p + XB + CB + SB);
        const uint32_t w[2] = {cw[(warp * 2) * 32 + lane], cw[(warp * 2 + 1) * 32 + lane]};
        float d[2][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
            ldmatrix4(a, p + r * 128 + swz<8>(r, ch) * 16);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t b0 = pairm(w[j], 2 * (2 * kt), 0x00030003u), b1 = pairm(w[j], 2 * (2 * kt + 1), 0x00030003u);
                if (kt == 0) mma0(d[j], a, b0, b1);
                else mma(d[j], a, b0, b1);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c0 = (warp * 2 + j) * 8 + (lane & 3) * 2;
            const float2 s0 = __bfloat1622float2(sc[c0]), s1 = __bfloat1622float2(sc[c0 + 1]);
            const float xv[2] = {px[lane >> 2], px[(lane >> 2) + 8]};
            acc[j][0] += d[j][0] * s0.x + xv[0] * s0.y;
            acc[j][1] += d[j][1] * s1.x + xv[0] * s1.y;
            acc[j][2] += d[j][2] * s0.x + xv[1] * s0.y;
            acc[j][3] += d[j][3] * s1.x + xv[1] * s1.y;
        }
    }
    wait<0>();
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = (lane >> 2) + h * 8, col = tile * 64 + (warp * 2 + j) * 8 + (lane & 3) * 2;
            if (row < rows)
                *reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(row) * ldo + col) =
                        __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
        }
}

__device__ __forceinline__ unsigned key16(unsigned short b) { return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u); }

__device__ __forceinline__ void select(const __nv_bfloat16* __restrict__ L, int V, int ldo, int K, int cap,
                                       float* __restrict__ PV, unsigned* __restrict__ PI) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_bin, s_above, s_count;
    const int row = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const unsigned short* x = reinterpret_cast<const unsigned short*>(L + static_cast<size_t>(row) * ldo);
    unsigned above = 0, prefix = 0;
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = t; i < 256; i += T) hist[i] = 0;
        __syncthreads();
        for (int c = t; c < V; c += T) {
            const unsigned k = key16(x[c]);
            const bool in = pass == 0 || (k >> 8) == prefix;
            const unsigned bin = pass == 0 ? k >> 8 : (k & 255u);
            const unsigned mask = __match_any_sync(__activemask(), in ? bin : 0x100u);
            if (in && (threadIdx.x & 31) == __ffs(mask) - 1) atomicAdd(&hist[bin], __popc(mask));
        }
        __syncthreads();
        if (t == 0) {
            unsigned a = above;
            int b = 255;
            for (; b > 0; --b) {
                if (a + hist[b] >= static_cast<unsigned>(K)) break;
                a += hist[b];
            }
            s_bin = b;
            s_above = a;
        }
        __syncthreads();
        prefix = pass == 0 ? s_bin : (prefix << 8 | s_bin);
        above = s_above;
        __syncthreads();
    }
    const unsigned cut = prefix;
    if (t == 0) s_count = 0;
    __syncthreads();
    float* pv = PV + static_cast<size_t>(row) * cap;
    unsigned* pi = PI + static_cast<size_t>(row) * cap;
#pragma unroll 1
    for (int tie = 0; tie < 2; ++tie) {
        for (int c = t; c < V; c += T) {
            const unsigned k = key16(x[c]);
            if (tie ? k == cut : k > cut) {
                const unsigned at = atomicAdd(&s_count, 1u);
                if (at < static_cast<unsigned>(cap)) {
                    pv[at] = __bfloat162float(__ushort_as_bfloat16(x[c]));
                    pi[at] = static_cast<unsigned>(c);
                }
            }
        }
        __syncthreads();
    }
    for (unsigned at = min(s_count, static_cast<unsigned>(cap)) + t; at < static_cast<unsigned>(cap); at += T) {
        pv[at] = __int_as_float(0xff800000);
        pi[at] = 0xFFFFFFFFu;
    }
}

__device__ __forceinline__ void rescore(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                               const uint32_t* __restrict__ w4, const __nv_bfloat162* __restrict__ sbcol,
                                               int KG, int M,
                                               float* __restrict__ PV, const unsigned* __restrict__ PI, int n,
                                               int first) {
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int at = blockIdx.y * 32 + warp * 8;
    if (at >= n) return;
    const size_t base = static_cast<size_t>(r) * n + at;
    const int pb = lane >> 2, pc = (lane & 3) * 2;
    auto colOf = [&](int p) -> int {
        if (at + p >= n) return -1;
        const unsigned c = PI[base + p];
        return c < static_cast<unsigned>(first) ? static_cast<int>(c) : -1;
    };
    const int cb = colOf(pb), cc0 = colOf(pc), cc1 = colOf(pc + 1);
    if (__all_sync(0xffffffffu, cb < 0)) return;
    const int ub = max(cb, 0), u0 = max(cc0, 0), u1 = max(cc1, 0);
    const uint32_t* wcol = w4 + static_cast<size_t>(ub / 64) * KG * TILE_WORDS + (((ub % 64) / 8) * 32 + (ub % 8) * 4 + (lane & 3)) * 2;
    const int g4 = lane >> 2, t4 = lane & 3;
    const bool r0 = g4 < M, r1 = g4 + 8 < M;
    const __nv_bfloat16* x0 = x + static_cast<size_t>(r0 ? g4 : 0) * KG * GS + 2 * t4;
    const __nv_bfloat16* x1 = x + static_cast<size_t>(r1 ? g4 + 8 : 0) * KG * GS + 2 * t4;
    const __nv_bfloat162* sb0 = sbcol + static_cast<size_t>(u0) * KG;
    const __nv_bfloat162* sb1 = sbcol + static_cast<size_t>(u1) * KG;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int g = 0; g < KG; ++g) {
        const uint32_t w[2] = {wcol[static_cast<size_t>(g) * TILE_WORDS], wcol[static_cast<size_t>(g) * TILE_WORDS + 1]};
        float d[4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int k = g * GS + kt * 16;
            uint32_t a[4];
            a[0] = r0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
            a[1] = r1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
            a[2] = r0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
            a[3] = r1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
            const uint32_t b0 = pairm(w[kt / 2], (kt & 1) * 8, 0x000F000Fu), b1 = pairm(w[kt / 2], (kt & 1) * 8 + 4, 0x000F000Fu);
            if (kt == 0) mma0(d, a, b0, b1);
            else mma(d, a, b0, b1);
        }
        const float2 p0 = __bfloat1622float2(sb0[g]), p1 = __bfloat1622float2(sb1[g]);
        const float sv[2] = {p0.x, p1.x}, bv[2] = {p0.y, p1.y};
        const float xv[2] = {r0 ? xs[static_cast<size_t>(g4) * KG + g] : 0.0f, r1 ? xs[static_cast<size_t>(g4 + 8) * KG + g] : 0.0f};
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[e], sv[e & 1], acc[e]));
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int row = g4 + (e >> 1) * 8, p = pc + (e & 1);
        const int c = (e & 1) ? cc1 : cc0;
        if (row == r && c >= 0) PV[base + p] = __bfloat162float(__float2bfloat16_rn(acc[e]));
    }
}

}

extern "C" __global__ void __launch_bounds__(256) tf_q35d_head2_pack(const uint32_t* __restrict__ w4,
        const __nv_bfloat16* __restrict__ s, const __nv_bfloat16* __restrict__ b, int npad, int KG,
        uint32_t* __restrict__ codes, __nv_bfloat162* __restrict__ scales, __nv_bfloat162* __restrict__ sbcol) {
    tf_q35d_head2::pack(w4, s, b, npad, KG, codes, scales, sbcol);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35d_head2_screen(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const uint32_t* __restrict__ codes, const __nv_bfloat162* __restrict__ scales, int M,
        int KG, __nv_bfloat16* __restrict__ out, int ldo) {
    tf_q35d_head2::screen(x, xs, codes, scales, M, KG, out, ldo);
}

extern "C" __global__ void __launch_bounds__(1024) tf_q35d_head2_select(const __nv_bfloat16* __restrict__ L, int V,
        int ldo, int K, int cap, float* __restrict__ PV, unsigned* __restrict__ PI) {
    tf_q35d_head2::select(L, V, ldo, K, cap, PV, PI);
}

extern "C" __global__ void __launch_bounds__(128) tf_q35d_head2_rescore(const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ xs, const uint32_t* __restrict__ w4, const __nv_bfloat162* __restrict__ sbcol, int KG,
        int M, float* __restrict__ PV, const unsigned* __restrict__ PI, int n, int first) {
    tf_q35d_head2::rescore(x, xs, w4, sbcol, KG, M, PV, PI, n, first);
}
