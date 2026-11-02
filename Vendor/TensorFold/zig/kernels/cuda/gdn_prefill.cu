// Ours: v0.6.6's gdn_prefill.cu chain (src/tensorfold/cuda/kernels) for a prompt chunk, beta bf16 (Qwen3.5) or fp32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace tf_gdn_prefill {

constexpr int DK = 128;
constexpr int DV = 128;

__device__ __forceinline__ float4 widen(uint2 w) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

__device__ __forceinline__ float4 widen(float4 w) { return w; }

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename QK, typename B, int THREADS, int ROWS, int STEPS>
struct Prefetch {
    using Raw = typename std::conditional<std::is_same<QK, float>::value, float4, uint2>::type;
    static constexpr int KQ = STEPS * (DK / 4) / THREADS, VV = STEPS * ROWS / 8 / THREADS;
    static_assert(KQ * THREADS == STEPS * (DK / 4) && VV * THREADS * 8 == STEPS * ROWS, "whole slots a thread");
    Raw kr[KQ], qr[KQ];
    uint4 vr[VV];
    float gr, br;

    __device__ __forceinline__ void fetch(const QK* q, const QK* k, const __nv_bfloat16* v, const float* g,
                                          const B* beta, int t0, int W, int hk, int hv, int head, int key_head,
                                          int col0) {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (DK / 4), c = i % (DK / 4);
            const size_t at = (static_cast<size_t>(min(t0 + st, W - 1)) * hk + key_head) * DK + 4 * c;
            kr[it] = *reinterpret_cast<const Raw*>(k + at);
            qr[it] = *reinterpret_cast<const Raw*>(q + at);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (ROWS / 8), c = i % (ROWS / 8);
            vr[it] = *reinterpret_cast<const uint4*>(v + (static_cast<size_t>(min(t0 + st, W - 1)) * hv + head) * DV
                                                     + col0 + 8 * c);
        }
        if (threadIdx.x < STEPS) {
            const size_t at = static_cast<size_t>(min(t0 + static_cast<int>(threadIdx.x), W - 1)) * hv + head;
            gr = g[at];
            br = to_float(beta[at]);
        }
    }

    // key column c = 8 j + 4 h + 2 p + e of a step at ((j / 2) 4 + 2 h + p) 4 + (j % 2) 2 + e: the chain's thread
    // (h, p) reads its 32 columns as 8 float4, a row's four threads one 64-byte run
    __device__ __forceinline__ static int slot(int c) {
        const int j = c >> 3, h = (c >> 2) & 1, i = c & 3;
        return ((j >> 1) * 4 + 2 * h + (i >> 1)) * 4 + (j & 1) * 2 + (i & 1);
    }

    __device__ __forceinline__ void store(float (*ks)[DK], float (*qs)[DK], float* gs, float* bs,
                                          __nv_bfloat16 (*vs)[ROWS]) const {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (DK / 4), c = 4 * (i % (DK / 4));
            const float4 kk = widen(kr[it]), qq = widen(qr[it]);
            const int a = slot(c), b = slot(c + 2);
            *reinterpret_cast<float2*>(&ks[st][a]) = make_float2(kk.x, kk.y);
            *reinterpret_cast<float2*>(&ks[st][b]) = make_float2(kk.z, kk.w);
            *reinterpret_cast<float2*>(&qs[st][a]) = make_float2(qq.x, qq.y);
            *reinterpret_cast<float2*>(&qs[st][b]) = make_float2(qq.z, qq.w);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            *reinterpret_cast<uint4*>(&vs[i / (ROWS / 8)][8 * (i % (ROWS / 8))]) = vr[it];
        }
        if (threadIdx.x < STEPS) {
            gs[threadIdx.x] = gr;
            bs[threadIdx.x] = br;
        }
    }
};

// Two value rows a thread, four threads a row: thread (h, p) keeps key columns 8 j + 4 h + 2 p + e of rows 2 (t / 4)
// and 2 (t / 4) + 1, i.e. lanes 2 p and 2 p + 1 of v0.6.6's four accumulators over key half h. Every state entry takes
// the same operations in the same order, and a row's sums keep v0.6.6's tree ((m0 + m1) + (m2 + m3)) per half, then
// the halves: the same bits as two threads a row, with half the threads' accumulator lanes and k shared by two rows.
// The stager fetches a stage's inputs during the stage before, stores them, and may prepare v, g and beta in place.
template <int ROWS, int STEPS>
struct Stage {
    float ks[STEPS][DK];
    float qs[STEPS][DK];
    float gs[STEPS], bs[STEPS];
    __align__(16) __nv_bfloat16 vs[STEPS][ROWS];
};

// A follower's progress word: the rows of y this block has written, published with release semantics (the block's
// earlier y stores are ordered before it by the barrier that precedes the publishing thread's store).
__device__ __forceinline__ void publish(int* flag, int rows) {
    asm volatile("st.global.release.gpu.b32 [%0], %1;\n" ::"l"(flag), "r"(rows) : "memory");
}

template <int ROWS, int STEPS, class Stager>
__device__ __forceinline__ void chain_body(Stager& pf, Stage<ROWS, STEPS>& sm, const float* state, float* last,
                                           __nv_bfloat16* __restrict__ y, int W, int hv, int* flag = nullptr) {
    const int sub = threadIdx.x & 3, h = sub >> 1, p = sub & 1;
    const int lr = 2 * (threadIdx.x >> 2);
    const int head = blockIdx.x, row = blockIdx.y * ROWS + lr;
    float s[2][32];                                               // s[r][2 j + e]: column 8 j + 4 h + 2 p + e
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float* s0 = state + (static_cast<size_t>(head) * DV + row + r) * DK + 4 * h + 2 * p;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float2 t = *reinterpret_cast<const float2*>(s0 + 8 * j);
            s[r][2 * j] = t.x;
            s[r][2 * j + 1] = t.y;
        }
    }
    pf.fetch(0);                                                  // each stage's loads in flight during the one before
    const float4* k4 = reinterpret_cast<const float4*>(&sm.ks[0][0]) + sub;
    const float4* q4 = reinterpret_cast<const float4*>(&sm.qs[0][0]) + sub;
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        // a followed chain: every second stage, the rows before it (all written: every thread passed the barrier
        // since). Here thread 0's fence waits only for its last y stores (its prefetch loads landed long ago).
        if (flag != nullptr && threadIdx.x == 0 && t0 > 0 && (t0 / STEPS) % 2 == 0) publish(flag, t0);
        pf.store(sm);
        __syncthreads();
        if (t0 + STEPS < W) pf.fetch(t0 + STEPS);
        pf.prepare(sm, t0);
        for (int tt = 0; tt < n; ++tt) {
            const float gt = sm.gs[tt], bt = sm.bs[tt];
            const __nv_bfloat162 vv = *reinterpret_cast<const __nv_bfloat162*>(&sm.vs[tt][lr]);
            const float vt[2] = {__low2float(vv), __high2float(vv)};
            const float4* kt = k4 + tt * (DK / 4);
            const float4* qt = q4 + tt * (DK / 4);
            float m0[2] = {0.0f, 0.0f}, m1[2] = {0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 kk = kt[4 * j];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    s[r][4 * j] = s[r][4 * j] * gt;
                    s[r][4 * j + 1] = s[r][4 * j + 1] * gt;
                    s[r][4 * j + 2] = s[r][4 * j + 2] * gt;
                    s[r][4 * j + 3] = s[r][4 * j + 3] * gt;
                    m0[r] = __fmaf_rn(s[r][4 * j], kk.x, m0[r]);
                    m1[r] = __fmaf_rn(s[r][4 * j + 1], kk.y, m1[r]);
                    m0[r] = __fmaf_rn(s[r][4 * j + 2], kk.z, m0[r]);
                    m1[r] = __fmaf_rn(s[r][4 * j + 3], kk.w, m1[r]);
                }
            }
            float delta[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float mem = m0[r] + m1[r];
                mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
                mem = mem + __shfl_xor_sync(0xffffffffu, mem, 2);
                delta[r] = (vt[r] - mem) * bt;
            }
            float o0[2] = {0.0f, 0.0f}, o1[2] = {0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 kk = kt[4 * j], qq = qt[4 * j];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    s[r][4 * j] = __fmaf_rn(kk.x, delta[r], s[r][4 * j]);
                    s[r][4 * j + 1] = __fmaf_rn(kk.y, delta[r], s[r][4 * j + 1]);
                    s[r][4 * j + 2] = __fmaf_rn(kk.z, delta[r], s[r][4 * j + 2]);
                    s[r][4 * j + 3] = __fmaf_rn(kk.w, delta[r], s[r][4 * j + 3]);
                    o0[r] = __fmaf_rn(s[r][4 * j], qq.x, o0[r]);
                    o1[r] = __fmaf_rn(s[r][4 * j + 1], qq.y, o1[r]);
                    o0[r] = __fmaf_rn(s[r][4 * j + 2], qq.z, o0[r]);
                    o1[r] = __fmaf_rn(s[r][4 * j + 3], qq.w, o1[r]);
                }
            }
            float out[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                out[r] = o0[r] + o1[r];
                out[r] = out[r] + __shfl_xor_sync(0xffffffffu, out[r], 1);
                out[r] = out[r] + __shfl_xor_sync(0xffffffffu, out[r], 2);
            }
            if (sub == 0)
                *reinterpret_cast<__nv_bfloat162*>(y + (static_cast<size_t>(t0 + tt) * hv + head) * DV + row) =
                        __floats2bfloat162_rn(out[0], out[1]);
        }
    }
    if (flag != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) publish(flag, W);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float* s1 = last + (static_cast<size_t>(head) * DV + row + r) * DK + 4 * h + 2 * p;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            *reinterpret_cast<float2*>(s1 + 8 * j) = make_float2(s[r][2 * j], s[r][2 * j + 1]);
    }
}

// The stager of gdn_pre's outputs: q, k, v, g and beta as written.
template <typename QK, typename B, int ROWS, int STEPS>
struct Written {
    Prefetch<QK, B, 2 * ROWS, ROWS, STEPS> pf;
    const QK* q;
    const QK* k;
    const __nv_bfloat16* v;
    const float* g;
    const B* beta;
    int W, hk, hv, head, key_head;
    __device__ __forceinline__ void fetch(int t0) {
        pf.fetch(q, k, v, g, beta, t0, W, hk, hv, head, key_head, blockIdx.y * ROWS);
    }
    __device__ __forceinline__ void store(Stage<ROWS, STEPS>& sm) const { pf.store(sm.ks, sm.qs, sm.gs, sm.bs, sm.vs); }
    __device__ __forceinline__ void prepare(Stage<ROWS, STEPS>&, int) const {}
};

template <typename QK, typename B, int ROWS, int STEPS>
__global__ void __launch_bounds__(2 * ROWS) chain_kernel(
        const QK* __restrict__ q, const QK* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const B* __restrict__ beta, const float* state,
        float* last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    __shared__ __align__(16) Stage<ROWS, STEPS> sm;
    Written<QK, B, ROWS, STEPS> st{{}, q, k, v, g, beta, W, hk, hv, (int)blockIdx.x, (int)blockIdx.x / (hv / hk)};
    chain_body<ROWS, STEPS>(st, sm, state, last, y, W, hv);
}

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The value heads' share of qwen35.cu's tf_q35_gdn_pre for a prompt chunk (one stream's consecutive rows), done in
// the chain's stage: each value channel's causal convolution over the stage's rows and the three before, SiLU and
// rounding; each row's decay g and strength beta. Every expression is tf_q35_gdn_pre's (this file builds as qwen35.cu
// does, -O3 with its contractions), so v, g and beta take its bits. Rows before the chunk come from the stream's tail.
template <int ROWS, int STEPS>
struct Convolved {
    static constexpr int THREADS = 2 * ROWS, TAPS = 4, XR = STEPS + TAPS - 1, XN = XR * ROWS / 8;
    static constexpr int KQ = STEPS * (DK / 4) / THREADS, XV = (XN + THREADS - 1) / THREADS;
    static_assert(KQ * THREADS == STEPS * (DK / 4) && THREADS % ROWS == 0 && STEPS % (THREADS / ROWS) == 0, "slots");
    __nv_bfloat16 (*xs)[ROWS];                                    // the stage's rows t0 - 3 .. t0 + STEPS - 1
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* qkv;
    const __nv_bfloat16* tail;
    const int* rowpar;
    const __nv_bfloat16* ain;
    const __nv_bfloat16* bin;
    int W, hk, hv, head, key_head, C, c0;
    float neg_a, dt;                                              // -expf(A_log), the head's dt bias
    uint2 wv;                                                     // this thread's channel's four taps
    uint2 kr[KQ], qr[KQ];
    uint4 xr[XV];
    __nv_bfloat16 ar, br;

    __device__ __forceinline__ void fetch(int t0) {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (DK / 4), c = i % (DK / 4);
            const size_t at = (static_cast<size_t>(min(t0 + st, W - 1)) * hk + key_head) * DK + 4 * c;
            kr[it] = *reinterpret_cast<const uint2*>(k + at);
            qr[it] = *reinterpret_cast<const uint2*>(q + at);
        }
#pragma unroll
        for (int it = 0; it < XV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            if (XN % THREADS == 0 || i < XN) {
                const int r = min(t0 - (TAPS - 1) + i / (ROWS / 8), W - 1), c = c0 + 8 * (i % (ROWS / 8));
                if (r >= 0) xr[it] = *reinterpret_cast<const uint4*>(qkv + static_cast<size_t>(r) * C + c);
                else {
                    int a = 0;                                    // tf_q35's back(rowpar, 0, -r)
                    for (int s = 1; s <= -r; s++) {
                        a = rowpar[a];
                        if (a < 0) { a = -1 - (TAPS - 2 - (-r - s)); break; }
                    }
                    xr[it] = *reinterpret_cast<const uint4*>(tail + static_cast<size_t>(-1 - a) * C + c);
                }
            }
        }
        if (threadIdx.x < STEPS) {
            const size_t at = static_cast<size_t>(min(t0 + static_cast<int>(threadIdx.x), W - 1)) * hv + head;
            ar = ain[at];
            br = bin[at];
        }
    }

    __device__ __forceinline__ void store(Stage<ROWS, STEPS>& sm) const {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (DK / 4), c = 4 * (i % (DK / 4));
            using P = Prefetch<__nv_bfloat16, __nv_bfloat16, THREADS, ROWS, STEPS>;
            const float4 kk = widen(kr[it]), qq = widen(qr[it]);
            const int a = P::slot(c), b = P::slot(c + 2);
            *reinterpret_cast<float2*>(&sm.ks[st][a]) = make_float2(kk.x, kk.y);
            *reinterpret_cast<float2*>(&sm.ks[st][b]) = make_float2(kk.z, kk.w);
            *reinterpret_cast<float2*>(&sm.qs[st][a]) = make_float2(qq.x, qq.y);
            *reinterpret_cast<float2*>(&sm.qs[st][b]) = make_float2(qq.z, qq.w);
        }
#pragma unroll
        for (int it = 0; it < XV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            if (XN % THREADS == 0 || i < XN) *reinterpret_cast<uint4*>(&xs[i / (ROWS / 8)][8 * (i % (ROWS / 8))]) = xr[it];
        }
        if (threadIdx.x < STEPS) {
            // tf_q35_gdn_pre's lane 0 of a value head
            const float s = rbf(bf(ar) + dt);
            const float sp = rbf(fmaxf(s, 0.0f) + logf(1.0f + expf(-fabsf(s))));
            sm.gs[threadIdx.x] = expf(neg_a * sp);
            sm.bs[threadIdx.x] = bf(__float2bfloat16_rn(1.0f / (1.0f + expf(-bf(br)))));
        }
    }

    __device__ __forceinline__ void prepare(Stage<ROWS, STEPS>& sm, int) const {
        const int ch = threadIdx.x % ROWS;
        const __nv_bfloat16* cw = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
        for (int st = threadIdx.x / ROWS; st < STEPS; st += THREADS / ROWS) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; tap++) acc += bf(cw[tap]) * bf(xs[st + tap][ch]);
            const float conv = rbf(acc);
            const float sig = rbf(1.0f / (1.0f + expf(-conv)));
            sm.vs[st][ch] = __float2bfloat16_rn(rbf(conv * sig));
        }
        __syncthreads();
    }
};

// A prompt chunk's value convolution, decay and strength (the value heads' tf_q35_gdn_pre) and its delta rule in one
// launch, rows 0 .. W - 1 of one stream; grid (value heads, DV / 64), 128 threads. The q and k heads' tf_q35_gdn_pre
// runs before it; the stream's convolution tail (conv_commit) after it.
extern "C" __global__ void __launch_bounds__(128) tf_gdn_chain_conv(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ qkv,
        const __nv_bfloat16* __restrict__ cw, const __nv_bfloat16* __restrict__ tail, const int* __restrict__ rowpar,
        const __nv_bfloat16* __restrict__ ain, const __nv_bfloat16* __restrict__ bin, const float* __restrict__ alog,
        const __nv_bfloat16* __restrict__ dt, float* state, __nv_bfloat16* __restrict__ y, int W, int hk, int hv,
        int* flags) {
    constexpr int ROWS = 64, STEPS = 16;
    // flags (may be null): each block's progress word for qwen35.cu's tf_q35_gdn_post_follow, cleared before the
    // follower can launch (programmatic dependent launch: once every block has passed launch_dependents)
    int* flag = flags == nullptr ? nullptr : flags + 32 * (blockIdx.x * gridDim.y + blockIdx.y);  // a 128-byte line each
    if (flag != nullptr) {
        if (threadIdx.x == 0) {
            *reinterpret_cast<volatile int*>(flag) = 0;
            __threadfence();
        }
        __syncthreads();
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    }
    __shared__ __align__(16) Stage<ROWS, STEPS> sm;
    __shared__ __align__(16) __nv_bfloat16 xs[STEPS + 3][ROWS];
    const int head = blockIdx.x, C = (2 * hk + hv) * DK, c0 = 2 * hk * DK + head * DK + blockIdx.y * ROWS;
    Convolved<ROWS, STEPS> st;
    st.xs = xs;
    st.q = q;
    st.k = k;
    st.qkv = qkv;
    st.tail = tail;
    st.rowpar = rowpar;
    st.ain = ain;
    st.bin = bin;
    st.W = W;
    st.hk = hk;
    st.hv = hv;
    st.head = head;
    st.key_head = head / (hv / hk);
    st.C = C;
    st.c0 = c0;
    st.neg_a = -expf(alog[head]);
    st.dt = bf(dt[head]);
    st.wv = *reinterpret_cast<const uint2*>(cw + static_cast<size_t>(c0 + threadIdx.x % ROWS) * 4);
    chain_body<ROWS, STEPS>(st, sm, state, state, y, W, hv, flag);
}

} // namespace tf_gdn_prefill

// Prompt chunks: 64 value rows and 16 staged steps; each value row keeps its exact recurrence order and sum tree.
template __global__ void tf_gdn_prefill::chain_kernel<__nv_bfloat16, __nv_bfloat16, 64, 16>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, const float* __restrict__,
    const __nv_bfloat16* __restrict__, const float*, float*, __nv_bfloat16* __restrict__, int, int, int);
