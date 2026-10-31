// Per-head RMSNorm of attention rows, out packed (rows, heads, HD); grid (rows * heads), 64 threads of 4 values.
extern "C" __global__ void __launch_bounds__(64) tf_q35_head_norm(const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ W,
                                                                    __nv_bfloat16* __restrict__ OUT0, int heads, int stride, int hstride, float eps,
                                                                    const int* __restrict__ P) {
    release_next();
    await_prior();
    const __nv_bfloat16* X = after_wait(X0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    __shared__ float sums[2];
    const int head = blockIdx.x % heads;
    if (past_rows(P, blockIdx.x / heads)) return;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = X + m * stride + head * hstride;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    if ((t & 31) == 0) sums[t >> 5] = acc;
    __syncthreads();
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        OUT[(m * heads + head) * HD + d] = __float2bfloat16_rn(bf(W[d]) * (v[i] * inv));
    }
}

// A key and value as bf16 rows in their cache slot (qwen35.cu's layout: a layer's keys then values, kvh blocks of cap).
struct StoreBf16 {
    int layer, kvh;
    __device__ __forceinline__ void operator()(const Seg& sg, int h, int slot, int d, __nv_bfloat16 k, __nv_bfloat16 v) const {
        __nv_bfloat16* kv = reinterpret_cast<__nv_bfloat16*>(sg.kv);
        kv[kv_row(sg, kvh, layer, 0, h, slot) + d] = k;
        kv[kv_row(sg, kvh, layer, 1, h, slot) + d] = v;
    }
};
