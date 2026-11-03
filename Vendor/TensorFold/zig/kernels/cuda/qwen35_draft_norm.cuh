__device__ __forceinline__ float group_sum(const __nv_bfloat16* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(x + j);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int k = 0; k < 8; k++) s += bf(b[k]);
    }
    return s;
}

// DFlash Qwen3 RMSNorm retains the normalized bf16 intermediate before applying the gain.
// Isolated from target Qwen3.5 arithmetic: the reduction, row copies and group sums are unchanged.
// Signature matches the existing row norm; DFlash passes R/P null and always requests XS.
extern "C" __global__ void __launch_bounds__(256) tf_q35d_norm(__nv_bfloat16* __restrict__ H0, const __nv_bfloat16* __restrict__ R0,
                                                               const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X0, float eps, int D,
                                                               const int* __restrict__ P, float* __restrict__ XS0) {
    constexpr int MAXE = 32;
    __shared__ float red[8];
    __shared__ __align__(16) __nv_bfloat16 xrow[8192];
    release_next();
    const int t = threadIdx.x;
    const int E = D / 256;
    // Immutable weights overlap the prior kernel; consumed row values remain behind its dependency wait.
    uint4 wv[4];
    if ((E & 3) == 0) {
#pragma unroll
        for (int j = 0; j < 4; j++)
            if ((t + 256 * j) * 8 < D) wv[j] = *reinterpret_cast<const uint4*>(W + (t + 256 * j) * 8);
#pragma unroll
        for (int j = 0; j < 4; j++)
            if ((t + 256 * j) * 8 < D) asm volatile("" ::"r"(wv[j].x), "r"(wv[j].y), "r"(wv[j].z), "r"(wv[j].w));
    }
    await_prior();
    __nv_bfloat16* H = after_wait(H0);
    const __nv_bfloat16* R = after_wait(R0);
    __nv_bfloat16* X = after_wait(X0);
    float* XS = after_wait(XS0);
    if ((E & 3) == 0) {
        const size_t row = blockIdx.x;
        uint4 hw[4];
#pragma unroll
        for (int j = 0; j < 4; j++)
            if ((t + 256 * j) * 8 < D) hw[j] = *reinterpret_cast<const uint4*>(H + row * D + (t + 256 * j) * 8);
        if ((P != nullptr && blockIdx.x >= P[2])) return;
        // h + r and w * (h * inv) move 8 entries a lane through xrow; the squares keep a thread's E entries in order
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            if (R != nullptr) {
                const uint4 rw = *reinterpret_cast<const uint4*>(R + row * D + d); // the residual at use: a shorter live range
                __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hw[j]);
                const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rw);
#pragma unroll
                for (int k = 0; k < 8; k++) hb[k] = __float2bfloat16_rn(bf(hb[k]) + bf(rb[k]));
                *reinterpret_cast<uint4*>(H + row * D + d) = hw[j];
            }
            *reinterpret_cast<uint4*>(xrow + d) = hw[j];
        }
        __syncthreads();
        float sq = 0.0f;
        const uint2* mine = reinterpret_cast<const uint2*>(xrow) + t * (E / 4);
#pragma unroll
        for (int i = 0; i < MAXE / 4; i++) {
            if (4 * i >= E) break;
            const uint2 v = mine[i];
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float h = bf(vb[k]);
                sq = fmaf(h, h, sq);
            }
        }
        sq = warp_sum(sq);
        if ((t & 31) == 0) red[t >> 5] = sq;
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += red[i];
        const float scale = rsqrtf(sum / (float)D + eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int d = (t + 256 * j) * 8;
            if (d >= D) break;
            const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hw[j]);
            const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
            uint4 xw;
            __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(&xw);
#pragma unroll
            for (int k = 0; k < 8; k++) xb[k] = __float2bfloat16_rn(bf(wb[k]) * rbf(bf(hb[k]) * scale));
            *reinterpret_cast<uint4*>(X + row * D + d) = xw;
            *reinterpret_cast<uint4*>(xrow + d) = xw;
        }
        if (XS == nullptr) return;
        __syncthreads();
        for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
        return;
    }
    if ((P != nullptr && blockIdx.x >= P[2])) return;
    const size_t row = blockIdx.x;
    // E not a multiple of 4: a thread's entries one at a time
    float hv[MAXE];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const size_t at = row * D + t * E + i;
        float h = bf(H[at]);
        if (R != nullptr) {
            const __nv_bfloat16 hb = __float2bfloat16_rn(h + bf(R[at]));
            H[at] = hb;
            h = bf(hb);
        }
        hv[i] = h;
        ss = fmaf(h, h, ss);
    }
    ss = warp_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < 8; i++) total += red[i];
    const float inv = rsqrtf(total / (float)D + eps);
#pragma unroll
    for (int i = 0; i < MAXE; i++) {
        if (i >= E) break;
        const int d = t * E + i;
        const __nv_bfloat16 x = __float2bfloat16_rn(bf(W[d]) * rbf(hv[i] * inv));
        X[row * D + d] = x;
        xrow[d] = x;
    }
    if (XS == nullptr) return;
    __syncthreads();
    for (int g = t; g < D / 64; g += blockDim.x) XS[row * (D / 64) + g] = group_sum(xrow + g * 64);
}

