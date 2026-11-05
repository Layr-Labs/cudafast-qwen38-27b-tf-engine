// Absorb-only fusion. The arithmetic below is verbatim tf_q35d_norm_rope;
// only the store address changes, and the corresponding value element is copied.
#pragma once
extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope_ring(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ V0, __nv_bfloat16* __restrict__ RK0,
    __nv_bfloat16* __restrict__ RV0, int heads, int stride, long long pos0,
    float eps, float theta, int ring_rows, const DroundState* __restrict__ st) {
    release_next();
    await_prior();
    const __nv_bfloat16* X = after_wait(X0);
    const auto* V = after_wait(V0);
    auto* RK = after_wait(RK0);
    auto* RV = after_wait(RV0);
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
    if (st != nullptr && m >= st->n) return;
    const __nv_bfloat16* x = X + m * stride + head * HD;
    float v[4], acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = warp_sum(acc);
    const float inv = rsqrtf(acc / (float)HD + eps);
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; i++) n[i] = rbf(rbf(v[i] * inv) * bf(W[4 * t + i]));
    // the pair of lane element d is d + HD / 2: this lane's partner values come from the lane HD / 8 = 16 away
    const float pos = (float)(pos0 + (long long)m);
    const size_t slot = (size_t)((pos0 + (long long)m) % ring_rows);
    __nv_bfloat16* out = RK + (slot * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);
        const int j = d < HD / 2 ? d : d - HD / 2;
        const float angle = pos * powf(theta, -(float)j / (float)(HD / 2));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        RV[(slot * heads + head) * HD + d] = V[(m * heads + head) * HD + d];
        out[d] = __float2bfloat16_rn(d < HD / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
    }
}
