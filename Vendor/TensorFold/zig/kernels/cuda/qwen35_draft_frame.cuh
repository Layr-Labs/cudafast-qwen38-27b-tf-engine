// Batched DFlash glue: stream-local convolution and rotary positions, then 256-key attention chunks.
#pragma once
struct DraftSegment { unsigned long long ring; long long begin, position; };
constexpr int DRAFT_CHUNKS = 9;

extern "C" __global__ void tf_q35d_dyn_conv_batch(const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ DYN0,
                                              const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* __restrict__ RES0,
                                              __nv_bfloat16* __restrict__ OUT0, int D, int group, int branch, int block_rows,
                                              float* __restrict__ XS0) {
    release_next();
    // ahead of the wait: this thread's first channel's two base taps (static); this frame's rows after it
    const int c0 = blockIdx.y * blockDim.x + threadIdx.x;
    float base0[2] = {0.0f, 0.0f};
    if (c0 < D) {
        base0[0] = bf(BASE[(branch * 2) * D + c0]);
        base0[1] = bf(BASE[(branch * 2 + 1) * D + c0]);
    }
    asm volatile("" ::"f"(base0[0]), "f"(base0[1]));
    await_prior();
    const __nv_bfloat16* X = after_wait(X0);
    const __nv_bfloat16* DYN = after_wait(DYN0);
    const __nv_bfloat16* RES = after_wait(RES0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    // XS (may be null): the rows' 64-group sums of OUT, tf_q35_sums' arithmetic (the host passes it only when the grid
    // covers D in one pass of 256-channel blocks, four whole groups a block)
    float* XS = after_wait(XS0);
    __shared__ __align__(16) __nv_bfloat16 outs[256];
    const int t = blockIdx.x;
    const int groups = D / group;
    // grid (rows, channel blocks): a thread a channel, so a 16-row window fills the device instead of 16 blocks
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < D; c += gridDim.y * blockDim.x) {
        const int g = c / group;
        float total = 0.0f;
#pragma unroll
        for (int l = 0; l < 2; l++) {
            if (t % block_rows - l < 0) continue;
            const float x = bf(X[(size_t)(t - l) * D + c]);
            total = rbf(total + rbf((c == c0 ? base0[l] : bf(BASE[(branch * 2 + l) * D + c])) * x));
            total = rbf(total + rbf(bf(DYN[(size_t)t * (4 * groups) + (branch * 2 + l) * groups + g]) * x));
        }
        if (RES != nullptr) total = rbf(total + bf(RES[(size_t)t * D + c]));
        const __nv_bfloat16 o = __float2bfloat16_rn(total);
        OUT[(size_t)t * D + c] = o;
        if (XS != nullptr) outs[threadIdx.x] = o;
    }
    if (XS == nullptr) return;
    __syncthreads();
    if (threadIdx.x < 4) {
        float s = 0.0f;  // group_sum's order: the group's 64 entries in index order
#pragma unroll
        for (int j = 0; j < 64; j++) s += bf(outs[threadIdx.x * 64 + j]);
        XS[(size_t)t * (D / 64) + blockIdx.y * 4 + threadIdx.x] = s;
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope_batch(const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT0, int heads, int stride, const DraftSegment* segments, int block_rows,
                                                                     float eps, float theta) {
    release_next();
    await_prior();
    const __nv_bfloat16* X = after_wait(X0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    const int head = blockIdx.x % heads;
    const size_t m = blockIdx.x / heads;
    const int t = threadIdx.x;
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
    const float pos = (float)(segments[m / block_rows].position + (long long)(m % block_rows));
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);
        const int j = d < HD / 2 ? d : d - HD / 2;
        const float angle = pos * powf(theta, -(float)j / (float)(HD / 2));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        out[d] = __float2bfloat16_rn(d < HD / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope_qk_batch(const __nv_bfloat16* __restrict__ XQ0,
        const __nv_bfloat16* __restrict__ WQ, __nv_bfloat16* __restrict__ OUTQ0, int q_heads,
        const __nv_bfloat16* __restrict__ XK0, const __nv_bfloat16* __restrict__ WK, __nv_bfloat16* __restrict__ OUTK0,
        int kv_heads, int rows, const DraftSegment* segments, int block_rows, float eps, float theta) {
    release_next();
    const int qb = rows * q_heads;
    const bool isq = blockIdx.x < qb;
    const int bx = isq ? blockIdx.x : blockIdx.x - qb;
    const int heads = isq ? q_heads : kv_heads;
    const __nv_bfloat16* W = isq ? WQ : WK;
    const int head = bx % heads;
    const size_t m = bx / heads;
    const int t = threadIdx.x;
    // ahead of the wait: the head's norm weights (static) and the row's rotary angles (its position comes from the
    // segments the host staged before the frame's first launch); the rows this frame wrote are read after it
    float wn[4], sn4[4], cs4[4];
#pragma unroll
    for (int i = 0; i < 4; i++) wn[i] = bf(W[4 * t + i]);
    {
        const float pos = (float)(segments[m / block_rows].position + (long long)(m % block_rows));
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int d = 4 * t + i;
            const int j = d < HD / 2 ? d : d - HD / 2;
            const float angle = pos * powf(theta, -(float)j / (float)(HD / 2));
            sincosf(angle, &sn4[i], &cs4[i]);
        }
    }
    asm volatile("" ::"f"(wn[0]), "f"(wn[1]), "f"(wn[2]), "f"(wn[3]), "f"(sn4[0]), "f"(cs4[0]), "f"(sn4[3]), "f"(cs4[3]));
    await_prior();
    const __nv_bfloat16* XQ = after_wait(XQ0);
    __nv_bfloat16* OUTQ = after_wait(OUTQ0);
    const __nv_bfloat16* XK = after_wait(XK0);
    __nv_bfloat16* OUTK = after_wait(OUTK0);
    const __nv_bfloat16* X = isq ? XQ : XK;
    __nv_bfloat16* OUT = isq ? OUTQ : OUTK;
    const __nv_bfloat16* x = X + m * ((size_t)heads * HD) + head * HD;
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
    for (int i = 0; i < 4; i++) n[i] = rbf(rbf(v[i] * inv) * wn[i]);
    __nv_bfloat16* out = OUT + (m * heads + head) * HD;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
        const float partner = __shfl_xor_sync(0xffffffffu, n[i], 16);
        out[d] = __float2bfloat16_rn(d < HD / 2 ? n[i] * cs4[i] - partner * sn4[i] : n[i] * cs4[i] + partner * sn4[i]);
    }
}

// Remove each stream's pending row before the output head and selector; rows stay grouped by stream.
extern "C" __global__ void tf_q35d_mask_rows(const __nv_bfloat16* X0, __nv_bfloat16* OUT0, int D, int block_rows) {
    release_next();
    await_prior();
    const __nv_bfloat16* X = after_wait(X0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    const int row = blockIdx.x, source = row / (block_rows - 1) * block_rows + 1 + row % (block_rows - 1);
    for (int c = threadIdx.x; c < D; c += blockDim.x) OUT[(size_t)row * D + c] = X[(size_t)source * D + c];
}

// 256 keys of one row, its stream's ring and block rows only: four warps' lanes score them, warp 0 merges the slices.
extern "C" __global__ void __launch_bounds__(128) tf_q35d_attn_partial(const __nv_bfloat16* Q0,
        const __nv_bfloat16* KB0, const __nv_bfloat16* VB0, const DraftSegment* segments,
        int block_rows, int heads, int kv_heads, int W, int window, int layer, float* PM0, float* PL0, float* PO0) {
    release_next();
    await_prior();
    const __nv_bfloat16* Q = after_wait(Q0);
    const __nv_bfloat16* KB = after_wait(KB0);
    const __nv_bfloat16* VB = after_wait(VB0);
    float* PM = after_wait(PM0);
    float* PL = after_wait(PL0);
    float* PO = after_wait(PO0);
    constexpr int DPL = HD / 32, CHUNK = 256, WARPS = 4, SLICE = CHUNK / WARPS, PER = SLICE / 32, BATCH = 16;
    __shared__ float qs[HD];
    __shared__ float ps[CHUNK];
    __shared__ float wm[WARPS], wl[WARPS];
    __shared__ float wo[WARPS][HD];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m = blockIdx.x, h = blockIdx.y, chunk = blockIdx.z;
    const int stream = m / block_rows, local = m % block_rows;
    const DraftSegment sg = segments[stream];
    const int kvh = h / (heads / kv_heads), width = kv_heads * HD;
    const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * W * width;
    const __nv_bfloat16* RV = RK + (size_t)W * width;
    const long long first = max(sg.begin, sg.position + local - window + 1);
    const int ring_keys = (int)(sg.position - first), keys = ring_keys + block_rows;
    const size_t slot = ((size_t)m * heads + h) * DRAFT_CHUNKS + chunk;
    if (chunk * CHUNK >= keys) {  // past the stream's keys: an empty partial, which the merge skips
        if (threadIdx.x == 0) { PM[slot] = -INFINITY; PL[slot] = 0.0f; }
        return;
    }
    const int start = chunk * CHUNK + warp * SLICE, n = max(min(keys - start, SLICE), 0);  // this warp's keys
    const int ring0 = start < ring_keys ? (int)((first + start) % W) : 0;  // the slice's first ring row, then wrapping
    auto row_at = [&](int e) -> size_t {  // key start + e's element offset in the ring (RK, RV) or the block (KB, VB)
        const int index = start + e, wrapped = ring0 + e;
        const size_t row = index < ring_keys ? (size_t)(wrapped >= W ? wrapped - W : wrapped) : (size_t)(stream * block_rows + index - ring_keys);
        return (row * kv_heads + kvh) * HD;
    };
    for (int i = threadIdx.x; i < HD; i += 32 * WARPS) qs[i] = bf(Q[((size_t)m * heads + h) * HD + i]);
    __syncthreads();
    float mx = -INFINITY, score[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        score[j] = -INFINITY;
        if (e >= n) continue;
        const int index = start + e;
        const uint4* k = reinterpret_cast<const uint4*>((index < ring_keys ? RK : KB) + row_at(e));
        float dot = 0.0f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint4 kw[HD / 16];
#pragma unroll
            for (int c = 0; c < HD / 16; ++c) kw[c] = k[half * (HD / 16) + c];
#pragma unroll
            for (int c = 0; c < HD / 16; ++c) {
                const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw[c]);
#pragma unroll
                for (int i = 0; i < 8; ++i) dot = fmaf(qs[8 * (half * (HD / 16) + c) + i], bf(kb[i]), dot);
            }
        }
        score[j] = dot * rsqrtf((float)HD);
        mx = fmaxf(mx, score[j]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const float p = lane + 32 * j < n ? __expf(score[j] - mx) : 0.0f;
        ps[warp * SLICE + lane + 32 * j] = p;
        sum += p;
    }
    sum = warp_sum(sum);
    __syncwarp();
    float o[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
    for (int e0 = 0; e0 < n; e0 += BATCH) {
        uint2 vw[BATCH];
#pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            const int index = start + e0 + b;
            if (e0 + b < n) vw[b] = *reinterpret_cast<const uint2*>((index < ring_keys ? RV : VB) + row_at(e0 + b) + lane * DPL);
        }
#pragma unroll
        for (int b = 0; b < BATCH; ++b) {
            if (e0 + b >= n) break;
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw[b]);
            const float p = ps[warp * SLICE + e0 + b];
#pragma unroll
            for (int i = 0; i < DPL; ++i) o[i] = fmaf(p, bf(vb[i]), o[i]);
        }
    }
    if (lane == 0) { wm[warp] = n > 0 ? mx : -INFINITY; wl[warp] = n > 0 ? sum : 0.0f; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) wo[warp][lane * DPL + i] = o[i];
    __syncthreads();
    if (warp != 0) return;
    float top = -INFINITY;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) if (wl[w] > 0.0f) top = fmaxf(top, wm[w]);
    float total = 0.0f, out[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        if (wl[w] == 0.0f) continue;
        const float scale = __expf(wm[w] - top);
        total = fmaf(wl[w], scale, total);
#pragma unroll
        for (int i = 0; i < DPL; ++i) out[i] = fmaf(wo[w][lane * DPL + i], scale, out[i]);
    }
    if (lane == 0) { PM[slot] = total > 0.0f ? top : -INFINITY; PL[slot] = total; }
#pragma unroll
    for (int i = 0; i < DPL; ++i) PO[slot * HD + lane * DPL + i] = out[i];
}

// tf_q35d_attn_partial's arithmetic, its key and value rows staged once in shared memory for R rows of a stream and the
// four query heads of a key head. A row's keys are the absolute positions first(row) .. position + block_rows (ring rows,
// then the block's own rows); a slice's union over the R rows (at most 64 + R - 1 keys, as first moves by at most one a
// row) is staged once, a slice's values loading under its scores and the next slice's keys under its value sums.
__device__ __forceinline__ void tf_q35d_cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
}

// A (stream, R-row group, key head, chunk) block, one warp a row: the warp runs the four query heads of the key head
// (four score chains a key, each staged key and value converted once), tf_q35d_attn_partial's slice warps one after
// another, and merges each head's slices as its warp 0 did.
template <int R>
__device__ __forceinline__ void attn_partial_heads(const __nv_bfloat16* Q,
        const __nv_bfloat16* KB, const __nv_bfloat16* VB, const DraftSegment* segments,
        int block_rows, int heads, int kv_heads, int W, int window, int layer, float* PM, float* PL, float* PO) {
    constexpr int DPL = HD / 32, CHUNK = 256, SLICES = 4, G = 4, SLICE = CHUNK / SLICES, PER = SLICE / 32, BATCH = 16;
    constexpr int KROW = HD * 2 + 16, ROWS = SLICE + R - 1;
    __shared__ __align__(16) unsigned char ks[ROWS * KROW];
    __shared__ __align__(16) __nv_bfloat16 vs[ROWS * HD];
    __shared__ __align__(16) float qs[R][HD][G];
    __shared__ __align__(16) float ps[R][SLICE][G];
    const int lane = threadIdx.x & 31, r = threadIdx.x >> 5;
    const int groups = block_rows / R, stream = blockIdx.x / groups, local0 = blockIdx.x % groups * R, local = local0 + r;
    const int m = stream * block_rows + local, kvh = blockIdx.y, chunk = blockIdx.z, h0 = kvh * G;
    const DraftSegment sg = segments[stream];
    const int width = kv_heads * HD;
    const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * W * width;
    const __nv_bfloat16* RV = RK + (size_t)W * width;
    const long long first = max(sg.begin, sg.position + local - window + 1);
    const long long lo = max(sg.begin, sg.position + local0 - window + 1), hi = max(sg.begin, sg.position + local0 + R - 1 - window + 1);
    const long long end = sg.position + block_rows;
    const int ring_keys = (int)(sg.position - first), keys = ring_keys + block_rows;
    const size_t slot0 = ((size_t)m * heads + h0) * DRAFT_CHUNKS + chunk;  // head h0 + g's: slot0 + g * DRAFT_CHUNKS
    const bool live = chunk * CHUNK < keys;
    if (lo + chunk * CHUNK >= end) {
        if (lane < G) { PM[slot0 + lane * DRAFT_CHUNKS] = -INFINITY; PL[slot0 + lane * DRAFT_CHUNKS] = 0.0f; }
        return;
    }
    for (int i = lane; i < HD * G; i += 32) qs[r][i % HD][i / HD] = bf(Q[((size_t)m * heads + h0 + i / HD) * HD + i % HD]);
    const auto stage = [&](int w, bool key) {
        const long long a0 = lo + chunk * CHUNK + w * SLICE, a1 = min(hi + chunk * CHUNK + w * SLICE + SLICE, end);
        const int staged = (int)max(a1 - a0, 0LL);
        for (int c = threadIdx.x; c < staged * (HD / 8); c += R * 32) {
            const int e = c / (HD / 8), part = c % (HD / 8);
            const long long a = a0 + e;
            const bool ring = a < sg.position;
            const size_t row = ring ? (size_t)(a % W) : (size_t)(stream * block_rows + (a - sg.position));
            const size_t at = (row * kv_heads + kvh) * HD + part * 8;
            if (key) tf_q35d_cp16(ks + e * KROW + part * 16, (ring ? RK : KB) + at);
            else tf_q35d_cp16(vs + e * HD + part * 8, (ring ? RV : VB) + at);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    stage(0, true);
    float wm[G][SLICES], wl[G][SLICES], wo[G][SLICES][DPL];
#pragma unroll
    for (int w = 0; w < SLICES; ++w) {
        const int start = chunk * CHUNK + w * SLICE, n = max(min(keys - start, SLICE), 0);
        const int off = (int)(first - lo);
        if (w != 0) __syncthreads();
        stage(w, false);
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();
        float mx[G], score[G][PER];
#pragma unroll
        for (int g = 0; g < G; ++g) mx[g] = -INFINITY;
        if (live) {
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int e = lane + 32 * j;
#pragma unroll
                for (int g = 0; g < G; ++g) score[g][j] = -INFINITY;
                if (e >= n) continue;
                const uint4* k = reinterpret_cast<const uint4*>(ks + (off + e) * KROW);
                float dot[G];
#pragma unroll
                for (int g = 0; g < G; ++g) dot[g] = 0.0f;
#pragma unroll 4
                for (int c = 0; c < HD / 8; ++c) {
                    const uint4 kw = k[c];
                    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float kv = bf(kb[i]);
                        const float4 q = *reinterpret_cast<const float4*>(&qs[r][8 * c + i][0]);
                        dot[0] = fmaf(q.x, kv, dot[0]);
                        dot[1] = fmaf(q.y, kv, dot[1]);
                        dot[2] = fmaf(q.z, kv, dot[2]);
                        dot[3] = fmaf(q.w, kv, dot[3]);
                    }
                }
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    score[g][j] = dot[g] * rsqrtf((float)HD);
                    mx[g] = fmaxf(mx[g], score[g][j]);
                }
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        if (w + 1 < SLICES) stage(w + 1, true);
        if (!live) continue;
        float sum[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
#pragma unroll
            for (int o = 16; o; o >>= 1) mx[g] = fmaxf(mx[g], __shfl_xor_sync(0xffffffffu, mx[g], o));
            sum[g] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            float p[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                p[g] = lane + 32 * j < n ? __expf(score[g][j] - mx[g]) : 0.0f;
                sum[g] += p[g];
            }
            *reinterpret_cast<float4*>(&ps[r][lane + 32 * j][0]) = make_float4(p[0], p[1], p[2], p[3]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) sum[g] = warp_sum(sum[g]);
        __syncwarp();
        float o[G][DPL];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < DPL; ++i) o[g][i] = 0.0f;
        for (int e0 = 0; e0 < n; e0 += BATCH) {
            uint2 vw[BATCH];
#pragma unroll
            for (int b = 0; b < BATCH; ++b)
                if (e0 + b < n) vw[b] = *reinterpret_cast<const uint2*>(vs + (off + e0 + b) * HD + lane * DPL);
#pragma unroll
            for (int b = 0; b < BATCH; ++b) {
                if (e0 + b >= n) break;
                const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw[b]);
                const float4 p = *reinterpret_cast<const float4*>(&ps[r][e0 + b][0]);
#pragma unroll
                for (int i = 0; i < DPL; ++i) {
                    const float v = bf(vb[i]);
                    o[0][i] = fmaf(p.x, v, o[0][i]);
                    o[1][i] = fmaf(p.y, v, o[1][i]);
                    o[2][i] = fmaf(p.z, v, o[2][i]);
                    o[3][i] = fmaf(p.w, v, o[3][i]);
                }
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            wm[g][w] = n > 0 ? mx[g] : -INFINITY;
            wl[g][w] = n > 0 ? sum[g] : 0.0f;
#pragma unroll
            for (int i = 0; i < DPL; ++i) wo[g][w][i] = o[g][i];
        }
    }
    if (!live) {
        if (lane < G) { PM[slot0 + lane * DRAFT_CHUNKS] = -INFINITY; PL[slot0 + lane * DRAFT_CHUNKS] = 0.0f; }
        return;
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float top = -INFINITY;
#pragma unroll
        for (int w = 0; w < SLICES; ++w) if (wl[g][w] > 0.0f) top = fmaxf(top, wm[g][w]);
        float total = 0.0f, out[DPL];
#pragma unroll
        for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
#pragma unroll
        for (int w = 0; w < SLICES; ++w) {
            if (wl[g][w] == 0.0f) continue;
            const float scale = __expf(wm[g][w] - top);
            total = fmaf(wl[g][w], scale, total);
#pragma unroll
            for (int i = 0; i < DPL; ++i) out[i] = fmaf(wo[g][w][i], scale, out[i]);
        }
        const size_t slot = slot0 + (size_t)g * DRAFT_CHUNKS;
        if (lane == 0) { PM[slot] = total > 0.0f ? top : -INFINITY; PL[slot] = total; }
#pragma unroll
        for (int i = 0; i < DPL; ++i) PO[slot * HD + lane * DPL + i] = out[i];
    }
}

// attn_partial_heads of four rows a block: tf_q35d_attn_partial's bits for heads = 4 * kv_heads.
extern "C" __global__ void __launch_bounds__(128) tf_q35d_attn_partial_rows4(const __nv_bfloat16* Q0,
        const __nv_bfloat16* KB0, const __nv_bfloat16* VB0, const DraftSegment* segments,
        int block_rows, int heads, int kv_heads, int W, int window, int layer, float* PM0, float* PL0, float* PO0) {
    release_next();
    await_prior();
    attn_partial_heads<4>(after_wait(Q0), after_wait(KB0), after_wait(VB0), segments, block_rows, heads, kv_heads, W, window, layer, after_wait(PM0), after_wait(PL0), after_wait(PO0));
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_attn_merge(const float* PM0, const float* PL0,
        const float* PO0, int heads, __nv_bfloat16* OUT0) {
    release_next();
    await_prior();
    const float* PM = after_wait(PM0);
    const float* PL = after_wait(PL0);
    const float* PO = after_wait(PO0);
    __nv_bfloat16* OUT = after_wait(OUT0);
    constexpr int DPL = HD / 32;
    const int lane = threadIdx.x, m = blockIdx.x, h = blockIdx.y;
    const size_t base = ((size_t)m * heads + h) * DRAFT_CHUNKS;
    float mx = -INFINITY, sum = 0.0f, out[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) out[i] = 0.0f;
    for (int chunk = 0; chunk < DRAFT_CHUNKS; ++chunk) {
        const size_t slot = base + chunk;
        if (PL[slot] == 0.0f) continue;
        const float next = fmaxf(mx, PM[slot]), a = __expf(mx - next), b = __expf(PM[slot] - next);
        sum = sum * a + PL[slot] * b;
#pragma unroll
        for (int i = 0; i < DPL; ++i) out[i] = fmaf(b, PO[slot * HD + lane * DPL + i], out[i] * a);
        mx = next;
    }
#pragma unroll
    for (int i = 0; i < DPL; ++i) OUT[((size_t)m * heads + h) * HD + lane * DPL + i] = __float2bfloat16_rn(out[i] / sum);
}
