// Batched DFlash glue: stream-local convolution and rotary positions, then 256-key attention chunks.
#pragma once
struct DraftSegment { unsigned long long ring; long long begin, position; };
constexpr int DRAFT_CHUNKS = 9;

extern "C" __global__ void tf_q35d_dyn_conv_batch(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DYN,
                                              const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* __restrict__ RES,
                                              __nv_bfloat16* __restrict__ OUT, int D, int group, int branch, int block_rows) {
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
            total = rbf(total + rbf(bf(BASE[(branch * 2 + l) * D + c]) * x));
            total = rbf(total + rbf(bf(DYN[(size_t)t * (4 * groups) + (branch * 2 + l) * groups + g]) * x));
        }
        if (RES != nullptr) total = rbf(total + bf(RES[(size_t)t * D + c]));
        OUT[(size_t)t * D + c] = __float2bfloat16_rn(total);
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_norm_rope_batch(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     __nv_bfloat16* __restrict__ OUT, int heads, int stride, const DraftSegment* segments, int block_rows,
                                                                     float eps, float theta) {
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

// Remove each stream's pending row before the output head and selector; rows stay grouped by stream.
extern "C" __global__ void tf_q35d_mask_rows(const __nv_bfloat16* X, __nv_bfloat16* OUT, int D, int block_rows) {
    const int row = blockIdx.x, source = row / (block_rows - 1) * block_rows + 1 + row % (block_rows - 1);
    for (int c = threadIdx.x; c < D; c += blockDim.x) OUT[(size_t)row * D + c] = X[(size_t)source * D + c];
}

// 256 keys of one stream's key list (its ring's window, then its block rows) for every block row and every query
// head of one kv head: the 4 query heads of a kv head and the block's rows read each key and value once from shared
// memory, not once a (row, head). Grid (streams * kv_heads, DRAFT_CHUNKS), 256 threads: warp w holds 8 queries (row,
// head) and lane j key j of each 32-key tile; each query keeps an online softmax over the chunk. A row's ring keys
// start at its own window's first; a key outside a row's window scores -inf for that row. The partials (PM, PL, PO,
// a (row, head, chunk) slot each) are what tf_q35d_attn_merge sums, so any partition of a row's keys over the chunks
// gives that row's attention.
extern "C" __global__ void __launch_bounds__(256) tf_q35d_attn_partial(const __nv_bfloat16* Q,
        const __nv_bfloat16* KB, const __nv_bfloat16* VB, const DraftSegment* segments,
        int block_rows, int heads, int kv_heads, int W, int window, int layer, float* PM, float* PL, float* PO) {
    constexpr int DPL = HD / 32, CHUNK = 256, TILE = 32, QW = 8, KP = HD + 8;  // KP: a key row's bf16 stride
    __shared__ __align__(16) __nv_bfloat16 qs[64][HD];
    __shared__ __align__(16) __nv_bfloat16 ks[TILE][KP];
    __shared__ __align__(16) __nv_bfloat16 vs[TILE][HD];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int G = heads / kv_heads, stream = blockIdx.x / kv_heads, kvh = blockIdx.x % kv_heads, chunk = blockIdx.y;
    const int nq = block_rows * G;  // queries of this kv head: (row, head) pairs, row-major
    const DraftSegment sg = segments[stream];
    const int width = kv_heads * HD;
    const __nv_bfloat16* RK = reinterpret_cast<const __nv_bfloat16*>(sg.ring) + (size_t)layer * 2 * W * width;
    const __nv_bfloat16* RV = RK + (size_t)W * width;
    const long long first0 = max(sg.begin, sg.position - window + 1);  // row 0's window: the earliest ring key
    const int ring_all = (int)(sg.position - first0), keys = ring_all + block_rows;
    const int k0 = chunk * CHUNK, kn = min(CHUNK, keys - k0);
    // this warp's queries
    int qrow[QW], qhead[QW];
    long long qfirst[QW];
#pragma unroll
    for (int i = 0; i < QW; ++i) {
        const int qi = warp * QW + i;
        qrow[i] = qi < nq ? qi / G : -1;
        qhead[i] = kvh * G + (qi < nq ? qi % G : 0);
        qfirst[i] = qrow[i] >= 0 ? max(sg.begin, sg.position + qrow[i] - window + 1) : 0;
    }
    if (kn <= 0) {  // past the stream's keys: empty partials, which the merge skips
#pragma unroll
        for (int i = 0; i < QW; ++i)
            if (qrow[i] >= 0 && lane == 0) {
                const size_t slot = ((size_t)(stream * block_rows + qrow[i]) * heads + qhead[i]) * DRAFT_CHUNKS + chunk;
                PM[slot] = -INFINITY;
                PL[slot] = 0.0f;
            }
        return;
    }
    for (int i = threadIdx.x; i < nq * (HD / 8); i += blockDim.x) {
        const int qi = i / (HD / 8), c = i % (HD / 8);
        const int row = qi / G, h = kvh * G + qi % G;
        *reinterpret_cast<uint4*>(&qs[qi][8 * c]) =
            *reinterpret_cast<const uint4*>(Q + ((size_t)(stream * block_rows + row) * heads + h) * HD + 8 * c);
    }
    float mx[QW], sum[QW], o[QW][DPL];
#pragma unroll
    for (int i = 0; i < QW; ++i) {
        mx[i] = -INFINITY;
        sum[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) o[i][d] = 0.0f;
    }
    const float scale = rsqrtf((float)HD);
    for (int t0 = 0; t0 < kn; t0 += TILE) {
        const int tn = min(TILE, kn - t0);
        __syncthreads();  // the last tile's readers are done (and, the first time, qs is written)
        for (int i = threadIdx.x; i < TILE * (HD / 8); i += blockDim.x) {
            const int e = i / (HD / 8), c = i % (HD / 8);
            uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
            if (e < tn) {
                const int index = k0 + t0 + e;
                const size_t row = index < ring_all ? (size_t)((first0 + index) % W) : (size_t)(stream * block_rows + index - ring_all);
                const size_t at = (row * kv_heads + kvh) * HD + 8 * c;
                kw = *reinterpret_cast<const uint4*>((index < ring_all ? RK : KB) + at);
                vw = *reinterpret_cast<const uint4*>((index < ring_all ? RV : VB) + at);
            }
            *reinterpret_cast<uint4*>(&ks[e][8 * c]) = kw;
            *reinterpret_cast<uint4*>(&vs[e][8 * c]) = vw;
        }
        __syncthreads();
        // lane `lane` scores key t0 + lane for each of the warp's queries
        const int index = k0 + t0 + lane;
        const long long kpos = first0 + index;  // a ring key's position (block keys follow every row's ring keys)
        float sc[QW];
#pragma unroll
        for (int i = 0; i < QW; ++i) sc[i] = 0.0f;
#pragma unroll 4
        for (int c = 0; c < HD / 8; ++c) {
            const uint4 kw = *reinterpret_cast<const uint4*>(&ks[lane][8 * c]);
            const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kw);
            float kf[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) kf[j] = bf(kb[j]);
#pragma unroll
            for (int i = 0; i < QW; ++i) {
                const uint4 qw = *reinterpret_cast<const uint4*>(&qs[(warp * QW + i) < nq ? warp * QW + i : 0][8 * c]);
                const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&qw);
#pragma unroll
                for (int j = 0; j < 8; ++j) sc[i] = fmaf(bf(qb[j]), kf[j], sc[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < QW; ++i) {
            const bool live = qrow[i] >= 0 && lane < tn && (index >= ring_all || kpos >= qfirst[i]);
            const float s_ = live ? sc[i] * scale : -INFINITY;
            float tmax = s_;
#pragma unroll
            for (int off = 16; off; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            const float next = fmaxf(mx[i], tmax);
            if (next == -INFINITY) { sc[i] = 0.0f; continue; }  // nothing live yet for this query
            const float a = __expf(mx[i] - next);
            const float p = live ? __expf(s_ - next) : 0.0f;
            sum[i] = sum[i] * a + warp_sum(p);
#pragma unroll
            for (int d = 0; d < DPL; ++d) o[i][d] *= a;
            mx[i] = next;
            sc[i] = p;
        }
        // o += p . v over the tile's keys: lane `lane` owns dims lane * DPL ..
        for (int e = 0; e < tn; ++e) {
            const uint2 vw = *reinterpret_cast<const uint2*>(&vs[e][lane * DPL]);
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vw);
            float vf[DPL];
#pragma unroll
            for (int d = 0; d < DPL; ++d) vf[d] = bf(vb[d]);
#pragma unroll
            for (int i = 0; i < QW; ++i) {
                const float p = __shfl_sync(0xffffffffu, sc[i], e);
#pragma unroll
                for (int d = 0; d < DPL; ++d) o[i][d] = fmaf(p, vf[d], o[i][d]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < QW; ++i) {
        if (qrow[i] < 0) continue;
        const size_t slot = ((size_t)(stream * block_rows + qrow[i]) * heads + qhead[i]) * DRAFT_CHUNKS + chunk;
        if (lane == 0) { PM[slot] = sum[i] > 0.0f ? mx[i] : -INFINITY; PL[slot] = sum[i]; }
#pragma unroll
        for (int d = 0; d < DPL; ++d) PO[slot * HD + lane * DPL + d] = o[i][d];
    }
}

extern "C" __global__ void __launch_bounds__(32) tf_q35d_attn_merge(const float* PM, const float* PL,
        const float* PO, int heads, __nv_bfloat16* OUT) {
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
