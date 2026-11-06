// One warp retains the two head-normalization sums; its block shares position-independent rotary frequencies.
extern "C" __global__ void __launch_bounds__(256) tf_q35_head_rope_warp(
        Geo geo, const __nv_bfloat16* __restrict__ Qraw0, const __nv_bfloat16* __restrict__ Kraw0,
        const __nv_bfloat16* __restrict__ Vraw0, const __nv_bfloat16* __restrict__ Qweight,
        const __nv_bfloat16* __restrict__ Kweight, __nv_bfloat16* __restrict__ Qn0,
        const Seg* __restrict__ segs, const int* __restrict__ rowseg,
        const Ancestor* __restrict__ rowanc, int layer, float eps,
        const int* __restrict__ P, const int* __restrict__ M) {
    release_next();
    await_prior();
    const __nv_bfloat16* Qraw = after_wait(Qraw0);
    const __nv_bfloat16* Kraw = after_wait(Kraw0);
    const __nv_bfloat16* Vraw = after_wait(Vraw0);
    __nv_bfloat16* Qn = after_wait(Qn0);
    const int* counts = after_wait(P);
    const int heads = geo.qh + geo.kvh, lane = threadIdx.x;
    const size_t first_row = blockIdx.x * blockDim.y / heads;
    if (past_rows(counts, first_row)) return;
    __shared__ float frequencies[32];
    if (threadIdx.y == 0 && lane < 32)
        frequencies[lane] = powf(10000000.0f, -(float)lane / 32.0f);
    __syncthreads();
    const size_t row_head = blockIdx.x * blockDim.y + threadIdx.y, row = row_head / heads;
    if (past_rows(counts, row)) return;
    const int head = row_head % heads;
    const bool query = head < geo.qh;
    const __nv_bfloat16* x = query ? Qraw + (row * geo.qh + head) * 2 * HD
                                  : Kraw + (row * geo.kvh + head - geo.qh) * HD;
    const __nv_bfloat16* weight = query ? Qweight : Kweight;
    float values[2][4], sums[2] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(x + half * 128 + 4 * lane);
        const __nv_bfloat16* bytes = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[half][i] = bf(bytes[i]);
            sums[half] += values[half][i] * values[half][i];
        }
        sums[half] = __shfl_sync(0xffffffffu, warp_sum(sums[half]), 0);
    }
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
    __nv_bfloat16 normalized[2][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(weight + half * 128 + 4 * lane);
        const __nv_bfloat16* weights = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            normalized[half][i] = __float2bfloat16_rn(bf(weights[i]) * (values[half][i] * inv));
    }
    const Seg sg = segs[rowseg[row]];
    int pos[3];
    row_positions(sg, rowanc[row], row, M, pos);
    const int slot = sg.len + (int)row - sg.row0;
    const StoreBf16 store{layer, geo.kvh};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * lane + i;
        const float own = bf(normalized[0][i]);
        const float partner = __shfl_xor_sync(0xffffffffu, own, 8);
        __nv_bfloat16 low = normalized[0][i];
        if (d < 64) {
            const int frequency = d % 32;
            const float angle = (float)pos[mrope_axis(frequency)] * frequencies[frequency];
            float sn, cs;
            sincosf(angle, &sn, &cs);
            const float a = d < 32 ? own : partner, b = d < 32 ? partner : own;
            low = __float2bfloat16_rn(d < 32 ? a * cs - b * sn : b * cs + a * sn);
        }
        const __nv_bfloat16 high = normalized[1][i];
        if (query) {
            Qn[(row * geo.qh + head) * HD + d] = low;
            Qn[(row * geo.qh + head) * HD + 128 + d] = high;
        } else {
            const int kh = head - geo.qh;
            store(sg, kh, slot, d, low, Vraw[(row * geo.kvh + kh) * HD + d]);
            store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
        }
    }
}

// TF_HROPE_PF=1: the same kernel with the rotate's sincosf folded per (d, d + 32) pair, the norm_rope PAIRFOLD
// way: the pair's angle, its frequency and its mrope axis are one (d % 32), and the d < 32 lane computes the one
// sincosf and writes both rotated lows (the same products and roundings, no sn or cs crossing lanes). Half the
// sincosf calls; every written bit the baseline's own.
extern "C" __global__ void __launch_bounds__(256) tf_q35_head_rope_warp_pf(
        Geo geo, const __nv_bfloat16* __restrict__ Qraw0, const __nv_bfloat16* __restrict__ Kraw0,
        const __nv_bfloat16* __restrict__ Vraw0, const __nv_bfloat16* __restrict__ Qweight,
        const __nv_bfloat16* __restrict__ Kweight, __nv_bfloat16* __restrict__ Qn0,
        const Seg* __restrict__ segs, const int* __restrict__ rowseg,
        const Ancestor* __restrict__ rowanc, int layer, float eps,
        const int* __restrict__ P, const int* __restrict__ M) {
    release_next();
    await_prior();
    const __nv_bfloat16* Qraw = after_wait(Qraw0);
    const __nv_bfloat16* Kraw = after_wait(Kraw0);
    const __nv_bfloat16* Vraw = after_wait(Vraw0);
    __nv_bfloat16* Qn = after_wait(Qn0);
    const int* counts = after_wait(P);
    const int heads = geo.qh + geo.kvh, lane = threadIdx.x;
    const size_t first_row = blockIdx.x * blockDim.y / heads;
    if (past_rows(counts, first_row)) return;
    __shared__ float frequencies[32];
    if (threadIdx.y == 0 && lane < 32)
        frequencies[lane] = powf(10000000.0f, -(float)lane / 32.0f);
    __syncthreads();
    const size_t row_head = blockIdx.x * blockDim.y + threadIdx.y, row = row_head / heads;
    if (past_rows(counts, row)) return;
    const int head = row_head % heads;
    const bool query = head < geo.qh;
    const __nv_bfloat16* x = query ? Qraw + (row * geo.qh + head) * 2 * HD
                                  : Kraw + (row * geo.kvh + head - geo.qh) * HD;
    const __nv_bfloat16* weight = query ? Qweight : Kweight;
    float values[2][4], sums[2] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(x + half * 128 + 4 * lane);
        const __nv_bfloat16* bytes = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[half][i] = bf(bytes[i]);
            sums[half] += values[half][i] * values[half][i];
        }
        sums[half] = __shfl_sync(0xffffffffu, warp_sum(sums[half]), 0);
    }
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
    __nv_bfloat16 normalized[2][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(weight + half * 128 + 4 * lane);
        const __nv_bfloat16* weights = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            normalized[half][i] = __float2bfloat16_rn(bf(weights[i]) * (values[half][i] * inv));
    }
    const Seg sg = segs[rowseg[row]];
    int pos[3];
    row_positions(sg, rowanc[row], row, M, pos);
    const int slot = sg.len + (int)row - sg.row0;
    const StoreBf16 store{layer, geo.kvh};
    // one lane per pair (the norm_rope PAIRFOLD pattern): the d < 32 lane computes the pair's single sincosf and
    // writes both rotated lows, so no sn or cs crosses lanes; every product, angle and rounding is the baseline's
    // own. The passthrough halves keep each lane's own entries (lane 8..15 write only their 128 + d highs; the
    // 128 + d + 32 highs are lane 8..15's own), so the writes stay disjoint and complete.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * lane + i;
        const float own = bf(normalized[0][i]);
        const float partner = __shfl_xor_sync(0xffffffffu, own, 8);
        const __nv_bfloat16 high = normalized[1][i];
        if (lane < 8) {
            const int frequency = d % 32;
            const float angle = (float)pos[mrope_axis(frequency)] * frequencies[frequency];
            float sn, cs;
            sincosf(angle, &sn, &cs);
            const __nv_bfloat16 low = __float2bfloat16_rn(own * cs - partner * sn);
            const __nv_bfloat16 low2 = __float2bfloat16_rn(partner * cs + own * sn);
            if (query) {
                Qn[(row * geo.qh + head) * HD + d] = low;
                Qn[(row * geo.qh + head) * HD + d + 32] = low2;
                Qn[(row * geo.qh + head) * HD + 128 + d] = high;
            } else {
                const int kh = head - geo.qh;
                store(sg, kh, slot, d, low, Vraw[(row * geo.kvh + kh) * HD + d]);
                store(sg, kh, slot, d + 32, low2, Vraw[(row * geo.kvh + kh) * HD + d + 32]);
                store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
            }
        } else if (lane < 16 || d >= 64) {
            // lanes 8..15 keep only their own 128 + d high (their d, d + 32 lows are the pair lane's); lanes 16..31
            // are past the rotate: their low and high are passthrough entries, as the baseline wrote them
            if (query) {
                if (lane >= 16) Qn[(row * geo.qh + head) * HD + d] = normalized[0][i];
                Qn[(row * geo.qh + head) * HD + 128 + d] = high;
            } else {
                const int kh = head - geo.qh;
                if (lane >= 16) store(sg, kh, slot, d, normalized[0][i], Vraw[(row * geo.kvh + kh) * HD + d]);
                store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
            }
        }
    }
}
// TF_HROPE_PK=1 (PEER-13): the same kernel with the query head's stores packed — the four
// 2-byte lows and the four 2-byte highs a lane already owns become one 8-byte store each, the
// same bytes at the same (8-byte aligned, lane-disjoint) addresses, rounded values bit-copied.
// The key/value cache stores keep the baseline's per-element StoreBf16 writes.; its block shares position-independent rotary frequencies.
extern "C" __global__ void __launch_bounds__(256) tf_q35_head_rope_warp_pk(
        Geo geo, const __nv_bfloat16* __restrict__ Qraw0, const __nv_bfloat16* __restrict__ Kraw0,
        const __nv_bfloat16* __restrict__ Vraw0, const __nv_bfloat16* __restrict__ Qweight,
        const __nv_bfloat16* __restrict__ Kweight, __nv_bfloat16* __restrict__ Qn0,
        const Seg* __restrict__ segs, const int* __restrict__ rowseg,
        const Ancestor* __restrict__ rowanc, int layer, float eps,
        const int* __restrict__ P, const int* __restrict__ M) {
    release_next();
    await_prior();
    const __nv_bfloat16* Qraw = after_wait(Qraw0);
    const __nv_bfloat16* Kraw = after_wait(Kraw0);
    const __nv_bfloat16* Vraw = after_wait(Vraw0);
    __nv_bfloat16* Qn = after_wait(Qn0);
    const int* counts = after_wait(P);
    const int heads = geo.qh + geo.kvh, lane = threadIdx.x;
    const size_t first_row = blockIdx.x * blockDim.y / heads;
    if (past_rows(counts, first_row)) return;
    __shared__ float frequencies[32];
    if (threadIdx.y == 0 && lane < 32)
        frequencies[lane] = powf(10000000.0f, -(float)lane / 32.0f);
    __syncthreads();
    const size_t row_head = blockIdx.x * blockDim.y + threadIdx.y, row = row_head / heads;
    if (past_rows(counts, row)) return;
    const int head = row_head % heads;
    const bool query = head < geo.qh;
    const __nv_bfloat16* x = query ? Qraw + (row * geo.qh + head) * 2 * HD
                                  : Kraw + (row * geo.kvh + head - geo.qh) * HD;
    const __nv_bfloat16* weight = query ? Qweight : Kweight;
    float values[2][4], sums[2] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(x + half * 128 + 4 * lane);
        const __nv_bfloat16* bytes = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[half][i] = bf(bytes[i]);
            sums[half] += values[half][i] * values[half][i];
        }
        sums[half] = __shfl_sync(0xffffffffu, warp_sum(sums[half]), 0);
    }
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
    __nv_bfloat16 normalized[2][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(weight + half * 128 + 4 * lane);
        const __nv_bfloat16* weights = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            normalized[half][i] = __float2bfloat16_rn(bf(weights[i]) * (values[half][i] * inv));
    }
    const Seg sg = segs[rowseg[row]];
    int pos[3];
    row_positions(sg, rowanc[row], row, M, pos);
    const int slot = sg.len + (int)row - sg.row0;
    const StoreBf16 store{layer, geo.kvh};
    __nv_bfloat16 lows[4], highs[4];  // the q path collects its stores; the k path stores per element
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * lane + i;
        const float own = bf(normalized[0][i]);
        const float partner = __shfl_xor_sync(0xffffffffu, own, 8);
        __nv_bfloat16 low = normalized[0][i];
        if (d < 64) {
            const int frequency = d % 32;
            const float angle = (float)pos[mrope_axis(frequency)] * frequencies[frequency];
            float sn, cs;
            sincosf(angle, &sn, &cs);
            const float a = d < 32 ? own : partner, b = d < 32 ? partner : own;
            low = __float2bfloat16_rn(d < 32 ? a * cs - b * sn : b * cs + a * sn);
        }
        const __nv_bfloat16 high = normalized[1][i];
        if (query) {
            lows[i] = low;
            highs[i] = high;
        } else {
            const int kh = head - geo.qh;
            store(sg, kh, slot, d, low, Vraw[(row * geo.kvh + kh) * HD + d]);
            store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
        }
    }
    if (query) {
        // the four lows and four highs are the same rounded bits the baseline stored one by one;
        // each lane's 8-byte block starts at Qn + (row * qh + head) * HD + 4 * lane (8-byte aligned)
        uint2 lo, hi;
        __nv_bfloat16* lb = reinterpret_cast<__nv_bfloat16*>(&lo);
        __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hi);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lb[j] = lows[j];
            hb[j] = highs[j];
        }
        __nv_bfloat16* qrow = Qn + (row * geo.qh + head) * HD;
        *reinterpret_cast<uint2*>(qrow + 4 * lane) = lo;
        *reinterpret_cast<uint2*>(qrow + 128 + 4 * lane) = hi;
    }
}


// TF_HROPE_PFPK=1 (composed for the integration tree): the pair-fold rotate of
// tf_q35_head_rope_warp_pf (one sincosf per (d, d + 32) pair, the pair's lane computes both
// rotated lows, no sn or cs crossing lanes) with the query head's stores packed as in
// tf_q35_head_rope_warp_pk (the same rounded bytes as 8-byte words). The two changes touch
// disjoint instructions: PF is value computation only, PK is store width only, so the composed
// kernel's written bits are the baseline's own. Query stores per lane: lanes with d < 32 emit
// their pair's two 8-byte low blocks plus their high block; lanes 8..15 only the high block
// (their lows are the pair lane's); lanes past the rotate the low and high blocks. The
// key/value cache stores keep the per-element StoreBf16 writes.
extern "C" __global__ void __launch_bounds__(256) tf_q35_head_rope_warp_pf_pk(
        Geo geo, const __nv_bfloat16* __restrict__ Qraw0, const __nv_bfloat16* __restrict__ Kraw0,
        const __nv_bfloat16* __restrict__ Vraw0, const __nv_bfloat16* __restrict__ Qweight,
        const __nv_bfloat16* __restrict__ Kweight, __nv_bfloat16* __restrict__ Qn0,
        const Seg* __restrict__ segs, const int* __restrict__ rowseg,
        const Ancestor* __restrict__ rowanc, int layer, float eps,
        const int* __restrict__ P, const int* __restrict__ M) {
    release_next();
    await_prior();
    const __nv_bfloat16* Qraw = after_wait(Qraw0);
    const __nv_bfloat16* Kraw = after_wait(Kraw0);
    const __nv_bfloat16* Vraw = after_wait(Vraw0);
    __nv_bfloat16* Qn = after_wait(Qn0);
    const int* counts = after_wait(P);
    const int heads = geo.qh + geo.kvh, lane = threadIdx.x;
    const size_t first_row = blockIdx.x * blockDim.y / heads;
    if (past_rows(counts, first_row)) return;
    __shared__ float frequencies[32];
    if (threadIdx.y == 0 && lane < 32)
        frequencies[lane] = powf(10000000.0f, -(float)lane / 32.0f);
    __syncthreads();
    const size_t row_head = blockIdx.x * blockDim.y + threadIdx.y, row = row_head / heads;
    if (past_rows(counts, row)) return;
    const int head = row_head % heads;
    const bool query = head < geo.qh;
    const __nv_bfloat16* x = query ? Qraw + (row * geo.qh + head) * 2 * HD
                                  : Kraw + (row * geo.kvh + head - geo.qh) * HD;
    const __nv_bfloat16* weight = query ? Qweight : Kweight;
    float values[2][4], sums[2] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(x + half * 128 + 4 * lane);
        const __nv_bfloat16* bytes = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[half][i] = bf(bytes[i]);
            sums[half] += values[half][i] * values[half][i];
        }
        sums[half] = __shfl_sync(0xffffffffu, warp_sum(sums[half]), 0);
    }
    const float inv = rsqrtf((sums[0] + sums[1]) / (float)HD + eps);
    __nv_bfloat16 normalized[2][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint2 packed = *reinterpret_cast<const uint2*>(weight + half * 128 + 4 * lane);
        const __nv_bfloat16* weights = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            normalized[half][i] = __float2bfloat16_rn(bf(weights[i]) * (values[half][i] * inv));
    }
    const Seg sg = segs[rowseg[row]];
    int pos[3];
    row_positions(sg, rowanc[row], row, M, pos);
    const int slot = sg.len + (int)row - sg.row0;
    const StoreBf16 store{layer, geo.kvh};
    __nv_bfloat16 lows[4], pairs[4], highs[4];  // the q path collects: own lows, the pair's lows, the highs
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * lane + i;
        const float own = bf(normalized[0][i]);
        const float partner = __shfl_xor_sync(0xffffffffu, own, 8);
        const __nv_bfloat16 high = normalized[1][i];
        if (lane < 8) {
            const int frequency = d % 32;
            const float angle = (float)pos[mrope_axis(frequency)] * frequencies[frequency];
            float sn, cs;
            sincosf(angle, &sn, &cs);
            lows[i] = __float2bfloat16_rn(own * cs - partner * sn);
            pairs[i] = __float2bfloat16_rn(partner * cs + own * sn);
            highs[i] = high;
        } else if (lane < 16) {
            highs[i] = high;  // the d and d + 32 lows are the pair lane's
        } else {
            lows[i] = normalized[0][i];
            highs[i] = high;
        }
        if (!query) {
            // the key path keeps the baseline's per-element stores of the same values
            const int kh = head - geo.qh;
            if (lane < 8) {
                store(sg, kh, slot, d, lows[i], Vraw[(row * geo.kvh + kh) * HD + d]);
                store(sg, kh, slot, d + 32, pairs[i], Vraw[(row * geo.kvh + kh) * HD + d + 32]);
                store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
            } else if (lane < 16) {
                store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
            } else {
                store(sg, kh, slot, d, lows[i], Vraw[(row * geo.kvh + kh) * HD + d]);
                store(sg, kh, slot, 128 + d, high, Vraw[(row * geo.kvh + kh) * HD + 128 + d]);
            }
        }
    }
    if (query) {
        // every block is the same rounded bytes the baseline stored one by one, at the same
        // 8-byte aligned, lane-disjoint addresses; each lane packs only the arrays it filled
        __nv_bfloat16* qrow = Qn + (row * geo.qh + head) * HD;
        if (lane < 8) {
            uint2 lo, pa, hi;
            __nv_bfloat16* lb = reinterpret_cast<__nv_bfloat16*>(&lo);
            __nv_bfloat16* pb = reinterpret_cast<__nv_bfloat16*>(&pa);
            __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hi);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lb[j] = lows[j];
                pb[j] = pairs[j];
                hb[j] = highs[j];
            }
            *reinterpret_cast<uint2*>(qrow + 4 * lane) = lo;
            *reinterpret_cast<uint2*>(qrow + 4 * lane + 32) = pa;
            *reinterpret_cast<uint2*>(qrow + 128 + 4 * lane) = hi;
        } else if (lane < 16) {
            uint2 hi;
            __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hi);
#pragma unroll
            for (int j = 0; j < 4; ++j) hb[j] = highs[j];
            *reinterpret_cast<uint2*>(qrow + 128 + 4 * lane) = hi;
        } else {
            uint2 lo, hi;
            __nv_bfloat16* lb = reinterpret_cast<__nv_bfloat16*>(&lo);
            __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(&hi);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lb[j] = lows[j];
                hb[j] = highs[j];
            }
            *reinterpret_cast<uint2*>(qrow + 4 * lane) = lo;
            *reinterpret_cast<uint2*>(qrow + 128 + 4 * lane) = hi;
        }
    }
}
