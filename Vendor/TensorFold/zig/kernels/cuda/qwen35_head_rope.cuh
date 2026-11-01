// One warp retains the two head-normalization sums; its block shares position-independent rotary frequencies.
extern "C" __global__ void __launch_bounds__(256) tf_q35_head_rope_warp(
        Geo geo, const __nv_bfloat16* __restrict__ Qraw0, const __nv_bfloat16* __restrict__ Kraw0,
        const __nv_bfloat16* __restrict__ Vraw0, const __nv_bfloat16* __restrict__ Qweight,
        const __nv_bfloat16* __restrict__ Kweight, __nv_bfloat16* __restrict__ Qn0,
        const Seg* __restrict__ segs, const int* __restrict__ rowseg,
        const Ancestor* __restrict__ rowanc, int layer, float eps,
        const int* __restrict__ P, const int* __restrict__ M) {
    release_next();
    // ahead of the wait: the block's rotary frequencies (position-independent) and this warp's head's norm weights
    // (static); the rows the kernel before wrote are read after it
    const int heads = geo.qh + geo.kvh, lane = threadIdx.x;
    __shared__ float frequencies[32];
    if (threadIdx.y == 0 && lane < 32)
        frequencies[lane] = powf(10000000.0f, -(float)lane / 32.0f);
    const size_t row_head = blockIdx.x * blockDim.y + threadIdx.y, row = row_head / heads;
    const int head = row_head % heads;
    const bool query = head < geo.qh;
    const __nv_bfloat16* wsrc = plain(query ? Qweight : Kweight);
    const uint2 wpre[2] = {*reinterpret_cast<const uint2*>(wsrc + 4 * lane), *reinterpret_cast<const uint2*>(wsrc + 128 + 4 * lane)};
    asm volatile("" ::"r"(wpre[0].x), "r"(wpre[0].y), "r"(wpre[1].x), "r"(wpre[1].y));
    __syncthreads();
    await_prior();
    const __nv_bfloat16* Qraw = after_wait(Qraw0);
    const __nv_bfloat16* Kraw = after_wait(Kraw0);
    const __nv_bfloat16* Vraw = after_wait(Vraw0);
    __nv_bfloat16* Qn = after_wait(Qn0);
    const int* counts = after_wait(P);
    const size_t first_row = blockIdx.x * blockDim.y / heads;
    if (past_rows(counts, first_row)) return;
    if (past_rows(counts, row)) return;
    const __nv_bfloat16* x = query ? Qraw + (row * geo.qh + head) * 2 * HD
                                  : Kraw + (row * geo.kvh + head - geo.qh) * HD;
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
        const __nv_bfloat16* weights = reinterpret_cast<const __nv_bfloat16*>(&wpre[half]);
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
