// Persistent prompt tiles 256 columns wide on eight column warps, one CTA an SM: GDN's qkv, z, a and b over a prompt's
// rows. Each warp keeps the 128-row x 32-column fragment of the 128-column tiles (the same per-warp registers, MMA chain
// and stores), so every output keeps prefill_kernel's bits; a K group's input rows are copied once for 256 output columns
// instead of 128. Weights take three stages (qmm_prompt_pipe.cuh WD = 3): 60,416 bytes of shared memory.
#pragma once
namespace tf_qmm_persistent_wide {
using namespace tf_qmm_prefill;

constexpr int BN = 256, WN = 8;

template <int BM>
__global__ void __launch_bounds__(WN * 32, 1) kernel(const __nv_bfloat16* __restrict__ x, const __grid_constant__ ManyParts parts,
        int M, int K, int ldx, int total) {
    constexpr int STAGES = 2;
    const int rows = (M + BM - 1) / BM;
    for (int work = blockIdx.x; work < total; work += gridDim.x) {
        int tile = work;
        // rotate the row bands between cohorts, as the 128-column persistent kernels do
        if (total % rows == 0 && (int)gridDim.x % rows == 0) tile = work / rows * rows + (work % rows + work / (int)gridDim.x) % rows;
        int q = 0;
#pragma unroll
        for (int i = 1; i < MANY; i++)
            if (i < parts.count && tile >= parts.p[i].first) q = i;
        const ManyPart& P = parts.p[q];
        const int2 at = tile_of(tile - P.first, M, P.n, BM, BN, rows);
        // a short last band runs the whole tile (its rows past M are neither read nor stored): 112 rows do not split
        // into whole input chunks over 256 threads
        tf_qmm_pipe::tile<64, BM, BN, 1, WN, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, at.x, at.y);
        __syncthreads();
    }
}
} // namespace tf_qmm_persistent_wide
