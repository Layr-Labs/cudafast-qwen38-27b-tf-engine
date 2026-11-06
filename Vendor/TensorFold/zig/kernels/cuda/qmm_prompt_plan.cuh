// A 496-row prompt's GEMMs on planned tiles: the persistent CTAs take items from a fixed plan (qmm_prompt_plan_tables.cuh)
// instead of tile_of's row bands, so every SM's two CTAs end together: the warp scheduler favours CTA i over CTA i + 48
// on their SM, so CTA i's items carry more rows. A plan mixes whole panels (row tiles 128, 128, 128, 112) with panels of
// five row tiles (112, 96, 96, 96, 96, or 128, 112, 96, 80, 80); a panel's tiles share a slot, so its weights stream
// once. Each tile is qmm_prompt_pipe.cuh's tile (or mlp_tile) at its own height: every output keeps prefill_kernel's
// bits, its m16 row block and n8 column block taking the same MMA chain over K in every tile shape.
#pragma once
#include "qmm_prompt_plan_tables.cuh"

namespace tf_qmm_plan {
using namespace tf_qmm_prefill;

struct Item {
    int m0, shape, n0, part;
};

__device__ __forceinline__ Item item(int at) {
    const unsigned w = tf_qmm_plan_items[at];
    return {static_cast<int>(w & 255u) * 16, static_cast<int>((w >> 8) & 31u), static_cast<int>((w >> 13) & 0x1ffffu) * 32,
            static_cast<int>(w >> 30)};
}

}  // namespace tf_qmm_plan

// Plan items first .. first + count - 1 of one or more projections of x (ManyParts' out each): CTA i takes items i,
// i + gridDim.x, ... (gridDim.x = 96).
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, int M, int K, int ldx, int first, int count) {
    using namespace tf_qmm_plan;
    // CAP3-05 D3: fire the dependents at entry, as the crown prompt bodies do (H25 parity; the plan
    // kernels replace those bodies' launches at the 496-row prompt). Scheduling only, no arithmetic.
    qmm_frag::grid_launch();
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        const Item t = item(first + w);
        const ManyPart& P = parts.p[t.part];
        switch (t.shape) {
            case 0: tf_qmm_pipe::tile<64, 128, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 1: tf_qmm_pipe::tile<64, 112, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 9: tf_qmm_pipe::tile<64, 96, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 10: tf_qmm_pipe::tile<64, 80, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            default: break;
        }
        // a faster warp cannot refill shared panels while a sibling still reads the preceding tile
        __syncthreads();
    }
}

namespace tf_qmm_plan {

// A plan item's tile height (its shape's BM in tf_qmm_prompt_plan's switch); 0 for a shape that kernel skips.
__device__ __forceinline__ int item_rows(int shape) {
    switch (shape) {
        case 0: return 128;
        case 1: return 112;
        case 9: return 96;
        case 10: return 80;
        default: return 0;
    }
}

__device__ __forceinline__ int relaxed_word(const int* f) {
    int v;
    asm volatile("ld.global.relaxed.gpu.b32 %0, [%1];\n" : "=r"(v) : "l"(f) : "memory");
    return v;
}

}  // namespace tf_qmm_plan

// tf_qmm_prompt_plan for the out projection of a GDN layer whose input x (the gated norm of the chain's y) is still
// being written: launched as the programmatic dependent of the chain's followers (qwen35.cu tf_q35_gdn_post_follow,
// then tf_q35_conv_commit), every block of which is resident by then. rows_done[b] counts the value heads whose rows
// 8b .. 8b + 7 the follower has stored (released after the stores). Before a tile, thread 0 waits until every 8-row
// block of the tile's rows (past M none) holds `need` heads, then an acquire fence and the block barrier order the
// tile's cp.async reads of x after those stores. The tile itself is tf_qmm_prompt_plan's: the same items in the same
// order, the same tile shapes and bits. Every block ends with griddepcontrol.wait, so the grid ends no earlier than
// the commit before it, and through it the follower and the chain: the kernels after it see a whole layer.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan_follow(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, int M, int K, int ldx, int first, int count,
        const int* rows_done, int need) {
    using namespace tf_qmm_plan;
    qmm_frag::grid_launch();
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        const Item t = item(first + w);
        const ManyPart& P = parts.p[t.part];
        const int h = item_rows(t.shape);
        if (threadIdx.x == 0 && h > 0) {
            const int last = min(t.m0 + h, M) - 1;
            for (int b = t.m0 / 8; b <= last / 8; ++b)
                while (relaxed_word(rows_done + b) < need) __nanosleep(1000);
            asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
        }
        __syncthreads();
        switch (t.shape) {
            case 0: tf_qmm_pipe::tile<64, 128, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 1: tf_qmm_pipe::tile<64, 112, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 9: tf_qmm_pipe::tile<64, 96, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 10: tf_qmm_pipe::tile<64, 80, 128, 1, 4, 2>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            default: break;
        }
        __syncthreads();
    }
    qmm_frag::grid_wait();
}

// The MLP's gate and up through silu(gate) * up into act on plan items (n0: the 64-column panel's first column).
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan_mlp(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, __nv_bfloat16* __restrict__ act, int M, int K, int ldx,
        int first, int count) {
    using namespace tf_qmm_plan;
    // CAP3-05 D3: fire the dependents at entry, as the crown prompt bodies do (H25 parity; the plan
    // kernels replace those bodies' launches at the 496-row prompt). Scheduling only, no arithmetic.
    qmm_frag::grid_launch();
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        const Item t = item(first + w);
        switch (t.shape) {
            case 0: tf_qmm_pipe::mlp_tile<64, 128, 1, 4, 2>(x, parts, act, M, K, ldx, t.m0, t.n0); break;
            case 1: tf_qmm_pipe::mlp_tile<64, 112, 1, 4, 2>(x, parts, act, M, K, ldx, t.m0, t.n0); break;
            case 9: tf_qmm_pipe::mlp_tile<64, 96, 1, 4, 2>(x, parts, act, M, K, ldx, t.m0, t.n0); break;
            default: break;
        }
        __syncthreads();
    }
}
