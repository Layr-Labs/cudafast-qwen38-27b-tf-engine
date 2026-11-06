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
// LOP3 threads child07's pair_lop3 through the planned path's tile calls (the §1b 2-switch on this
// tree); the default keeps the base instantiation. MLP-side has its own twin (TF_PAIRLOP3MLP).
template <bool LOP3>
__device__ __forceinline__ void plan_body(const __nv_bfloat16* __restrict__ x,
        const tf_qmm_prefill::ManyParts& parts, int M, int K, int ldx, int first, int count) {
    using namespace tf_qmm_plan;
    // CAP3-05 D3: fire the dependents at entry, as the crown prompt bodies do (H25 parity; the plan
    // kernels replace those bodies' launches at the 496-row prompt). Scheduling only, no arithmetic.
    qmm_frag::grid_launch();
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        const Item t = item(first + w);
        const ManyPart& P = parts.p[t.part];
        switch (t.shape) {
            case 0: tf_qmm_pipe::tile<64, 128, 128, 1, 4, 2, LOP3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 1: tf_qmm_pipe::tile<64, 112, 128, 1, 4, 2, LOP3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 9: tf_qmm_pipe::tile<64, 96, 128, 1, 4, 2, LOP3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            case 10: tf_qmm_pipe::tile<64, 80, 128, 1, 4, 2, LOP3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, t.m0, t.n0); break;
            default: break;
        }
        // a faster warp cannot refill shared panels while a sibling still reads the preceding tile
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, int M, int K, int ldx, int first, int count) {
    plan_body<false>(x, parts, M, K, ldx, first, count);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan_l(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, int M, int K, int ldx, int first, int count) {
    plan_body<true>(x, parts, M, K, ldx, first, count);
}

// The MLP's gate and up through silu(gate) * up into act on plan items (n0: the 64-column panel's first column).
template <bool MLPLOP3>
__device__ __forceinline__ void plan_mlp_body(const __nv_bfloat16* __restrict__ x,
        const tf_qmm_prefill::ManyParts& parts, __nv_bfloat16* __restrict__ act, int M, int K, int ldx,
        int first, int count) {
    using namespace tf_qmm_plan;
    // CAP3-05 D3: fire the dependents at entry, as the crown prompt bodies do (H25 parity; the plan
    // kernels replace those bodies' launches at the 496-row prompt). Scheduling only, no arithmetic.
    qmm_frag::grid_launch();
    for (int w = blockIdx.x; w < count; w += gridDim.x) {
        const Item t = item(first + w);
        switch (t.shape) {
            case 0: tf_qmm_pipe::mlp_tile<64, 128, 1, 4, 2, MLPLOP3>(x, parts, act, M, K, ldx, t.m0, t.n0); break;
            case 1: tf_qmm_pipe::mlp_tile<64, 112, 1, 4, 2, MLPLOP3>(x, parts, act, M, K, ldx, t.m0, t.n0); break;
            case 9: tf_qmm_pipe::mlp_tile<64, 96, 1, 4, 2, MLPLOP3>(x, parts, act, M, K, ldx, t.m0, t.n0); break;
            default: break;
        }
        __syncthreads();
    }
}

// PFC-22 MLP twin (TF_PAIRLOP3MLP=1): mlp_tile's weight pair through the same and-or lop3; MLP rounding untouched.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan_mlp(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, __nv_bfloat16* __restrict__ act, int M, int K, int ldx,
        int first, int count) {
    plan_mlp_body<false>(x, parts, act, M, K, ldx, first, count);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_plan_mlp_l(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts, __nv_bfloat16* __restrict__ act, int M, int K, int ldx,
        int first, int count) {
    plan_mlp_body<true>(x, parts, act, M, K, ldx, first, count);
}
