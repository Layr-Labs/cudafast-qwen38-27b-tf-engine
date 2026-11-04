// A balanced persistent schedule for one 5120-column prompt projection in four row bands on 96 slots (the out, o and
// down projections of a 385..512-row prompt on a GB10's 48 SMs, two CTAs each). The host passes each slot's tile list:
// one 128-column tile of the first 24 panels, then 64- and 32-column tiles of the last 16 panels chosen so each SM's two
// slots carry about the same work (their 160 whole tiles left a second wave a third full), with three weight stages so
// the narrow tiles keep their weight copies two groups ahead. Every tile is
// qmm_prompt_pipe.cuh's body with its MMA chain over K, so each output keeps prefill_kernel's bits.
#pragma once
namespace tf_qmm_persistent_bal {
using namespace tf_qmm_prefill;

constexpr int SLOTS = 96, MAX_TILES = 320;

// start[s] .. start[s + 1]: slot s's entries in desc; an entry is band (bits 0-1), width (bits 2-3: 128, 64, 32
// columns) and first column / 32 (bits 4-11).
struct Sched {
    unsigned short start[SLOTS + 1];
    unsigned short desc[MAX_TILES];
};

template <int BM, int WM>
__global__ void __launch_bounds__(WM * 128) kernel(const __nv_bfloat16* __restrict__ x, const __grid_constant__ ManyParts parts,
        const __grid_constant__ Sched sched, int M, int K, int ldx) {
    constexpr int BN = 128, STAGES = 2;
    const ManyPart& P = parts.p[0];
    const int end = sched.start[blockIdx.x + 1];
    for (int i = sched.start[blockIdx.x]; i < end; ++i) {
        const int d = sched.desc[i];
        const int m0 = (d & 3) * BM, width = (d >> 2) & 3, n0 = (d >> 4) * 32;
        if (m0 >= M) continue;
        const bool sh = tf_qmm_rows0::short_tile<BM, WM>(M - m0);
        if (width == 0) {
            if (sh) tf_qmm_pipe::tile<64, BM - 16 * WM, BN, WM, 4, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, m0, n0);
            else tf_qmm_pipe::tile<64, BM, BN, WM, 4, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, m0, n0);
        } else if (width == 1) {
            if (sh) tf_qmm_pipe::tile<64, BM - 16 * WM, BN / 2, WM, 4, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, m0, n0);
            else tf_qmm_pipe::tile<64, BM, BN / 2, WM, 4, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, m0, n0);
        } else {
            if (sh) tf_qmm_pipe::tile<64, BM - 16 * WM, BN / 4, WM, 4, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, m0, n0);
            else tf_qmm_pipe::tile<64, BM, BN / 4, WM, 4, STAGES, 3>(x, P.w, P.scales, P.biases, P.out, P.n, P.npad, M, K, ldx, m0, n0);
        }
        // the next tile's first copies refill the stages this tile's slower warps may still read
        __syncthreads();
    }
}
} // namespace tf_qmm_persistent_bal
