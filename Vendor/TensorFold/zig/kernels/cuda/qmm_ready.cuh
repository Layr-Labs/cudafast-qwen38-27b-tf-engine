// The decode MLP's tile readiness (cuda_mlp_ready.zig): the gate/up kernel publishes each finished intermediate tile
// (its act columns and their group sums), and the down kernel's K slices acquire a tile before they load its groups,
// in place of a wait on the whole gate/up grid. The words: [0] the forward's generation (the down launches finished
// in this forward), [1] the down launch's finished CTAs, [2 + t] tile t's generation once published. A forward
// starts from zeros (cuda_forward.zig clears them); every launch pair of it moves the generation by one.
#pragma once
#include <stdint.h>

namespace qmm_ready {

constexpr int GEN = 0, DONE = 1, FLAGS = 2;

// PTX ISA memory-consistency forms (sm_70 and later: every target of these fatbins, sm_80 up).
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

// One thread: until tile t carries `want`. Every published value of this forward is at most the current generation
// plus one, and a tile is published once a launch, so equality is the only ready state.
__device__ __forceinline__ void await_tile(const unsigned* ready, int t, unsigned want) {
    while (ld_acquire(ready + FLAGS + t) != want) {
    }
}

// Thread 0 of a gate/up CTA, after the CTA's act and group-sum stores and a __syncthreads: tile t ready.
__device__ __forceinline__ void publish_tile(unsigned* ready, int t, unsigned want) {
    __threadfence();
    st_release(ready + FLAGS + t, want);
}

// Thread 0 of each down CTA at its end: the last one to finish clears the count and moves the generation, after
// every CTA of the launch has read it (each reads it at its start, before it arrives here).
__device__ __forceinline__ void finish_down(unsigned* ready, unsigned want) {
    __threadfence();
    if (atomicAdd(ready + DONE, 1u) == gridDim.x - 1) {
        ready[DONE] = 0;
        __threadfence();
        st_release(ready + GEN, want);
    }
}

}  // namespace qmm_ready
