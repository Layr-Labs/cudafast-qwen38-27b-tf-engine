// CTA-local 1D bulk copies. Activation loads and all arithmetic stay on the native path.
#pragma once
#include "qmm_frag.cuh"

namespace qmm_bulk {
#if __CUDA_ARCH__ >= 1200
__device__ __forceinline__ void init(uint64_t* bar) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"(qmm_frag::smem(bar)) : "memory");
}
// Call after initialization, before the CTA rendezvous. Publish the barrier to the async proxy.
__device__ __forceinline__ void publish() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void expect(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(qmm_frag::smem(bar)), "r"(bytes) : "memory");
    publish();
}
__device__ __forceinline__ void copy(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(qmm_frag::smem(dst)), "l"(src), "r"(bytes), "r"(qmm_frag::smem(bar)) : "memory");
}
// One issuing thread waits; the caller's CTA barrier then publishes the completed bytes to all consumers.
__device__ __forceinline__ void wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "qmm_bulk_wait:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1;\n"
                 "@!ready bra qmm_bulk_wait;\n"
                 "}\n" :: "r"(qmm_frag::smem(bar)), "r"(phase) : "memory");
}
__device__ __forceinline__ void invalidate(uint64_t* bar) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(qmm_frag::smem(bar)) : "memory");
}
#endif
}
