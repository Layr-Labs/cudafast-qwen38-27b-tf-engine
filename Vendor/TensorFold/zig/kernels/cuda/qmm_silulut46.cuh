#pragma once
// Immutable per-module BF16-domain SiLU values, initialized once during engine open.
// Keep nonfinite gates on the original expression, including its NaN semantics.
namespace tf_silulut46 {
__device__ float values[65536];
__device__ __forceinline__ float original(float g) { return g / (1.0f + expf(-g)); }
__device__ __forceinline__ float gate(float g) {
    const unsigned bits = __float_as_uint(g);
    if ((bits & 0x7f800000u) == 0x7f800000u) return original(g);
    return values[bits >> 16];
}
}
extern "C" __global__ void tf_silulut46_init() {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) tf_silulut46::values[i] = tf_silulut46::original(__uint_as_float(i << 16));
}
