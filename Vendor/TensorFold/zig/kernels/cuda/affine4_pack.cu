// The MLX affine-4 repack into the lane matmul's tiles (cuda/qlinear.zig), moved verbatim from nemotron_ops.cu.

#include <stdint.h>
#include <stddef.h>

// MLX (n, k/8) words -> the tiled lane-matmul layout [npad/64][kg][8][32][2] (qmm.py's pack, groups of 64).
extern "C" __global__ void tf_pack_dense(const uint32_t* __restrict__ w, int n, int k8, int kg, uint32_t* __restrict__ out,
                                         long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = idx & 1, lane = (idx >> 1) & 31, j = (idx >> 6) & 7;
    const long long tg = idx >> 9;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const long long row = t * 64 + j * 8 + (lane >> 2);
    const int c = lane & 3;
    const int offs[8] = {0, 8, 16, 24, 1, 9, 17, 25};
    uint32_t word = 0;
    if (row < n) {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int input = g * 64 + 32 * v + 2 * c + offs[p];
            const uint32_t src = w[row * k8 + (input >> 3)];
            word |= ((src >> (4 * (input & 7))) & 0xFu) << (4 * p);
        }
    }
    out[idx] = word;
}

// (n, kg) 16-bit -> (kg, npad) with zero columns past n: the tiled scales' and biases' layout.
extern "C" __global__ void tf_transpose_pad16(const uint16_t* __restrict__ x, int n, int kg, int npad,
                                              uint16_t* __restrict__ out) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(kg) * npad) return;
    const int g = static_cast<int>(idx / npad), col = static_cast<int>(idx % npad);
    out[idx] = col < n ? x[static_cast<long long>(col) * kg + g] : 0;
}


// A second, value-exact view of the packed words, with group before column panel.
extern "C" __global__ void tf_pack_wide_kmajor(const uint32_t* words, uint32_t* out, int groups, int tiles) {
 const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
 if (i >= static_cast<size_t>(groups) * tiles * 512) return;
 const size_t tile = i / (groups * 512), group = (i / 512) % groups;
 out[(group * tiles + tile) * 512 + i % 512] = words[i];
}

extern "C" __global__ void tf_pack_kmajor(const uint32_t* input, uint32_t* output, int groups, int tiles) {
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t words=size_t(groups)*tiles*512;
    if(at>=words)return;
    const size_t panel=at/512, tile=panel/groups, group=panel%groups;
    output[(group*tiles+tile)*512+at%512]=input[at];
}

extern "C" __global__ void tf_dict_mark(const uint16_t* __restrict__ v, long long count, uint32_t* __restrict__ bitmap) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const uint32_t x = v[i];
        atomicOr(bitmap + (x >> 5), 1u << (x & 31));
    }
}

extern "C" __global__ void __launch_bounds__(1024) tf_dict_rank(const uint32_t* __restrict__ bitmap, uint32_t* __restrict__ rank,
                                                                int* __restrict__ count) {
    __shared__ uint32_t sums[1024];
    const uint32_t* bm = bitmap + 2048 * blockIdx.x;
    const int t = threadIdx.x;
    const uint32_t a = __popc(bm[2 * t]), b = __popc(bm[2 * t + 1]);
    sums[t] = a + b;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const uint32_t v = t >= off ? sums[t - off] : 0u;
        __syncthreads();
        sums[t] += v;
        __syncthreads();
    }
    const uint32_t before = sums[t] - a - b;
    rank[2048 * blockIdx.x + 2 * t] = before;
    rank[2048 * blockIdx.x + 2 * t + 1] = before + a;
    if (t == 1023) count[blockIdx.x] = static_cast<int>(sums[1023]);
}

extern "C" __global__ void tf_dict_table(const uint32_t* __restrict__ bitmap, const uint32_t* __restrict__ rank, int ns,
                                         uint16_t* __restrict__ table) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4096) return;
    const int which = i >> 11, w = i & 2047;
    uint32_t m = bitmap[i];
    uint32_t at = rank[i] + (which ? ns : 0);
    while (m) {
        const int bit = __ffs(m) - 1;
        table[at++] = static_cast<uint16_t>(w * 32 + bit);
        m &= m - 1;
    }
}

__device__ __forceinline__ uint32_t dict_index(const uint32_t* __restrict__ bitmap, const uint32_t* __restrict__ rank, uint32_t v) {
    return rank[v >> 5] + __popc(bitmap[v >> 5] & ((1u << (v & 31)) - 1u));
}

extern "C" __global__ void tf_dict_pack(const uint16_t* __restrict__ s, const uint16_t* __restrict__ b, int kg, int npad,
                                        const uint32_t* __restrict__ bitmap, const uint32_t* __restrict__ rank, int bits,
                                        uint32_t* __restrict__ out, long long total, int kmajor) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = 4 * bits, lw = static_cast<int>(i % per);
    const long long tg = i / per;
    const int tiles = npad / 64;
    const int g = static_cast<int>(kmajor ? tg / tiles : tg % kg), t = static_cast<int>(kmajor ? tg % tiles : tg / kg);
    const int lo = 32 * lw / bits, hi = min(127, (32 * lw + 31) / bits);
    uint32_t word = 0;
    for (int p = lo; p <= hi; ++p) {
        const int which = p >> 6, col = t * 64 + (p & 63);
        const uint32_t v = (which ? b : s)[static_cast<long long>(g) * npad + col];
        const uint32_t idx = dict_index(bitmap + 2048 * which, rank + 2048 * which, v);
        const int start = p * bits - 32 * lw;
        word |= start >= 0 ? idx << start : idx >> -start;
    }
    out[i] = word;
}
