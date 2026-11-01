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

extern "C" __global__ void __launch_bounds__(128) tf_fastpool_read(const unsigned char* __restrict__ p, unsigned long long slab,
                                                                    uint32_t* __restrict__ out) {
    constexpr int STAGES = 8, SB = 128 * 16;
    __shared__ __align__(16) unsigned char sm[STAGES * SB];
    const unsigned char* base = p + blockIdx.x * slab;
    const int steps = static_cast<int>(slab / SB);
    uint32_t acc = 0;
    auto load = [&](int s, int i) {
        const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(sm + s * SB + threadIdx.x * 16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(base + static_cast<size_t>(i) * SB + threadIdx.x * 16));
    };
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < steps) load(s, s);
        asm volatile("cp.async.commit_group;");
    }
    for (int i = 0; i < steps; ++i) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncthreads();
        const int next = i + STAGES - 1;
        if (next < steps) load(next % STAGES, next);
        asm volatile("cp.async.commit_group;");
        acc ^= *reinterpret_cast<const uint32_t*>(sm + (i % STAGES) * SB + threadIdx.x * 16);
    }
    asm volatile("cp.async.wait_all;");
    if (acc == 0x9e3779b9u) out[0] = acc;
}

extern "C" __global__ void tf_pack_kmajor(const uint32_t* input, uint32_t* output, int groups, int tiles) {
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t words=size_t(groups)*tiles*512;
    if(at>=words)return;
    const size_t panel=at/512, tile=panel/groups, group=panel%groups;
    output[(group*tiles+tile)*512+at%512]=input[at];
}

// Dictionary coding of the tiled scales and biases (qmm_dict.cuh), built at engine open from their own values.
// tf_dict_mark: a presence bit for every 16-bit value of a (kg, npad) array (2048 words, zeroed by the caller).
extern "C" __global__ void tf_dict_mark(const uint16_t* __restrict__ v, long long count, uint32_t* __restrict__ bitmap) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const uint32_t x = v[i];
        atomicOr(bitmap + (x >> 5), 1u << (x & 31));
    }
}

// tf_dict_rank: block b (0: scales, 1: biases) turns its bitmap into each word's count of smaller present values
// (rank + 2048 * b) and its total (count[b]); 1024 threads, two words a thread.
extern "C" __global__ void __launch_bounds__(1024) tf_dict_rank(const uint32_t* __restrict__ bitmap, uint32_t* __restrict__ rank,
                                                                int* __restrict__ count) {
    __shared__ uint32_t sums[1024];
    const uint32_t* bm = bitmap + 2048 * blockIdx.x;
    const int t = threadIdx.x;
    const uint32_t a = __popc(bm[2 * t]), b = __popc(bm[2 * t + 1]);
    sums[t] = a + b;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {           // inclusive scan (Hillis-Steele)
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

// tf_dict_table: the ns present scale values in increasing bit order, then the nb bias values; one thread a word.
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

// tf_dict_pack: each 64-column tile's group of the (kg, npad) scales s and biases b as 128 indices of `bits` bits
// (the tile's 64 scales, then its 64 biases), packed little-endian into 4 * bits words: out [npad / 64][kg][4 * bits],
// or group-major [kg][npad / 64][4 * bits] (`kmajor`: tf_qmm_mlp_single16's K-major panels).
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

extern "C" __global__ void tf_dict_pair_mark(const uint16_t* __restrict__ s, const uint16_t* __restrict__ b, long long count,
                                             const uint32_t* __restrict__ bitmap, const uint32_t* __restrict__ rank,
                                             uint32_t* __restrict__ pairs) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const uint32_t key = dict_index(bitmap, rank, s[i]) << 12 | dict_index(bitmap + 2048, rank + 2048, b[i]);
        atomicOr(pairs + (key >> 5), 1u << (key & 31));
    }
}

extern "C" __global__ void __launch_bounds__(1024) tf_dict_pair_rank(const uint32_t* __restrict__ pairs, uint32_t* __restrict__ rank,
                                                                     uint32_t* __restrict__ totals) {
    __shared__ uint32_t sums[1024];
    const int t = threadIdx.x;
    const size_t w0 = static_cast<size_t>(blockIdx.x) * 2048 + 2 * t;
    const uint32_t a = __popc(pairs[w0]), c = __popc(pairs[w0 + 1]);
    sums[t] = a + c;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const uint32_t v = t >= off ? sums[t - off] : 0u;
        __syncthreads();
        sums[t] += v;
        __syncthreads();
    }
    const uint32_t before = sums[t] - a - c;
    rank[w0] = before;
    rank[w0 + 1] = before + a;
    if (t == 1023) totals[blockIdx.x] = sums[1023];
}

extern "C" __global__ void __launch_bounds__(256) tf_dict_pair_offsets(uint32_t* __restrict__ totals) {
    __shared__ uint32_t sums[256];
    const int t = threadIdx.x;
    const uint32_t own = totals[t];
    sums[t] = own;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        const uint32_t v = t >= off ? sums[t - off] : 0u;
        __syncthreads();
        sums[t] += v;
        __syncthreads();
    }
    totals[t] = sums[t] - own;
    if (t == 255) totals[256] = sums[255];
}

__device__ __forceinline__ uint32_t dict_pair_index(const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ rank,
                                                    const uint32_t* __restrict__ offsets, uint32_t key) {
    const uint32_t w = key >> 5;
    return offsets[w >> 11] + rank[w] + __popc(pairs[w] & ((1u << (key & 31)) - 1u));
}

extern "C" __global__ void tf_dict_pair_table(const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ rank,
                                              const uint32_t* __restrict__ offsets, const uint16_t* __restrict__ stab, int ns,
                                              uint32_t* __restrict__ table) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= (1 << 19)) return;
    uint32_t m = pairs[w];
    uint32_t at = offsets[w >> 11] + rank[w];
    while (m) {
        const uint32_t key = static_cast<uint32_t>(w) * 32 + (__ffs(m) - 1);
        table[at++] = static_cast<uint32_t>(stab[key >> 12]) | static_cast<uint32_t>(stab[ns + (key & 4095)]) << 16;
        m &= m - 1;
    }
}

extern "C" __global__ void tf_dict_pair_pack(const uint16_t* __restrict__ s, const uint16_t* __restrict__ b, int kg, int npad,
                                             const uint32_t* __restrict__ bitmap, const uint32_t* __restrict__ rank,
                                             const uint32_t* __restrict__ pairs, const uint32_t* __restrict__ prank,
                                             const uint32_t* __restrict__ offsets, int pbits, uint32_t* __restrict__ out,
                                             long long total, int kmajor) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = 2 * pbits, lw = static_cast<int>(i % per);
    const long long tg = i / per;
    const int tiles = npad / 64;
    const int g = static_cast<int>(kmajor ? tg / tiles : tg % kg), t = static_cast<int>(kmajor ? tg % tiles : tg / kg);
    const int lo = 32 * lw / pbits, hi = min(63, (32 * lw + 31) / pbits);
    uint32_t word = 0;
    for (int p = lo; p <= hi; ++p) {
        const long long at = static_cast<long long>(g) * npad + t * 64 + p;
        const uint32_t key = dict_index(bitmap, rank, s[at]) << 12 | dict_index(bitmap + 2048, rank + 2048, b[at]);
        const uint32_t idx = dict_pair_index(pairs, prank, offsets, key);
        const int start = p * pbits - 32 * lw;
        word |= start >= 0 ? idx << start : idx >> -start;
    }
    out[i] = word;
}

extern "C" __global__ void tf_dict_pair_check(const uint16_t* __restrict__ s, const uint16_t* __restrict__ b, int kg, int npad,
                                              const uint32_t* __restrict__ stream, const uint32_t* __restrict__ table, int pbits,
                                              int kmajor, unsigned long long* __restrict__ bad) {
    const long long count = static_cast<long long>(kg) * npad;
    const int tiles = npad / 64;
    const uint32_t mask = (1u << pbits) - 1u;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int g = static_cast<int>(i / npad), col = static_cast<int>(i % npad), t = col / 64, c = col % 64;
        const long long tg = kmajor ? static_cast<long long>(g) * tiles + t : static_cast<long long>(t) * kg + g;
        const long long bit = tg * 64 * pbits + static_cast<long long>(c) * pbits;
        const uint32_t idx = __funnelshift_r(stream[bit >> 5], stream[(bit >> 5) + 1], static_cast<int>(bit & 31)) & mask;
        const uint32_t e = table[idx];
        if ((e & 0xFFFFu) != s[i] || (e >> 16) != b[i]) atomicAdd(bad, 1ull);
    }
}
