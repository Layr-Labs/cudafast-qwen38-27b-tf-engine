// Dictionary-coded affine-4 scales and biases for the decode kernels (qmm_group.cu, lane_gemv.cu).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace qmm_dict {

using qmm_frag::cp16;

// Dictionary-coded scales and biases (built at engine open from the checkpoint's own values, lossless): a part's
// `scales` points at its index words, `biases` at its table, `npad` holds bits | ns << 5 | nb << 18. A 64-column
// tile's group is 128 indices of `bits` bits (its 64 scales', then its 64 biases'), packed little-endian into
// 16 * bits bytes; a part's tiles lie [npad / 64][kg][16 * bits] (kg * bits a multiple of 4: every tile starts on a
// 64-byte line), 256 bytes of slack after the last. The table is the ns distinct scale values, then the nb distinct
// bias values. A warp's lanes turn its 16 columns' indices back into the bf16 values the (kg, npad) arrays hold, one
// value a lane, so every fma reads the same scale and bias it reads from those arrays.
// Index windows start on a 64-byte line; streamed consumers copy only the prefix their decoder reads.
constexpr int DICT_DEC = 256;                    // shared bytes of one group's decoded scales and biases (bf16, 2 x 64)

constexpr int DICT_PAIR_SMEM = 10240;

struct DictLanes {
    const unsigned char* src;                    // this thread's 16-byte chunk at the tile's first index group
    const uint32_t* gtab;
    int step, bit0, base, pos;                   // bytes a group; the lane's index's bit in its group; table half
    int shift;
    uint32_t mask;
    bool ok, pair;
};

// The group's 256-byte window: from the 64-byte line its indices start in.
__device__ __forceinline__ size_t dict_window(const DictLanes& D, int g) {
    return static_cast<size_t>(g) * D.step & ~static_cast<size_t>(63);
}

// Include the funnel-shift decoder's lookahead word, without extending the original 256-byte window.
__host__ __device__ constexpr int dict_window_chunks(int step, int g) {
    const int chunks = (((g * step) & 63) + step + 4 + 15) / 16;
    return chunks < 16 ? chunks : 16;
}

__device__ __forceinline__ int dict_bits(int code) { return code & 31; }
__device__ __forceinline__ int dict_ns(int code) { return code >> 5 & 0x1FFF; }
__device__ __forceinline__ int dict_nb(int code) { return code >> 18 & 0x1FFF; }
__device__ __forceinline__ bool dict_pair(int code) { return dict_nb(code) == 0; }
__device__ __forceinline__ int dict_np(int code) { return dict_ns(code); }
__device__ __forceinline__ int dict_step(int code) { return (dict_pair(code) ? 8 : 16) * dict_bits(code); }
__device__ __forceinline__ int dict_tab_bytes(int code) {
    if (!dict_pair(code)) return (dict_ns(code) + dict_nb(code) + 7) / 8 * 16;
    const int b = (dict_np(code) * 4 + 15) / 16 * 16;
    return b <= DICT_PAIR_SMEM ? b : 0;
}
__device__ __forceinline__ int dict_tab_bytes_upto(int code, int room) {
    if (!dict_pair(code)) return (dict_ns(code) + dict_nb(code) + 7) / 8 * 16;
    const int b = (dict_np(code) * 4 + 15) / 16 * 16;
    return b <= room ? b : 0;
}
__device__ __forceinline__ uint32_t dict_pair_half(uint32_t entry, int shift) { return entry >> shift & 0xFFFFu; }
__device__ __forceinline__ uint32_t dict_pair_entry(const unsigned char* tab, const uint32_t* gtab, uint32_t idx) {
    return gtab != nullptr ? __ldg(gtab + idx) : reinterpret_cast<const uint32_t*>(tab)[idx];
}

template <int BN, int WN, typename PartT>
__device__ __forceinline__ DictLanes dict_lanes(const PartT& P, int n0, int KG, int tid) {
    static_assert(BN == 64 && BN / WN == 16, "a warp decodes its 16 columns' scales and biases, a lane each");
    const int code = P.npad, bits = dict_bits(code), lane = tid & 31, wn = tid >> 5;
    DictLanes D;
    D.pair = dict_pair(code);
    D.step = dict_step(code);
    D.ok = tid < 16;
    D.src = reinterpret_cast<const unsigned char*>(P.scales) + static_cast<size_t>(n0 / 64) * KG * D.step + tid * 16;
    const int col = wn * (BN / WN) + (lane & 15);
    D.pos = (lane >> 4) * BN + col;
    D.bit0 = (D.pair ? col : D.pos) * bits;
    D.base = !D.pair && (lane >> 4) ? dict_ns(code) : 0;
    D.shift = D.pair ? (lane >> 4) * 16 : 0;
    D.gtab = D.pair && dict_tab_bytes(code) == 0 ? reinterpret_cast<const uint32_t*>(P.biases) : nullptr;
    D.mask = (1u << bits) - 1u;
    return D;
}

template <typename PartT>
__device__ __forceinline__ void dict_room(DictLanes& D, const PartT& P, int room) {
    D.gtab = D.pair && dict_tab_bytes_upto(P.npad, room) == 0 ? reinterpret_cast<const uint32_t*>(P.biases) : nullptr;
}

// The part's table (ns + nb bf16, 16-byte padded) into shared memory: static data, so it may load before the wait.
template <int THREADS, typename PartT>
__device__ __forceinline__ void dict_table(const PartT& P, unsigned char* tab, int tid, int room = DICT_PAIR_SMEM) {
    const int chunks = dict_tab_bytes_upto(P.npad, room) / 16;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(P.biases);
    for (int c = tid; c < chunks; c += THREADS) cp16(tab + c * 16, src + c * 16);
}

__device__ __forceinline__ uint32_t dict_fetch(const DictLanes& D, const unsigned char* packed, const unsigned char* tab,
                                               int g) {
    const uint32_t* pk = reinterpret_cast<const uint32_t*>(packed);
    const int off = (g * D.step & 63) * 8 + D.bit0;
    const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & D.mask;
    if (D.pair) return dict_pair_half(dict_pair_entry(tab, D.gtab, idx), D.shift);
    return reinterpret_cast<const unsigned short*>(tab)[D.base + idx];
}

__device__ __forceinline__ uint32_t dict_fetch_raw(const DictLanes& D, const unsigned char* packed, const unsigned char* tab,
                                                   int g) {
    if (D.pair && D.shift != 0) return 0u;
    const uint32_t* pk = reinterpret_cast<const uint32_t*>(packed);
    const int off = (g * D.step & 63) * 8 + D.bit0;
    const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & D.mask;
    if (D.pair) return dict_pair_entry(tab, D.gtab, idx);
    return reinterpret_cast<const unsigned short*>(tab)[D.base + idx];
}

__device__ __forceinline__ void dict_raw_vals(const DictLanes& D, uint32_t dv, float& sf, float& bf) {
    const __nv_bfloat162 sb = *reinterpret_cast<const __nv_bfloat162*>(&dv);
    sf = __low2float(sb);
    bf = D.pair ? __high2float(sb) : sf;
}

__device__ __forceinline__ void x_keep_l2(const __nv_bfloat16* x, int rows, int K, int ldx) {
#if __CUDA_ARCH__ >= 900
    const int lines = K / 64, total = rows * lines;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(x + static_cast<size_t>(i / lines) * ldx) + (i % lines) * 128;
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p));
    }
#endif
}

__device__ __forceinline__ void dict_put(const DictLanes& D, unsigned char* dec, uint32_t v) {
    reinterpret_cast<unsigned short*>(dec)[D.pos] = static_cast<unsigned short>(v);
}

__device__ __forceinline__ void dict_decode(const DictLanes& D, const unsigned char* packed, unsigned char* dec, int g) {
    dict_put(D, dec, dict_fetch(D, packed, dec + DICT_DEC, g));
}

} // namespace qmm_dict
