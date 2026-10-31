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
// GB10 reads DRAM in 64-byte pieces: a stage loads the 256 bytes from its group's 64-byte line down, so each piece is
// fetched whole once and the next group's indices in it come from L2.
constexpr int DICT_DEC = 256;                    // shared bytes of one group's decoded scales and biases (bf16, 2 x 64)

// Packed words: a coded view whose code is C3_CODE (the kmajor bit aside) holds, a group, its 64 columns' 16-bit words
// in column order (128 bytes: the 8-bit dictionary's window, so every load is the dictionary's), each word
// sign | 7 mantissa bits | exponent - ebase (3 bits) | c (5 bits), c = (n - nbase) * 3 + d + 1, and its 16-byte table
// starts with eb = ebase | nbase << 8. c3_bits gives a word's scale bits (half 0) or bias bits (half 1): the scale's,
// and the bf16 bits of -n * scale (fp32, rounded to nearest even) moved by d ulps. affine4_pack.cu's tf_c3_pack keeps
// a projection's word view only when c3_bits gives every stored scale and bias bit for bit.
constexpr int C3_CODE = 8 | 1 << 5 | 1 << 18;

__device__ __forceinline__ bool dict_c3(int code) { return (code & 0x7FFFFFFF) == C3_CODE; }

__device__ __forceinline__ unsigned short c3_bits(unsigned u, unsigned eb, int half) {
    const unsigned sbits = (u & 0x8000u) | ((eb & 0xffu) + ((u >> 5) & 7u)) << 7 | ((u >> 8) & 0x7fu);
    if (!half) return static_cast<unsigned short>(sbits);
    const float sv = __uint_as_float(sbits << 16);
    const unsigned c = u & 31u, q = (c * 11u) >> 5;   // c / 3 for c < 32
    const unsigned code = ((eb >> 8) & 0xffu) + q | (c - 3u * q) << 5;
    const unsigned short r = __bfloat16_as_ushort(__float2bfloat16_rn(static_cast<float>(-static_cast<int>(code & 31u)) * sv));
    return static_cast<unsigned short>(r + (code >> 5) - 1u);
}

struct DictLanes {
    const unsigned char* src;                    // this thread's 16-byte chunk at the tile's first index group
    int step, bit0, base, pos;                   // bytes a group; the lane's index's bit in its group; table half
    uint32_t mask;
    bool ok;
    bool c3;                                     // packed words (C3_CODE): the lane's column word at bit (pos % 64) * 16
};

// The group's 256-byte window: from the 64-byte line its indices start in.
__device__ __forceinline__ size_t dict_window(const DictLanes& D, int g) {
    return static_cast<size_t>(g) * D.step & ~static_cast<size_t>(63);
}

__device__ __forceinline__ int dict_bits(int code) { return code & 31; }
__device__ __forceinline__ int dict_ns(int code) { return code >> 5 & 0x1FFF; }
__device__ __forceinline__ int dict_nb(int code) { return code >> 18 & 0x1FFF; }

template <int BN, int WN, typename PartT>
__device__ __forceinline__ DictLanes dict_lanes(const PartT& P, int n0, int KG, int tid) {
    static_assert(BN == 64 && BN / WN == 16, "a warp decodes its 16 columns' scales and biases, a lane each");
    const int code = P.npad, bits = dict_bits(code), lane = tid & 31, wn = tid >> 5;
    DictLanes D;
    D.step = 16 * bits;
    D.ok = tid < 16;
    D.src = reinterpret_cast<const unsigned char*>(P.scales) + static_cast<size_t>(n0 / 64) * KG * D.step + tid * 16;
    D.pos = (lane >> 4) * BN + wn * (BN / WN) + (lane & 15);
    D.bit0 = D.pos * bits;
    D.base = (lane >> 4) ? dict_ns(code) : 0;
    D.mask = (1u << bits) - 1u;
    D.c3 = dict_c3(code);
    if (D.c3) {
        D.bit0 = (D.pos & 63) * 16;
        D.mask = 0xFFFFu;
    }
    return D;
}

// The part's table (ns + nb bf16, 16-byte padded) into shared memory: static data, so it may load before the wait.
template <int THREADS, typename PartT>
__device__ __forceinline__ void dict_table(const PartT& P, unsigned char* tab, int tid) {
    const int chunks = (dict_ns(P.npad) + dict_nb(P.npad) + 7) / 8;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(P.biases);
    for (int c = tid; c < chunks; c += THREADS) cp16(tab + c * 16, src + c * 16);
}

// Group g's stage window -> the lane's bf16 value at dec[pos] (dec: 64 scales, 64 biases; the table right after it).
__device__ __forceinline__ void dict_decode(const DictLanes& D, const unsigned char* packed, unsigned char* dec, int g) {
    const uint32_t* pk = reinterpret_cast<const uint32_t*>(packed);
    const int off = (g * D.step & 63) * 8 + D.bit0;
    const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & D.mask;
    const unsigned short* tab = reinterpret_cast<const unsigned short*>(dec + DICT_DEC);
    reinterpret_cast<unsigned short*>(dec)[D.pos] = D.c3 ? c3_bits(idx, tab[0], D.pos >> 6) : tab[D.base + idx];
}

} // namespace qmm_dict
