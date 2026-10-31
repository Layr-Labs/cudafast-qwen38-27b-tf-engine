#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace qmm_dict {

using qmm_frag::cp16;

constexpr int DICT_DEC = 256;

struct DictLanes {
    const unsigned char* src;
    int step, bit0, base, pos;
    uint32_t mask;
    bool ok;
};

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
    return D;
}

template <int THREADS, typename PartT>
__device__ __forceinline__ void dict_table(const PartT& P, unsigned char* tab, int tid) {
    const int chunks = (dict_ns(P.npad) + dict_nb(P.npad) + 7) / 8;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(P.biases);
    for (int c = tid; c < chunks; c += THREADS) cp16(tab + c * 16, src + c * 16);
}

__device__ __forceinline__ void dict_decode(const DictLanes& D, const unsigned char* packed, unsigned char* dec, int g) {
    const uint32_t* pk = reinterpret_cast<const uint32_t*>(packed);
    const int off = (g * D.step & 63) * 8 + D.bit0;
    const uint32_t idx = __funnelshift_r(pk[off >> 5], pk[(off >> 5) + 1], off & 31) & D.mask;
    const unsigned short* tab = reinterpret_cast<const unsigned short*>(dec + DICT_DEC);
    reinterpret_cast<unsigned short*>(dec)[D.pos] = tab[D.base + idx];
}

}
