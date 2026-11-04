// Lossless BP51/v1 panel view. Model-load-owned immutable weights, never activation state.
#pragma once
#include <stdint.h>
namespace tf_bp54 {
struct View { const uint8_t* blob; const uint32_t* offsets; const uint16_t* table; const uint8_t* mapping; };
struct Pair { View p[2]; };
// All lanes of one warp decode one 2048-byte panel into a disjoint shared region.
// The caller's existing CTA stage barrier publishes the synchronous stores.
__device__ __forceinline__ void decode_panel(const View& view,unsigned panel,uint32_t* output) {
    const unsigned lane=threadIdx.x&31;
    const uint8_t* blob=view.blob;
    const uint32_t* offsets=view.offsets;
    const uint16_t* table=view.table;
    const uint8_t* mapping=view.mapping;
    const uint8_t* p=blob+offsets[panel];
    // 5-bit stream length-minus-one field; last lane uses at most directory byte19.
    const unsigned db=lane*5, di=db/8;
    unsigned field=p[1536+di];
    if(di+1<20)field|=unsigned(p[1536+di+1])<<8;
    const unsigned length=((field>>(db&7))&31)+1;
    unsigned scan=length;
#pragma unroll
    for(unsigned d=1;d<32;d<<=1){unsigned v=__shfl_up_sync(0xffffffffu,scan,d);if(lane>=d)scan+=v;}
    const uint8_t* stream=p+1556+scan-length;
    unsigned bitpos=0;
#pragma unroll
    for(unsigned j=0;j<16;++j){
        const unsigned b=bitpos/8;
        // No read crosses this lane's serialized stream. Padding supplies high
        // lookup bits in its final symbol without consuming subsequent streams.
        unsigned window=0;
#pragma unroll
        for(unsigned q=0;q<3;++q)if(b+q<length)window|=unsigned(stream[b+q])<<(8*q);
        const uint16_t entry=table[(window>>(bitpos&7))&65535];
        bitpos+=entry>>8;const unsigned mask=entry&255;
        const unsigned word=lane*16+j;
        unsigned low=unsigned(p[word*3])|(unsigned(p[word*3+1])<<8)|(unsigned(p[word*3+2])<<16);
        uint32_t value=0;
#pragma unroll
        for(unsigned k=0;k<8;++k){unsigned code=((low>>(3*k))&7)|(((mask>>k)&1)<<3);value|=unsigned(mapping[code])<<(4*k);}
        output[word]=value;
    }
}
}
