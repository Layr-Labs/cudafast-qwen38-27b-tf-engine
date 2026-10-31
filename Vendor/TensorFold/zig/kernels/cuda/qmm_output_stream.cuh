// Four native K slices share a CTA, with their original ordered sum in shared memory.
#pragma once
#include "qmm_frag.cuh"
#include "qmm_dict.cuh"
namespace tf_qmm_output_stream {
using namespace qmm_frag;
constexpr int OUT_PAIR_SMEM=14336;

// Each 64-column dictionary window occupies 256 bytes after the weight words.
// Each slice decodes in registers; the shared table follows all pipeline stages.
__device__ __forceinline__ void dict_decode_at(const qmm_dict::DictLanes& D,const unsigned char* packed,unsigned char* dec,
        const unsigned char* table,int g){
    qmm_dict::dict_put(D,dec,qmm_dict::dict_fetch(D,packed,table,g));
}

template<int FutureGroups,int Slices=4,int Rolling=0,bool DICT=false,int BN=64>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ x,const float* __restrict__ xs,
        const tf_qmm_group::Parts& parts,int M,int K,int ldx) {
    static_assert(!DICT||Rolling==0,"coded rolling prefetch is not implemented");
    static_assert(BN==64||(BN==128&&DICT),"supported output tile");
    constexpr int Passes=BN/64,NT=BN/32,Words=BN*32,Meta=2048+Words,Sums=Meta+BN*4;
    constexpr int Stage=Sums+64,SliceBytes=Stage*2,Bytes=SliceBytes*Slices;
    static_assert(Slices*NT*4*128*sizeof(float)<=Bytes,"ordered partials fit after the pipeline drains");
    extern __shared__ __align__(128) unsigned char buf[];
    const auto p=parts.p[0];
    const int slice=threadIdx.x/128,tid=threadIdx.x%128,lane=tid&31,wn=tid>>5;
    const int KG=K/64,per=KG/Slices,g0=slice*per,tile=blockIdx.x,n0=tile*BN,rows=min(M,16);
    const int tiles=DICT?int(gridDim.x)*Passes:p.npad/64;
    auto* local=buf+slice*SliceBytes;
    auto stage=[&](int it){return local+(it&1)*Stage;};
    auto weight=[&](int g){return reinterpret_cast<const unsigned char*>(p.w)+(size_t(g)*tiles+tile*Passes)*2048;};
    qmm_dict::DictLanes D[Passes]{};
    if constexpr(DICT){
#pragma unroll
        for(int pass=0;pass<Passes;++pass){
            const int logical=wn*Passes+pass;
            D[pass]=qmm_dict::dict_lanes<64,4>(p,n0+(logical/4)*64,KG,(logical%4)*32+lane);
            qmm_dict::dict_room(D[pass],p,OUT_PAIR_SMEM);
        }
    }
    unsigned char* dec=buf+Bytes;
    unsigned char* table=buf+Bytes;
    auto load_w=[&](int it){
        auto* dst=stage(it);const int g=g0+it;
#pragma unroll
        for(int part=0;part<Passes;++part)cp16(dst+2048+part*2048+tid*16,weight(g)+part*2048+tid*16);
        if constexpr(DICT){
            if(tid<16){
#pragma unroll
                for(int part=0;part<Passes;++part){
                    const auto* src=reinterpret_cast<const unsigned char*>(p.scales)+size_t(n0/64+part)*KG*D[0].step+tid*16;
                    cp16(dst+Meta+part*256+tid*16,src+qmm_dict::dict_window(D[0],g));
                }
            }
        }else if(tid<16){const auto* table=tid<8?p.scales:p.biases;cp16(dst+Meta+tid*16,table+size_t(g)*p.npad+n0+(tid%8)*8);}
    };
    auto load_x=[&](int it){
        auto* dst=stage(it);const int row=tid/8,ch=tid%8,g=g0+it;
        cp16z(dst+row*128+swz<8>(row,ch)*16,x+size_t(min(row,rows-1))*ldx+g*64+ch*8,row<rows);
        if(tid<16){if(tid<rows)cp4(dst+Sums+tid*4,xs+size_t(tid)*KG+g);else *reinterpret_cast<float*>(dst+Sums+tid*4)=0;}
    };
    if constexpr(DICT)qmm_dict::dict_table<Slices*128>(p,table,threadIdx.x,OUT_PAIR_SMEM);
    load_w(0);commit();
#if __CUDA_ARCH__ >= 900
    if(tid==0) {
#pragma unroll
        for(int g=1;g<=FutureGroups;++g)if(g<per){
            const auto* w=weight(g0+g);
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 2048;" :: "l"(w) : "memory");
            if constexpr(DICT){
                const auto* idx=D[0].src+qmm_dict::dict_window(D[0],g0+g);
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 256;" :: "l"(idx) : "memory");
            }else{
                const auto* s=p.scales+size_t(g0+g)*p.npad+n0;const auto* b=p.biases+size_t(g0+g)*p.npad+n0;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(s) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(b) : "memory");
            }
        }
    }
#endif
    grid_wait();qmm_dict::x_keep_l2(x,rows,K,ldx);load_x(0);commit();grid_launch();
    uint32_t mask;asm volatile("mov.b32 %0, 0x000F000F;\n":"=r"(mask));
    float acc[NT][4]{};
    const int b_off=DICT?(D[0].pair?0:16):0;
    for(int it=0;it<per;++it){
        #if __CUDA_ARCH__ >= 900
        if constexpr(Rolling>0)if(tid==0&&it+Rolling<per){
            const int g=g0+it+Rolling;const auto* w=weight(g);
            const auto* s=p.scales+size_t(g)*p.npad+n0;const auto* b=p.biases+size_t(g)*p.npad+n0;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 2048;" :: "l"(w) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(s) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(b) : "memory");
        }
#endif
        wait<0>();__syncthreads();
        if(it+1<per){load_w(it+1);load_x(it+1);}commit();
        const auto* at=stage(it);const auto* wp=reinterpret_cast<const uint32_t*>(at+2048);
        uint32_t dv[Passes]{};
        if constexpr(DICT){
#pragma unroll
            for(int pass=0;pass<Passes;++pass)dv[pass]=qmm_dict::dict_fetch_raw(D[pass],at+Meta+((wn*Passes+pass)/4)*256,table,g0+it);
        }
        const auto* sp=reinterpret_cast<const __nv_bfloat16*>(DICT?dec:at+Meta);const auto* sums=reinterpret_cast<const float*>(at+Sums);
        uint32_t words[NT][2];
#pragma unroll
        for(int j=0;j<NT;++j)
#pragma unroll
            for(int v=0;v<2;++v)words[j][v]=wp[((wn*NT+j)*32+lane)*2+v];
        float d[NT][4];
#pragma unroll
        for(int kt=0;kt<4;++kt){
            uint32_t a[4];const int r=(lane&7)+((lane>>3)&1)*8,ch=kt*2+(lane>>4);
            ldmatrix4(a,at+r*128+swz<8>(r,ch)*16);
#pragma unroll
            for(int j=0;j<NT;++j){const auto b0=tf_qmm_group::pairm(words[j][kt/2],(kt&1)*8,mask),b1=tf_qmm_group::pairm(words[j][kt/2],(kt&1)*8+4,mask);
                if(kt==0)mma0(d[j],a,b0,b1);else mma(d[j],a,b0,b1);
            }
        }
        [[maybe_unused]] float sf[Passes]{},bf[Passes]{};
        if constexpr(DICT){
#pragma unroll
            for(int pass=0;pass<Passes;++pass)qmm_dict::dict_raw_vals(D[pass],dv[pass],sf[pass],bf[pass]);
        }
        const int r=lane>>2;const float xv[2]={sums[r],sums[r+8]};
#pragma unroll
        for(int j=0;j<NT;++j){
            float sv[2],bv[2];
            if constexpr(DICT){
                const int pass=j/2,c0=(j%2)*8+(lane&3)*2;
                sv[0]=__shfl_sync(0xffffffffu,sf[pass],c0);sv[1]=__shfl_sync(0xffffffffu,sf[pass],c0+1);
                bv[0]=__shfl_sync(0xffffffffu,bf[pass],b_off+c0);bv[1]=__shfl_sync(0xffffffffu,bf[pass],b_off+c0+1);
            }else{
                const int col=wn*(BN/4)+j*8+(lane&3)*2;
                const auto s2=*reinterpret_cast<const __nv_bfloat162*>(sp+col),b2=*reinterpret_cast<const __nv_bfloat162*>(sp+BN+col);
                sv[0]=__low2float(s2);sv[1]=__high2float(s2);bv[0]=__low2float(b2);bv[1]=__high2float(b2);
            }
#pragma unroll
            for(int e=0;e<4;++e)acc[j][e]=__fmaf_rn(xv[e>>1],bv[e&1],__fmaf_rn(d[j][e],sv[e&1],acc[j][e]));
        }
    }
    wait<0>();__syncthreads();auto* partial=reinterpret_cast<float*>(buf);
    if(slice!=0){
#pragma unroll
        for(int j=0;j<NT;++j)
#pragma unroll
            for(int e=0;e<4;++e)partial[((slice*NT+j)*4+e)*128+tid]=acc[j][e];
    }
    __syncthreads();
    if(slice==0){
        for(int peer=1;peer<Slices;++peer)
#pragma unroll
            for(int j=0;j<NT;++j)
#pragma unroll
                for(int e=0;e<4;++e)acc[j][e]=acc[j][e]+partial[((peer*NT+j)*4+e)*128+tid];
#pragma unroll
        for(int j=0;j<NT;++j)
#pragma unroll
            for(int h=0;h<2;++h){const int r=(lane>>2)+h*8,c=n0+wn*(BN/4)+j*8+(lane&3)*2;
                if(r<M&&c+1<p.n)*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.out)+size_t(r)*p.n+c)=__floats2bfloat162_rn(acc[j][2*h],acc[j][2*h+1]);
            }
    }
}
}

extern "C" __global__ void __launch_bounds__(512) tf_qmm_output_stream16(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_qmm_output_stream::body<0>(x,xs,parts,M,K,ldx);
}
// tf_qmm_output_stream16 on a dictionary-coded part: the same loads, MMA chains, fmas and slice-ordered sum.
extern "C" __global__ void __launch_bounds__(512) tf_qmm_output_stream16_dict(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    if(K==6144)tf_qmm_output_stream::body<4,4,0,true>(x,xs,parts,M,K,ldx);
    else tf_qmm_output_stream::body<0,4,0,true>(x,xs,parts,M,K,ldx);
}
extern "C" __global__ void __launch_bounds__(512) tf_qmm_output_down128_dict(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_qmm_output_stream::body<0,4,0,true,128>(x,xs,parts,M,K,ldx);
}
extern "C" __global__ void __launch_bounds__(512) tf_qmm_output_prefetch16(const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,int M,int K,int ldx){
    tf_qmm_output_stream::body<4>(x,xs,parts,M,K,ldx);
}
