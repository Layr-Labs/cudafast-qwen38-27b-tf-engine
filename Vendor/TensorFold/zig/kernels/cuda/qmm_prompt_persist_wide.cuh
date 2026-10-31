// Persistent prompt schedules on a 128 x 256 tile: eight warps of the same 128 x 32 warp tile, three stages, one CTA an SM.
// Each output keeps its native MMA chain: the warp tile, group order, K16 steps and affine fmas of the 128 x 128 tile's,
// so every bit is the 4-warp schedule's; only how many columns a CTA covers (and how far ahead it loads) changes.
#pragma once
namespace tf_qmm_persistent_wide {
using namespace tf_qmm_prefill;

template<int GS,int BM,int BN,int WM,int WN,int STAGES,bool DEEP>
__device__ __forceinline__ void tile_body(const __nv_bfloat16* __restrict__ x,const ManyPart& P,int M,int K,int ldx,int m0,int n0){
    if constexpr(DEEP) tf_qmm_rowsdeep0::prompt_tile<GS,BM,BN,WM,WN,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,m0,n0);
    else tf_qmm_rows0::prompt_tile<GS,BM,BN,WM,WN,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,m0,n0);
}

template<int BM,int WM,bool SPLIT,bool DEEP>
__global__ void __launch_bounds__(WM*256,1) kernel(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    constexpr int BN=256, WN=8, STAGES=3;
    for(int work=blockIdx.x;work<total;work+=gridDim.x){
        int tile=work;
        const int rows=(M+BM-1)/BM;
        if(rotate && group==rows && total%rows==0 && (int)gridDim.x%rows==0){
            int base=work/rows*rows;
            tile=base+(work%rows+(work/gridDim.x))%rows;
        }
        int q=0;
#pragma unroll
        for(int i=1;i<MANY;i++)if(i<parts.count && tile>=parts.p[i].first)q=i;
        const ManyPart& P=parts.p[q];
        const int local=tile-P.first;
        const int full=rows*(full_cols/BN);
        bool tail=false;
        int2 at;
        if constexpr(SPLIT){
            tail=local>=full;
            at=tail?tile_of(local-full,M,P.n-full_cols,BM,BN/2,group):tile_of(local,M,full_cols,BM,BN,group);
            if(tail)at.y+=full_cols;
        }else at=tile_of(local,M,P.n,BM,BN,group);
        // a short last row tile runs the full tile: its rows past M load as zeros and are not stored (a 112-row tile
        // would not split into whole input chunks over 256 threads)
        if(tail)tile_body<64,BM,BN/2,WM,WN,STAGES,DEEP>(x,P,M,K,ldx,at.x,at.y);
        else tile_body<64,BM,BN,WM,WN,STAGES,DEEP>(x,P,M,K,ldx,at.x,at.y);
        // a faster warp cannot reuse shared panels while a sibling still reads the preceding tile
        __syncthreads();
    }
}
} // namespace tf_qmm_persistent_wide
