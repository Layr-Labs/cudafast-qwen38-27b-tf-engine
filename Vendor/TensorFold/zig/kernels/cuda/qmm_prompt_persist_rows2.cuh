// Persistent row/column tile scheduling; each output retains the native MMA chain.
#pragma once
namespace tf_qmm_persistent_rows2 {
using namespace tf_qmm_prefill;

// The queue entry point's work counter: module storage of its own, reset by CTA 0 before the grid sync.
__device__ unsigned queue_counter;

template<int BM,int WM,bool MLP,bool SPLIT>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ x,
        const ManyParts& parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate,bool queue) {
    constexpr int BN=128, STAGES=2;
    // Dynamic mode: a cooperative grid claims ascending work numbers from queue_counter.
    __shared__ int next_work;
    if(queue){
        if(blockIdx.x==0 && threadIdx.x==0)queue_counter=0u;
        cooperative_groups::this_grid().sync();
    }
    // Thread 0 claims; the barrier publishes the claim and orders it after the previous tile's last shared read.
    auto claim=[&]()->int{
        if(threadIdx.x==0)next_work=(int)atomicAdd(&queue_counter,1u);
        __syncthreads();
        return next_work;
    };
    for(int work=queue?claim():(int)blockIdx.x;work<total;work=queue?claim():work+(int)gridDim.x){
        int tile=work;
        // Rotate row assignment between complete cohorts so short rows do
        // not remain assigned to the same persistent workers each iteration.
        const int rows=(M+BM-1)/BM;
        if(rotate && group==rows && total%rows==0 && (int)gridDim.x%rows==0){
            int base=work/rows*rows;
            tile=base+(work%rows+(work/gridDim.x))%rows;
        }
        if constexpr(MLP){
            const int2 at=tile_of(tile,M,parts.p[0].n,BM,64,group);
            if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                tf_qmm_rows0::prompt_mlp_tile<64,BM-16*WM,WM,4,STAGES>(x,parts,act,M,K,ldx,at.x,at.y);
            else tf_qmm_rows0::prompt_mlp_tile<64,BM,WM,4,STAGES>(x,parts,act,M,K,ldx,at.x,at.y);
        }else{
            int q=0;
#pragma unroll
            for(int i=1;i<MANY;i++)if(i<parts.count && tile>=parts.p[i].first)q=i;
            const ManyPart& P=parts.p[q];
            const int local=tile-P.first;
            const int full=rows*(full_cols/BN);
            if constexpr(SPLIT){
                if(local>=full){
                    const int2 at=tile_of(local-full,M,P.n-full_cols,BM,BN/2,group);
                    if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                        tf_qmm_rows0::prompt_tile<64,BM-16*WM,BN/2,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,full_cols+at.y);
                    else tf_qmm_rows0::prompt_tile<64,BM,BN/2,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,full_cols+at.y);
                }else{
                    const int2 at=tile_of(local,M,full_cols,BM,BN,group);
                    if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                        tf_qmm_rows0::prompt_tile<64,BM-16*WM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                    else tf_qmm_rows0::prompt_tile<64,BM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                }
            }else{
                const int2 at=tile_of(local,M,P.n,BM,BN,group);
                if(tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                    tf_qmm_rows0::prompt_tile<64,BM-16*WM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                else tf_qmm_rows0::prompt_tile<64,BM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
            }
        }
        // A faster warp cannot reuse shared panels while a sibling still
        // reads the preceding tile. Each native tile already waits for copies.
        __syncthreads();
    }
}

template<int BM,int WM,bool MLP,bool SPLIT>
__global__ void __launch_bounds__(WM*128) kernel(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate){
    body<BM,WM,MLP,SPLIT>(x,parts,act,M,K,ldx,group,full_cols,total,rotate,false);
}
} // namespace tf_qmm_persistent_rows2
