// PTILE: the crown persistent prompt schedules (qmm_prompt_persist*.cuh) running the tf_qmm_pipe3 bodies at
// STAGES=3. Bit-safety by construction: the tile assignment, the rotation, the short-tile cuts and every
// accumulation loop are the crown persistent kernels'; only the cp.async pipeline depth differs (3 stages of
// the same stage bytes instead of 2, so two more groups of weights stay in flight). Plain extern "C" entry
// names (no C++ mangling) so the Zig side resolves them by string.
//
// Included after qmm_prompt_rows2.cuh / qmm_prompt_rows3.cuh (tf_qmm_rows0 / tf_qmm_rowsdeep0 short_tile).
#pragma once

namespace tf_qmm_persist3 {
using namespace tf_qmm_prefill;

template<int BM,int WM,bool MLP,bool SPLIT,bool DEEP,int STAGES=3>
__device__ __forceinline__ void kernel_body(const __nv_bfloat16* __restrict__ x,
        const ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    constexpr int BN=128;
    for(int work=blockIdx.x;work<total;work+=gridDim.x){
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
            if(short_tile<BM,WM>(M-at.x))
                tf_qmm_pipe3::mlp_tile3<64,BM-16*WM,WM,4,STAGES>(x,parts,act,M,K,ldx,at.x,at.y);
            else tf_qmm_pipe3::mlp_tile3<64,BM,WM,4,STAGES>(x,parts,act,M,K,ldx,at.x,at.y);
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
                    if(DEEP ? tf_qmm_rowsdeep0::short_tile<BM,WM>(M-at.x) : tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                        tf_qmm_pipe3::tile3<64,BM-16*WM,BN/2,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,full_cols+at.y);
                    else tf_qmm_pipe3::tile3<64,BM,BN/2,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,full_cols+at.y);
                }else{
                    const int2 at=tile_of(local,M,full_cols,BM,BN,group);
                    if(DEEP ? tf_qmm_rowsdeep0::short_tile<BM,WM>(M-at.x) : tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                        tf_qmm_pipe3::tile3<64,BM-16*WM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                    else tf_qmm_pipe3::tile3<64,BM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                }
            }else{
                const int2 at=tile_of(local,M,P.n,BM,BN,group);
                if(DEEP ? tf_qmm_rowsdeep0::short_tile<BM,WM>(M-at.x) : tf_qmm_rows0::short_tile<BM,WM>(M-at.x))
                    tf_qmm_pipe3::tile3<64,BM-16*WM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
                else tf_qmm_pipe3::tile3<64,BM,BN,WM,4,STAGES>(x,P.w,P.scales,P.biases,P.out,P.n,P.npad,M,K,ldx,at.x,at.y);
            }
        }
        // A faster warp cannot reuse shared panels while a sibling still
        // reads the preceding tile. Each native tile already waits for copies.
        __syncthreads();
    }
}

} // namespace tf_qmm_persist3

extern "C" __global__ void __launch_bounds__(128) tf_qmm_persist3_mlp(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    tf_qmm_persist3::kernel_body<128,1,true,false,false>(x,parts,act,M,K,ldx,group,full_cols,total,rotate);
}

// The rows2 cut (tf_qmm_rows0::short_tile): native rows (SPLIT=false) and split last panels (SPLIT=true).
extern "C" __global__ void __launch_bounds__(128) tf_qmm_persist3_rows2(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    tf_qmm_persist3::kernel_body<128,1,false,false,false>(x,parts,act,M,K,ldx,group,full_cols,total,rotate);
}

extern "C" __global__ void __launch_bounds__(128) tf_qmm_persist3_rows2s(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    tf_qmm_persist3::kernel_body<128,1,false,true,false>(x,parts,act,M,K,ldx,group,full_cols,total,rotate);
}

// The rows3 cut (tf_qmm_rowsdeep0::short_tile), native rows.
extern "C" __global__ void __launch_bounds__(128) tf_qmm_persist3_rows3(const __nv_bfloat16* __restrict__ x,
        const __grid_constant__ tf_qmm_prefill::ManyParts parts,__nv_bfloat16* __restrict__ act,
        int M,int K,int ldx,int group,int full_cols,int total,int rotate) {
    tf_qmm_persist3::kernel_body<128,1,false,false,true>(x,parts,act,M,K,ldx,group,full_cols,total,rotate);
}
