// One consumed row, with the native per-output arithmetic and K16 MMA chain.
#pragma once
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_terminal_many(
 const __nv_bfloat16* __restrict__ x,const __grid_constant__ tf_qmm_prefill::ManyParts parts,int M,int K,int ldx,int group){
 int q=0;
#pragma unroll
 for(int i=1;i<tf_qmm_prefill::MANY;i++)if(i<parts.count && int(blockIdx.x)>=parts.p[i].first)q=i;
 const auto& p=parts.p[q];
 tf_qmm_prefill::prompt_tile<64,16,128,1,4,2>(x,p.w,p.scales,p.biases,p.out,p.n,p.npad,M,K,ldx,0,(int(blockIdx.x)-p.first)*128);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmm_prompt_terminal_mlp(
 const __nv_bfloat16* __restrict__ x,const __grid_constant__ tf_qmm_prefill::ManyParts parts,__nv_bfloat16* __restrict__ act,int M,int K,int ldx,int group){
 tf_qmm_prefill::prompt_mlp_tile<64,16,1,4,2>(x,parts,act,M,K,ldx,0,int(blockIdx.x)*64);
}
