#pragma once
#include "width_common.cuh"
template<int BM,int BN,int S,bool KM> __global__ void __launch_bounds__(128) pair_kernel(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,
 __nv_bfloat16* act,float* sums,int M,int K,int ldx,int rows_t) {
 tf_qmm_group::mlp_pair_body<S,BM,BN,KM>(x,xs,parts,act,sums,M,K,ldx,rows_t);
}
template<int BM,int BN,int S,bool KM> void invoke_pair(cudaStream_t stream,tf_qmm_group::Parts parts,
 const __nv_bfloat16* x,const float* xs,__nv_bfloat16* act,float* sums,int rows,int k) {
 cudaLaunchAttribute attrs[2]{};
 attrs[0].id=cudaLaunchAttributeClusterDimension;attrs[0].val.clusterDim={2,1,1};
 attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;attrs[1].val.programmaticStreamSerializationAllowed=1;
 const int rowtiles=(rows+BM-1)/BM;
 constexpr int shared=qmm_frag::LaneTile<64,BM,BN,1,4,S>::SMEM+BM*BN*2;
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(2*rowtiles*parts.p[0].n/BN);cfg.blockDim=dim3(128);
 cfg.dynamicSmemBytes=shared;cfg.stream=stream;cfg.attrs=attrs;cfg.numAttrs=2;
 CU(cudaLaunchKernelEx(&cfg,pair_kernel<BM,BN,S,KM>,x,xs,parts,act,sums,rows,k,k,rowtiles));
}
