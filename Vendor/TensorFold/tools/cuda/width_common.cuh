#pragma once
// Exact row-width and weight-layout screen over checkpoint projection groups.
#include "../../zig/kernels/cuda/qmm_group.cu"
#include "../../zig/kernels/cuda/affine4_pack.cu"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#define CU(call) do { auto e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(2); } } while(0)
template<class T> T* alloc(size_t n) { T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p; }
template<class T> T* load(std::string path,size_t n) {
 std::vector<T> h(n);std::ifstream f(path,std::ios::binary);
 if(!f.read(reinterpret_cast<char*>(h.data()),n*sizeof(T)))exit(3);
 auto p=alloc<T>(n);CU(cudaMemcpy(p,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
__global__ void repack(const uint32_t* w,uint32_t* out,int kg,int tiles) {
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=size_t(tiles)*kg*512)return;
 const size_t tile=i/(kg*512),group=i/512%kg,word=i%512;
 out[(group*tiles+tile)*512+word]=w[i];
}
__global__ void flush_read(const uint4* p,unsigned* sums,size_t n) {
 const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;unsigned s=0;
 for(size_t i=tid;i<n;i+=size_t(gridDim.x)*blockDim.x) { auto v=p[i];s+=v.x+v.y+v.z+v.w; }
 sums[tid]=s;
}
