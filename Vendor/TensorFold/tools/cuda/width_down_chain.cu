// Native fused gate/up/activation followed by alternative exact 32-row down projections.
#include "width_pair_helper.cuh"
template<int BN,int S,bool KM> __global__ void __launch_bounds__(128) down_kernel(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,
 int M,int K,int ldx,int rows_t,int C) {
 tf_qmm_kmajor::project<64,32,BN,1,4,S,false>(x,xs,parts,M,K,ldx,rows_t,C);
}
template<int BN,int S,bool KM> void down(cudaStream_t stream,tf_qmm_group::Part p,
 const __nv_bfloat16* act,const float* sums) {
 constexpr int shared=qmm_frag::LaneTile<64,32,BN,1,4,S>::SMEM;
 tf_qmm_group::Parts parts{};parts.count=1;p.tiles=(p.n+BN-1)/BN;parts.p[0]=p;
 cudaLaunchAttribute attrs[2]{};
 attrs[0].id=cudaLaunchAttributeClusterDimension;attrs[0].val.clusterDim={4,1,1};
 attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;attrs[1].val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(p.tiles*4);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=shared;
 cfg.stream=stream;cfg.attrs=attrs;cfg.numAttrs=2;
 if constexpr(KM) CU(cudaLaunchKernelEx(&cfg,down_kernel<BN,S,true>,act,sums,parts,32,17408,17408,1,4));
 else CU(cudaLaunchKernelEx(&cfg,tf_qmm_group::group_kernel<64,32,BN,1,4,S,false>,act,sums,parts,32,17408,17408,1,4));
}
__global__ void fill_inputs(__nv_bfloat16* x,float* xs,int phase) {
 const int i=blockIdx.x*64+threadIdx.x;
 x[i]=__float2bfloat16_rn((int((unsigned(i)*17u+unsigned(phase)*31u)%129u)-64)/128.f);
 __syncthreads();
 if(threadIdx.x==0){float sum=0;for(int j=0;j<64;++j)sum+=__bfloat162float(x[blockIdx.x*64+j]);xs[blockIdx.x]=sum;}
}
int main(int argc,char** argv) {
 if(argc!=4)return 1;
 constexpr int n=17408,k=5120,rows=32;
 cudaStream_t stream;CU(cudaStreamCreate(&stream));
 tf_qmm_group::Parts pair{};pair.count=2;tf_qmm_group::Part dp{},packed{};
 auto act=alloc<__nv_bfloat16>(rows*n),out=alloc<__nv_bfloat16>(rows*k);
 auto x=alloc<__nv_bfloat16>(rows*k);auto xs=alloc<float>(rows*k/64),sums=alloc<float>(rows*n/64);
 for(int i=0;i<3;++i){
  const int nn=i==2?k:n,kk=i==2?n:k;
  size_t nw=size_t(nn)*kk/8,ns=size_t(nn)*kk/64;
  auto w=load<uint32_t>(std::string(argv[1+i])+".weight",nw),pw=alloc<uint32_t>(nw);
  auto s=load<uint16_t>(std::string(argv[1+i])+".scales",ns),b=load<uint16_t>(std::string(argv[1+i])+".biases",ns);
  auto ps=alloc<uint16_t>(ns),pb=alloc<uint16_t>(ns);
  tf_pack_dense<<<(nw+255)/256,256>>>(w,nn,kk/8,kk/64,pw,nw);
  tf_transpose_pad16<<<(ns+255)/256,256>>>(s,nn,kk/64,nn,ps);tf_transpose_pad16<<<(ns+255)/256,256>>>(b,nn,kk/64,nn,pb);
  CU(cudaDeviceSynchronize());
  tf_qmm_group::Part p{pw,reinterpret_cast<__nv_bfloat16*>(ps),reinterpret_cast<__nv_bfloat16*>(pb),out,nn,nn,i==2?4:1,nn/64,0};
  if(i<2)pair.p[i]=p;
  else{dp=p;packed=p;auto kw=alloc<uint32_t>(nw);repack<<<(nw+255)/256,256>>>(pw,kw,kk/64,nn/64);packed.w=kw;CU(cudaDeviceSynchronize());}
  CU(cudaFree(w));CU(cudaFree(s));CU(cudaFree(b));
 }
 CU(cudaFuncSetAttribute(down_kernel<256,4,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,qmm_frag::LaneTile<64,32,256,1,4,4>::SMEM));
 auto run=[&](int mode){
  invoke_pair<16,64,8,false>(stream,pair,x,xs,act,sums,rows,k);
  switch(mode){
  case 0:down<64,5,false>(stream,dp,act,sums);break;
  case 1:down<256,2,true>(stream,packed,act,sums);break;
  case 2:down<128,3,true>(stream,packed,act,sums);break;
  case 3:down<128,4,true>(stream,packed,act,sums);break;
  case 4:down<64,3,true>(stream,packed,act,sums);break;
  case 5:down<64,5,true>(stream,packed,act,sums);break;
  case 6:down<128,3,false>(stream,dp,act,sums);break;
  case 7:down<128,4,false>(stream,dp,act,sums);break;
  case 8:down<256,2,false>(stream,dp,act,sums);break;
  case 9:down<128,2,true>(stream,packed,act,sums);break;
  case 10:down<256,3,true>(stream,packed,act,sums);break;
  case 11:down<256,4,true>(stream,packed,act,sums);break;
  case 12:down<128,5,true>(stream,packed,act,sums);break;
  }
 };
 auto fill=[&](int phase){fill_inputs<<<rows*k/64,64,0,stream>>>(x,xs,phase);};
 std::vector<unsigned char>want(rows*(n*2+n/64*4+k*2)),got(want.size());
 auto collect=[&](std::vector<unsigned char>& h){size_t at=0;auto get=[&](const void* p,size_t count){CU(cudaMemcpy(h.data()+at,p,count,cudaMemcpyDeviceToHost));at+=count;};get(act,rows*n*2);get(sums,rows*n/64*4);get(out,rows*k*2);};
 fill(1);run(0);CU(cudaStreamSynchronize(stream));collect(want);
 for(int mode=0;mode<13;++mode){
  cudaGraph_t graph;cudaGraphExec_t exec;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  fill(0);run(mode);fill(1);run(mode);CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&exec,graph,0));
  CU(cudaGraphLaunch(exec,stream));CU(cudaStreamSynchronize(stream));collect(got);
  if(got!=want){fprintf(stderr,"changing input mismatch %d\n",mode);return 4;}
  CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
 }
 cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));cudaGraphExec_t graphs[13]{};
 for(int mode=0;mode<13;++mode){
  cudaGraph_t graph;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  CU(cudaEventRecordWithFlags(begin,stream,cudaEventRecordExternal));run(mode);CU(cudaEventRecordWithFlags(end,stream,cudaEventRecordExternal));
  CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&graphs[mode],graph,0));CU(cudaGraphDestroy(graph));
 }
 auto flush=alloc<uint4>((128<<20)/16);auto scratch=alloc<unsigned>(192*256);CU(cudaMemset(flush,1,128<<20));
 for(int rep=-2;rep<13;++rep)for(int j=0;j<13;++j){
  int mode=(j+rep+13)%13;flush_read<<<192,256,0,stream>>>(flush,scratch,(128<<20)/16);
  CU(cudaGraphLaunch(graphs[mode],stream));CU(cudaStreamSynchronize(stream));float ms;CU(cudaEventElapsedTime(&ms,begin,end));collect(got);
  if(got!=want){fprintf(stderr,"timed mismatch %d\n",mode);return 5;}
  if(rep>=0)printf("{\"mode\":%d,\"rep\":%d,\"ms\":%.9f,\"exact\":true}\n",mode,rep,ms);
 }
 return 0;
}
