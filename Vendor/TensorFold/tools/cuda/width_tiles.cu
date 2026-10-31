#include "width_common.cuh"
template<int BM,int BN,int S,bool KM,int WM=1,int WN=4> __global__ void __launch_bounds__(128) alternative(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,
 int M,int K,int ldx,int rows_t,int C) {
 tf_qmm_kmajor::project<64,BM,BN,WM,WN,S,false>(x,xs,parts,M,K,ldx,rows_t,C);
}
template<class F> void launch(F fn,cudaStream_t stream,int blocks,int shared,int cluster,
 const __nv_bfloat16* x,const float* xs,tf_qmm_group::Parts parts,int rows,int k) {
 cudaLaunchAttribute a[2]{};unsigned count=0;
 if(cluster>1){a[count].id=cudaLaunchAttributeClusterDimension;a[count++].val.clusterDim={unsigned(cluster),1,1};}
 a[count].id=cudaLaunchAttributeProgrammaticStreamSerialization;a[count++].val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t c{};c.gridDim=dim3(blocks);c.blockDim=dim3(128);c.dynamicSmemBytes=shared;c.stream=stream;c.attrs=a;c.numAttrs=count;
 CU(cudaLaunchKernelEx(&c,fn,x,xs,parts,rows,k,k,1,cluster));
}
template<int BM,int BN,int S,bool KM,int WM=1,int WN=4> void run(cudaStream_t stream,tf_qmm_group::Parts parts,
 const __nv_bfloat16* x,const float* xs,int k,bool grouped) {
 for(int batch=1;batch<=8;batch*=2) {
  tf_qmm_group::Parts group{};int blocks=0;
  auto fire=[&]() {
   if(!group.count)return;
   constexpr int shared=qmm_frag::LaneTile<64,BM,BN,WM,WN,S>::SMEM;
   if constexpr(KM) launch(alternative<BM,BN,S,true,WM,WN>,stream,blocks*batch,shared,batch,x,xs,group,BM,k);
   else launch(tf_qmm_group::group_kernel<64,BM,BN,WM,WN,S,false>,stream,blocks*batch,shared,batch,x,xs,group,BM,k);
  };
  for(int i=0;i<parts.count;++i)if(parts.p[i].sk==batch) {
   auto p=parts.p[i];p.tiles=(p.n+BN-1)/BN;p.first=blocks;group.p[group.count++]=p;blocks+=p.tiles;
   if(!grouped){fire();group={};blocks=0;}
  }
  fire();
 }
}
int main(int argc,char** argv) {
 if(argc<5 || (argc-1)%4)return 1;
 cudaStream_t stream;CU(cudaStreamCreate(&stream));
 tf_qmm_group::Parts parts{},packed{};parts.count=(argc-1)/4;
 const int k=atoi(argv[3]);
 std::vector<__nv_bfloat16*> outputs;
 for(int i=0;i<parts.count;++i) {
  std::string prefix=argv[1+i*4];int n=atoi(argv[2+i*4]),ki=atoi(argv[3+i*4]),sk=atoi(argv[4+i*4]);
  if(ki!=k)return 1;
  const int npad=(n+255)/256*256;
  size_t nw=size_t(n)*k/8,ns=size_t(n)*k/64;
  const size_t npw=size_t(npad)*k/8,nps=size_t(npad)*k/64;
  auto w=load<uint32_t>(prefix+".weight",nw),pw=alloc<uint32_t>(npw),kw=alloc<uint32_t>(npw);
  auto s=load<uint16_t>(prefix+".scales",ns),b=load<uint16_t>(prefix+".biases",ns);
  auto ps=alloc<uint16_t>(nps),pb=alloc<uint16_t>(nps),out=alloc<uint16_t>(32*n);
  tf_pack_dense<<<(npw+255)/256,256>>>(w,n,k/8,k/64,pw,npw);
  tf_transpose_pad16<<<(nps+255)/256,256>>>(s,n,k/64,npad,ps);tf_transpose_pad16<<<(nps+255)/256,256>>>(b,n,k/64,npad,pb);
  repack<<<(npw+255)/256,256>>>(pw,kw,k/64,npad/64);CU(cudaDeviceSynchronize());
  parts.p[i]={pw,reinterpret_cast<__nv_bfloat16*>(ps),reinterpret_cast<__nv_bfloat16*>(pb),out,n,npad,sk,(n+63)/64,0};
  packed.p[i]=parts.p[i];packed.p[i].w=kw;outputs.push_back(reinterpret_cast<__nv_bfloat16*>(out));
  CU(cudaFree(w));CU(cudaFree(s));CU(cudaFree(b));
 }
 packed.count=parts.count;
 auto x=alloc<__nv_bfloat16>(32*k);auto xs=alloc<float>(32*k/64);
 std::vector<__nv_bfloat16> hx(32*k);std::vector<float> hs(32*k/64);
 unsigned rng=337;
 for(size_t i=0;i<hx.size();++i){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;hx[i]=__float2bfloat16_rn((int(rng&65535)-32768)/16384.f);hs[i/64]+=__bfloat162float(hx[i]);}
 CU(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));CU(cudaMemcpy(xs,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
 auto invoke=[&](int mode){switch(mode){
 case 0:run<16,64,8,false>(stream,parts,x,xs,k,true);break;
 case 1:run<32,64,5,false>(stream,parts,x,xs,k,false);break;
 case 2:run<32,64,5,false>(stream,parts,x,xs,k,true);break;
 case 3:run<32,128,2,false>(stream,parts,x,xs,k,true);break;
 case 4:run<32,128,3,false>(stream,parts,x,xs,k,true);break;
 case 5:run<32,128,2,true>(stream,packed,x,xs,k,true);break;
 case 6:run<32,128,3,true>(stream,packed,x,xs,k,true);break;
 case 7:run<32,64,3,true>(stream,packed,x,xs,k,true);break;
 case 8:run<32,64,3,false>(stream,parts,x,xs,k,true);break;
 case 9:run<32,128,4,true>(stream,packed,x,xs,k,true);break;
 case 10:run<32,256,2,true>(stream,packed,x,xs,k,true);break;
 }};
 // A pair of native m16 calls is the byte reference for every m32 candidate.
 auto reference=parts;
 run<16,64,8,false>(stream,reference,x,xs,k,true);
 for(int i=0;i<reference.count;++i)reference.p[i].out=outputs[i]+16*reference.p[i].n;
 run<16,64,8,false>(stream,reference,x+16*k,xs+16*k/64,k,true);CU(cudaStreamSynchronize(stream));
 std::vector<std::vector<uint16_t>> want(parts.count),got(parts.count);
 for(int i=0;i<parts.count;++i){want[i].resize(32*parts.p[i].n);got[i].resize(want[i].size());CU(cudaMemcpy(want[i].data(),outputs[i],want[i].size()*2,cudaMemcpyDeviceToHost));}
 cudaEvent_t start,end;CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));cudaGraphExec_t graphs[11]{};
 for(int mode=0;mode<11;++mode){
  cudaGraph_t graph;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  CU(cudaEventRecordWithFlags(start,stream,cudaEventRecordExternal));invoke(mode);CU(cudaEventRecordWithFlags(end,stream,cudaEventRecordExternal));
  CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&graphs[mode],graph,0));CU(cudaGraphDestroy(graph));
 }
 auto flush=alloc<uint4>((128<<20)/16);auto sums=alloc<unsigned>(192*256);CU(cudaMemset(flush,1,128<<20));
 for(int rep=-2;rep<13;++rep)for(int j=0;j<11;++j){
  int mode=(j+rep+11)%11;
  for(int i=0;i<parts.count;++i)CU(cudaMemsetAsync(outputs[i],0x7f,want[i].size()*2,stream));
  flush_read<<<192,256,0,stream>>>(flush,sums,(128<<20)/16);
  CU(cudaGraphLaunch(graphs[mode],stream));CU(cudaStreamSynchronize(stream));float ms;CU(cudaEventElapsedTime(&ms,start,end));
  for(int i=0;i<parts.count;++i){CU(cudaMemcpy(got[i].data(),outputs[i],got[i].size()*2,cudaMemcpyDeviceToHost));
   size_t count=size_t(mode==0?16:32)*parts.p[i].n;
   if(!std::equal(want[i].begin(),want[i].begin()+count,got[i].begin())){fprintf(stderr,"mismatch mode%d part%d rep%d\n",mode,i,rep);return 4;}
  }
  if(rep>=0)printf("{\"mode\":%d,\"rep\":%d,\"ms\":%.9f,\"exact\":true}\n",mode,rep,ms);
 }
 return 0;
}
