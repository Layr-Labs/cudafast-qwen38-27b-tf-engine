#include "width_pair_helper.cuh"
#include "width_single.cuh"
template<int BM,int BN,int S,int WM=1,int WN=4>__global__ void __launch_bounds__(128) single(
 const __nv_bfloat16* x,const float* xs,const __grid_constant__ tf_qmm_group::Parts parts,
 __nv_bfloat16* act,float* sums,int M,int K,int ldx,int rows_t){
 tf_width_single::mlp_pair_body<S,BN,true,BM,WM,WN>(x,xs,parts,act,sums,M,K,ldx,rows_t);
}
template<int BM,int BN,int S,int WM=1,int WN=4>void invoke_single(cudaStream_t stream,tf_qmm_group::Parts parts,
 const __nv_bfloat16* x,const float* xs,__nv_bfloat16* act,float* sums,int rows,int k){
 const int rowtiles=(rows+BM-1)/BM,tiles=parts.p[0].n/(BN/2);
 constexpr int shared=qmm_frag::LaneTile<64,BM,BN,WM,WN,S>::SMEM;
 // Set opt-in capacity outside graph capture in main.
 parts.p[0].tiles=parts.p[1].tiles=tiles;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t config{};config.gridDim=dim3(rowtiles*tiles);config.blockDim=dim3(128);config.dynamicSmemBytes=shared;
 config.stream=stream;config.attrs=&attr;config.numAttrs=1;
 CU(cudaLaunchKernelEx(&config,single<BM,BN,S,WM,WN>,x,xs,parts,act,sums,rows,k,k,rowtiles));
}

int main(int argc,char** argv) {
 if(argc!=3)return 1;
 constexpr int n=17408,k=5120;
 cudaStream_t stream;CU(cudaStreamCreate(&stream));
 tf_qmm_group::Parts parts{},packed{};parts.count=packed.count=2;
 for(int i=0;i<2;++i){
  size_t nw=size_t(n)*k/8,ns=size_t(n)*k/64;
  auto w=load<uint32_t>(std::string(argv[1+i])+".weight",nw),pw=alloc<uint32_t>(nw),kw=alloc<uint32_t>(nw);
  auto s=load<uint16_t>(std::string(argv[1+i])+".scales",ns),b=load<uint16_t>(std::string(argv[1+i])+".biases",ns);
  auto ps=alloc<uint16_t>(ns),pb=alloc<uint16_t>(ns);
  tf_pack_dense<<<(nw+255)/256,256>>>(w,n,k/8,k/64,pw,nw);
  tf_transpose_pad16<<<(ns+255)/256,256>>>(s,n,k/64,n,ps);tf_transpose_pad16<<<(ns+255)/256,256>>>(b,n,k/64,n,pb);
  repack<<<(nw+255)/256,256>>>(pw,kw,k/64,n/64);CU(cudaDeviceSynchronize());
  parts.p[i]={pw,reinterpret_cast<__nv_bfloat16*>(ps),reinterpret_cast<__nv_bfloat16*>(pb),nullptr,n,n,1,n/64,0};
  packed.p[i]=parts.p[i];packed.p[i].w=kw;CU(cudaFree(w));CU(cudaFree(s));CU(cudaFree(b));
 }
 auto x=alloc<__nv_bfloat16>(32*k),act=alloc<__nv_bfloat16>(32*n);auto xs=alloc<float>(32*k/64),sums=alloc<float>(32*n/64);
 std::vector<__nv_bfloat16> hx(32*k);std::vector<float> hs(32*k/64);unsigned rng=19;
 for(size_t i=0;i<hx.size();++i){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;hx[i]=__float2bfloat16_rn((int(rng&65535)-32768)/16384.f);hs[i/64]+=__bfloat162float(hx[i]);}
 CU(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));CU(cudaMemcpy(xs,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
 for(auto f:{single<32,256,4>,single<32,512,3>})CU(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,68000));
 auto invoke=[&](int mode,int rows){switch(mode){
 case 0:case 1:invoke_pair<16,64,8,false>(stream,parts,x,xs,act,sums,rows,k);break;
 case 2:invoke_single<16,512,2,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 3:invoke_single<32,128,2,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 4:invoke_single<32,128,3,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 5:invoke_single<32,128,4,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 6:invoke_single<32,256,2,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 7:invoke_single<32,256,3,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 8:invoke_single<32,256,4,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 9:invoke_single<32,512,2,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 10:invoke_single<32,512,3,1,4>(stream,packed,x,xs,act,sums,rows,k);break;
 case 11:invoke_single<32,128,3,2,2>(stream,packed,x,xs,act,sums,rows,k);break;
 case 12:invoke_single<32,256,2,2,2>(stream,packed,x,xs,act,sums,rows,k);break;
 case 13:invoke_single<32,256,3,2,2>(stream,packed,x,xs,act,sums,rows,k);break;
 }};
 std::vector<uint16_t> want(32*n),got(32*n);std::vector<float> swant(32*n/64),sgot(swant.size());
 auto collect=[&](auto& h,auto& sh){CU(cudaMemcpy(h.data(),act,h.size()*2,cudaMemcpyDeviceToHost));CU(cudaMemcpy(sh.data(),sums,sh.size()*4,cudaMemcpyDeviceToHost));};
 for(int rows:{1,16,17,24,31,32}){
  invoke(0,rows);CU(cudaStreamSynchronize(stream));collect(want,swant);
  for(int mode=1;mode<14;++mode){
   CU(cudaMemsetAsync(act,0x7f,32*n*2,stream));CU(cudaMemsetAsync(sums,0x7f,32*n/64*4,stream));
   invoke(mode,rows);CU(cudaStreamSynchronize(stream));collect(got,sgot);
   if(!std::equal(want.begin(),want.begin()+rows*n,got.begin()) ||
      !std::equal(swant.begin(),swant.begin()+rows*n/64,sgot.begin())){fprintf(stderr,"pair mismatch rows%d mode%d\n",rows,mode);return 4;}
  }
 }
 cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));cudaGraphExec_t graphs[14]{};
 for(int mode=0;mode<14;++mode){
  cudaGraph_t graph;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));CU(cudaEventRecordWithFlags(begin,stream,cudaEventRecordExternal));
  invoke(mode,mode==1?16:32);CU(cudaEventRecordWithFlags(end,stream,cudaEventRecordExternal));
  CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&graphs[mode],graph,0));CU(cudaGraphDestroy(graph));
 }
 invoke(0,32);CU(cudaStreamSynchronize(stream));collect(want,swant);
 auto flush=alloc<uint4>((128<<20)/16);auto scratch=alloc<unsigned>(192*256);CU(cudaMemset(flush,1,128<<20));
 for(int rep=-2;rep<13;++rep)for(int j=0;j<14;++j){
  int mode=(j+rep+14)%14;flush_read<<<192,256,0,stream>>>(flush,scratch,(128<<20)/16);
  CU(cudaGraphLaunch(graphs[mode],stream));CU(cudaStreamSynchronize(stream));float ms;CU(cudaEventElapsedTime(&ms,begin,end));collect(got,sgot);
  int rows=mode==1?16:32;
  if(!std::equal(want.begin(),want.begin()+rows*n,got.begin()) || !std::equal(swant.begin(),swant.begin()+rows*n/64,sgot.begin()))return 5;
  if(rep>=0)printf("{\"mode\":%d,\"rep\":%d,\"ms\":%.9f,\"exact\":true}\n",mode,rep,ms);
 }
}
